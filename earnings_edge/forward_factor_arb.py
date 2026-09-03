"""Forward Factor Arbitrage Strategy

Documentation & Setup:
This strategy is a pure relative-value volatility arbitrage. It does not try to guess 
the historical magnitude of an earnings gap. Instead, it looks for structural mispricing 
in the term structure of implied volatility. 

The Edge:
Often, near-term options (Front Expiration, 30-60 DTE) are structurally overbid due to 
hedging demand, while slightly further out options (Back Expiration, 60-90 DTE) revert 
too quickly to a lower "background" mean. By analyzing the Forward Volatility (the implied 
volatility strictly between T1 and T2), we can calculate the Forward Factor.
If the Front IV is significantly richer than the Forward Volatility (Factor > 1.1), the 
calendar spread is mathematically underpriced. 

Crucially, because an upcoming earnings event naturally inflates the Front IV, we must 
strip out the implied earnings premium to calculate an "Ex-Earnings" IV and Forward Factor.
Only if the Ex-Earnings Forward Factor is > 1.1 do we consider the calendar spread truly 
underpriced on a relative-value basis.

Execution:
1. Scan Front (30-60 DTE) and Back (60-90 DTE) expirations.
2. Filter for Ex-Earnings Forward Factor > 1.1.
3. Setup a limit order ladder. Start the ladder at a debit price representing a theoretical
   Factor of 1.5, and slowly step down (concede price) to a Factor of 1.25.
4. If filled, hold the calendar spread until the Front Expiry (ScheduledExit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from .option_math import black_scholes_price, implied_volatility

RISK_FREE_RATE = 0.045
ET = timezone(timedelta(hours=-4), name="EDT")

def occ_parse(symbol: str) -> dict:
    root = symbol[:-15]
    yy, mm, dd = symbol[-15:-13], symbol[-13:-11], symbol[-11:-9]
    return {
        "root": root,
        "expiry": date(2000 + int(yy), int(mm), int(dd)),
        "option_type": "call" if symbol[-9] == "C" else "put",
        "strike": int(symbol[-8:]) / 1000.0,
    }

def forward_volatility(iv_near: float, T1: float, iv_far: float, T2: float) -> Optional[float]:
    if T2 <= T1 or iv_near <= 0 or iv_far <= 0:
        return None
    var = iv_far ** 2 * T2 - iv_near ** 2 * T1
    if var <= 0:
        return None
    return math.sqrt(var / (T2 - T1))

def calculate_forward_factor(iv_near: float, fwd_vol: float) -> Optional[float]:
    if fwd_vol <= 0:
        return None
    return (iv_near / fwd_vol) - 1.0

def required_near_iv_for_factor(fwd_vol: float, target_factor: float) -> float:
    """Calculates what the near IV should be to hit a specific forward factor."""
    return fwd_vol * (1.0 + target_factor)

def target_debit_for_factor(
    far_price: float,
    spot: float,
    strike: float,
    T1: float,
    fwd_vol: float,
    target_factor: float,
    r: float = RISK_FREE_RATE,
) -> Optional[float]:
    """Max calendar debit consistent with a specific Forward Factor."""
    iv_star = required_near_iv_for_factor(fwd_vol, target_factor)
    near_star = black_scholes_price(spot, strike, T1, r, iv_star, "call")
    if not math.isfinite(near_star):
        return None
    return far_price - near_star

# Ex-Earnings Math requires stripping the event variance from the near leg
def calculate_ex_earnings_iv(iv_near: float, T1: float, hist_rms_move: float) -> Optional[float]:
    """Strip the historical earnings variance from the near leg."""
    total_var = (iv_near ** 2) * T1
    event_var = hist_rms_move ** 2
    ex_event_var = total_var - event_var
    if ex_event_var <= 0:
        return None
    return math.sqrt(ex_event_var / T1)


def build_candidate(alpaca, ticker: str, *, today: Optional[date] = None):
    """Scan the option chain and evaluate if the ticker qualifies for Forward Factor Arbitrage."""
    from .fwd_factor_ladder import occ_parse, _pick_pair_tenor, CalendarCandidate, _reject, hist_rms_move
    from .fwd_factor import combo_debit
    from .db.repositories import snapshots_next_earnings_date
    
    if today is None:
        today = datetime.now(timezone.utc).date()
        
    next_earnings_str = snapshots_next_earnings_date(ticker, today=today.isoformat()) or ""
    try:
        next_earnings_date = date.fromisoformat(next_earnings_str) if next_earnings_str else None
    except ValueError:
        next_earnings_date = None
        
    fake_ed = next_earnings_date or date.min
        
    chain = alpaca.get_options_chain_snapshots(ticker)
    if not chain:
        return _reject(ticker, fake_ed, 0.0, "no chain")
        
    spot = alpaca.get_stock_latest_trade(ticker)
    if not spot:
        return _reject(ticker, fake_ed, 0.0, "no spot")
    
    t1, t2 = _pick_pair_tenor(chain, spot, today)
    if not t1 or not t2:
        return _reject(ticker, fake_ed, spot, "no T1/T2 pair")
        
    T1 = (t1["expiry"] - today).days / 365.0
    T2 = (t2["expiry"] - today).days / 365.0
    
    q1, q2 = chain[t1["symbol"]], chain[t2["symbol"]]
    mid1 = (q1["bid"] + q1["ask"]) / 2.0
    mid2 = (q2["bid"] + q2["ask"]) / 2.0
    iv1 = implied_volatility(mid1, spot, t1["strike"], T1, RISK_FREE_RATE, "call")
    iv2 = implied_volatility(mid2, spot, t2["strike"], T2, RISK_FREE_RATE, "call")
    
    if not iv1 or not iv2 or math.isnan(iv1) or math.isnan(iv2):
        return _reject(ticker, fake_ed, spot, "no IV")
        
    # Conditional ex-earnings correction: the ONLY earnings use. Strip the
    # event variance from the near-leg IV when an event falls strictly
    # inside T1; otherwise price off the raw near-leg IV.
    event_inside_t1 = bool(next_earnings_date and today < next_earnings_date <= t1["expiry"])
    ex_iv1 = iv1
    rms = 0.0
    
    if event_inside_t1:
        # Hist-RMS gate ONLY — no backfill here. ensure_hist_moves hits
        # Yahoo/LSE synchronously; calling it per-ticker inside the bot's
        # async proposal loop starves the event loop (see the 2026-07-31
        # event-loop-starvation postmortem). Coverage repair for the arb
        # universe is a separate offline drive (warm_hist_coverage.py).
        rms, n_hist = hist_rms_move(ticker=ticker)
        if rms is None:
            return _reject(ticker, fake_ed, spot, "event inside T1, no hist rms")
            
        ex_iv_calc = calculate_ex_earnings_iv(iv1, T1, rms)
        if ex_iv_calc is None:
            return _reject(ticker, fake_ed, spot, "ex-earnings var <= 0")
        ex_iv1 = ex_iv_calc

    fwd_vol = forward_volatility(ex_iv1, T1, iv2, T2)
    if not fwd_vol:
        return _reject(ticker, fake_ed, spot, "fwd_vol <= 0")
        
    factor = calculate_forward_factor(ex_iv1, fwd_vol)
    if not factor or factor < 0.10: # < 1.1 ratio
        return _reject(ticker, fake_ed, spot, f"factor {factor+1:.2f} < 1.1")
        
    near_bid = q1.get("bid", 0)
    far_ask = q2.get("ask", 0)
    if near_bid <= 0 or far_ask <= 0:
        return _reject(ticker, fake_ed, spot, "invalid quotes")
        
    debit_start = target_debit_for_factor(far_ask, spot, t1["strike"], T1, fwd_vol, 0.50) # factor 1.5
    debit_cap = target_debit_for_factor(far_ask, spot, t1["strike"], T1, fwd_vol, 0.25)   # factor 1.25
    
    if debit_start is None or debit_cap is None:
        return _reject(ticker, fake_ed, spot, "math domain error")
        
    mid_debit = combo_debit(q1.get("bid", 0), q1.get("ask", 0), q2.get("bid", 0), q2.get("ask", 0), executable=False)
    if not mid_debit:
        return _reject(ticker, fake_ed, spot, "no mid")
        
    # Earnings-agnostic candidates carry no earnings date; T1 expiry IS the
    # trade's exit horizon, so use it wherever a date is required downstream
    # (ff_candidate_to_trade does date.fromisoformat(cand.earnings_date) —
    # an empty string would crash the whole proposal batch, and the LadderRunner
    # arm/step guards compare against this date to expire unfilled ladders).
    ed_out = next_earnings_str if event_inside_t1 else t1["expiry"].isoformat()
    
    return CalendarCandidate(
        ticker=ticker, spot=spot, earnings_date=ed_out,
        strike=t1["strike"],
        near_symbol=t1["symbol"], far_symbol=t2["symbol"],
        near_expiry=t1["expiry"].isoformat(), far_expiry=t2["expiry"].isoformat(),
        near_bid=near_bid, near_ask=q1.get("ask", 0),
        far_bid=q2.get("bid", 0), far_ask=far_ask,
        sigma_fwd=fwd_vol, hist_rms_move=rms or 0.0, tau_days=0,
        d_cap=round(debit_cap, 2), mid_debit=round(mid_debit, 2),
        d_start=round(debit_start, 2)
    )
