"""Realism enrichment for strategy backtest results.

Applies the realism layer (``realism.py``) and statistics layer (``stats.py``)
to a finished :class:`StrategyResult`, without changing how strategies find or
fill trades:

  - per-trade IBKR commissions are deducted from the dollar P&L
    (entry fills only — positions are held to expiration, so there are no exit
    commissions; assignment/exercise fees are ignored, worst-case direction
    unknown);
  - REG-T / defined-risk margin (supplied by the strategy as
    ``features["margin_dollars"]`` per 1-lot) gives return-on-margin per trade;
  - ``stats.trade_stats`` and a chronological 70/30 train/test split
    (``stats.train_test_report``) are computed over the per-trade
    return-on-margin series.

Strategies opt in by recording two extra fields in ``Trade.features``:

  - ``leg_premiums``: list of per-share fill prices for each contract fill at
    entry (duplicated for multi-contract legs, e.g. the 2x body of a fly) —
    used to tier commissions;
  - ``margin_dollars``: initial margin per 1-lot in dollars (defined-risk
    spread max loss, or the REG-T naked formula for undefined risk).

Trades without those features get no commission/margin metrics (there is
nothing to tier on); the dollar P&L metrics are always reported.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from .realism import CONTRACT_MULTIPLIER, ibkr_commission
from .stats import trade_stats, train_test_report
from ..trading_types import StrategyResult, Trade

DEFAULT_TRAIN_SPLIT = 0.7


def _trade_commission(trade: Trade, contracts: int) -> Optional[float]:
    """Total entry commission for one trade, or None when premiums unknown."""
    premiums = trade.features.get("leg_premiums")
    if not premiums:
        return None
    return float(sum(ibkr_commission(float(p), contracts) for p in premiums))


def enrich_result(
    result: StrategyResult,
    *,
    contracts: int = 1,
    train_split: float = DEFAULT_TRAIN_SPLIT,
) -> StrategyResult:
    """Return a copy of *result* whose summary carries realism metrics.

    Added summary keys (when computable): ``commissions_total``,
    ``gross_total_pnl_dollars``, ``net_total_pnl_dollars``,
    ``net_avg_pnl_dollars``, ``net_win_rate``, ``avg_return_on_margin``,
    ``median_return_on_margin``, ``trade_stats`` (dict over return-on-margin),
    ``train_stats`` / ``test_stats`` (chronological split over return-on-margin).
    """
    if not result.trades:
        return result

    # chronological order for the split discipline
    trades = sorted(result.trades, key=lambda t: (t.scan_date, t.ticker))

    gross: list[float] = []
    net: list[float] = []
    roms: list[float] = []
    commissions_total = 0.0
    commissions_known = False

    for t in trades:
        gross_dollars = t.pnl * CONTRACT_MULTIPLIER * contracts
        commission = _trade_commission(t, contracts)
        net_dollars = gross_dollars - (commission or 0.0)
        gross.append(gross_dollars)
        net.append(net_dollars)
        if commission is not None:
            commissions_total += commission
            commissions_known = True
        margin = t.features.get("margin_dollars")
        if margin:
            roms.append(net_dollars / (float(margin) * contracts))

    summary: dict[str, Any] = dict(result.summary)
    summary["gross_total_pnl_dollars"] = float(sum(gross))
    summary["net_total_pnl_dollars"] = float(sum(net))
    summary["net_avg_pnl_dollars"] = float(sum(net) / len(net))
    summary["net_win_rate"] = float(sum(1 for p in net if p > 0) / len(net))
    if commissions_known:
        summary["commissions_total"] = float(commissions_total)
    if roms:
        summary["avg_return_on_margin"] = float(sum(roms) / len(roms))
        summary["median_return_on_margin"] = trade_stats(roms).median
        summary["trade_stats"] = asdict(trade_stats(roms))
        split = train_test_report(roms, split=train_split)
        summary["train_stats"] = asdict(split["train"])
        summary["test_stats"] = asdict(split["test"])

    return StrategyResult(result.name, result.trades, summary)
