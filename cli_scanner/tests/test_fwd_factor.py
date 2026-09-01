"""Tests for the forward-factor target-price math and limit ladder."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from earnings_edge.fwd_factor import (
    ET,
    LadderSpec,
    combo_debit,
    forward_iv,
    occ_parse,
    occ_symbol,
    required_near_iv,
    target_debit,
    within_fill_range,
)


# ── OCC helpers ------------------------------------------------------------

def test_occ_roundtrip():
    sym = occ_symbol("AAPL", date(2026, 8, 21), 330.0)
    assert sym == "AAPL260821C00330000"
    parsed = occ_parse(sym)
    assert parsed["root"] == "AAPL"
    assert parsed["expiry"] == date(2026, 8, 21)
    assert parsed["option_type"] == "call"
    assert parsed["strike"] == 330.0


def test_occ_put_and_fractional_strike():
    sym = occ_symbol("KSS", date(2026, 6, 26), 12.5, "put")
    parsed = occ_parse(sym)
    assert parsed["option_type"] == "put"
    assert parsed["strike"] == 12.5


# ── required_near_iv / target_debit -----------------------------------------

def test_required_near_iv_above_fwd():
    # 6% RMS move, 20% premium, T1=45d, tau=2d, fwd 30%
    iv = required_near_iv(sigma_fwd=0.30, T1=45 / 365, tau=2 / 365,
                          hist_rms_move=0.06, premium=0.20)
    assert iv is not None
    # event window adds variance → near IV must exceed the forward vol
    assert iv > 0.30


def test_required_near_iv_scales_with_premium():
    base = dict(sigma_fwd=0.30, T1=45 / 365, tau=2 / 365, hist_rms_move=0.06)
    iv20 = required_near_iv(**base, premium=0.20)
    iv25 = required_near_iv(**base, premium=0.25)
    assert iv25 > iv20  # stricter premium → richer required near leg


def test_target_debit_cheaper_for_higher_premium():
    # realistic: far 80d @ ~35% IV ≈ $6.54, required near (20%) ≈ $5.34 → D*20 ≈ +1.20
    base = dict(far_price=6.54, spot=100.0, strike=100.0, T1=45 / 365,
                sigma_fwd=0.35, tau=2 / 365, hist_rms_move=0.05)
    d20 = target_debit(**base, premium=0.20)
    d25 = target_debit(**base, premium=0.25)
    assert d20 is not None and d25 is not None
    assert d25 < d20  # ladder starts cheaper (25%) and concedes to the cap (20%)
    assert d20 > 0


def test_target_debit_negative_when_threshold_demands_credit():
    # 6% RMS move, fwd only 30%, far leg cheap: the 20% threshold requires the
    # near leg to cost MORE than the far leg → D* < 0. Means: even a free
    # calendar meets the bar — extreme backwardation territory, skip in practice.
    d20 = target_debit(far_price=3.00, spot=100.0, strike=100.0, T1=45 / 365,
                       sigma_fwd=0.30, tau=2 / 365, hist_rms_move=0.06, premium=0.20)
    assert d20 is not None and d20 < 0


def test_target_debit_degenerate_inputs():
    assert target_debit(3.0, 100.0, 100.0, 0, 0.30, 0, 0.06, 0.20) is None
    assert target_debit(3.0, 100.0, 100.0, 45 / 365, 0.30, 50 / 365, 0.06, 0.20) is None  # tau > T1
    assert target_debit(3.0, 100.0, 100.0, 45 / 365, 0.30, 2 / 365, 0.0, 0.20) is None


def test_forward_iv_basic():
    # flat term structure → forward equals the common IV
    fwd = forward_iv(0.30, 45 / 365, 0.30, 80 / 365)
    assert fwd == pytest.approx(0.30, rel=1e-6)
    # elevated front → forward below far IV (but positive forward variance)
    fwd2 = forward_iv(0.40, 45 / 365, 0.35, 80 / 365)
    assert fwd2 is not None and fwd2 < 0.35
    # front so rich the implied forward variance is negative → None
    assert forward_iv(0.50, 45 / 365, 0.32, 80 / 365) is None
    # inverted inputs → None
    assert forward_iv(0.30, 80 / 365, 0.30, 45 / 365) is None


# ── combo_debit / within_fill_range ------------------------------------------

def test_combo_debit_mid_vs_executable():
    mid = combo_debit(near_bid=1.00, near_ask=1.10, far_bid=2.00, far_ask=2.20)
    assert mid == pytest.approx(2.10 - 1.05)
    exe = combo_debit(1.00, 1.10, 2.00, 2.20, executable=True)
    assert exe == pytest.approx(2.20 - 1.00)
    assert exe > mid  # combo ask is the conservative fill cost


def test_within_fill_range():
    assert within_fill_range(mid_debit=1.00, cap_debit=1.10)       # already cheaper than cap
    assert within_fill_range(mid_debit=1.20, cap_debit=1.10, f=0.15)  # within 15%
    assert not within_fill_range(mid_debit=1.30, cap_debit=1.10, f=0.15)  # too far
    assert not within_fill_range(mid_debit=1.00, cap_debit=0.0)     # degenerate cap


# ── LadderSpec ---------------------------------------------------------------

def _et(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 27, hour, minute, tzinfo=ET)  # a Monday


def test_ladder_rungs_and_cap():
    spec = LadderSpec()
    # window closed before 14:00 ET and after 15:45 ET
    assert spec.rung_index(_et(13, 59)) is None
    assert spec.rung_index(_et(15, 46)) is None
    assert spec.rung_index(_et(14, 0)) == 0
    assert spec.rung_index(_et(14, 14)) == 0
    assert spec.rung_index(_et(14, 15)) == 1
    assert spec.rung_index(_et(15, 45)) == 7

    # price walk: +tick per rung, hard cap
    assert spec.limit_at(0, 1.00, 1.05) == 1.00
    assert spec.limit_at(3, 1.00, 1.05) == 1.03
    assert spec.limit_at(7, 1.00, 1.05) == 1.05   # clamped at cap, never above
    assert spec.limit_at(7, 1.00, 1.20) == 1.07   # tick walk if cap is far


def test_ladder_current_limit():
    spec = LadderSpec()
    assert spec.current_limit(_et(14, 30), 1.00, 1.10) == 1.02
    assert spec.current_limit(_et(12, 0), 1.00, 1.10) is None


def test_ladder_et_window_matches_us_session():
    # 14:00 ET start and 15:45 ET last rung are inside 09:30-16:00 ET
    spec = LadderSpec()
    assert spec.start_et.hour == 14
    assert spec.last_rung_et.hour == 15 and spec.last_rung_et.minute == 45
