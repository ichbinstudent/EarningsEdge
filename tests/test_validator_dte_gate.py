"""DTE gate must use Settings MAX_DAYS_TO_EXPIRY (30), not a leftover literal 9."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd

from earnings_edge.config import MAX_DAYS_TO_EXPIRY
from earnings_edge.models import EarningsCandidate
from earnings_edge.validator import StockValidator


def _validator(first_expiry: str) -> tuple[StockValidator, MagicMock]:
    provider = MagicMock()
    provider.history.return_value = pd.DataFrame({"Close": [50.0]})
    far = (datetime.now(UTC).date() + timedelta(days=51)).isoformat()
    provider.options_expiries.return_value = [first_expiry, far]
    chain = MagicMock()
    chain.oi_available = False
    chain.source = "yahoo"
    chain.calls = pd.DataFrame()
    chain.puts = pd.DataFrame()
    provider.option_chain.return_value = chain
    analyzer = MagicMock()
    analysis = MagicMock()
    analysis.ok = False
    analysis.error = "stop-after-dte"
    analyzer.compute_recommendation.return_value = analysis
    return StockValidator(analyzer, MagicMock(), provider=provider), analyzer


def test_settings_max_days_to_expiry_is_30():
    assert MAX_DAYS_TO_EXPIRY == 30


def test_monthly_first_expiry_23d_is_not_rejected_as_too_far():
    first = (datetime.now(UTC).date() + timedelta(days=23)).isoformat()
    validator, analyzer = _validator(first)
    result = validator.validate(
        EarningsCandidate(ticker="PAYX", timing="Unknown", earnings_date=datetime.now(UTC).date())
    )
    assert "Expiry too far" not in result.reason
    analyzer.compute_recommendation.assert_called()


def test_first_expiry_beyond_settings_cap_is_rejected():
    first = (datetime.now(UTC).date() + timedelta(days=MAX_DAYS_TO_EXPIRY + 1)).isoformat()
    validator, analyzer = _validator(first)
    result = validator.validate(
        EarningsCandidate(ticker="PAYX", timing="Unknown", earnings_date=datetime.now(UTC).date())
    )
    assert result.passed is False
    assert result.reason.startswith("Expiry too far:")
    analyzer.compute_recommendation.assert_not_called()
