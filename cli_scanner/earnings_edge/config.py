"""Shared configuration: logging, HTTP session, constants.

Filter constants are now sourced from ``settings.Settings`` but re-exported
here under the same names for backward compatibility.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from curl_cffi import requests as curl_requests

from .settings import get_settings

# ── Shared HTTP session (curl_cffi impersonating Chrome) ──────────────
def _build_session() -> curl_requests.Session:
    """Chrome-impersonating session; honours YFINANCE_PROXY when set."""
    proxy = get_settings().yfinance_proxy
    kwargs: dict = {"impersonate": "chrome"}
    if proxy:
        kwargs["proxies"] = {"http": proxy, "https": proxy}
    sess = curl_requests.Session(**kwargs)
    
    # Enforce default timeout to prevent indefinite hangs if yfinance misses it
    orig_request = sess.request
    def _request_with_timeout(*args, **req_kwargs):
        req_kwargs.setdefault("timeout", 30.0)
        return orig_request(*args, **req_kwargs)
    sess.request = _request_with_timeout
    return sess


session = _build_session()

# ── Default filter thresholds (sourced from Settings) ─────────────────
_filters = get_settings().filters

DEFAULT_IV_RV_PASS = _filters.iv_rv_pass
DEFAULT_IV_RV_NEAR_MISS = _filters.iv_rv_near_miss

TERM_STRUCTURE_HARD_LIMIT = _filters.term_structure_hard_limit
TERM_STRUCTURE_TIER2 = _filters.term_structure_tier2

MIN_PRICE = _filters.min_price
MIN_VOLUME = _filters.min_volume
NEAR_MISS_VOLUME = _filters.near_miss_volume
MAX_DAYS_TO_EXPIRY = _filters.max_days_to_expiry
MIN_OPEN_INTEREST = _filters.min_open_interest
MAX_ATM_DELTA = _filters.max_atm_delta
MIN_EXPECTED_MOVE = _filters.min_expected_move

# ── Logging ───────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``earnings_edge`` namespace."""
    return logging.getLogger(f"earnings_edge.{name}")


def setup_logging(log_dir: str = "logs") -> None:
    """Configure root ``earnings_edge`` logger with file + console output."""
    logger = logging.getLogger("earnings_edge")
    if logger.handlers:          # already configured
        return

    logger.setLevel(logging.INFO)

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(
        f"{log_dir}/scanner_{datetime.now().strftime('%Y%m%d')}.log"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
