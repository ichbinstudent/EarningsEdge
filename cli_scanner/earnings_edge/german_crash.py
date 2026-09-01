"""German-venue 5-minute crash alerts (Gettex, Tradegate, Xetra, Frankfurt).

Drop definition
---------------
Peak-to-trough of the monitored price inside a rolling window (default 5
minutes):

    drop_pct = (window_high - last_price) / window_high

An alert fires when ``drop_pct > threshold`` (default 0.20, i.e. more than
20%). ``window_high`` is the maximum valid price in ``[now - window, now]``;
``last_price`` is the most recent valid price in that window. At least two
valid samples are required. A bounce that recovers before the latest print
does not alert.

Price used: last trade when it sits within 25% of the live mid and the
trade is fresh; otherwise the bid/ask mid (so a book crash without a print
still counts; Tradegate index JSON has no last). Quotes without a live
two-sided, uncrossed book are rejected (fail closed). Zero / NaN / stale
snapshots are rejected.

Alerts only — this module never places orders.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")

VENUE_BY_SUFFIX = {
    "GTX": "Gettex",
    "DE": "Xetra",
    "F": "Frankfurt",
    "D": "Dusseldorf",
    "H": "Hamburg",
    "MU": "Munich",
    "SG": "Stuttgart",
    "HA": "Hannover",
    "BE": "Berlin",
    "DEU": "Tradegate",
}

# Last-vs-mid band: last more than this far from mid is treated as stale.
_LAST_VS_MID_MAX = 0.25


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    return parts or default


@dataclass(frozen=True)
class CrashAlertConfig:
    """Thresholds and universe. Defaults match the requested 20% / 5 minutes."""

    threshold: float = 0.20
    window_secs: int = 300
    cooldown_secs: int = 1800
    stale_secs: int = 180
    min_price: float = 1.0
    exchanges: tuple[str, ...] = ("GTX", "GER", "FRA")
    tradegate_enabled: bool = True
    snapshot_open: bool = True
    enabled: bool = True
    last_vs_mid_max: float = _LAST_VS_MID_MAX

    @classmethod
    def from_env(cls) -> "CrashAlertConfig":
        return cls(
            threshold=_env_float("GERMAN_CRASH_THRESHOLD", 0.20),
            window_secs=_env_int("GERMAN_CRASH_WINDOW_SECS", 300),
            cooldown_secs=_env_int("GERMAN_CRASH_COOLDOWN_SECS", 1800),
            stale_secs=_env_int("GERMAN_CRASH_STALE_SECS", 180),
            min_price=_env_float("GERMAN_CRASH_MIN_PRICE", 1.0),
            exchanges=_env_csv("GERMAN_CRASH_EXCHANGES", ("GTX", "GER", "FRA")),
            tradegate_enabled=_env_bool("GERMAN_CRASH_TRADEGATE", True),
            snapshot_open=_env_bool("GERMAN_CRASH_SNAPSHOT", True),
            enabled=_env_bool("GERMAN_CRASH_ENABLED", True),
        )


@dataclass(frozen=True)
class GermanQuote:
    ticker: str
    venue: str
    price: float
    last: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    ts: datetime
    ric: str = ""
    name: str = ""
    source: str = ""


@dataclass(frozen=True)
class CrashAlert:
    ticker: str
    venue: str
    drop_pct: float
    high: float
    last: float
    window_secs: int
    ts: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    name: str = ""
    ric: str = ""
    high_ts: Optional[datetime] = None
    source: str = ""


@dataclass
class _Sample:
    ts: datetime
    price: float
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    name: str
    ric: str
    source: str


def parse_lseg_number(value) -> Optional[float]:
    """Parse LSEG widget numbers (``'+183.2'``, ``'+0'``, ``'-'``, ``'n.a.'``)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    s = str(value).strip()
    if not s or s in {"-", "n.a.", "NA", "null", "None"}:
        return None
    if s[0] in "+":
        s = s[1:]
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None
    if v != v:
        return None
    return v


def parse_tg_number(value) -> Optional[float]:
    """Parse Tradegate JSON numbers (float or German ``'1 076,00'``)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v != v:
            return None
        return v
    s = str(value).strip().replace("\xa0", " ").replace(" ", "")
    if not s or s in {"-", ".", "./."}:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    if v != v:
        return None
    return v


def ric_parts(ric: str) -> tuple[str, str]:
    """Return (ticker, venue name) from a RIC like ``SAPG.GTX``."""
    ric = (ric or "").strip()
    if "." in ric:
        root, suffix = ric.rsplit(".", 1)
        venue = VENUE_BY_SUFFIX.get(suffix.upper(), suffix.upper())
        return root or ric, venue
    return ric, "unknown"


def parse_trade_ts(
    trade_date: Optional[str],
    trade_time: Optional[str],
    captured_at: datetime,
) -> Optional[datetime]:
    """Parse ``01 SEP 2026`` + ``17:47:33`` as Europe/Berlin. None on failure."""
    if not trade_date or not trade_time:
        return None
    date_s = str(trade_date).strip()
    time_s = str(trade_time).strip()
    if time_s.count(":") == 1:
        time_s = time_s + ":00"
    for fmt in ("%d %b %Y %H:%M:%S", "%d %B %Y %H:%M:%S"):
        try:
            naive = datetime.strptime(f"{date_s} {time_s}", fmt)
            return naive.replace(tzinfo=BERLIN).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def validate_quote(
    *,
    ticker: str,
    venue: str,
    last: Optional[float],
    bid: Optional[float],
    ask: Optional[float],
    ts: datetime,
    now: datetime,
    cfg: CrashAlertConfig,
    ric: str = "",
    name: str = "",
    source: str = "",
    trade_ts: Optional[datetime] = None,
) -> Optional[GermanQuote]:
    """Fail closed: live two-sided uncrossed book, finite price, not stale.

    ``ts`` is observation time (when we captured the quote). A stale
    observation is rejected. An old last-trade print against a live book
    falls back to mid so a dumped book still alerts.
    """
    if not ticker:
        return None
    now = _aware(now)
    ts = _aware(ts)
    if now - ts > timedelta(seconds=cfg.stale_secs):
        return None
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    if bid > ask:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0 or mid != mid:
        return None
    price = mid
    used_last = last if (last is not None and last > 0 and last == last) else None
    if used_last is not None and abs(used_last - mid) / mid <= cfg.last_vs_mid_max:
        last_fresh = True
        if trade_ts is not None:
            last_fresh = (now - _aware(trade_ts)) <= timedelta(seconds=cfg.stale_secs)
        if last_fresh:
            price = used_last
    if price < cfg.min_price:
        return None
    return GermanQuote(
        ticker=ticker,
        venue=venue,
        price=price,
        last=used_last,
        bid=bid,
        ask=ask,
        ts=ts,
        ric=ric,
        name=name or "",
        source=source,
    )


def quote_from_lseg(
    raw: dict,
    captured_at: datetime,
    cfg: CrashAlertConfig,
) -> Optional[GermanQuote]:
    ric = str(raw.get("q.RIC") or raw.get("x.RIC") or "").strip()
    if not ric:
        return None
    ticker, venue = ric_parts(ric)
    last = parse_lseg_number(raw.get("q._TRDPRC_1"))
    bid = parse_lseg_number(raw.get("q._BID"))
    ask = parse_lseg_number(raw.get("q._ASK"))
    trade_ts = parse_trade_ts(raw.get("q._TRADE_DATE"), raw.get("q._TRDTIM_1"), captured_at)
    name = str(raw.get("q._DSPLY_NAME") or "").strip()
    return validate_quote(
        ticker=ticker, venue=venue, last=last, bid=bid, ask=ask,
        ts=captured_at, now=captured_at, cfg=cfg, ric=ric, name=name,
        source="lseg", trade_ts=trade_ts,
    )


def quote_from_tradegate(
    raw: dict,
    captured_at: datetime,
    cfg: CrashAlertConfig,
) -> Optional[GermanQuote]:
    isin = str(raw.get("isin") or "").strip()
    if not isin:
        return None
    bid = parse_tg_number(raw.get("bid"))
    ask = parse_tg_number(raw.get("ask"))
    last = parse_tg_number(raw.get("last"))  # usually absent on index JSON
    ts = captured_at
    unix_ts = raw.get("_timestamp")
    if isinstance(unix_ts, (int, float)) and unix_ts > 0:
        ts = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    name = str(raw.get("name") or "").strip()
    return validate_quote(
        ticker=isin, venue="Tradegate", last=last, bid=bid, ask=ask,
        ts=ts, now=captured_at, cfg=cfg, ric=isin, name=name, source="tradegate",
    )


class CrashDetector:
    """In-memory rolling windows keyed by (ticker, venue)."""

    def __init__(self, cfg: Optional[CrashAlertConfig] = None):
        self.cfg = cfg or CrashAlertConfig()
        self._windows: dict[tuple[str, str], deque[_Sample]] = defaultdict(deque)

    def ingest(
        self,
        quotes: list[GermanQuote],
        now: Optional[datetime] = None,
    ) -> list[CrashAlert]:
        now = _aware(now or datetime.now(timezone.utc))
        cutoff = now - timedelta(seconds=self.cfg.window_secs)
        alerts: list[CrashAlert] = []
        for q in quotes:
            if q is None:
                continue
            key = (q.ticker, q.venue)
            buf = self._windows[key]
            buf.append(_Sample(
                ts=q.ts, price=q.price, bid=q.bid, ask=q.ask, last=q.last,
                name=q.name, ric=q.ric, source=q.source,
            ))
            while buf and buf[0].ts < cutoff:
                buf.popleft()
            if len(buf) < 2:
                continue
            high_s = max(buf, key=lambda s: s.price)
            last_s = buf[-1]
            high = high_s.price
            last_px = last_s.price
            if high <= 0:
                continue
            drop = (high - last_px) / high
            if drop > self.cfg.threshold:
                alerts.append(CrashAlert(
                    ticker=q.ticker,
                    venue=q.venue,
                    drop_pct=drop,
                    high=high,
                    last=last_px,
                    window_secs=self.cfg.window_secs,
                    ts=now,
                    bid=last_s.bid,
                    ask=last_s.ask,
                    name=last_s.name,
                    ric=last_s.ric,
                    high_ts=high_s.ts,
                    source=last_s.source,
                ))
        return alerts


class Cooldown:
    """Per-ticker suppress window. Optional JSON persistence across restarts."""

    def __init__(self, cooldown_secs: int, path: Optional[str] = None):
        self.cooldown = timedelta(seconds=cooldown_secs)
        self.path = path
        self._last: dict[str, datetime] = {}
        if path:
            self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("crash cooldown load failed: %s", exc)
            return
        for key, val in (raw or {}).items():
            try:
                ts = datetime.fromisoformat(val)
                self._last[str(key)] = _aware(ts)
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            payload = {k: v.isoformat() for k, v in self._last.items()}
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.path)
        except Exception as exc:
            logger.warning("crash cooldown save failed: %s", exc)

    def allow(self, key: str, now: datetime) -> bool:
        now = _aware(now)
        prev = self._last.get(key)
        if prev is not None and now - prev < self.cooldown:
            return False
        self._last[key] = now
        self._save()
        return True

    def filter(self, alerts: list[CrashAlert], now: datetime) -> list[CrashAlert]:
        """Keep the largest drop per ticker, then apply cooldown."""
        best: dict[str, CrashAlert] = {}
        for a in alerts:
            cur = best.get(a.ticker)
            if cur is None or a.drop_pct > cur.drop_pct:
                best[a.ticker] = a
        out: list[CrashAlert] = []
        for ticker, a in best.items():
            if self.allow(ticker, now):
                out.append(a)
        out.sort(key=lambda a: a.drop_pct, reverse=True)
        return out


def format_alert(alert: CrashAlert) -> str:
    """Telegram HTML body. Caller sends via approval chats."""
    import html as html_mod

    def esc(s: str) -> str:
        return html_mod.escape(s, quote=False)

    berlin = _aware(alert.ts).astimezone(BERLIN)
    tzname = berlin.tzname() or "CET"
    window_m = max(1, int(round(alert.window_secs / 60)))
    name = f" ({esc(alert.name)})" if alert.name else ""
    bidask = ""
    if alert.bid and alert.ask:
        bidask = f"\nBid/ask: €{alert.bid:.4g} / €{alert.ask:.4g}"
    high_bit = ""
    if alert.high_ts is not None:
        ht = _aware(alert.high_ts).astimezone(BERLIN).strftime("%H:%M:%S")
        high_bit = f" at {ht}"
    return (
        f"🚨 <b>German crash</b>: {esc(alert.ticker)} on {esc(alert.venue)}{name}\n"
        f"Drop: <b>{alert.drop_pct * 100:.1f}%</b> in {window_m}m "
        f"(peak €{alert.high:.4g}{high_bit} → last €{alert.last:.4g})\n"
        f"Window: {window_m}m  |  last €{alert.last:.4g}  |  "
        f"{berlin.strftime('%Y-%m-%d %H:%M:%S')} {tzname}"
        f"{bidask}"
    )


def in_open_snapshot_window(now: datetime) -> bool:
    """True during 07:30–08:00 Europe/Berlin (original Gettex capture slot)."""
    local = _aware(now).astimezone(BERLIN).time()
    return (local.hour == 7 and local.minute >= 30) or (
        local.hour == 8 and local.minute == 0
    )


@dataclass
class CrashMonitor:
    """One poll: fetch German quotes, detect drops, apply cooldown."""

    cfg: CrashAlertConfig = field(default_factory=CrashAlertConfig)
    collector: object = None
    tradegate: object = None
    data_dir: str = ""
    cooldown_path: Optional[str] = None
    detector: CrashDetector = field(init=False)
    cooldown: Cooldown = field(init=False)
    _watch_rics: set[str] = field(default_factory=set)
    _last_full_mono: float = 0.0

    def __post_init__(self) -> None:
        self.detector = CrashDetector(self.cfg)
        path = self.cooldown_path
        if path is None and self.data_dir:
            path = os.path.join(self.data_dir, "german_crash_cooldown.json")
        self.cooldown = Cooldown(self.cfg.cooldown_secs, path=path)

    def _full_refresh_due(self, full_every_secs: float = 900.0) -> bool:
        import time
        now = time.monotonic()
        if not self._watch_rics or (now - self._last_full_mono) >= full_every_secs:
            self._last_full_mono = now
            return True
        return False

    def poll(self, now: Optional[datetime] = None) -> dict:
        if not self.cfg.enabled:
            return {"skipped": "disabled", "alerts": []}
        now = _aware(now or datetime.now(timezone.utc))
        n_fetched = 0
        valid: list[GermanQuote] = []
        raw_lseg: list[dict] = []

        collector = self.collector
        if collector is not None:
            try:
                snapshot = self.cfg.snapshot_open and in_open_snapshot_window(now)
                if snapshot or self._full_refresh_due():
                    rics = collector.fetch_instruments(self.cfg.exchanges)
                else:
                    rics = list(self._watch_rics)
                raw_lseg = collector.fetch_quotes(rics) if rics else []
                n_fetched += len(raw_lseg)
                for row in raw_lseg:
                    q = quote_from_lseg(row, now, self.cfg)
                    if q:
                        valid.append(q)
            except Exception as exc:
                logger.error("LSEG german quotes failed: %s", exc)

        if self.cfg.tradegate_enabled and self.tradegate is not None:
            try:
                rows = self.tradegate.fetch_index_quotes()
                n_fetched += len(rows)
                for row in rows:
                    q = quote_from_tradegate(row, now, self.cfg)
                    if q:
                        valid.append(q)
            except Exception as exc:
                logger.error("Tradegate quotes failed: %s", exc)

        self._watch_rics = {q.ric for q in valid if q.source == "lseg" and q.ric}

        if (
            self.cfg.snapshot_open
            and raw_lseg
            and collector is not None
            and in_open_snapshot_window(now)
            and hasattr(collector, "write_snapshot")
        ):
            try:
                collector.write_snapshot(raw_lseg, now)
            except Exception as exc:
                logger.warning("Gettex snapshot write failed: %s", exc)

        raw_alerts = self.detector.ingest(valid, now)
        alerts = self.cooldown.filter(raw_alerts, now)
        return {
            "n_fetched": n_fetched,
            "n_valid": len(valid),
            "n_raw_alerts": len(raw_alerts),
            "n_alerts": len(alerts),
            "tickers": [a.ticker for a in alerts],
            "alerts": alerts,
        }


def build_monitor(data_dir: str, cfg: Optional[CrashAlertConfig] = None) -> CrashMonitor:
    """Live wiring: Gettex LSEG collector + Tradegate JSON."""
    from earnings_edge.collectors.gettex import GettexCollector
    from earnings_edge.collectors.tradegate import TradegateCollector

    cfg = cfg or CrashAlertConfig.from_env()
    collector = GettexCollector(data_dir, exchanges=list(cfg.exchanges))
    tg = TradegateCollector() if cfg.tradegate_enabled else None
    return CrashMonitor(cfg=cfg, collector=collector, tradegate=tg, data_dir=data_dir)
