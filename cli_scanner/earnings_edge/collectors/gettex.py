"""LSEG widget quotes for German venues (Gettex, Xetra, Frankfurt, regionals).

Auth is the public gettex.de SAML → JWT flow used by the exchange's own
widgets. No API key. Instruments are STO equities on the configured
LSEG exchange codes; quotes are last/bid/ask plus trade date/time.
"""

from __future__ import annotations

import json
import logging
import os
import re
import base64
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GETTEX_ORIGIN = "https://www.gettex.de"
LSEG_AUTH_SAML = (
    "https://lseg-widgets.financial.com/auth/api/v1/sessions/samllogin"
    "?profile=wlib_gettex"
)
LSEG_AUTH_TOKEN = "https://lseg-widgets.financial.com/auth/api/v1/tokens"
LSEG_FIND = "https://lseg-widgets.financial.com/rest/api/find/securities"
LSEG_QUOTE = "https://lseg-widgets.financial.com/rest/api/quote/info"

# Exchange codes that this widget actually returns (probed 2026-09-01).
# GTX = Gettex (.GTX), GER = Xetra (.DE), FRA = Frankfurt (.F),
# DUS/HAM/MUN/STU/HAN = regional German floors.
KNOWN_EXCHANGES = ("GTX", "GER", "FRA", "DUS", "HAM", "MUN", "STU", "HAN")
DEFAULT_EXCHANGES = ("GTX", "GER", "FRA")

QUOTE_FIDS = (
    "q.RIC,q._TRDPRC_1,q._BID,q._ASK,q._BIDSIZE,q._ASKSIZE,"
    "q._TRDTIM_1,q._TRADE_DATE,q._DSPLY_NAME"
)
INSTRUMENT_FIDS = "x.RIC"

_UA = "Mozilla/5.0"


class GettexCollector:
    """Fetch German-venue equity quotes via the Gettex LSEG widget."""

    def __init__(
        self,
        data_dir: str,
        exchanges: Optional[list[str]] = None,
        batch_size: int = 80,
        max_workers: int = 8,
        session: Optional[requests.Session] = None,
    ):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.exchanges = tuple(
            e.strip().upper() for e in (exchanges or DEFAULT_EXCHANGES) if e.strip()
        ) or DEFAULT_EXCHANGES
        self.batch_size = max(1, int(batch_size))
        self.max_workers = max(1, int(max_workers))
        self._jwt = ""
        self._session = session or requests.Session()
        self._rics_cache: dict[str, list[str]] = {}
        self._rics_cached_at: float = 0.0

    def _headers(self, jwt: Optional[str] = None) -> dict[str, str]:
        token = jwt if jwt is not None else self._jwt
        return {
            "accept": "application/json",
            "jwt": token,
            "origin": GETTEX_ORIGIN,
            "user-agent": _UA,
        }

    def _get_jwt(self) -> str:
        try:
            res = self._session.get(GETTEX_ORIGIN + "/", timeout=15)
            match = re.search(r"const samlRequest=`(.*?)`;", res.text, re.DOTALL)
            if not match:
                raise ValueError("SAML not found in gettex.de HTML")
            saml_b64 = base64.b64encode(match.group(1).strip().encode("utf-8")).decode("utf-8")
            saml_enc = urllib.parse.quote(saml_b64)
            res2 = self._session.post(
                LSEG_AUTH_SAML,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": GETTEX_ORIGIN,
                    "Referer": GETTEX_ORIGIN + "/",
                    "User-Agent": _UA,
                },
                data=f"SAMLResponse={saml_enc}",
                timeout=15,
            )
            sid = res2.json().get("sid")
            if not sid:
                raise ValueError("No sid returned")
            res3 = self._session.post(
                LSEG_AUTH_TOKEN,
                headers={
                    "Accept": "application/json",
                    "Origin": GETTEX_ORIGIN,
                    "Referer": GETTEX_ORIGIN + "/",
                    "User-Agent": _UA,
                    "sid": sid,
                },
                timeout=15,
            )
            jwt = res3.text.strip()
            if not jwt:
                raise ValueError("No JWT returned")
            return jwt
        except Exception as exc:
            logger.error("Failed to get Gettex JWT: %s", exc)
            return ""

    def get_headers(self) -> dict[str, str]:
        if not self._jwt:
            self._jwt = self._get_jwt()
        return self._headers()

    def _get_json(self, url: str) -> Optional[dict]:
        res = self._session.get(url, headers=self.get_headers(), timeout=20)
        if res.status_code == 401:
            self._jwt = self._get_jwt()
            res = self._session.get(url, headers=self.get_headers(), timeout=20)
        if res.status_code != 200:
            logger.warning("LSEG GET %s -> %s", url.split("?", 1)[0], res.status_code)
            return None
        try:
            return res.json()
        except ValueError:
            logger.warning("LSEG non-JSON from %s", url.split("?", 1)[0])
            return None

    def fetch_sto_instruments(self, exchange: Optional[str] = None) -> list[str]:
        """All STO RICs on one exchange (or the first configured exchange)."""
        ex = (exchange or self.exchanges[0]).upper()
        rics: list[str] = []
        page = 0
        while True:
            url = (
                f"{LSEG_FIND}?fids={INSTRUMENT_FIDS}&exchanges={ex}"
                f"&secTypes=STO&pageSize=5000&pageNo={page}"
            )
            payload = self._get_json(url)
            if not payload:
                break
            data = payload.get("data") or []
            if not data:
                break
            rics.extend(d["x.RIC"] for d in data if d.get("x.RIC"))
            if len(data) < 5000:
                break
            page += 1
        return rics

    def fetch_instruments(
        self,
        exchanges: Optional[tuple[str, ...]] = None,
        force: bool = False,
        ttl_secs: float = 3600.0,
    ) -> list[str]:
        """Cached union of STO RICs across configured German exchanges."""
        import time

        exchanges = exchanges or self.exchanges
        now = time.monotonic()
        cache_key = ",".join(exchanges)
        if (
            not force
            and self._rics_cache.get(cache_key)
            and (now - self._rics_cached_at) < ttl_secs
        ):
            return list(self._rics_cache[cache_key])
        rics: list[str] = []
        seen: set[str] = set()
        for ex in exchanges:
            for ric in self.fetch_sto_instruments(ex):
                if ric not in seen:
                    seen.add(ric)
                    rics.append(ric)
        self._rics_cache[cache_key] = rics
        self._rics_cached_at = now
        logger.info("Gettex universe %s: %d RICs", cache_key, len(rics))
        return list(rics)

    def _quote_batch(self, batch: list[str]) -> list[dict]:
        url = f"{LSEG_QUOTE}?rics={','.join(batch)}&fids={QUOTE_FIDS}"
        payload = self._get_json(url)
        if not payload:
            return []
        data = payload.get("data") or []
        return [row for row in data if isinstance(row, dict)]

    def fetch_quotes(self, rics: list[str]) -> list[dict]:
        if not rics:
            return []
        batches = [
            rics[i:i + self.batch_size] for i in range(0, len(rics), self.batch_size)
        ]
        if len(batches) == 1 or self.max_workers == 1:
            out: list[dict] = []
            for batch in batches:
                out.extend(self._quote_batch(batch))
            return out
        out = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = [pool.submit(self._quote_batch, b) for b in batches]
            for fut in as_completed(futs):
                try:
                    out.extend(fut.result())
                except Exception as exc:
                    logger.warning("Gettex quote batch failed: %s", exc)
        return out

    def write_snapshot(self, quotes: list[dict], now: Optional[datetime] = None) -> str:
        """Append raw LSEG rows to today's jsonl. Returns the path."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        today = now.astimezone().strftime("%Y-%m-%d")
        filepath = os.path.join(self.data_dir, f"gettex_quotes_{today}.jsonl")
        ts = now.isoformat()
        with open(filepath, "a") as f:
            for q in quotes:
                row = dict(q)
                row["timestamp"] = ts
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        logger.info("Captured %d gettex quotes to %s", len(quotes), filepath)
        return filepath

    def capture_snapshot(self) -> Optional[str]:
        """Full-universe fetch + jsonl append (original 07:30–08:00 CET job)."""
        rics = self.fetch_instruments()
        if not rics:
            logger.error("No RICs found for Gettex capture.")
            return None
        quotes = self.fetch_quotes(rics)
        if not quotes:
            logger.error("No quotes fetched.")
            return None
        return self.write_snapshot(quotes)
