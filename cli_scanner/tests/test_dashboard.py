"""Smoke tests for the live dashboard service."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from dashboard.server import PANELS, app, build_panels


@pytest.fixture(autouse=True)
def _dash_tmp_db(tmp_path, monkeypatch):
    """Keep dashboard CLI paths off the production DB."""
    from earnings_edge.db import engine as db_engine

    path = tmp_path / "dash.db"
    db_engine.configure(path)
    monkeypatch.setenv("DASH_DB", str(path))


def test_panel_registry_is_wellformed():
    ids = [p.id for p in PANELS]
    assert len(ids) == len(set(ids)), "panel ids must be unique"
    for p in PANELS:
        assert p.kind in ("stats", "table", "text")


def test_build_panels_tolerates_missing_tables():
    # the real DB may lack lazily-created tables (ff_ladders, trade_events…)
    # — every provider must still return a payload
    panels = build_panels()
    assert set(panels) == {p.id for p in PANELS}
    for pid, state in panels.items():
        assert state["kind"] in ("stats", "table", "text")
        assert "payload" in state


def test_index_and_state_endpoints():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Earnings Edge" in r.text
        assert "telegram-web-app.js" in r.text
        r = client.get("/api/state")
        assert r.status_code == 200
        data = r.json()
        assert "bot_health" in data
        assert data["bot_health"]["kind"] == "stats"


def test_index_explains_browser_open():
    with TestClient(app) as client:
        html = client.get("/").text
        assert "Open desk" in html
        assert "initData" in html


def test_websocket_hello_and_initial_state():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            hello = json.loads(ws.receive_text())
            assert hello["type"] == "hello"
            assert {p["id"] for p in hello["panels"]} == {p.id for p in PANELS}
            # initial state for every panel follows the hello
            seen = set()
            for _ in PANELS:
                msg = json.loads(ws.receive_text())
                assert msg["type"] == "panel"
                seen.add(msg["id"])
            assert seen == {p.id for p in PANELS}
