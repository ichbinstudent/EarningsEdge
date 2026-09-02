"""Telegram Mini App: initData HMAC, operator gate, desk actions."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from starlette.testclient import TestClient

from dashboard.desk import load_desk, run_desk_action
from dashboard.tg_auth import (
    InitDataError, require_operator, sign_init_data, verify_init_data, webapp_url,
)
from framework.positions.book_actions import adopt_orphan
from earnings_edge.db import configure


TOKEN = "123456:TEST-TOKEN"
NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
OP = 256565866


def _init(uid=OP, auth_date=None, token=TOKEN):
    user = json.dumps({"id": uid, "first_name": "Op"})
    ts = auth_date if auth_date is not None else int(NOW.timestamp())
    return sign_init_data({"auth_date": str(ts), "user": user, "query_id": "aa"}, token)


def test_webapp_url_https_only():
    assert webapp_url({}) is None
    assert webapp_url({"TELEGRAM_WEBAPP_URL": "http://x"}) is None
    assert webapp_url({"TELEGRAM_WEBAPP_URL": "https://desk.example/"}) == "https://desk.example/"
    assert webapp_url({"TELEGRAM_WEBAPP_URL": "https://desk.example"}) == "https://desk.example/"


def test_verify_init_data_accepts_signed_and_rejects_tamper():
    raw = _init()
    parsed = verify_init_data(raw, TOKEN, now=NOW)
    assert parsed["user_id"] == OP
    bad = raw[:-4] + "0000"
    with pytest.raises(InitDataError, match="bad hash"):
        verify_init_data(bad, TOKEN, now=NOW)
    stale = _init(auth_date=int(NOW.timestamp()) - 90_000)
    with pytest.raises(InitDataError, match="expired"):
        verify_init_data(stale, TOKEN, now=NOW, max_age_s=3600)


def test_quoted_token_and_chat_id_operator():
    raw = _init()
    parsed = verify_init_data(raw, f'"{TOKEN}"', now=NOW)
    assert parsed["user_id"] == OP
    chat_raw = sign_init_data(
        {
            "auth_date": str(int(NOW.timestamp())),
            "user": json.dumps({"id": 1, "first_name": "X"}),
            "chat": json.dumps({"id": OP, "type": "private"}),
        },
        TOKEN,
    )
    assert require_operator(
        chat_raw, TOKEN, env={"TELEGRAM_APPROVAL_CHAT_ID": str(OP)}, now=NOW
    ) == 1


def test_require_operator_fail_closed(monkeypatch):
    raw = _init()
    with pytest.raises(InitDataError, match="unconfigured"):
        require_operator(raw, TOKEN, env={}, now=NOW)
    env = {"TELEGRAM_APPROVAL_CHAT_ID": str(OP)}
    assert require_operator(raw, TOKEN, env=env, now=NOW) == OP
    other = _init(uid=1)
    with pytest.raises(InitDataError, match="not an operator"):
        require_operator(other, TOKEN, env=env, now=NOW)


def test_load_desk_readonly_does_not_write(tmp_path, monkeypatch):
    path = tmp_path / "ro.db"
    configure(path)
    from earnings_edge.db import risk_state_get
    risk_state_get(ensure=True)
    monkeypatch.setenv("DASH_DB", str(path))
    snap = load_desk(get_positions=lambda: [])
    assert snap["kill"]["halted"] is False


def test_desk_snapshot_and_adopt(tmp_path, monkeypatch):
    path = tmp_path / "d.db"
    configure(path)
    monkeypatch.setenv("DASH_DB", str(path))
    broker = [{
        "symbol": "ATLO260918C00030000", "qty": "-1", "side": "short",
        "avg_entry_price": "1", "current_price": "4", "unrealized_pl": "-300",
    }]
    snap = load_desk(get_positions=lambda: broker)
    assert any(i["symbol"] == "ATLO260918C00030000" for i in snap["book"]["orphan"])
    assert snap["kill"]["halted"] is False
    rec = adopt_orphan(broker[0], by="test")
    assert rec["ok"]
    snap2 = load_desk(get_positions=lambda: broker)
    assert any(i["symbol"] == "ATLO260918C00030000" for i in snap2["book"]["managed"])
    assert not any(i["symbol"] == "ATLO260918C00030000" for i in snap2["book"]["orphan"])


def test_run_desk_halt_and_unknown(tmp_path, monkeypatch):
    path = tmp_path / "k.db"
    configure(path)
    monkeypatch.setenv("DASH_DB", str(path))
    out = run_desk_action("halt", {}, by="webapp:1")
    assert out["ok"] and "Kill switch" in out["banner"]
    from framework.risk.killswitch import KillSwitch
    assert KillSwitch().is_halted()
    bad = run_desk_action("nope", {}, by="webapp:1")
    assert bad["ok"] is False


def test_action_endpoint_requires_init_data():
    from dashboard.server import app
    with TestClient(app) as client:
        r = client.post("/api/action", json={"op": "halt"})
        assert r.status_code == 403
        r = client.get("/api/desk")
        assert r.status_code == 403


def test_action_halt_with_valid_init(tmp_path, monkeypatch):
    from dashboard import server as srv
    db = tmp_path / "act.db"
    configure(db)
    monkeypatch.setenv("DASH_DB", str(db))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_APPROVAL_CHAT_ID", str(OP))
    with TestClient(srv.app) as client:
        real_now = int(datetime.now(timezone.utc).timestamp())
        r = client.post(
            "/api/action",
            json={"op": "halt"},
            headers={"X-Telegram-Init-Data": _init(auth_date=real_now)},
        )
        assert r.status_code == 200
        # Funnel-safe path: initData in the query string, no custom header
        r2 = client.get("/api/desk", params={"initData": _init(auth_date=real_now)})
        assert r2.status_code == 200
        assert r2.json()["kill"]["halted"] is True
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        desk = client.get("/api/desk", headers={"X-Telegram-Init-Data": _init(auth_date=real_now)})
        assert desk.status_code == 200
        assert desk.json()["kill"]["halted"] is True
        assert desk.json()["user_id"] == OP


def test_main_reply_kb_adds_webapp_row(monkeypatch):
    from bot import MAIN_KB, _main_reply_kb
    monkeypatch.delenv("TELEGRAM_WEBAPP_URL", raising=False)
    kb = _main_reply_kb()
    rows = [[b.text for b in row] for row in kb.keyboard]
    assert rows == MAIN_KB
    monkeypatch.setenv("TELEGRAM_WEBAPP_URL", "https://desk.example")
    kb2 = _main_reply_kb()
    rows2 = [[b.text for b in row] for row in kb2.keyboard]
    assert rows2[-1] == ["🖥 Open desk"]
    # Reply key must be plain text — web_app on a reply keyboard
    # launches without initData on several Telegram clients.
    assert getattr(kb2.keyboard[-1][0], "web_app", None) is None
    from bot import _desk_webapp_markup
    ikb = _desk_webapp_markup()
    assert ikb.inline_keyboard[0][0].web_app.url == "https://desk.example/"
