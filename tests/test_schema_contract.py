import pytest
from earnings_edge import bot_views
from earnings_edge import rich_msg
from earnings_edge.db import engine as db_engine

class TrackingDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed_keys = set()
    
    def __getitem__(self, key):
        self.accessed_keys.add(key)
        return super().__getitem__(key)
        
    def get(self, key, default=None):
        self.accessed_keys.add(key)
        return super().get(key, default)
    
    def items(self):
        for k in self.keys():
            self.accessed_keys.add(k)
        return super().items()

def test_rich_msg_orders(seeded_db):
    from earnings_edge.db.repositories import trade_events_list
    events = trade_events_list(limit=1)
    tracked = [TrackingDict(e) for e in events]
    import unittest.mock
    with unittest.mock.patch("earnings_edge.db.trade_events_list", return_value=tracked):
        rich_msg.orders_rich_view()
    for t in tracked:
        assert t.accessed_keys <= set(t.keys()), f"view accessed unknown key(s): {t.accessed_keys - set(t.keys())}"

def test_rich_msg_jobs(seeded_db):
    from earnings_edge.db.repositories import job_runs_list
    runs = job_runs_list(limit=1)
    tracked = [TrackingDict(e) for e in runs]
    import unittest.mock
    with unittest.mock.patch("earnings_edge.db.job_runs_list", return_value=tracked):
        rich_msg.jobs_rich_view()
    for t in tracked:
        assert t.accessed_keys <= set(t.keys()), f"view accessed unknown key(s): {t.accessed_keys - set(t.keys())}"

def test_rich_msg_equity(seeded_db):
    from earnings_edge.db.repositories import equity_snapshots_daily_avg
    eqs = equity_snapshots_daily_avg(days=1)
    tracked = [TrackingDict(e) for e in eqs]
    import unittest.mock
    with unittest.mock.patch("earnings_edge.db.equity_snapshots_daily_avg", return_value=tracked), \
         unittest.mock.patch("framework.risk.equity.latest_equity", return_value={"equity": 10000.50, "buying_power": 5000.0, "ts": "2026-09-04"}), \
         unittest.mock.patch("framework.risk.equity.day_start_equity", return_value=9900.0), \
         unittest.mock.patch("framework.risk.equity.daily_pnl", return_value=100.50):
        rich_msg.equity_rich_view()
    for t in tracked:
        assert t.accessed_keys <= set(t.keys()), f"view accessed unknown key(s): {t.accessed_keys - set(t.keys())}"

def test_bot_views_orders(seeded_db):
    from earnings_edge.db.repositories import trade_events_list
    events = trade_events_list(limit=1)
    tracked = [TrackingDict(e) for e in events]
    import unittest.mock
    with unittest.mock.patch("earnings_edge.bot_views.trade_events_list", return_value=tracked):
        bot_views.orders_view()
    for t in tracked:
        assert t.accessed_keys <= set(t.keys()), f"view accessed unknown key(s): {t.accessed_keys - set(t.keys())}"

def test_bot_views_jobs(seeded_db):
    from earnings_edge.db.repositories import job_runs_list
    runs = job_runs_list(limit=1)
    tracked = [TrackingDict(e) for e in runs]
    import unittest.mock
    with unittest.mock.patch("earnings_edge.bot_views.job_runs_list", return_value=tracked):
        bot_views.jobs_view()
    for t in tracked:
        assert t.accessed_keys <= set(t.keys()), f"view accessed unknown key(s): {t.accessed_keys - set(t.keys())}"

def test_bot_views_equity(seeded_db):
    from earnings_edge.db.repositories import equity_snapshots_daily_avg
    eqs_avg = equity_snapshots_daily_avg(days=14)
    eqs_avg.append({"d": "2026-09-02", "e": 9000.0})
    tracked_avg = [TrackingDict(e) for e in eqs_avg]
    import unittest.mock
    with unittest.mock.patch("earnings_edge.bot_views.equity_snapshots_daily_avg", return_value=tracked_avg):
        bot_views.equity_view()
    for t in tracked_avg:
        assert t.accessed_keys <= set(t.keys()), f"view accessed unknown key(s): {t.accessed_keys - set(t.keys())}"

def test_bot_views_strategy_state(seeded_db):
    from earnings_edge.db.repositories import strategy_state_list
    states = strategy_state_list()
    tracked = [TrackingDict(e) for e in states]
    import unittest.mock
    with unittest.mock.patch("framework.execution.lifecycle.strategy_state_list", return_value=tracked):
        bot_views.strategies_view()
    for t in tracked:
        assert t.accessed_keys <= set(t.keys()), f"view accessed unknown key(s): {t.accessed_keys - set(t.keys())}"
