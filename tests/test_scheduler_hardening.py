from datetime import datetime
from unittest.mock import MagicMock

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

from bot import TradingBot


class _FakeBot:
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.scanners = {"Earnings Calendar": MagicMock()}

    _setup_scheduler = TradingBot._setup_scheduler
    _run_sync = MagicMock()
    _ff_propose_sync = MagicMock()
    _ff_step_sync = MagicMock()
    _equity_snapshot_sync = MagicMock()
    _reconcile_sync = MagicMock()
    _guard_eval_sync = MagicMock()
    _exit_eval_sync = MagicMock()
    _backup_sync = MagicMock()
    _db_health_sync = MagicMock()
    _picks_sync = MagicMock()
    _chain_cache_sync = MagicMock()

def test_scheduler_hardening_properties():
    bot = _FakeBot()
    bot._setup_scheduler()

    jobs = bot.scheduler.get_jobs()
    assert len(jobs) > 0
    for job in jobs:
        assert job.max_instances == 1
        assert job.coalesce is True
        # Scanners get 300s misfire_grace_time, others get 120s
        if job.id.startswith("scanner_"):
            assert job.misfire_grace_time == 300
        else:
            assert job.misfire_grace_time == 120

def test_scheduler_hardening_dst_safety():
    bot = _FakeBot()
    bot._setup_scheduler()

    ny_tz = pytz.timezone("America/New_York")

    # Check winter week (Standard Time)
    winter_base = datetime(2026, 1, 12, tzinfo=pytz.UTC) # A Monday
    # Check summer week (Daylight Saving Time)
    summer_base = datetime(2026, 7, 13, tzinfo=pytz.UTC) # A Monday

    # Expected hours in ET (wall-clock)
    expected_hours = {
        "scanner_Earnings Calendar": [14],
        "ff_ladder_propose": [13],
        "ff_ladder_step": [14, 15],
        "equity_snapshot": [9, 10, 11, 12, 13, 14, 15, 16],
        "db_backup": [0],
        "daily_picks": [7],
    }

    for base_date in [winter_base, summer_base]:
        for job in bot.scheduler.get_jobs():
            trig = job.trigger

            # Fire 5 times and check the hour in America/New_York
            current_time = base_date
            for _ in range(5):
                next_time = trig.get_next_fire_time(None, current_time)
                if next_time:
                    ny_time = next_time.astimezone(ny_tz)
                    if job.id in expected_hours:
                        assert ny_time.hour in expected_hours[job.id], f"{job.id} fired at wrong hour {ny_time.hour} on {ny_time}"
                    current_time = next_time
