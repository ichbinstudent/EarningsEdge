import json
import threading
import time
from urllib.request import urlopen
from http.server import HTTPServer
import pytest
from datetime import datetime, timezone

from bot import _HealthHandler
from earnings_edge.db import configure
from earnings_edge.db.repositories import (
    job_runs_start,
    job_runs_finish,
    equity_snapshots_insert,
    risk_state_set_halted
)

@pytest.fixture
def test_db(tmp_path):
    p = tmp_path / "metrics_test.db"
    configure(p)
    return p

def test_metrics_endpoint(test_db):
    # Seed data
    now = datetime.now(timezone.utc).isoformat()
    # 1. Job runs
    run_id = job_runs_start("earnings_scan", started_at=now)
    job_runs_finish(run_id, success=True, stats_json='{"scanned": 1}')
    
    # 2. Equity
    equity_snapshots_insert(ts=now, equity=100000.0, buying_power=95000.0, portfolio_value=100000.0)
    
    # 3. Risk state
    risk_state_set_halted(True, reason="test halt")
    
    # Spin up server on port 0
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_port
    
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    
    try:
        # Fetch metrics
        resp = urlopen(f"http://127.0.0.1:{port}/metrics")
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        
        metrics = {}
        for line in content.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                metrics[parts[0]] = float(parts[1])
                
        # Assert required metrics present
        assert "process_up" in metrics
        assert any(k.startswith("python_version{") for k in metrics)
        assert 'job_last_success_timestamp{job="earnings_scan"}' in metrics
        assert 'job_success_rate_1d{job="earnings_scan"}' in metrics
        assert metrics['job_success_rate_1d{job="earnings_scan"}'] == 1.0
        assert 'job_last_run_age_seconds{job="earnings_scan"}' in metrics
        
        assert metrics["equity_latest"] == 100000.0
        assert metrics["buying_power_latest"] == 95000.0
        assert metrics["killswitch_halted"] == 1.0
        
    finally:
        server.shutdown()
        t.join(timeout=1.0)
