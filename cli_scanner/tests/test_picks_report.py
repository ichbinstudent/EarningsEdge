import pytest
import sqlite3
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import datetime

import picks_report

def test_picks_report_no_db(capsys):
    with patch("picks_report.DEFAULT_DB", Path("/does/not/exist.db")):
        with patch("sys.argv", ["picks_report.py"]):
            ret = picks_report.main()
            assert ret == 1
            captured = capsys.readouterr()
            assert "database not found" in captured.out

def test_picks_report_with_date(capsys, tmp_path):
    db_path = tmp_path / "test.db"
    db_path.touch()
    
    with patch("picks_report.generate_picks") as mock_gen:
        mock_gen.return_value = {
            "earnings": pd.DataFrame({"ticker": ["AAPL"]})
        }
        with patch("sys.argv", ["picks_report.py", "--db", str(db_path), "--date", "2026-08-19"]):
            ret = picks_report.main()
            assert ret == 0
            captured = capsys.readouterr()
            assert "OQUANTS PICKS AS OF 2026-08-19" in captured.out
            assert "AAPL" in captured.out

def test_picks_report_auto_date(capsys, tmp_path):
    from earnings_edge.db import configure
    db_path = tmp_path / "test.db"
    configure(db_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO snapshots (ticker, earnings_date, scan_date) "
        "VALUES ('X', '2026-08-19', '2026-08-19 16:00:00')"
    )
    con.commit()
    con.close()

    with patch("picks_report.generate_picks") as mock_gen:
        mock_gen.return_value = {
            "momentum_skew": pd.DataFrame()
        }
        with patch("sys.argv", ["picks_report.py", "--db", str(db_path)]):
            ret = picks_report.main()
            assert ret == 0
            captured = capsys.readouterr()
            assert "Using latest snapshot date: 2026-08-19" in captured.out
            assert "No picks found." in captured.out
