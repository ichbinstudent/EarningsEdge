import pytest
from unittest.mock import patch
import designer_cli
import sys

def test_designer_cli_basic(capsys):
    # simulate: designer_cli.py --spot 100 --leg "buy call 100 2026-10-16 1 5.0 0.3" --forecast-rv 0.4
    args = ["designer_cli.py", "--spot", "100", "--leg", "buy call 100 2026-10-16 1 5.0 0.3", "--forecast-rv", "0.4"]
    with patch("sys.argv", args):
        ret = designer_cli.main()
        assert ret == 0
        captured = capsys.readouterr()
        assert "POSITION SUMMARY" in captured.out
        assert "RV SCENARIO SIMULATION" in captured.out
        assert "max_loss: -500.0" in captured.out

def test_designer_cli_parse_error(capsys):
    args = ["designer_cli.py", "--spot", "100", "--leg", "buy call 100"]
    with patch("sys.argv", args):
        ret = designer_cli.main()
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error parsing legs" in captured.out


def test_designer_cli_auto_price_iv_from_chain(tmp_path, capsys):
    from earnings_edge.db import configure, insert_options_chain_rows

    db = tmp_path / "c.db"
    configure(db)
    insert_options_chain_rows([{
        "ticker": "XYZ",
        "scan_date": "2026-08-21",
        "contract_ticker": "XYZ261016C00100000",
        "expiry": "2026-10-16",
        "strike": 100.0,
        "contract_type": "call",
        "volume": 100,
        "implied_volatility": 0.30,
        "delta": 0.50,
        "midpoint": 5.0,
        "close": 5.0,
    }])

    args = ["designer_cli.py", "--spot", "100", "--ticker", "XYZ",
            "--db", str(db),
            "--leg", "buy call 100 2026-10-16 1 auto auto", "--forecast-rv", "0.3"]
    with patch("sys.argv", args):
        ret = designer_cli.main()
        assert ret == 0
        out = capsys.readouterr().out
        assert "POSITION SUMMARY" in out
        assert "max_loss: -500.0" in out  # 100 * $5.00 mid from the chain


def test_designer_cli_auto_requires_ticker(capsys):
    args = ["designer_cli.py", "--spot", "100", "--leg", "buy call 100 2026-10-16 1 auto 0.3"]
    with patch("sys.argv", args):
        ret = designer_cli.main()
        assert ret == 1
        assert "--ticker" in capsys.readouterr().out
