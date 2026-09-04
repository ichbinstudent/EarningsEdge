from unittest.mock import MagicMock, patch

from bot import _watchdog_loop, sd_notify


def test_sd_notify_no_socket(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    with patch("socket.socket") as mock_sock:
        sd_notify("READY=1")
        mock_sock.assert_not_called()


def test_sd_notify_with_socket(monkeypatch):
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    mock_sock_instance = MagicMock()
    with patch("socket.socket", return_value=mock_sock_instance) as mock_sock:
        # We need mock_sock_instance to be returned by context manager __enter__
        mock_sock_instance.__enter__.return_value = mock_sock_instance
        sd_notify("READY=1")
        mock_sock_instance.sendto.assert_called_once_with(b"READY=1", "/run/systemd/notify")


def test_sd_notify_abstract_socket(monkeypatch):
    monkeypatch.setenv("NOTIFY_SOCKET", "@/run/systemd/notify")
    mock_sock_instance = MagicMock()
    with patch("socket.socket", return_value=mock_sock_instance) as mock_sock:
        mock_sock_instance.__enter__.return_value = mock_sock_instance
        sd_notify("READY=1")
        mock_sock_instance.sendto.assert_called_once_with(b"READY=1", "\0/run/systemd/notify")


def test_watchdog_loop_noop(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    with patch("time.sleep") as mock_sleep:
        _watchdog_loop()
        mock_sleep.assert_not_called()
