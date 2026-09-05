import threading
import time
from pathlib import Path

from sqlalchemy import text

from earnings_edge.db.engine import configure, session_scope


def test_busy_timeout(tmp_path: Path):
    db_path = tmp_path / "test_busy.db"

    configure(db_path)
    with session_scope() as session:
        session.execute(text("CREATE TABLE IF NOT EXISTS test_busy (id INTEGER PRIMARY KEY)"))

    start_event = threading.Event()

    def writer_one():
        with session_scope() as session:
            # Execute raw SQL so it goes straight to the driver
            session.execute(text("INSERT INTO test_busy (id) VALUES (999)"))
            start_event.set()
            time.sleep(1.0)

    t1 = threading.Thread(target=writer_one)
    t1.start()

    start_event.wait(timeout=2)

    start_time = time.time()
    with session_scope() as session:
        session.execute(text("INSERT INTO test_busy (id) VALUES (2)"))

    duration = time.time() - start_time
    t1.join()

    assert 0.5 < duration < 2.0
