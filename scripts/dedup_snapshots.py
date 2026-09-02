#!/usr/bin/env python3
"""Dedup snapshots by (ticker, earnings_date, scan_date, timing, data_source), keeping lowest id."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earnings_edge.db import DEFAULT_DB_PATH, snapshots_dedup


def main() -> None:
    db_path = str(DEFAULT_DB_PATH)
    shutil.copy2(db_path, db_path + ".dedup.bak")
    deleted, remaining = snapshots_dedup()
    print(f"Deleted {deleted} duplicate rows")
    print(f"Remaining duplicate groups: {remaining}")


if __name__ == "__main__":
    main()
