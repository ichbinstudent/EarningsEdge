"""Framework-owned database tables.

Lives in the same SQLite database as ``earnings_edge.db`` (single-file
simplicity, WAL mode). Schema is created by ``earnings_edge.db.engine.configure``
(create_all + run_migrations). Query via ``earnings_edge.db.repositories``.
"""
