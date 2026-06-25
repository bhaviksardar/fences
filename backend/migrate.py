"""
Migration: add max_iterations and max_duration_ms to runs table.

Run this once against your production database before deploying feature 2.

Usage (from Railway console):
    /opt/venv/bin/python3 migrate.py
"""
import os
import sys

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./fences.db")

# Rewrite to sync driver for this script
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if "+asyncpg" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("+asyncpg", "", 1)
if "+aiosqlite" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("+aiosqlite", "", 1)

from sqlalchemy import create_engine, text

engine = create_engine(DATABASE_URL)
is_sqlite = DATABASE_URL.startswith("sqlite")

def add_column_if_missing(conn, table, column, col_type, default):
    """
    Adds a column only if it doesn't already exist.
    Postgres supports IF NOT EXISTS natively.
    SQLite doesn't, so we check the schema first.
    """
    if is_sqlite:
        # Check if column already exists
        result = conn.execute(text(f"PRAGMA table_info({table})"))
        existing = [row[1] for row in result.fetchall()]
        if column in existing:
            print(f"SKIP: {column} already exists")
            return
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} NOT NULL DEFAULT {default}"))
    else:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type} NOT NULL DEFAULT {default}"))
    conn.commit()
    print(f"OK: added {column}")


def run():
    with engine.connect() as conn:
        # Feature 2: iteration and time limits
        add_column_if_missing(conn, "runs", "max_iterations", "INTEGER", 100)
        add_column_if_missing(conn, "runs", "max_duration_ms", "INTEGER", 300000)
        add_column_if_missing(conn, "runs", "iterations", "INTEGER", 0)

        # Feature 3: decision audit trail — create decisions table if missing
        if is_sqlite:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    reasoning TEXT NOT NULL,
                    action TEXT
                )
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    reasoning TEXT NOT NULL,
                    action TEXT
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_decisions_run_id ON decisions (run_id)
            """))
        conn.commit()
        print("OK: decisions table ready")

    print("\nMigration complete.")

if __name__ == "__main__":
    run()