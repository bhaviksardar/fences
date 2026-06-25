"""
Database setup for Fences.

Uses SQLite locally (zero setup — no separate DB server needed to run tests
or develop) and Postgres in production. Swapping is a single environment
variable change (DATABASE_URL) — the SQLAlchemy models and queries don't
need to change.
"""
import os
import time
import secrets
import hashlib
from sqlalchemy import Column, String, Float, Integer, Boolean, select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# ── Connection ────────────────────────────────────────────────────────────────
import sys

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./fences.db")

print(f"[fences] Raw DATABASE_URL prefix: {DATABASE_URL[:30]}...", file=sys.stderr)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

print(f"[fences] Using DATABASE_URL prefix: {DATABASE_URL[:30]}...", file=sys.stderr)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class Run(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    agent_name = Column(String, nullable=False)
    budget_usd = Column(Float, nullable=False)
    max_iterations = Column(Integer, nullable=False, default=100)
    max_duration_ms = Column(Integer, nullable=False, default=300_000)
    spent_usd = Column(Float, nullable=False, default=0.0)
    iterations = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="running")  # running | success | breached | error
    error = Column(String, nullable=True)
    started_at = Column(Float, nullable=False, default=time.time)
    ended_at = Column(Float, nullable=True)


class Decision(Base):
    """
    One row per log_decision() call. Builds the human-readable audit trail
    of WHY the agent did what it did, not just what happened.
    """
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    timestamp = Column(Float, nullable=False, default=time.time)
    iteration = Column(Integer, nullable=False, default=0)
    reasoning = Column(String, nullable=False)
    action = Column(String, nullable=True)


class ApiKey(Base):
    """
    Stores only a HASH of each key, never the plaintext key itself.
    The raw key is shown to the user exactly once, at creation time,
    and can never be retrieved again — same pattern Stripe/GitHub use.
    """
    __tablename__ = "api_keys"

    key_hash = Column(String, primary_key=True)
    label = Column(String, nullable=True)
    prefix = Column(String, nullable=False)
    created_at = Column(Float, nullable=False, default=time.time)
    revoked = Column(Boolean, nullable=False, default=False)
    last_used_at = Column(Float, nullable=True)


# ── Key generation / hashing helpers ─────────────────────────────────────────

def generate_api_key() -> str:
    """Generates a new random API key, prefixed for easy visual identification."""
    return f"fc_{secrets.token_urlsafe(32)}"


def hash_api_key(raw_key: str) -> str:
    """One-way hash used for storage and lookup — never store the raw key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ── Setup helper ──────────────────────────────────────────────────────────────

async def init_db():
    """Create tables if they don't exist. Call once at app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """FastAPI dependency — yields a session per request, closes it after."""
    async with SessionLocal() as session:
        yield session