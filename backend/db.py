import os
import time
import secrets
import hashlib
from sqlalchemy import Column, String, Float, Integer, Boolean, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./fences.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("sqlite://") and "+aiosqlite" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://", 1)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    # Hash of the API key that started the run. Every run query is scoped to it.
    owner_key_hash = Column(String, nullable=True, index=True)
    agent_name = Column(String, nullable=False)
    budget_usd = Column(Float, nullable=False)
    max_iterations = Column(Integer, nullable=False, default=100)
    max_duration_ms = Column(Integer, nullable=False, default=300_000)
    max_tokens = Column(Integer, nullable=False, default=0)
    spent_usd = Column(Float, nullable=False, default=0.0)
    iterations = Column(Integer, nullable=False, default=0)
    tokens_used = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="running")
    error = Column(String, nullable=True)
    started_at = Column(Float, nullable=False, default=time.time)
    ended_at = Column(Float, nullable=True)


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    timestamp = Column(Float, nullable=False, default=time.time)
    iteration = Column(Integer, nullable=False, default=0)
    reasoning = Column(String, nullable=False)
    action = Column(String, nullable=True)


class ApiKey(Base):
    __tablename__ = "api_keys"

    key_hash = Column(String, primary_key=True)
    label = Column(String, nullable=True)
    prefix = Column(String, nullable=False)
    created_at = Column(Float, nullable=False, default=time.time)
    revoked = Column(Boolean, nullable=False, default=False)
    last_used_at = Column(Float, nullable=True)


def generate_api_key() -> str:
    return f"fc_{secrets.token_urlsafe(32)}"


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _add_missing_columns(conn):
    # create_all() never alters existing tables, so columns added later are added here.
    # Runs created before ownership existed keep a NULL owner and are visible to no key.
    if "owner_key_hash" not in {c["name"] for c in inspect(conn).get_columns("runs")}:
        conn.execute(text("ALTER TABLE runs ADD COLUMN owner_key_hash VARCHAR"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_runs_owner_key_hash ON runs (owner_key_hash)"))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session