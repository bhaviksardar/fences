import sys
import os
import time
import secrets
import hashlib
from sqlalchemy import create_engine, text, Column, String, Float, Boolean
from sqlalchemy.orm import declarative_base, Session

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./fences.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if "+asyncpg" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("+asyncpg", "", 1)
if "+aiosqlite" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("+aiosqlite", "", 1)

engine = create_engine(DATABASE_URL)
Base = declarative_base()


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


def ensure_tables():
    Base.metadata.create_all(engine)


def create_key(label: str):
    ensure_tables()
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    prefix = raw_key[:12]

    with Session(engine) as session:
        session.add(ApiKey(
            key_hash=key_hash,
            label=label,
            prefix=prefix,
            created_at=time.time(),
            revoked=False,
        ))
        session.commit()

    print(f"\nCreated API key for '{label}':\n")
    print(f"  {raw_key}\n")
    print("Save this now — it will not be shown again.")


def list_keys():
    ensure_tables()
    with Session(engine) as session:
        keys = session.query(ApiKey).all()

    if not keys:
        print("No API keys found.")
        return

    print(f"{'Prefix':<16}{'Label':<24}{'Revoked':<10}{'Last used'}")
    for k in keys:
        last_used = "never" if not k.last_used_at else str(round(k.last_used_at))
        print(f"{k.prefix:<16}{(k.label or ''):<24}{str(k.revoked):<10}{last_used}")


def revoke_key(raw_key_or_prefix: str):
    ensure_tables()
    with Session(engine) as session:
        if len(raw_key_or_prefix) > 20:
            target_hash = hash_api_key(raw_key_or_prefix)
            key = session.query(ApiKey).filter(ApiKey.key_hash == target_hash).first()
        else:
            key = session.query(ApiKey).filter(ApiKey.prefix == raw_key_or_prefix).first()

        if not key:
            print("No matching key found.")
            return

        key.revoked = True
        session.commit()
        print(f"Revoked key with prefix {key.prefix}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "create":
        label = sys.argv[2] if len(sys.argv) > 2 else "unlabeled"
        create_key(label)
    elif command == "list":
        list_keys()
    elif command == "revoke":
        if len(sys.argv) < 3:
            print("Usage: python3 manage_keys.py revoke <key-or-prefix>")
            sys.exit(1)
        revoke_key(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print(__doc__)