"""
CLI for managing API keys — run this locally to create your first key,
since there's no signup flow yet.

Usage:
    python3 manage_keys.py create "my laptop"
    python3 manage_keys.py list
    python3 manage_keys.py revoke fc_a1b2c3...
"""
import asyncio
import sys
from sqlalchemy import select
from db import SessionLocal, init_db, ApiKey, generate_api_key, hash_api_key


async def create_key(label: str):
    await init_db()
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    prefix = raw_key[:12]

    async with SessionLocal() as session:
        session.add(ApiKey(key_hash=key_hash, label=label, prefix=prefix))
        await session.commit()

    print(f"\nCreated API key for '{label}':\n")
    print(f"  {raw_key}\n")
    print("Save this now — it will not be shown again. Only a hash is stored.")


async def list_keys():
    await init_db()
    async with SessionLocal() as session:
        result = await session.execute(select(ApiKey))
        keys = result.scalars().all()

    if not keys:
        print("No API keys found.")
        return

    print(f"{'Prefix':<16}{'Label':<24}{'Revoked':<10}{'Last used'}")
    for k in keys:
        last_used = "never" if not k.last_used_at else str(k.last_used_at)
        print(f"{k.prefix:<16}{(k.label or ''):<24}{str(k.revoked):<10}{last_used}")


async def revoke_key(raw_key_or_prefix: str):
    await init_db()
    async with SessionLocal() as session:
        # Allow revoking by full key (hash it) or by matching the stored prefix
        if raw_key_or_prefix.startswith("fc_") and len(raw_key_or_prefix) > 20:
            target_hash = hash_api_key(raw_key_or_prefix)
            result = await session.execute(select(ApiKey).where(ApiKey.key_hash == target_hash))
        else:
            result = await session.execute(select(ApiKey).where(ApiKey.prefix == raw_key_or_prefix))

        key = result.scalar_one_or_none()
        if not key:
            print("No matching key found.")
            return

        key.revoked = True
        await session.commit()
        print(f"Revoked key with prefix {key.prefix}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "create":
        label = sys.argv[2] if len(sys.argv) > 2 else "unlabeled"
        asyncio.run(create_key(label))
    elif command == "list":
        asyncio.run(list_keys())
    elif command == "revoke":
        if len(sys.argv) < 3:
            print("Usage: python3 manage_keys.py revoke <key-or-prefix>")
            sys.exit(1)
        asyncio.run(revoke_key(sys.argv[2]))
    else:
        print(f"Unknown command: {command}")
        print(__doc__)