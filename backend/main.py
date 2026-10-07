import os
import time
import secrets
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, Field
from typing import Optional, Literal
from contextlib import asynccontextmanager

from db import init_db, get_session, Run, Decision, ApiKey, hash_api_key, generate_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Fences API", lifespan=lifespan)

# The SDK calls the API from servers, which CORS doesn't affect. Only browser apps
# need an entry here: CORS_ORIGINS=https://app.example.com,https://admin.example.com
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "600"))
_window = 0
_hits: dict = {}  # key hash -> requests in the current minute


def rate_limit(who: str, limit: int):
    # ponytail: in-memory fixed window, per process. Move to Redis if you run more than one instance.
    global _window
    window = int(time.time() // 60)
    if window != _window:  # new minute: drop every old count
        _window = window
        _hits.clear()
    _hits[who] = _hits.get(who, 0) + 1
    if _hits[who] > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again in a minute")


async def verify_api_key(
    x_api_key: str = Header(...),
    session: AsyncSession = Depends(get_session),
) -> str:
    """Returns the key's hash, which identifies the caller's tenant."""
    key_hash = hash_api_key(x_api_key)
    result = await session.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
    key_record = result.scalar_one_or_none()

    if key_record is None or key_record.revoked:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    rate_limit(key_hash, RATE_LIMIT_PER_MINUTE)

    key_record.last_used_at = time.time()
    await session.commit()
    return key_hash


async def owned_run(session: AsyncSession, run_id: str, key_hash: str) -> Run:
    # Another tenant's run is indistinguishable from a missing one
    run = await session.get(Run, run_id)
    if run is None or run.owner_key_hash != key_hash:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


class StartRunPayload(BaseModel):
    run_id: str
    agent_name: str
    budget_usd: float = Field(gt=0)
    max_iterations: int = Field(default=100, gt=0)
    max_duration_ms: int = Field(default=300_000, gt=0)
    max_tokens: int = Field(default=0, ge=0)


class CheckpointPayload(BaseModel):
    cost_delta_usd: float = Field(ge=0)
    iterations: int = Field(ge=0)
    duration_ms: int = Field(default=0, ge=0)  # accepted from older SDKs, ignored: the server times runs itself
    tokens_used: int = Field(default=0, ge=0)


class EndRunPayload(BaseModel):
    status: Literal["success", "error", "breached"]
    error: Optional[str] = None


class DecisionPayload(BaseModel):
    iteration: int = Field(ge=0)
    reasoning: str = Field(min_length=1, max_length=2000)
    action: Optional[str] = Field(default=None, max_length=200)


class CreateKeyPayload(BaseModel):
    label: str = Field(min_length=1, max_length=64)


class RevokeKeyPayload(BaseModel):
    prefix: str = Field(min_length=4, max_length=16)


def run_to_dict(run: Run) -> dict:
    return {
        "run_id": run.run_id,
        "agent_name": run.agent_name,
        "budget_usd": run.budget_usd,
        "max_iterations": run.max_iterations,
        "max_duration_ms": run.max_duration_ms,
        "max_tokens": run.max_tokens,
        "spent_usd": run.spent_usd,
        "iterations": run.iterations,
        "tokens_used": run.tokens_used,
        "status": run.status,
        "error": run.error,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
    }


@app.post("/api/runs/start")
async def start_run(
    payload: StartRunPayload,
    key_hash: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    existing = await session.get(Run, payload.run_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"run_id '{payload.run_id}' already exists")

    run = Run(
        run_id=payload.run_id,
        owner_key_hash=key_hash,
        agent_name=payload.agent_name,
        budget_usd=payload.budget_usd,
        max_iterations=payload.max_iterations,
        max_duration_ms=payload.max_duration_ms,
        max_tokens=payload.max_tokens,
        spent_usd=0.0,
        iterations=0,
        tokens_used=0,
        status="running",
        started_at=time.time(),
    )
    session.add(run)
    await session.commit()
    return {"ok": True}


@app.post("/api/runs/{run_id}/checkpoint")
async def checkpoint(
    run_id: str,
    payload: CheckpointPayload,
    key_hash: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await owned_run(session, run_id, key_hash)

    if run.status != "running":
        raise HTTPException(status_code=409, detail=f"Run is already '{run.status}', cannot checkpoint")

    await session.execute(
        update(Run)
        .where(Run.run_id == run_id)
        .values(
            spent_usd=Run.spent_usd + payload.cost_delta_usd,
            iterations=payload.iterations,
            tokens_used=Run.tokens_used + payload.tokens_used,
        )
    )
    await session.commit()
    await session.refresh(run)

    # Time is measured on the server, so a client can't extend its own time limit
    duration_ms = int((time.time() - run.started_at) * 1000)

    breach = None
    if round(run.spent_usd, 9) > run.budget_usd:  # round away float drift (0.02*5 != 0.10)
        breach = "budget_exceeded"
    elif run.iterations > run.max_iterations:
        breach = "iteration_limit"
    elif duration_ms > run.max_duration_ms:
        breach = "time_limit"
    elif run.max_tokens > 0 and run.tokens_used > run.max_tokens:
        breach = "token_limit"

    if breach:
        run.status = "breached"
        await session.commit()
        return {
            "ok": False,
            "breach": breach,
            "spent_usd": run.spent_usd,
            "budget_usd": run.budget_usd,
            "iterations": run.iterations,
            "max_iterations": run.max_iterations,
            "tokens_used": run.tokens_used,
            "max_tokens": run.max_tokens,
        }

    return {
        "ok": True,
        "spent_usd": run.spent_usd,
        "iterations": run.iterations,
        "tokens_used": run.tokens_used,
    }


@app.post("/api/runs/{run_id}/end")
async def end_run(
    run_id: str,
    payload: EndRunPayload,
    key_hash: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await owned_run(session, run_id, key_hash)

    if run.status != "breached":
        run.status = payload.status
    run.error = payload.error
    run.ended_at = time.time()
    await session.commit()
    return {"ok": True}


@app.get("/api/runs")
async def list_runs(
    key_hash: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Run).where(Run.owner_key_hash == key_hash).order_by(Run.started_at.desc())
    )
    runs = result.scalars().all()
    return {"runs": [run_to_dict(r) for r in runs]}


@app.get("/api/runs/{run_id}")
async def get_run(
    run_id: str,
    key_hash: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await owned_run(session, run_id, key_hash)
    return {"run": run_to_dict(run)}


@app.post("/api/runs/{run_id}/decisions")
async def log_decision(
    run_id: str,
    payload: DecisionPayload,
    key_hash: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await owned_run(session, run_id, key_hash)

    session.add(Decision(
        run_id=run_id,
        iteration=payload.iteration,
        reasoning=payload.reasoning,
        action=payload.action,
        timestamp=time.time(),
    ))
    await session.commit()
    return {"ok": True}


@app.get("/api/runs/{run_id}/decisions")
async def get_decisions(
    run_id: str,
    key_hash: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await owned_run(session, run_id, key_hash)

    result = await session.execute(
        select(Decision)
        .where(Decision.run_id == run_id)
        .order_by(Decision.timestamp.asc())
    )
    decisions = result.scalars().all()
    return {
        "run_id": run_id,
        "agent_name": run.agent_name,
        "status": run.status,
        "decisions": [
            {
                "iteration": d.iteration,
                "timestamp": d.timestamp,
                "reasoning": d.reasoning,
                "action": d.action,
            }
            for d in decisions
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


ADMIN_PASSWORD_MIN_LENGTH = 32


def verify_admin(x_admin_password: str = Header(...)) -> str:
    # No rate limit here: behind a proxy every caller shares one IP, so a limit would let
    # anyone lock the admin out. A long random password makes guessing hopeless instead.
    expected = os.environ.get("ADMIN_PASSWORD")
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    if len(expected) < ADMIN_PASSWORD_MIN_LENGTH:
        raise HTTPException(
            status_code=503,
            detail=f"ADMIN_PASSWORD must be at least {ADMIN_PASSWORD_MIN_LENGTH} characters, e.g. `openssl rand -hex 32`",
        )
    if not secrets.compare_digest(x_admin_password, expected):
        raise HTTPException(status_code=401, detail="Invalid admin password")
    return x_admin_password


@app.post("/admin/keys/create")
async def admin_create_key(
    payload: CreateKeyPayload,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(verify_admin),
):
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    prefix = raw_key[:12]

    existing = await session.get(ApiKey, key_hash)
    if existing:
        raise HTTPException(status_code=409, detail="Key collision — try again")

    session.add(ApiKey(
        key_hash=key_hash,
        label=payload.label,
        prefix=prefix,
        created_at=time.time(),
        revoked=False,
    ))
    await session.commit()

    return {
        "key": raw_key,
        "prefix": prefix,
        "label": payload.label,
        "warning": "Save this now — it will not be shown again"
    }


@app.get("/admin/keys")
async def admin_list_keys(
    session: AsyncSession = Depends(get_session),
    _: str = Depends(verify_admin),
):
    result = await session.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
    keys = result.scalars().all()
    return {"keys": [
        {
            "prefix": k.prefix,
            "label": k.label,
            "revoked": k.revoked,
            "created_at": k.created_at,
            "last_used_at": k.last_used_at,
        }
        for k in keys
    ]}


@app.post("/admin/keys/revoke")
async def admin_revoke_key(
    payload: RevokeKeyPayload,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(verify_admin),
):
    result = await session.execute(
        select(ApiKey).where(ApiKey.prefix == payload.prefix)
    )
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    if key.revoked:
        raise HTTPException(status_code=409, detail="Key already revoked")

    key.revoked = True
    await session.commit()
    return {"ok": True, "revoked": payload.prefix}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)