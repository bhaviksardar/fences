from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import os
import time
import secrets

from db import init_db, get_session, Run, Decision, ApiKey, hash_api_key, generate_api_key

# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()  # creates tables on startup if they don't exist
    yield

app = FastAPI(title="Fences API — v1: budget enforcement", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


async def verify_api_key(
    x_api_key: str = Header(...),
    session: AsyncSession = Depends(get_session),
) -> str:
    """
    Looks up the key by its hash — the raw key is never stored, so this
    is the only way to validate it. Updates last_used_at as a side effect,
    which is useful for spotting keys that are no longer in use.
    """
    key_hash = hash_api_key(x_api_key)
    result = await session.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
    key_record = result.scalar_one_or_none()

    if key_record is None or key_record.revoked:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    key_record.last_used_at = time.time()
    await session.commit()
    return x_api_key


# ── Schemas ───────────────────────────────────────────────────────────────────

class StartRunPayload(BaseModel):
    run_id: str
    agent_name: str
    budget_usd: float = Field(gt=0)
    max_iterations: int = Field(default=100, gt=0)
    max_duration_ms: int = Field(default=300_000, gt=0)


class CheckpointPayload(BaseModel):
    cost_delta_usd: float = Field(ge=0)
    iterations: int = Field(ge=0)
    duration_ms: int = Field(ge=0)


class EndRunPayload(BaseModel):
    status: str
    error: Optional[str] = None


def run_to_dict(run: Run) -> dict:
    return {
        "run_id": run.run_id,
        "agent_name": run.agent_name,
        "budget_usd": run.budget_usd,
        "max_iterations": run.max_iterations,
        "max_duration_ms": run.max_duration_ms,
        "spent_usd": run.spent_usd,
        "iterations": run.iterations,
        "status": run.status,
        "error": run.error,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/runs/start")
async def start_run(
    payload: StartRunPayload,
    api_key: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    existing = await session.get(Run, payload.run_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"run_id '{payload.run_id}' already exists")

    run = Run(
        run_id=payload.run_id,
        agent_name=payload.agent_name,
        budget_usd=payload.budget_usd,
        max_iterations=payload.max_iterations,
        max_duration_ms=payload.max_duration_ms,
        spent_usd=0.0,
        iterations=0,
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
    api_key: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != "running":
        # Run already ended/breached — don't let a late checkpoint resurrect it
        # or quietly add more spend to a closed run.
        raise HTTPException(status_code=409, detail=f"Run is already '{run.status}', cannot checkpoint")

    # ATOMIC spend update
    await session.execute(
        update(Run)
        .where(Run.run_id == run_id)
        .values(
            spent_usd=Run.spent_usd + payload.cost_delta_usd,
            iterations=payload.iterations,
        )
    )
    await session.commit()
    await session.refresh(run)

    # Server-side enforcement of all three limits
    breach = None
    if run.spent_usd >= run.budget_usd:
        breach = "budget_exceeded"
    elif run.iterations >= run.max_iterations:
        breach = "iteration_limit"
    elif payload.duration_ms >= run.max_duration_ms:
        breach = "time_limit"

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
        }

    return {"ok": True, "spent_usd": run.spent_usd, "iterations": run.iterations}


@app.post("/api/runs/{run_id}/end")
async def end_run(
    run_id: str,
    payload: EndRunPayload,
    api_key: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Don't overwrite a server-detected "breached" status with the client's
    # opinion that it ended normally — the server's view wins.
    if run.status != "breached":
        run.status = payload.status
    run.error = payload.error
    run.ended_at = time.time()
    await session.commit()
    return {"ok": True}


@app.get("/api/runs")
async def list_runs(
    api_key: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Run).order_by(Run.started_at.desc()))
    runs = result.scalars().all()
    return {"runs": [run_to_dict(r) for r in runs]}


@app.get("/api/runs/{run_id}")
async def get_run(
    run_id: str,
    api_key: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run": run_to_dict(run)}


class DecisionPayload(BaseModel):
    iteration: int = Field(ge=0)
    reasoning: str = Field(min_length=1, max_length=2000)
    action: Optional[str] = Field(default=None, max_length=200)


@app.post("/api/runs/{run_id}/decisions")
async def log_decision(
    run_id: str,
    payload: DecisionPayload,
    api_key: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    run = await session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

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
    api_key: str = Depends(verify_api_key),
    session: AsyncSession = Depends(get_session),
):
    """Returns the full audit trail for a run in chronological order."""
    run = await session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

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


# ── Admin endpoints ───────────────────────────────────────────────────────────
# Protected by ADMIN_PASSWORD env var set in Railway.
# Never exposed in code — if ADMIN_PASSWORD is not set, these endpoints
# are completely disabled so a misconfigured deploy can't be exploited.

def verify_admin(x_admin_password: str = Header(...)) -> str:
    """
    Checks the X-Admin-Password header against the ADMIN_PASSWORD env var.
    Three protections:
    1. Uses secrets.compare_digest — prevents timing attacks where an
       attacker could guess the password one character at a time by
       measuring how long the comparison takes.
    2. Returns 404 (not 401) if ADMIN_PASSWORD isn't set in the environment
       — the endpoint appears to not exist rather than advertising itself
       as a real but locked door.
    3. Always compares both strings fully even if the first char differs
       — again, timing attack prevention.
    """
    expected = os.environ.get("ADMIN_PASSWORD")
    if not expected:
        # Env var not set — act like the endpoint doesn't exist
        raise HTTPException(status_code=404, detail="Not found")
    if not secrets.compare_digest(x_admin_password, expected):
        raise HTTPException(status_code=401, detail="Invalid admin password")
    return x_admin_password


class CreateKeyPayload(BaseModel):
    label: str = Field(min_length=1, max_length=64)


class RevokeKeyPayload(BaseModel):
    prefix: str = Field(min_length=4, max_length=16)


@app.post("/admin/keys/create")
async def admin_create_key(
    payload: CreateKeyPayload,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(verify_admin),
):
    """
    Generate a new API key. The raw key is returned exactly once —
    only its hash is stored, so it cannot be retrieved again.
    """
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
    """List all keys — shows prefix and metadata only, never the raw key."""
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
    """Revoke a key by its prefix. Takes effect immediately."""
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