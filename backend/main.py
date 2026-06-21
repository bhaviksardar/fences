from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import time

from db import init_db, get_session, Run, ApiKey, hash_api_key

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
    budget_usd: float = Field(gt=0, description="Must be a positive amount")


class CheckpointPayload(BaseModel):
    # Disallow negative deltas — otherwise an agent could "refund" its way
    # out of a breach by reporting negative spend.
    cost_delta_usd: float = Field(ge=0)


class EndRunPayload(BaseModel):
    status: str
    error: Optional[str] = None


def run_to_dict(run: Run) -> dict:
    return {
        "run_id": run.run_id,
        "agent_name": run.agent_name,
        "budget_usd": run.budget_usd,
        "spent_usd": run.spent_usd,
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
        spent_usd=0.0,
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

    # ATOMIC update: increments spent_usd directly in the database in one
    # statement, so two simultaneous checkpoint calls for the same run_id
    # can't race and silently drop one of the increments (which they would
    # if we did "read value, add in Python, write value back" instead).
    await session.execute(
        update(Run)
        .where(Run.run_id == run_id)
        .values(spent_usd=Run.spent_usd + payload.cost_delta_usd)
    )
    await session.commit()

    # Re-fetch to see the authoritative post-update value
    await session.refresh(run)

    if run.spent_usd >= run.budget_usd:
        run.status = "breached"
        await session.commit()
        return {"ok": False, "spent_usd": run.spent_usd, "budget_usd": run.budget_usd}

    return {"ok": True, "spent_usd": run.spent_usd, "budget_usd": run.budget_usd}


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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)