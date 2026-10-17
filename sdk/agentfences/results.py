"""What checkpoint() returns, and the words a breach comes with: for the user, and for the model."""
from dataclasses import dataclass
from typing import Optional


# ── CheckpointResult ──────────────────────────────────────────────────────────

@dataclass
class CheckpointResult:
    """
    Returned by checkpoint() after every call.

    If breached is False, the agent is within all limits and should continue.
    If breached is True, the agent has crossed a limit and should stop or
    handle gracefully — the message field is ready to return to the user
    or inject into the LLM conversation as a system message.

    Usage — programmatic check:
        result = await checkpoint(cost_delta_usd=0.02, tokens_used=450)
        if result.breached:
            return result.message

    Usage — inject into LLM conversation:
        result = await checkpoint(cost_delta_usd=0.02, tokens_used=450)
        if result.breached:
            messages.append({"role": "system", "content": result.system_prompt})
            final = client.chat.completions.create(model=MODEL, messages=messages)
            return final.choices[0].message.content
    """
    breached: bool = False
    breach_type: Optional[str] = None   # "budget_exceeded" | "iteration_limit" | "time_limit" | "token_limit"
    message: str = ""                   # human-readable, ready to return to user
    system_prompt: str = ""             # inject into LLM conversation context
    spent_usd: float = 0.0              # the run's total spend so far

    @property
    def ok(self) -> bool:
        return not self.breached


def breach_result(breach_type: str, **kwargs) -> CheckpointResult:
    messages = {
        "budget_exceeded": (
            f"I've reached my budget limit (spent ${kwargs.get('spent', 0):.4f} "
            f"of ${kwargs.get('limit', 0):.4f}). I'll summarize what I found so far."
        ),
        "iteration_limit": (
            f"I've reached my iteration limit ({kwargs.get('iterations', 0)} steps). "
            "I'll summarize what I found so far."
        ),
        "time_limit": (
            f"I've reached my time limit ({kwargs.get('duration_ms', 0)}ms). "
            "I'll summarize what I found so far."
        ),
        "token_limit": (
            f"I've reached my token limit ({kwargs.get('tokens_used', 0):,} tokens). "
            "I'll summarize what I found so far."
        ),
        "stopped_by_user": "I was stopped from the Fences dashboard. I'll summarize what I found so far.",
        "key_daily_budget": "My daily spending cap has been reached. I'll summarize what I found so far.",
        "key_monthly_budget": "My monthly spending cap has been reached. I'll summarize what I found so far.",
        "fences_unreachable": (
            "I can't reach Fences to confirm I'm within my limits, so I'm stopping to be safe. "
            "I'll summarize what I found so far."
        ),
        "paused": "I was paused from the Fences dashboard and not resumed in time. I'll summarize what I found so far.",
    }

    system_prompts = {
        "budget_exceeded": (
            f"You have reached your budget limit (${kwargs.get('spent', 0):.4f} of "
            f"${kwargs.get('limit', 0):.4f} spent). Stop your current task immediately "
            "and provide a clear summary of what you have found or completed so far. "
            "Tell the user you stopped due to budget and what you accomplished."
        ),
        "iteration_limit": (
            f"You have reached your iteration limit ({kwargs.get('iterations', 0)} steps). "
            "Stop your current task immediately and summarize what you have found or "
            "completed so far. Tell the user you stopped due to the iteration limit."
        ),
        "time_limit": (
            "You have reached your time limit. Stop your current task immediately "
            "and summarize what you have found or completed so far. "
            "Tell the user you stopped due to the time limit."
        ),
        "token_limit": (
            f"You have reached your token limit ({kwargs.get('tokens_used', 0):,} tokens used). "
            "Stop your current task immediately and summarize what you have found or "
            "completed so far. Tell the user you stopped due to the token limit."
        ),
        "stopped_by_user": (
            "A person stopped this run from the Fences dashboard. Stop your current task immediately "
            "and summarize what you have found or completed so far. Tell the user you were stopped by an operator."
        ),
        "key_daily_budget": (
            "Your daily spending cap, across all of your runs, has been reached. Stop your current task "
            "immediately and summarize what you have found or completed so far. Tell the user you stopped "
            "because the daily cap was reached."
        ),
        "key_monthly_budget": (
            "Your monthly spending cap, across all of your runs, has been reached. Stop your current task "
            "immediately and summarize what you have found or completed so far. Tell the user you stopped "
            "because the monthly cap was reached."
        ),
        "fences_unreachable": (
            "Your governance service can't be reached and you are configured to stop when that happens. "
            "Stop your current task immediately and summarize what you have found or completed so far."
        ),
        "paused": (
            "A person paused this run from the Fences dashboard and it was not resumed in time. Stop your current "
            "task and summarize what you have found or completed so far. Tell the user the run was paused by an operator."
        ),
    }

    msg = messages.get(breach_type, "Governance limit reached.")
    sys_prompt = system_prompts.get(breach_type, "A governance limit has been reached. Summarize what you have done so far.")

    return CheckpointResult(
        breached=True,
        breach_type=breach_type,
        message=msg,
        system_prompt=sys_prompt,
    )
