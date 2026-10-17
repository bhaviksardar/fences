# Enforcement guarantees

What Fences guarantees about an agent's limits, at each of its three levels, and what it can't. Each guarantee names the check that proves it, or is marked **(server)** when it rests on the Fences server implementing [PROTOCOL.md](PROTOCOL.md). These live in [`tests/test_enforcement.py`](tests/test_enforcement.py) unless marked otherwise, and run in CI on Python 3.9 to 3.14. This covers the Python SDK 0.9.0. The TypeScript SDK 0.9.0 has the same three levels, tested in `sdk-js/test/enforcement.test.mjs` and `sdk-js/test/instrument.test.mjs`, with two differences. It prechecks model calls (level 1) made through `instrument(client)` (OpenAI and Anthropic), as it has no framework integrations yet. And JavaScript can't stop running code, so a stuck run's governed call rejects with `FencesStop` and the run's `signal` aborts, but the agent's work only stops if it uses that signal (see the [support table](README.md#python-and-typescript)).

## The three levels

| Level | What it does | Strength |
|---|---|---|
| 1. Budgets and checkpoints | Each model call is prechecked against the budget before it's sent; `checkpoint()` after each step checks spend, steps, time and tokens | **Prevents** overspend on model calls Fences sees; **detects** the other limits after the step that crossed them |
| 2. Live control | A heartbeat brings stop, pause and limit changes from the dashboard; a stuck run is cancelled or halted | **Interrupts** a run, including one that isn't checkpointing |
| 3. Guarded tools | `@tool(guard=True)` asks Fences before the tool runs, and reserves its cost | **Prevents** the action: the tool's code never runs unless allowed |

Levels 1 and 2 are cooperative: they work through the agent's own loop. Level 3 is the boundary for actions with consequences. The model can't get past it, because the check sits in the tool's own code, ahead of anything it does.

## Level 1: budgets and checkpoints

**Guaranteed**
- **A model call that could take the run over its budget isn't made.** With `instrument=True` (OpenAI and Anthropic clients), the LangChain handler or the OpenAI Agents integration, Fences works out each call's worst case before it's sent. That's its input, estimated high from the request, at the dearest input rate, plus its output cap at the output rate. The output cap is `max_tokens` (or `max_completion_tokens`, `max_output_tokens`) if the call sets one, otherwise the model's maximum. If spend plus that worst case could pass the budget, the call raises `FencesStop` (`budget_exceeded`) and never reaches the provider. Checks: `test_instrument.py`: `prechecks`; `test_langchain.py`: `check_graph_stops_on_budget`; `test_openai_agents.py`: `check_stops_on_budget`; `check_model_call_worst_case`.
- **Parallel model calls can't overspend together.** Each call's worst case is held while it's in flight, then replaced by its actual cost. A failed call is released, since providers don't bill it. Check: `check_parallel_model_calls_cannot_overspend`.
- A run stops being allowed once a limit is **exceeded**. The rules are the same locally and on the server: spend is compared after rounding to 9 decimals, and when several limits trip at once the order is spend, steps, time, tokens. Checks: `test_local.py`: `check_limits_trip_once_exceeded`, `check_time_limit`.
- In cloud mode the server is authoritative. It sees spend from other processes and the key's daily and monthly caps across all of an agent's runs, and the SDK adopts its totals and limits on every reply. Checks: `test_local.py`: `check_server_limits_are_adopted`, `check_dashboard_owned_limits`; key caps **(server)**, tested in `fences-platform` `tests/test_auth.py`.
- **Network partition:** if the server can't be reached, the SDK enforces the same limits locally with the numbers it last had, or, with `init(fail_closed=True)`, stops the run with `fences_unreachable`. The most a run can overshoot meanwhile is one step per process for a limit only the server sees in full (a daily or monthly cap shared across runs), plus any model call that wasn't prechecked. Check: `check_network_partition`.
- **An outage doesn't slow every step.** After 3 failed calls in a row, a circuit breaker skips calls to Fences for about 30 seconds (with jitter), then lets one through to test the server; meanwhile checkpoints are enforced locally at once and guarded tools are blocked at once. Heartbeats back off with jitter. Checks: `test_local.py`: `check_circuit_breaker`; `check_heartbeats_back_off_while_failing`.
- **A retried step is counted once.** Every checkpoint carries a `step_id`. Once the server shows it deduplicates them (by echoing the id), a checkpoint that hits a connection error is sent once more with the same id; a timeout is never retried. Guard calls are retried the same way, under their `guard_id`. Check: `test_local.py`: `check_step_ids_and_one_retry`.
- **No double counting:** a checkpoint is retried only under its `step_id`, on a server that counts a step once; otherwise one lost in transit is enforced locally instead of being sent twice.
- **Costs can't switch the budget off.** A cost that isn't a finite number of zero or more (`NaN`, negative, infinite, a string) raises instead of being added, in `checkpoint()` and in a guarded tool's `cost_usd`; a custom price without an input and an output price is refused at `init()`. A null or non-numeric value in a server reply is ignored rather than adopted. Checks: `test_local.py`: `check_bad_amounts_are_refused`, `check_custom_prices_are_validated`, `check_revoked_key_and_odd_replies`.
- **A key revoked during a run ends it.** Its next checkpoint returns `fences_unreachable`, and guarded tools are blocked; nothing raises over the agent's own result or exception. Check: `test_local.py`: `check_revoked_key_and_odd_replies`.

**Not guaranteed**
- **Model calls Fences doesn't see aren't prechecked.** If you call a model yourself and only report it with `checkpoint(response)`, the step that crosses the budget has already been paid for, so the run can overspend by one call. Use `instrument=True` or a framework integration to have every call prechecked.
- **The input estimate is generous, not exact.** It assumes 2 bytes per token (real tokenizers average 3 to 4 for English and code) and 2,000 tokens per embedded image or file. A request whose real token count beats that, such as a very large image, can cost more than its worst case. Output is bounded exactly.
- **Prechecks are strict.** A call that *could* overspend is refused even if it wouldn't have. Without `max_tokens`, the worst case uses the model's whole output limit (16,384 tokens for gpt-4o), so small budgets need `max_tokens` on each call. The refusal says so in the logs.
- **A model with no known price isn't prechecked.** It's counted in tokens but not cost, with a warning; add its price with `init(prices=...)`.
- **Key caps aren't prechecked.** Concurrent runs sharing a key's daily or monthly cap can each overshoot it by a call. The server checks the cap at every checkpoint, but it can't refuse a call it doesn't see in advance.

## Level 2: live control

**Guaranteed**
- **A stop reaches the agent within one heartbeat** (15 seconds by default), even between checkpoints. Its next checkpoint returns `stopped_by_user`, and a stop sticks whatever later replies say. Check: `test_local.py`: `check_heartbeat_stop_and_limits`.
- **A stop during a hanging tool:** an async run that doesn't reach a checkpoint within `stop_grace_s` (default 30 seconds) of a stop is cancelled. The governed call raises `FencesStop`, the run ends as `breached`, and the heartbeat has already told the server the stop was received. Check: `check_stop_during_hanging_tool`.
- **A stuck sync run is halted too.** `FencesStop` is raised in its thread: at once in Python code, even under an `except Exception`, or when a blocking call it's in returns. A halt is never left to land after the run's own code has finished. A streaming run stuck in its own code is cancelled, but a slow consumer isn't. Checks: `check_stuck_sync_run_is_halted`, `check_stuck_stream_is_cancelled`.
- **An agent that saw its stop isn't cut off.** Once a checkpoint (or a blocked tool) has handed the agent its stop, it's never cancelled, so it can write its summary. Check: `check_stop_seen_at_checkpoint_is_not_cancelled`.
- **Tool timeouts:** `@tool(timeout_s=...)` raises `TimeoutError` in time, whether the tool is async or sync. Check: `check_hanging_tool_times_out`.
- **Pause** holds the agent at its next checkpoint and expires after `pause_timeout_s`. Check: `test_local.py`: `check_pause_resume_and_timeout`.
- **The dashboard can tell received from unanswered.** The SDK acknowledges every command it receives (stop, pause, quarantine, resume), by its id, in the next heartbeat, which it sends at once. A lost heartbeat's acknowledgements go with the next one. A command with no acknowledgement within 30 seconds marks the agent as not acknowledging **(server)**. Check: `test_local.py`: `check_acks_and_containment_hooks`.
- **Containment hooks run, and report back.** `on_quarantine` runs once per quarantine in each process, and `on_stop` once per run. Whether each worked, failed (with the error) or timed out is acknowledged to the server. A process that exits right after a stop finishes its hooks and sends their results first. Checks: `check_acks_and_containment_hooks`, `check_hooks_finish_before_exit`.
- **A quarantine blocks guarded tools at once,** in the SDK as well as on the server, without waiting for the pause it brings. Check: `test_local.py`: `check_quarantine_blocks_guarded_tools`.

**Not guaranteed**
- **A sync run stuck inside a blocking call is halted only when that call returns.** Python can't interrupt a thread waiting in C (a socket read with no timeout, a lock), and the halt needs CPython: elsewhere a sync run stops at its next checkpoint or guarded tool. Code that catches `BaseException` and carries on can swallow the halt. A timed-out sync tool keeps running in a background thread; the agent just stops waiting for it.
- **A dead or cut-off process** can't hear a stop or run its hooks. The server sees no heartbeat and no acknowledgement; it can't end the process. Containment that must happen anyway belongs on the server side: webhooks the server fires, and credentials it can revoke itself.
- A streaming run is cancelled only while its own code runs. If its consumer is the part that's stuck, the stream stops at its next checkpoint.

## Level 3: guarded tools

**Guaranteed**
- **Blocked means not run.** A guarded tool runs only after Fences allows it. A stopped, paused, over-limit or quarantined run, or a missing approval, raises `ToolBlocked` before any of the tool's code runs. Check: `check_blocked_tools_never_run`.
- **Parallel calls can't overspend together.** The cost is reserved before the call, and checking and reserving is one atomic step: in the SDK for one process (async tasks or threads), and **(server)** across processes. Five parallel $0.40 calls against a $1 budget: exactly two run. In cloud mode the SDK also counts its own in-flight model calls before asking the server, which can't see them. Checks: `check_parallel_calls_cannot_overspend`, `check_guard_counts_model_calls_in_flight`.
- **Settled whatever happens.** A call that raises is released uncharged. A call that times out or is cancelled is charged in full, because it may have happened. A reservation never settled (a lost connection, a dead process) is charged in full when the run ends **(server)**. Check: `check_failed_and_timed_out_calls_are_settled`.
- **Fails closed.** If the server can't be reached, answers with an error, or doesn't support guards, the tool is blocked, even without `fail_closed=True`. Check: `check_guard_fails_closed`.
- **Approvals are checked by the server, not just the SDK.** When the server requires approval for a tool, the SDK asks a person and checks again with the approval's id. Check: `check_server_required_approval`. The server accepts an approval only once, for the run it was granted to **(server)**.
- **Retries are safe.** Every check carries a `guard_id`, and the server answers a repeat the same way instead of reserving twice **(server)**.

**Not guaranteed**
- **Fences guards against the model, not against your own code.** The gate lives in the agent's process. It stops an agent from acting past its limits however it's prompted, but code that calls the payment API directly, without the guarded tool, isn't checked. Route every consequential action through a guarded tool, and give the agent no other way to perform it.
- **A guarded tool called outside a governed run just runs:** there's no run to check it against.
- **`cost_usd` is your estimate.** Fences reserves and charges what you declare, not what the payment provider bills.

## Waiting at a limit

With `on_breach="hold"` (opt-in), a crossed limit pauses the run instead of ending it.

**Guaranteed**
- **A held run doesn't continue until a person acts.** It waits until someone raises the limit and resumes it (`resume`), it's stopped, or `pause_timeout_s` passes. A server never auto-resumes a held run **(server)**. Checks: `check_hold_at_a_breach_until_raised`, `check_hold_a_call_until_the_budget_is_raised`, and their TypeScript twins.
- **A held call is checked again before it runs.** A model call or guarded tool refused for budget re-checks with the raised limit and waits again if it still doesn't fit. Nothing is spent while held.
- **A stop or a timeout still ends it.** A stop gives `stopped_by_user`; a timeout gives the breach it was held at, so the agent wraps up as it would have. Check: `check_hold_ends_on_stop_or_timeout`.
- **It only holds when someone can raise the limit.** Never in local mode, and never at a breach found while the server was unreachable. Check: `check_hold_only_when_someone_can_raise_it`.

**Not guaranteed**
- **A held agent holds its resources:** a thread for a synchronous agent, and the caller's request for anything a person is waiting on. Use it for background agents.

## Server requirements

Levels 1 and 2 work against any Fences server that implements [PROTOCOL.md](PROTOCOL.md). Level 3 needs the `/guard` endpoint. Against a server without it, guarded tools are blocked (they fail closed), with one warning.
