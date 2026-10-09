"""
Record every OpenAI and Anthropic model call made inside a governed run, with no code
changes: model, tokens, cost, latency, and the error if it failed. Recording only: calls
don't count as checkpoints and don't add to the run's spend (checkpoint(response) does).
"""
import functools
import importlib
import time
from typing import List, Optional

from .core import record_event, get_active_run
from .pricing import cost_of, read_usage

# (provider, module, class) whose `create` is patched. Async classes return awaitables.
TARGETS = [
    ("openai", "openai.resources.chat", "Completions"),
    ("openai", "openai.resources.chat", "AsyncCompletions"),
    ("openai", "openai.resources.responses", "Responses"),
    ("openai", "openai.resources.responses", "AsyncResponses"),
    ("anthropic", "anthropic.resources", "Messages"),
    ("anthropic", "anthropic.resources", "AsyncMessages"),
]


def _record(provider: str, kwargs: dict, started: float, response=None, error: Optional[BaseException] = None):
    event = {
        "type": "llm_call",
        "provider": provider,
        "model": kwargs.get("model"),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "ok": error is None,
    }
    if error is not None:
        event["error"] = (f"{type(error).__qualname__}: {error}" if str(error) else type(error).__qualname__)[:1000]
        event["status"] = getattr(error, "status_code", None)  # e.g. 429 rate limit, 529 overloaded
    elif kwargs.get("stream"):
        event["stream"] = True  # usage arrives with the stream; pass the final message to checkpoint()
    else:
        usage = read_usage(response)
        if usage:
            event["model"] = usage["model"] or event["model"]
            event.update(input_tokens=usage["input"], cache_read_tokens=usage["cache_read"],
                         cache_write_tokens=usage["cache_write"], output_tokens=usage["output"])
            event["cost_usd"] = cost_of(response)[0]
    record_event(event)


def _wrap(create, provider: str, is_async: bool):
    if getattr(create, "_fences_recorded", False):
        return create

    if is_async:
        @functools.wraps(create)
        async def wrapper(self, *args, **kwargs):
            if get_active_run() is None:
                return await create(self, *args, **kwargs)
            started = time.perf_counter()
            try:
                response = await create(self, *args, **kwargs)
            except Exception as e:
                _record(provider, kwargs, started, error=e)
                raise
            _record(provider, kwargs, started, response=response)
            return response
    else:
        @functools.wraps(create)
        def wrapper(self, *args, **kwargs):
            if get_active_run() is None:
                return create(self, *args, **kwargs)
            started = time.perf_counter()
            try:
                response = create(self, *args, **kwargs)
            except Exception as e:
                _record(provider, kwargs, started, error=e)
                raise
            _record(provider, kwargs, started, response=response)
            return response

    wrapper._fences_recorded = True
    return wrapper


def instrument() -> List[str]:
    """
    Patch the installed OpenAI and Anthropic clients so every model call inside a governed
    run is recorded. Safe to call more than once. Returns the classes it patched.
    """
    patched = []
    for provider, module, cls_name in TARGETS:
        try:
            cls = getattr(importlib.import_module(module), cls_name)
        except (ImportError, AttributeError):
            continue  # provider not installed, or a version without this API
        cls.create = _wrap(cls.__dict__["create"], provider, cls_name.startswith("Async"))
        patched.append(f"{module}.{cls_name}")
    return patched
