"""
@tool: record each call of an agent's tool (name, a short summary of its arguments,
success or failure, and latency) as evidence on the active run.
"""
import functools
import inspect
import reprlib
import time
from typing import Callable, Optional

from .core import record_event, get_active_run

MAX_ARGS = 300

_short = reprlib.Repr()
_short.maxstring, _short.maxother, _short.maxlist, _short.maxdict = 60, 60, 5, 5


def args_summary(func: Callable, args: tuple, kwargs: dict) -> str:
    """`query='cats', limit=5`, each value shortened, the whole thing capped."""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        items = [(k, v) for k, v in bound.arguments.items() if k not in ("self", "cls")]
    except (TypeError, ValueError):  # builtins or odd signatures: fall back to positions
        items = [(str(i), v) for i, v in enumerate(args)] + list(kwargs.items())
    text = ", ".join(f"{k}={_short.repr(v)}" for k, v in items)
    return text if len(text) <= MAX_ARGS else text[:MAX_ARGS - 1] + "…"


def _record(name: str, args: str, started: float, error: Optional[BaseException]):
    record_event({
        "type": "tool_call",
        "name": name,
        "args": args,
        "ok": error is None,
        "error": None if error is None else (f"{type(error).__qualname__}: {error}" if str(error) else type(error).__qualname__)[:1000],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    })


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Record every call of an agent tool on the active run:

        @agentfences.tool
        def web_search(query: str) -> list: ...

        @agentfences.tool(name="search")
        async def search_docs(q: str): ...

    Records the tool's name, a short summary of its arguments, whether it succeeded (and
    the error if not) and how long it took. Exceptions still reach the caller. Works on
    sync and async functions; outside a governed run it just calls the function.
    """
    def decorate(f: Callable) -> Callable:
        label = name or f.__name__

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                if get_active_run() is None:
                    return await f(*args, **kwargs)
                summary, started = args_summary(f, args, kwargs), time.perf_counter()
                try:
                    out = await f(*args, **kwargs)
                except Exception as e:
                    _record(label, summary, started, e)
                    raise
                _record(label, summary, started, None)
                return out
            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if get_active_run() is None:
                return f(*args, **kwargs)
            summary, started = args_summary(f, args, kwargs), time.perf_counter()
            try:
                out = f(*args, **kwargs)
            except Exception as e:
                _record(label, summary, started, e)
                raise
            _record(label, summary, started, None)
            return out
        return wrapper

    return decorate(func) if func is not None else decorate
