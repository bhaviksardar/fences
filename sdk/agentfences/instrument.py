"""
Record every OpenAI and Anthropic model call made inside a governed run, with no code
changes: model, tokens, cost, latency, and the error if it failed. Calls don't count as
checkpoints and don't add to the run's spend (checkpoint(response) does), but in a run with
a budget each one is prechecked: refused with FencesStop if its worst case doesn't fit.
"""
import functools
import importlib
import time
from typing import List, Optional

from . import precheck
from .core import get_active_run
from .spans import record_llm_call

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
    record_llm_call(provider, kwargs.get("model"), started, response, error, stream=bool(kwargs.get("stream")))


def _wrap(create, provider: str, is_async: bool):
    if getattr(create, "_fences_recorded", False):
        return create

    if is_async:
        @functools.wraps(create)
        async def wrapper(self, *args, **kwargs):
            if get_active_run() is None:
                return await create(self, *args, **kwargs)
            h = await precheck.hold_async(kwargs.get("model"), kwargs, kwargs)
            started = time.perf_counter()
            try:
                response = await create(self, *args, **kwargs)
            except BaseException as e:
                precheck.finish(h, failed=True)
                _record(provider, kwargs, started, error=e)
                raise
            precheck.finish(h, response)
            _record(provider, kwargs, started, response=response)
            return response
    else:
        @functools.wraps(create)
        def wrapper(self, *args, **kwargs):
            if get_active_run() is None:
                return create(self, *args, **kwargs)
            h = precheck.hold(kwargs.get("model"), kwargs, kwargs)  # raises FencesStop if it might not fit the budget
            started = time.perf_counter()
            try:
                response = create(self, *args, **kwargs)
            except BaseException as e:
                precheck.finish(h, failed=True)
                _record(provider, kwargs, started, error=e)
                raise
            precheck.finish(h, response)
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
