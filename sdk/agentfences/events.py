"""
What a run carries besides numbers: its context (environment, release, user, session),
the exception that ended it, and the redaction hook every outgoing event passes through.
"""
import contextlib
import contextvars
import logging
import re
import traceback
from typing import Callable, Optional

log = logging.getLogger("agentfences")

MAX_CONTEXT_KEYS, MAX_CONTEXT_VALUE = 32, 256
MAX_MESSAGE, MAX_FRAMES = 1000, 15

_process_context: dict = {}  # environment / release, from init()
_context = contextvars.ContextVar("agentfences_context", default={})
_redact: Optional[Callable[[dict], Optional[dict]]] = None
_redact_failed = False
CAPTURE = ("full", "metadata")
capture = "full"  # init(capture=...): "metadata" sends nothing the agent or its users wrote


def configure(environment: Optional[str], release: Optional[str], redact: Optional[Callable], capture_: str = "full"):
    global _redact, _redact_failed, capture
    if capture_ not in CAPTURE:
        raise ValueError(f"capture must be one of {CAPTURE}, not {capture_!r}")
    _process_context.clear()
    _process_context.update({k: str(v)[:MAX_CONTEXT_VALUE] for k, v in (("environment", environment), ("release", release)) if v})
    _redact, _redact_failed, capture = redact, False, capture_


@contextlib.contextmanager
def context(**values):
    """
    Tag every run started inside this block, e.g. with the end user or session it serves:

        with agentfences.context(user_id="u_123", session_id="s_9", trace_id=span_id):
            await support_agent(question)

    Blocks nest; inner values win. Values are stored as strings (up to 256 characters,
    32 keys); None leaves a key out.
    """
    merged = dict(_context.get())
    merged.update({str(k): str(v)[:MAX_CONTEXT_VALUE] for k, v in values.items() if v is not None})
    if len(merged) > MAX_CONTEXT_KEYS:
        raise ValueError(f"agentfences.context() takes at most {MAX_CONTEXT_KEYS} keys")
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> dict:
    return {**_process_context, **_context.get()}


MAX_TAGS, MAX_TAG_VALUE = 20, 100
_TAG_KEY = re.compile(r"[a-z0-9_.-]{1,40}")


def check_tags(tags: Optional[dict]) -> dict:
    """init(tags=) and @governed(tags=): up to 20 keys of a-z, 0-9, _ . - (40 at most), string values of 100 at most."""
    if tags is None:
        return {}
    if not isinstance(tags, dict):
        raise TypeError(f"tags must be a dict, not {type(tags).__name__}")
    if len(tags) > MAX_TAGS:
        raise ValueError(f"tags takes at most {MAX_TAGS} keys, not {len(tags)}")
    for k, v in tags.items():
        if not isinstance(k, str) or not _TAG_KEY.fullmatch(k):
            raise ValueError(f"tag key {k!r} must be 1 to 40 of a-z, 0-9, _ . -")
        if not isinstance(v, str) or len(v) > MAX_TAG_VALUE:
            raise ValueError(f"tag {k!r} must be a string of at most {MAX_TAG_VALUE} characters, not {v!r}")
    return dict(tags)


def with_tags(context: dict, tags: dict) -> dict:
    """
    The run start's context: its own keys plus each tag as `tag.<key>` (the run start has no field for
    tags, and older servers may refuse new ones). Servers take 32 keys at most: past that, environment,
    release and the tags are kept first, then as many context() values as fit, with a warning.
    """
    tagged = {f"tag.{k}": v for k, v in tags.items()}
    merged = {**context, **tagged}
    if len(merged) <= MAX_CONTEXT_KEYS:
        return merged
    kept = {k: context[k] for k in ("environment", "release") if k in context}
    kept.update(tagged)
    for k, v in context.items():
        if len(kept) >= MAX_CONTEXT_KEYS:
            break
        kept.setdefault(k, v)
    global _trim_warned
    if not _trim_warned:
        _trim_warned = True
        log.warning("agentfences: a run's context and tags come to %d keys, more than the %d Fences takes; "
                    "sent %d, its tags first", len(merged), MAX_CONTEXT_KEYS, len(kept))
    return kept


_trim_warned = False  # once per process


def error_text(e: BaseException) -> str:
    """`Type: message`, or just the type when there's no message."""
    return (f"{type(e).__qualname__}: {e}" if str(e) else type(e).__qualname__)[:MAX_MESSAGE]


CATEGORIES = ("rate_limit", "overloaded", "timeout", "auth", "bad_request", "server", "network", "cancelled",
              "blocked", "tool_error", "model_error", "unknown")
# For errors without an HTTP status: words in the class's name or its bases', checked in this order
# (an OpenAI APITimeoutError is also an APIConnectionError: a timeout first)
_NAMED = (("timeout", ("Timeout", "TimedOut", "DeadlineExceeded")),
          ("rate_limit", ("RateLimit", "TooManyRequests", "ResourceExhausted")),
          ("overloaded", ("Overloaded", "ServiceUnavailable")),
          ("auth", ("Authentication", "Unauthorized", "Unauthenticated", "PermissionDenied", "Forbidden")),
          ("network", ("Connect", "Network")),
          ("cancelled", ("Cancel", "Abort", "Halt")),
          ("bad_request", ("BadRequest", "InvalidRequest", "InvalidArgument", "NotFound", "UnprocessableEntity")),
          ("server", ("InternalServer", "ServerError")))


def http_status(e: BaseException) -> Optional[int]:
    """The HTTP status an error carries: status_code (OpenAI, Anthropic), code (Gemini), response.status_code (requests, httpx)."""
    for v in (getattr(e, "status_code", None), getattr(e, "status", None), getattr(e, "code", None),
              getattr(getattr(e, "response", None), "status_code", None)):
        if isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 599:
            return int(v)
    return None


def categorize(exc: BaseException) -> str:
    """
    What kind of failure this is, from its HTTP status and class names (no provider SDK is imported):
    rate_limit, overloaded, timeout, auth, bad_request, server, network, cancelled, blocked (Fences
    refused it) or unknown. Spans of failed tool and model calls say tool_error and model_error for
    what's unknown here.
    """
    from .exceptions import FencesAuthError, FencesError
    try:
        if isinstance(exc, FencesAuthError):
            return "auth"
        if isinstance(exc, FencesError):
            return "blocked"
        status = http_status(exc)
        if status is not None and status >= 400:
            return ("rate_limit" if status == 429 else "overloaded" if status in (503, 529) else "auth" if status in (401, 403)
                    else "timeout" if status in (408, 504) else "bad_request" if status < 500 else "server")
        # Python's own classes only for timeouts and connections: FileNotFoundError is no HTTP 404
        names = " ".join(c.__name__ for c in type(exc).__mro__
                         if c.__module__ != "builtins" or issubclass(c, (TimeoutError, ConnectionError)))
        return next((cat for cat, words in _NAMED if any(w in names for w in words)), "unknown")
    except Exception:  # an odd error object (a property that raises) must not break recording it
        return "unknown"


def exception_info(exc: BaseException) -> dict:
    """Type, message and the last frames of the stack, for the run's error report."""
    frames = traceback.extract_tb(exc.__traceback__)[-MAX_FRAMES:]
    return {
        "type": type(exc).__qualname__,
        "message": str(exc)[:MAX_MESSAGE],
        "stack": [f"{f.filename}:{f.lineno} in {f.name}" for f in frames],
    }


# capture="metadata": span attributes that carry what the agent, its users or its tools wrote
_FREE_TEXT = ("fences.error.message", "gen_ai.tool.call.arguments", "fences.approval.reason", "fences.approval.note")


def metadata_only(event: dict) -> Optional[dict]:
    """
    The event with nothing the agent or its users wrote: names, models, numbers, statuses and
    error types stay; messages, reasoning, arguments and context values go. A new dict, never
    the one passed in (spans are also kept on the run). None: don't send it at all.
    """
    kind = event.get("type")
    if kind == "decision":  # sent as a fences.decision span with its length instead (core.log_decision)
        return None
    if kind == "run_start" and event.get("context"):  # the keys, not the values; environment, release and tags stay
        return {**event, "context": {k: v if k in ("environment", "release") or k.startswith("tag.") else ""
                                     for k, v in event["context"].items()}}
    if kind == "run_end" and event.get("exception"):
        exc = event["exception"]
        return {**event, "error": exc.get("type"), "exception": {**exc, "message": ""}}
    if kind == "span":
        attrs = {k: v for k, v in event.get("attributes", {}).items() if k not in _FREE_TEXT}
        if attrs.get("fences.fallback.reason") not in (None, *CATEGORIES):  # a category stays, a label of the caller's goes
            del attrs["fences.fallback.reason"]
        return {**event, "attributes": attrs}
    return event  # a guarded tool's check keeps its arguments: rules need them


def redact(event: dict) -> Optional[dict]:
    """
    Pass an outgoing event through the metadata filter (capture="metadata"), then the user's
    redact hook. Returns the event to send, or None to drop it. If the hook raises, the event
    is dropped: never send unredacted data. The hook only sees what the filter kept, so it
    can't add back what was dropped.
    """
    global _redact_failed
    if capture == "metadata":
        event = metadata_only(event)
        if event is None:
            return None
    if _redact is None:
        return event
    try:
        out = _redact(dict(event))
        if out is not None and not isinstance(out, dict):
            raise TypeError(f"returned {type(out).__name__}, not a dict or None")
        return out
    except Exception as e:  # a broken hook must not crash the agent, or leak what it was meant to hide
        if not _redact_failed:
            _redact_failed = True
            log.warning("agentfences: redact hook failed (%s: %s); dropping events it fails on", type(e).__name__, e)
        return None
