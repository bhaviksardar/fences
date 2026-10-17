"""
Attributes for model and tool call spans, named as in the OpenTelemetry GenAI semantic
conventions (gen_ai.*), plus Fences' own (fences.*). Recorded with core.record_span.
"""
import time
from typing import Optional

from .core import record_span
from .pricing import cost_of, read_usage

MAX_ARGS = 300  # characters of a tool call's argument summary
TOOL_ARGUMENTS = ("names", "full", "none")
tool_arguments = "names"  # init(tool_arguments=...): what tool call spans carry of the arguments


def record_llm_call(provider: str, request_model: Optional[str], started: float, response=None,
                    error: Optional[BaseException] = None, stream: bool = False, integration: Optional[str] = None):
    """A `chat {model}` span. started is a time.perf_counter() reading; response carries the usage."""
    attrs = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": provider,
             "gen_ai.request.model": request_model, "fences.integration": integration}
    if stream:
        attrs["gen_ai.request.stream"] = True  # usage arrives with the stream, so none is recorded here
    usage = read_usage(response) if response is not None and error is None else None
    if usage:
        cached = usage["cache_read"] + usage["cache_write"]
        attrs.update({
            "gen_ai.response.model": usage["model"] or None,
            "gen_ai.usage.input_tokens": usage["input"] + cached,  # OTel counts cached input in the total
            "gen_ai.usage.cache_read.input_tokens": usage["cache_read"],
            "gen_ai.usage.cache_write.input_tokens": usage["cache_write"],
            "gen_ai.usage.output_tokens": usage["output"],
            "fences.cost_usd": cost_of(response)[0],
        })
    model = attrs.get("gen_ai.response.model") or request_model or "unknown"
    record_span(f"chat {model}", time.perf_counter() - started, attrs, error)


def record_tool_call(name: str, args: str, started: float, error: Optional[BaseException] = None,
                     call_id: Optional[str] = None, integration: Optional[str] = None, arg_names: Optional[str] = None):
    """
    An `execute_tool {name}` span. args is a short summary of the call's arguments and arg_names
    just their names: the span carries the names by default, the values only with
    init(tool_arguments="full"), and neither with "none".
    """
    sent = args if tool_arguments == "full" else arg_names if tool_arguments == "names" else None
    record_span(f"execute_tool {name}", time.perf_counter() - started, {
        "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name, "gen_ai.tool.call.id": call_id,
        "gen_ai.tool.call.arguments": sent if sent is None or len(sent) <= MAX_ARGS else sent[:MAX_ARGS - 1] + "…",
        "fences.integration": integration,
    }, error)


def names_of(arguments) -> Optional[str]:
    """`query, limit` for a dict of arguments; None when there are no names to give."""
    return ", ".join(str(k) for k in arguments) if isinstance(arguments, dict) else None
