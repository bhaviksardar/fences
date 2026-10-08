"""
Turns a model response into (cost in USD, tokens). Prices come from prices.json, a
snapshot of LiteLLM's price list refreshed by scripts/update_prices.py at each release.
"""
import json
import logging
import re
from importlib import resources
from typing import Optional, Tuple

log = logging.getLogger("agentfences")

_table: Optional[dict] = None
_custom: dict = {}   # init(prices=...) entries, same shape as the table
_warned: set = set()


def set_custom_prices(prices: Optional[dict]):
    """prices: {"model": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75}}, USD per 1M tokens."""
    _custom.clear()
    for model, p in (prices or {}).items():
        _custom[model.lower()] = {k: p[src] / 1_000_000 for k, src in
                                  (("in", "input"), ("out", "output"), ("cache_read", "cache_read"), ("cache_write", "cache_write"))
                                  if p.get(src) is not None}


def _prices() -> dict:
    global _table
    if _table is None:
        _table = json.loads(resources.files(__package__).joinpath("prices.json").read_text())["models"]
    return _table


def price_of(model: str) -> Optional[dict]:
    name = model.lower().split("/")[-1]                     # "models/gemini-2.5-pro", "openai/gpt-4o"
    undated = re.sub(r"-(\d{4}-\d{2}-\d{2}|\d{8})$", "", name)  # "claude-sonnet-4-5-20250929"
    for candidate in (name, undated):
        for table in (_custom, _prices()):
            if candidate in table:
                return table[candidate]
    return None


def _get(obj, *path):
    for key in path:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj


def _num(*values) -> int:
    return next((int(v) for v in values if isinstance(v, (int, float))), 0)


def read_usage(response) -> Optional[dict]:
    """
    Normalise usage from the common response shapes into uncached input, cache reads,
    cache writes and output tokens. Returns None if the response carries no usage.
    """
    model = (_get(response, "model") or _get(response, "model_version")
             or _get(response, "response_metadata", "model_name") or _get(response, "response_metadata", "model") or "")

    usage = _get(response, "usage")
    if usage is not None:
        # OpenAI: the input total includes cache reads and writes
        if _get(usage, "prompt_tokens") is not None:          # Chat Completions
            details, total, out = _get(usage, "prompt_tokens_details"), _get(usage, "prompt_tokens"), _get(usage, "completion_tokens")
        else:                                                  # Responses API, if it has input details
            details, total, out = _get(usage, "input_tokens_details"), _get(usage, "input_tokens"), _get(usage, "output_tokens")
        if _get(usage, "prompt_tokens") is not None or details is not None:
            read, write = _num(_get(details, "cached_tokens")), _num(_get(details, "cache_write_tokens"))
            u = dict(input=_num(total) - read - write, cache_read=read, cache_write=write, output=_num(out))
        else:                                                  # Anthropic: input excludes cache reads and writes
            u = dict(input=_num(_get(usage, "input_tokens")), cache_read=_num(_get(usage, "cache_read_input_tokens")),
                     cache_write=_num(_get(usage, "cache_creation_input_tokens")), output=_num(_get(usage, "output_tokens")))
        cost = _get(usage, "cost")                             # gateways like OpenRouter report the billed cost
        u["reported_cost"] = cost if isinstance(cost, (int, float)) else None
        return dict(u, model=model)

    meta = _get(response, "usage_metadata")
    if meta is None:
        return None
    if _get(meta, "prompt_token_count") is not None:          # Gemini: thinking tokens are billed as output
        cached = _num(_get(meta, "cached_content_token_count"))
        return dict(model=model, input=_num(_get(meta, "prompt_token_count")) - cached, cache_read=cached, cache_write=0,
                    output=_num(_get(meta, "candidates_token_count")) + _num(_get(meta, "thoughts_token_count")),
                    reported_cost=None)
    read = _num(_get(meta, "input_token_details", "cache_read"))  # LangChain: input includes cache reads and writes
    write = _num(_get(meta, "input_token_details", "cache_creation"))
    return dict(model=model, input=_num(_get(meta, "input_tokens")) - read - write, cache_read=read, cache_write=write,
                output=_num(_get(meta, "output_tokens")), reported_cost=None)


def _warn_once(key: str, msg: str, *args):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def cost_of(response) -> Tuple[float, int]:
    """(cost in USD, tokens) for one model response."""
    u = read_usage(response)
    if u is None:
        _warn_once("no-usage:" + type(response).__name__,
                   "agentfences: %s has no token usage, so this step costs $0. For streams, pass the final "
                   "message or a chunk that includes usage.", type(response).__name__)
        return 0.0, 0
    tokens = u["input"] + u["cache_read"] + u["cache_write"] + u["output"]
    if u["reported_cost"] is not None:
        return float(u["reported_cost"]), tokens

    p = price_of(u["model"]) if u["model"] else None
    if p is None:
        _warn_once("no-price:" + u["model"],
                   "agentfences: no price for model %r. Its tokens are counted but its cost isn't, so budget_usd "
                   "can't stop it. Add it with agentfences.init(prices={%r: {\"input\": ..., \"output\": ...}}) "
                   "(USD per 1M tokens).", u["model"], u["model"] or "model-name")
        return 0.0, tokens

    prompt = u["input"] + u["cache_read"] + u["cache_write"]
    rates = dict(p, **p["tier"]) if "above" in p and prompt > p["above"] else p  # long-context pricing
    cost = (u["input"] * rates["in"]
            + u["cache_read"] * rates.get("cache_read", rates["in"])
            + u["cache_write"] * rates.get("cache_write", rates["in"])
            + u["output"] * rates["out"])
    return cost, tokens
