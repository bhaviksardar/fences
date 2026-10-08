"""
Refresh agentfences/prices.json from LiteLLM's maintained model price list (MIT).
Run before each release:

    python scripts/update_prices.py
"""
import json
import time
from pathlib import Path

import requests

SOURCE = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
OUT = Path(__file__).resolve().parent.parent / "agentfences" / "prices.json"

# First-party APIs only, in priority order: when two providers list the same bare
# model name, the earlier one's price wins.
PROVIDERS = ["openai", "anthropic", "gemini", "vertex_ai-language-models", "mistral", "deepseek", "xai", "groq"]

RATES = {  # our short name -> LiteLLM field, USD per token, standard tier
    "in": "input_cost_per_token",
    "out": "output_cost_per_token",
    "cache_read": "cache_read_input_token_cost",
    "cache_write": "cache_creation_input_token_cost",
}


def entry(v: dict) -> dict:
    e = {k: v[f] for k, f in RATES.items() if v.get(f) is not None}
    for threshold in ("200k", "272k"):  # long-context tier: higher rates once the prompt passes it
        tier = {k: v[f"{f}_above_{threshold}_tokens"] for k, f in RATES.items() if v.get(f"{f}_above_{threshold}_tokens") is not None}
        if tier:
            e["above"] = int(threshold[:-1]) * 1000
            e["tier"] = tier
            break
    return e


def main():
    resp = requests.get(SOURCE, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    models = {}
    for provider in PROVIDERS:
        for key, v in data.items():
            if not isinstance(v, dict) or v.get("litellm_provider") != provider:
                continue
            if v.get("mode") not in ("chat", "responses") or v.get("input_cost_per_token") is None:
                continue
            name = key.split("/")[-1].lower()  # "gemini/gemini-2.5-pro" -> "gemini-2.5-pro", as the API reports it
            models.setdefault(name, entry(v))

    OUT.write_text(json.dumps({
        "source": "LiteLLM model_prices_and_context_window.json, MIT License, Copyright (c) 2023 Berri AI",
        "fetched": time.strftime("%Y-%m-%d"),
        "models": dict(sorted(models.items())),
    }, separators=(",", ":")) + "\n")
    print(f"wrote {len(models)} models to {OUT}")


if __name__ == "__main__":
    main()
