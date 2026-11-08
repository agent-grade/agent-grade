import argparse
import hashlib
import io
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import anthropic
import pypdf

from . import check, manifest

PRICES = manifest.PROVIDERS.parent / "prices"
CACHE = PRICES / "judged"
MODEL = "claude-haiku-5-5"
EFFORT = "low"
RUBRIC_ID = "r6-price-1"
MAX_CHARS = 200_000
USD_PER_TOKEN = {"input_tokens": 0.10e-6, "output_tokens": 0.50e-6, "cache_creation_input_tokens": 0.125e-6,
                 "cache_read_input_tokens": 0.01e-6}
RUBRIC = """You read one payment provider's pricing page for Agent-Grade, a catalog that measures how well \
a provider serves an AI agent building a small online shop for a non-technical owner.

The requirement being scored is R6, "a price the agent can say in one sentence": the provider SHALL have a \
price an agent can state to the owner as one line, with no matrix, no "contact sales", and no surprise at the \
end of the month. It is met when the price is published, flat or per unit, and readable without interpretation.

Your task: say whether the price a small online shop would pay to accept card payments can be told to the \
owner in one plain sentence, using only what this page states.

Rules:
1. Use only the page text you are given. Do not use what you know about the provider. If the page does not \
state a price, the answer is that it cannot be said.
2. One sentence means one line an owner understands without a table: a percentage per payment, a fixed fee \
per payment, a monthly fee, or a combination of at most two or three such terms, with the currency. Example: \
"1.3% per card payment from Ukrainian cards, 2% from foreign cards, no monthly fee."
3. It cannot be said in one sentence when the price depends on negotiation or volume tiers the owner must \
look up ("individual tariff", "contact sales"), when it needs a table of many cases, when the page sends the \
owner elsewhere for the actual number, or when fees for a small shop are not stated at all.
4. Write the sentence in English, as the agent would say it to the owner. Name currencies as the page does.
5. Give the exact excerpts of the page text your answer rests on, copied character for character, each short \
(under 200 characters). If you say the price cannot be said, quote what shows it, or give no excerpt if the \
page states no price at all.
6. If the page is not a pricing page at all, say so in why_not and answer that the price cannot be said.

Return JSON only, matching the schema."""
SCHEMA = {
    "type": "object",
    "properties": {
        "fits_one_sentence": {"type": "boolean"},
        "sentence": {"type": ["string", "null"]},
        "why_not": {"type": ["string", "null"]},
        "excerpts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["fits_one_sentence", "sentence", "why_not", "excerpts"],
    "additionalProperties": False,
}


def page_text(body: bytes, content_type: str) -> str:
    if body.startswith(b"%PDF"):
        pdf = pypdf.PdfReader(io.BytesIO(body))
        return re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in pdf.pages)).strip()
    if "html" in content_type:
        return check.visible_text(body)
    return re.sub(r"\s+", " ", body.decode("utf-8", "replace")).strip()


def method_sha256() -> str:
    return hashlib.sha256(json.dumps([MODEL, EFFORT, RUBRIC, SCHEMA], sort_keys=True).encode()).hexdigest()


def params(text: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "max_tokens": 4000,
        "system": [{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": f"<pricing_page>\n{text}\n</pricing_page>"}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def cost(usage: dict[str, Any]) -> float:
    return sum((usage.get(kind) or 0) * rate for kind, rate in USD_PER_TOKEN.items())


def verified(excerpts: list[str], text: str) -> bool:
    flat = re.sub(r"\s+", " ", text).lower()
    return bool(excerpts) and all(re.sub(r"\s+", " ", e).strip().lower() in flat for e in excerpts)


def judgment(api, text: str, cache: Path) -> dict[str, Any]:
    key = hashlib.sha256((method_sha256() + text).encode()).hexdigest()
    path = cache / f"{key}.json"
    if path.exists():
        return {**json.loads(path.read_text()), "cached": True}
    try:
        message = api.messages.create(**params(text)).to_dict()
    except anthropic.APIError as error:
        return {"status": "not judged", "note": f"the API call failed: {type(error).__name__}", "cached": False}
    if message["stop_reason"] != "end_turn":
        return {"status": "unreadable", "note": f"the model stopped with {message['stop_reason']}",
                "usage": message["usage"], "cost_usd": cost(message["usage"]), "cached": False}
    answer = json.loads(next(block["text"] for block in message["content"] if block["type"] == "text"))
    found = {"status": "ok", "rubric": RUBRIC_ID, "method_sha256": method_sha256(), "model": message["model"],
             "effort": EFFORT, "usage": message["usage"], "cost_usd": cost(message["usage"]), **answer,
             "excerpts_found_on_page": verified(answer["excerpts"], text)}
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {**found, "cached": False}


def price(client: check.Client, api, provider: dict, cache: Path) -> dict[str, Any]:
    slot = provider["urls"]["pricing"]
    unread = {"fits_one_sentence": None}
    if slot["status"] != "ok":
        return {**check._carried(slot), **unread}
    fetched = check.page(client, slot, None)
    if fetched["status"] != "ok":
        return {"url": slot["url"], **check._carried(fetched), **unread}
    try:
        text = page_text(fetched["body"], fetched["content_type"])
    except pypdf.errors.PyPdfError as error:
        return {"url": slot["url"], "status": "unreadable", "note": f"the PDF could not be read: {error}", **unread}
    if not text:
        return {"url": slot["url"], "status": "unreadable", "note": "the page has no readable text", **unread}
    found = {"url": slot["url"], "sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text),
             "truncated": len(text) > MAX_CHARS}
    if not re.search(r"\d", text):
        return {**found, "status": "ok", "fits_one_sentence": False, "decided_by": "rule",
                "why_not": "the page states no number at all"}
    judged = judgment(api, text[:MAX_CHARS], cache)
    return {**found, **check._carried(judged), "fits_one_sentence": judged.get("fits_one_sentence"),
            "decided_by": "model", "judgment": judged}


def api_client(api_key: str) -> anthropic.Anthropic:
    saved = dict(os.environ)
    os.environ.clear()  # no proxy or ANTHROPIC_* header variable may reroute or add to the request carrying the key
    try:
        return anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com")
    finally:
        os.environ.update(saved)


def run(client: check.Client, api, providers: list[dict], day: str, cache: Path = CACHE) -> dict[str, Any]:
    prices = {p["id"]: price(client, api, p, cache) for p in providers
              if any(market["takes"] for market in p["markets"].values())}
    spent = sum(p.get("judgment", {}).get("cost_usd", 0) for p in prices.values()
                if not p.get("judgment", {}).get("cached"))
    return {"date": day, "rubric": RUBRIC_ID, "model": MODEL, "cost_usd_this_run": round(spent, 6),
            "providers": prices}


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-grade-price",
                                     description="Read each provider's price as one sentence, or say why it can't be.")
    parser.add_argument("--key-file", type=Path, required=True, help="a file holding only the Anthropic API key")
    args = parser.parse_args()
    api_key = args.key_file.read_text().strip()
    if not api_key:
        parser.error(f"{args.key_file} is empty")
    day = date.today().isoformat()
    web = check.Client()
    results = run(web, api_client(api_key), manifest.load(), day)
    if not web.answered:
        sys.exit("no host answered a single request, so nothing was written")
    PRICES.mkdir(exist_ok=True)
    out = PRICES / f"{day}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"{len(results['providers'])} prices read for ${results['cost_usd_this_run']:.4f}, "
          f"written to {out.relative_to(PRICES.parents[2])}")
