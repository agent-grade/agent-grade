import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import anthropic
import pypdf

from . import check, manifest, price

RUNS = manifest.PROVIDERS.parent / "models"
CAP_USD = 4.50
EFFORT = "medium"
USD_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0),
                "claude-fable-5-2": (8.0, 40.0)}
MIN_PAGE_CHARS = 1000

R1_ID = "r1-first-try-1"
R1_MODEL = "claude-sonnet-5-5"
R1_JUDGE = "claude-sonnet-5-5"
R1_TASK = """You are an AI agent building a small online shop for a non-technical owner. The owner already has \
a merchant account with {name}, and wants customers to pay for an order of 100.00 by card.

Using only the documentation page below, write the first HTTP request your code would send to {name}'s API to \
take that payment. Use placeholders for keys, signatures and IDs you do not have. Do not use what you remember \
about {name}: if the page does not tell you something you need, list it under gaps.

Return JSON only, matching the schema."""
R1_SCHEMA = {
    "type": "object",
    "properties": {
        "method": {"type": "string"},
        "url": {"type": "string"},
        "headers": {"type": "array", "items": {
            "type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
            "required": ["name", "value"], "additionalProperties": False}},
        "body": {"type": "string"},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["method", "url", "headers", "body", "gaps"],
    "additionalProperties": False,
}
R1_RUBRIC = """You check one HTTP request an AI agent wrote from a payment provider's documentation page, for \
Agent-Grade. The agent was asked to write the first request that takes a card payment for an order of 100.00.

Judge it against the page only, not against what you know about the provider. It is correct when a merchant \
with valid credentials could send it as written, after filling in the placeholders, and the provider would \
accept it as the start of that payment: the right method and endpoint, the required fields present and in the \
documented format, and the documented authentication. Placeholders for keys, signatures and IDs are fine; a \
signature scheme described but left as a placeholder is fine.

List each error as one short line. Quote the excerpts of the page your judgment rests on, copied character for \
character, each under 200 characters.

Return JSON only, matching the schema."""
R1_VERDICT = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "errors": {"type": "array", "items": {"type": "string"}},
                   "excerpts": {"type": "array", "items": {"type": "string"}}},
    "required": ["correct", "errors", "excerpts"],
    "additionalProperties": False,
}

R4_ID = "r4-ask-1"
R4_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2"]
R4_SAMPLES = 3
R4_PROMPTS = {
    "ME": "I'm a sole trader registered in Montenegro and I'm opening a small online shop. Which payment "
          "providers can I use to accept card payments online? Name the ones you would recommend.",
    "UA": "I'm a sole trader (FOP) registered in Ukraine and I'm opening a small online shop. Which payment "
          "providers can I use to accept card payments online? Name the ones you would recommend.",
}
R4_ALIASES = {"ckb-otp-banka": ["otp banka"], "plata-by-mono": ["plata by mono"], "rozetkapay": ["rozetka pay"],
              "wayforpay": ["way for pay"], "liqpay": ["liq pay"], "allsecure": ["all secure"]}
R4_UNMEASURED = ("OpenAI, Google, xAI, Meta, Mistral and every other vendor: not measured, because only the "
                 "Anthropic API was available to this run.")


class Budget:
    def __init__(self, spent: float = 0.0, cap: float = CAP_USD):
        self.cap, self.spent = cap, spent


def worst_usd(model: str, input_tokens: int, max_tokens: int) -> float:
    rate_in, rate_out = USD_PER_MTOK[model]
    return (input_tokens * rate_in + max_tokens * rate_out) / 1e6


def cost(model: str, usage: dict[str, Any]) -> float:
    rate_in, rate_out = USD_PER_MTOK[model]
    read_in = (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
    written = 1.25 * (usage.get("cache_creation_input_tokens") or 0)
    return ((read_in + written) * rate_in + (usage.get("output_tokens") or 0) * rate_out) / 1e6


def params(model: str, system: str, user: str, max_tokens: int, schema: dict | None = None) -> dict[str, Any]:
    output = {"effort": EFFORT, **({"format": {"type": "json_schema", "schema": schema}} if schema else {})}
    return {"model": model, "max_tokens": max_tokens, **({"system": system} if system else {}),
            "messages": [{"role": "user", "content": user}], "output_config": output}


def ask(api, budget: Budget, request: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    try:
        tokens = api.messages.count_tokens(**{k: v for k, v in request.items() if k != "max_tokens"}).input_tokens
        worst = worst_usd(request["model"], tokens, request["max_tokens"])
        if dry_run:
            return {"status": "dry run", "input_tokens": tokens, "worst_usd": round(worst, 4)}
        if budget.spent + worst > budget.cap:
            return {"status": "not measured", "note": f"the call could pass the ${budget.cap:.2f} cap"}
        message = api.messages.create(**request).to_dict()
    except anthropic.APIError as error:
        return {"status": "not measured", "note": f"the API call failed: {type(error).__name__}"}
    spent = cost(request["model"], message["usage"])
    budget.spent += spent
    found = {"model": message["model"], "usage": message["usage"], "cost_usd": round(spent, 6)}
    if message["stop_reason"] != "end_turn":
        return {**found, "status": "not measured", "note": f"the model stopped with {message['stop_reason']}"}
    text = "".join(b["text"] for b in message["content"] if b["type"] == "text")
    if "format" not in request["output_config"]:
        return {**found, "status": "ok", "text": text}
    try:
        return {**found, "status": "ok", "json": json.loads(text)}
    except ValueError:
        return {**found, "status": "not measured", "note": "the answer was not the JSON asked for", "text": text}


def task_page(provider: dict) -> tuple[str | None, str | None]:
    probe = provider.get("probe")
    if probe:
        return probe["source"], None
    docs = provider["urls"]["docs"]
    if docs["status"] == "ok":
        return docs["url"], "no task page in the manifest, so the docs entry page"
    return None, "no public docs"


def read_page(client: check.Client, url: str) -> tuple[dict[str, Any] | None, str | None]:
    fetched = check.page(client, {"url": url, "status": "ok"}, None)
    if fetched["status"] != "ok":
        return None, fetched.get("note") or fetched["status"]
    try:
        text = price.page_text(fetched["body"], fetched["content_type"])
    except pypdf.errors.PyPdfError as error:
        return None, f"the PDF could not be read: {error}"
    if len(text) < MIN_PAGE_CHARS:
        return None, f"under {MIN_PAGE_CHARS} characters of readable text, likely a page built by script"
    if len(text) > price.MAX_CHARS:
        return None, f"longer than {price.MAX_CHARS} characters, more than one page can carry"
    return {"text": text, "sha256": fetched["sha256"]}, None


def first_try(client: check.Client, api, budget: Budget, provider: dict, dry_run: bool) -> dict[str, Any]:
    url, note = task_page(provider)
    if url is None:
        return {"status": "not measured", "note": note}
    read, unread = read_page(client, url)
    found = {"page": url, **({"page_note": note} if note else {})}
    if read is None:
        return {**found, "status": "not measured", "note": unread}
    page = f"<documentation_page url=\"{url}\">\n{read['text']}\n</documentation_page>"
    found.update(chars=len(read["text"]), page_sha256=read["sha256"])
    attempt = ask(api, budget, params(R1_MODEL, R1_TASK.format(name=provider["name"]), page, 12000, R1_SCHEMA),
                  dry_run)
    request = attempt.get("json", {})
    verdict = ask(api, budget, params(R1_JUDGE, R1_RUBRIC, f"{page}\n<request>\n"
                                      f"{json.dumps(request, ensure_ascii=False)}\n</request>", 12000, R1_VERDICT),
                  dry_run) if attempt["status"] in ("ok", "dry run") else None
    found.update(attempt=attempt, verdict=verdict)
    if attempt["status"] != "ok" or verdict["status"] != "ok":
        return {**found, "status": attempt["status"] if attempt["status"] != "ok" else "not judged"}
    judged = verdict["json"]
    return {**found, "status": "ok", "correct": judged["correct"], "errors": judged["errors"],
            "excerpts": judged["excerpts"], "excerpts_found_on_page": price.verified(judged["excerpts"], read["text"])}


def named(text: str, providers: list[dict]) -> list[str]:
    return sorted(p["id"] for p in providers
                  if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.I)
                         for term in [p.get("search") or p["id"], *R4_ALIASES.get(p["id"], [])]))


def unprompted(api, budget: Budget, providers: list[dict], key: str, dry_run: bool) -> dict[str, Any]:
    market, model, _ = key.split("/")
    answer = ask(api, budget, params(model, "", R4_PROMPTS[market], 4000), dry_run)
    return {**answer, "named": named(answer["text"], providers)} if answer["status"] == "ok" else answer


def start(day: str) -> dict[str, Any]:
    return {"date": day, "cap_usd": CAP_USD, "spent_usd": 0.0, "effort": EFFORT,
            "r1": {"id": R1_ID, "model": R1_MODEL, "judge": R1_JUDGE, "task": R1_TASK, "rubric": R1_RUBRIC,
                   "providers": {}},
            "r4": {"id": R4_ID, "prompts": R4_PROMPTS, "unmeasured": R4_UNMEASURED, "answers": {}}}


def run(client: check.Client, api, providers: list[dict], day: str, out: Path | None = None,
        dry_run: bool = False) -> dict[str, Any]:
    results = json.loads(out.read_text()) if out and out.exists() else start(day)
    budget = Budget(results["spent_usd"])
    first_tries, answers = results["r1"]["providers"], results["r4"]["answers"]
    for provider in providers:
        if any(market["takes"] for market in provider["markets"].values()) and provider["id"] not in first_tries:
            first_tries[provider["id"]] = first_try(client, api, budget, provider, dry_run)
            save(results, budget, out)
    for key in (f"{market}/{model}/{sample}" for market in R4_PROMPTS for model in R4_MODELS
                for sample in range(1, R4_SAMPLES + 1)):
        if key not in answers:
            answers[key] = unprompted(api, budget, providers, key, dry_run)
            save(results, budget, out)
    return results


def save(results: dict[str, Any], budget: Budget, out: Path | None) -> None:
    results["spent_usd"] = round(budget.spent, 6)
    if out:
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-grade-models",
                                     description="Run the R1 first-try and R4 named-unprompted model checks.")
    parser.add_argument("--key-file", type=Path, required=True, help="a file holding only the Anthropic API key")
    parser.add_argument("--dry-run", action="store_true", help="read the pages and count tokens; call no model")
    args = parser.parse_args()
    api_key = args.key_file.read_text().strip()
    if not api_key:
        parser.error(f"{args.key_file} is empty")
    day = date.today().isoformat()
    out = None if args.dry_run else RUNS / f"{day}.json"
    results = run(check.Client(), price.api_client(api_key), manifest.load(), day, out, args.dry_run)
    if out is None:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    print(f"spent ${results['spent_usd']:.4f} of ${results['cap_usd']:.2f}, written to "
          f"{out.relative_to(RUNS.parents[2])}")
