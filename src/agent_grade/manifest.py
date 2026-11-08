import json
import re
from datetime import date
from pathlib import Path

PROVIDERS = Path(__file__).resolve().parents[2] / "data" / "payments" / "providers"
RECEIPTS = PROVIDERS.parent / "receipts"
KINDS = {"bank-acquirer", "gateway", "aggregator", "merchant-of-record", "wallet"}
URL_SLOTS = ("docs", "api_reference", "status", "pricing", "signup", "llms_txt")
ANSWERS = {"yes", "not found"}
STATUSES = {"ok", "not found", "unreadable"}
COUNTRY = re.compile(r"[A-Z]{2}")


def load(root: Path = PROVIDERS) -> list[dict]:
    manifests = []
    for path in sorted(root.glob("*.json")):
        manifest = json.loads(path.read_text())
        problems = check(manifest)
        if manifest.get("id") != path.stem:
            problems.append(f"id {manifest.get('id')!r} differs from the file name")
        if problems:
            raise ValueError(f"{path.name}: " + "; ".join(problems))
        manifests.append(manifest)
    return manifests


def check(manifest: dict) -> list[str]:
    problems = []
    if manifest.get("kind") not in KINDS:
        problems.append(f"kind {manifest.get('kind')!r} is not one of {sorted(KINDS)}")
    if not manifest.get("name"):
        problems.append("no name")
    markets = manifest.get("markets") or {}
    if not markets:
        problems.append("no markets")
    for country, market in markets.items():
        problems += [f"{country}: {p}" for p in _market_problems(country, market)]
    if manifest.get("probe") is not None:
        problems += [f"probe: {p}" for p in _probe_problems(manifest["probe"])]
    if manifest.get("github") and not _sourced(manifest["github"], "account"):
        problems.append("github needs an account and an https source")
    problems += ["npm: a package needs a name and an https source"
                 for package in manifest.get("npm", []) if not _sourced(package, "name")]
    if any(market.get("takes") for market in markets.values()):
        if not manifest.get("search"):
            problems.append("no search term for the registries")
        urls = manifest.get("urls") or {}
        for slot in URL_SLOTS:
            problems += [f"urls.{slot}: {p}" for p in _url_problems(urls.get(slot))]
    return problems


def _market_problems(country: str, market: dict) -> list[str]:
    problems = [] if COUNTRY.fullmatch(country) else ["not an ISO 3166 alpha-2 code"]
    if not isinstance(market.get("takes"), bool):
        problems.append("takes is not true or false")
    if not market.get("why"):
        problems.append("no why")
    problems += _evidence_problems("evidence", market.get("evidence"))
    if market.get("takes"):
        for who in ("sole_trader", "company"):
            answer = market.get(who) or {}
            if answer.get("answer") not in ANSWERS:
                problems.append(f"{who}.answer is not one of {sorted(ANSWERS)}")
            if answer.get("answer") != "not found":
                problems += _evidence_problems(who, answer)
    return problems


def _evidence_problems(name: str, evidence: dict | None) -> list[str]:
    evidence = evidence or {}
    return [f"{name}.{field} missing" for field in ("source", "quote", "english") if not evidence.get(field)]


def _url_problems(slot: dict | None) -> list[str]:
    if slot is None:
        return ["missing"]
    problems = []
    url, status = slot.get("url", ""), slot.get("status")
    if status not in STATUSES:
        problems.append(f"status is not one of {sorted(STATUSES)}")
    if url is not None and not str(url).startswith("https://"):
        problems.append("url is neither an https URL nor null")
    if status == "ok" and url is None:
        problems.append("an ok slot has no url")
    if status == "not found" and url is not None:
        problems.append("a not-found slot has a url")
    if status == "unreadable" and not slot.get("note"):
        problems.append("an unreadable slot has no note saying why")
    try:
        date.fromisoformat(slot.get("checked", ""))
    except (TypeError, ValueError):
        problems.append("checked is not an ISO date")
    return problems


def _probe_problems(probe: dict) -> list[str]:
    problems = [] if probe.get("method") in ("GET", "POST") else ["method is not GET or POST"]
    if not str(probe.get("url", "")).startswith("https://"):
        problems.append("url is not an https URL")
    if probe.get("format") not in ("json", "form"):
        problems.append("format is not json or form")
    if not probe.get("fields"):
        problems.append("no fields to look for in the error")
    if probe.get("base") is not None and not _sourced(probe["base"], "quote"):
        problems.append("a probe's base needs a quote and an https source")
    return problems + ([] if _sourced(probe, "quote") else ["a probe needs a quote and an https source"])


def _sourced(entry: dict, key: str) -> bool:
    return bool(entry.get(key)) and str(entry.get("source", "")).startswith("https://")


def load_receipts(root: Path = RECEIPTS) -> dict[str, dict]:
    notes = {}
    for path in sorted(root.glob("*.json")):
        note = json.loads(path.read_text())
        problems = receipts_problems(note)
        if note.get("country") != path.stem:
            problems.append(f"country {note.get('country')!r} differs from the file name")
        if problems:
            raise ValueError(f"{path.name}: " + "; ".join(problems))
        notes[path.stem] = note
    return notes


def receipts_problems(note: dict) -> list[str]:
    problems = [] if COUNTRY.fullmatch(str(note.get("country"))) else ["country is not an ISO 3166 alpha-2 code"]
    try:
        date.fromisoformat(note.get("checked", ""))
    except (TypeError, ValueError):
        problems.append("checked is not an ISO date")
    if not note.get("summary"):
        problems.append("no summary")
    if not note.get("claims"):
        problems.append("no claims")
    for number, claim in enumerate(note.get("claims", []), 1):
        problems += _evidence_problems(f"claim {number}", claim)
        if not claim.get("statement"):
            problems.append(f"claim {number}: no statement")
        if not isinstance(claim.get("official"), bool):
            problems.append(f"claim {number}: official is not true or false")
    problems += [f"service {service.get('name')!r} needs a name and an https source"
                 for service in note.get("services", []) if not _sourced(service, "name")]
    return problems
