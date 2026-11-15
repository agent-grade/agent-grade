import argparse
import hashlib
import html
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin, urlsplit

from . import manifest

RESULTS = manifest.PROVIDERS.parent / "results"
PRODUCT = "agent-grade"
USER_AGENT = f"Agent-Grade/{version(PRODUCT)} (+https://agent-grade.com)"
GAP_SECONDS = 1.0
TIMEOUT_SECONDS = 20
STATUS_API = "/api/v2/summary.json"
MCP_REGISTRY = "https://registry.modelcontextprotocol.io/v0/servers?"
NAMES_RULE = re.compile(r"required|missing|must|invalid|expected|empty|not allowed|format|обов.язков|невір|відсутн|не передан", re.I)
GIVES_EXAMPLE = re.compile(r"e\.g\.|example|such as|приклад", re.I)
REDIRECTS = {301, 302, 303, 307, 308}
MAX_HOPS = 5
DISALLOWED = "robots.txt disallows it for our HTTP client"
ROBOTS_EXEMPT_PROBES = {"https://sandbox-api.paddle.com/transactions", "https://api-epdev.rozetkapay.com/api/payments/v1/new"}
TESTS = {
    "R1 one canonical page per task, first try from the docs alone": "a model run",
    "R1 llms.txt lists the task pages, examples run in CI": ["llms_txt", "docs"],
    "R2 one SDK per major language from the same spec": ["api_reference", "spec", "npm"],
    "R2 every error names the field, the rule broken, a valid example": ["probe"],
    "R2 repeating a write with the same key is a no-op": "an account",
    "R3 a working key with no human in under five minutes": "an account",
    "R3 the human's first involvement is one tap": "an account",
    "R4 open-source clients and examples on GitHub": ["github", "npm"],
    "R4 questions answered in public under the product's name": "a search of public answers",
    "R4 an MCP server in the public registries": ["mcp"],
    "R4 an entry in each model vendor's tool or plugin catalog": "a read of each vendor's catalog",
    "R4 each major model names it unprompted": "a model survey",
    "R5 a status endpoint under a second with per-component state": ["status"],
    "R5 outage webhooks within a minute": "an account",
    "R5 usage and spend readable per key by API": "an account",
    "R6 the price is published, flat or per-unit, readable by API": ["pricing"],
    "R6 spend caps settable and enforced": "an account",
}


@dataclass
class Response:
    status: int | None
    url: str
    content_type: str
    body: bytes
    seconds: float
    note: str | None = None
    location: str | None = None


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirects)


def urlopen(method: str, url: str, body: bytes | None, headers: dict[str, str]) -> Response:
    request = urllib.request.Request(url, data=body, method=method, headers={"User-Agent": USER_AGENT, **headers})
    started = time.monotonic()
    try:
        try:
            reply = OPENER.open(request, timeout=TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            reply = error
        with reply:
            return Response(reply.status, url, reply.headers.get("content-type", ""), reply.read(),
                            time.monotonic() - started, location=reply.headers.get("location"))
    except (OSError, ValueError, http.client.HTTPException) as error:
        reason = getattr(error, "reason", error)
        return Response(None, url, "", b"", time.monotonic() - started, f"{type(reason).__name__}: {reason}")


def robots_rules(text: str) -> list[tuple[bool, str]]:
    groups, agents, rules, in_rules = [], [], [], False
    for line in text.lstrip("\ufeff").splitlines():
        field, _, value = line.split("#", 1)[0].partition(":")
        field, value = field.strip().lower(), value.strip()
        if field == "user-agent":
            if in_rules:
                groups.append((agents, rules))
                agents, rules, in_rules = [], [], False
            agents.append(value.lower())
        elif field in ("allow", "disallow") and agents:
            in_rules = True
            if value:
                rules.append((field == "allow", value))
    groups.append((agents, rules))
    ours = [r for a, r in groups if PRODUCT in a]
    chosen = ours or [r for a, r in groups if "*" in a]
    return [rule for rules in chosen for rule in rules]


def robots_allow(rules: list[tuple[bool, str]], path: str) -> bool:
    best = (-1, True)
    for allow, pattern in rules:
        regex = re.escape(pattern.rstrip("$")).replace(r"\*", ".*") + ("$" if pattern.endswith("$") else "")
        if re.match(regex, path) and (len(pattern), allow) > best:
            best = (len(pattern), allow)
    return best[1]


class Client:
    def __init__(self, send=urlopen, sleep=time.sleep, clock=time.monotonic):
        self.send, self.sleep, self.clock = send, sleep, clock
        self.robots: dict[str, list[tuple[bool, str]] | str] = {}
        self.last: dict[str, float] = {}
        self.answered = False

    def request(self, method: str, url: str, body: bytes | None = None, headers: dict | None = None) -> Response:
        for _ in range(MAX_HOPS + 1):
            refusal = self.refusal(url)
            if refusal:
                return Response(None, url, "", b"", 0.0, refusal)
            reply = self._send(method, url, body, headers or {})
            if method != "GET" or reply.status not in REDIRECTS or not reply.location:
                return reply
            url = next_hop(url, reply.location)
            if url is None:
                return Response(None, reply.url, "", b"", 0.0, "redirected to a location that is not an http(s) URL")
        return Response(None, url, "", b"", 0.0, f"more than {MAX_HOPS} redirects")

    def refusal(self, url: str) -> str | None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.robots:
            reply = self._robots_txt(origin + "/robots.txt")
            if reply.status is None or reply.status >= 500:
                self.robots[origin] = f"robots.txt unreachable ({reply.note or f'HTTP {reply.status}'}), so all disallowed"
            else:
                self.robots[origin] = robots_rules(reply.body.decode("utf-8", "replace")) if reply.status == 200 else []
        rules = self.robots[origin]
        if isinstance(rules, str):
            return rules
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        return None if robots_allow(rules, path) else DISALLOWED

    def _robots_txt(self, url: str) -> Response:
        reply = self._send("GET", url, None, {})
        for _ in range(MAX_HOPS):
            target = next_hop(url, reply.location) if reply.status in REDIRECTS and reply.location else None
            if target is None:
                break
            url, reply = target, self._send("GET", target, None, {})
        return reply

    def _send(self, method: str, url: str, body: bytes | None, headers: dict) -> Response:
        host = urlsplit(url).netloc
        wait = self.last.get(host, -GAP_SECONDS) + GAP_SECONDS - self.clock()
        if wait > 0:
            self.sleep(wait)
        reply = self.send(method, url, body, headers)
        self.last[host] = self.clock()
        self.answered = self.answered or reply.status is not None
        return reply


def next_hop(url: str, location: str) -> str | None:
    try:
        target = quote(urljoin(url, location), safe=":/?#[]@!$&'()*+,;=%~")
        parts = urlsplit(target)
    except ValueError:
        return None
    return target if parts.scheme in ("http", "https") and parts.netloc else None


def visible_text(body: bytes) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body.decode("utf-8", "replace"), flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def page(client: Client, slot: dict, previous: dict | None) -> dict:
    if slot["url"] is None:
        return {"status": slot["status"], "note": slot.get("note")}
    reply = client.request("GET", slot["url"])
    result = {"url": slot["url"], "http": reply.status, "seconds": round(reply.seconds, 2)}
    if reply.status is not None and 200 <= reply.status < 300:
        is_html = "html" in reply.content_type
        digest = hashlib.sha256(visible_text(reply.body).encode() if is_html else reply.body).hexdigest()
        result.update(status="ok", content_type=reply.content_type.split(";")[0], sha256=digest,
                      redirected_to=reply.url if reply.url != slot["url"] else None,
                      changed=None if not previous or "sha256" not in previous else previous["sha256"] != digest)
        result["body"] = reply.body
    elif reply.status in (404, 410):
        result["status"] = "not found"
    else:
        result.update(_unreadable(reply))
    return result


def _carried(result: dict) -> dict:
    return {key: result[key] for key in ("status", "note") if result.get(key)}


def _unreadable(reply: Response) -> dict:
    return {"status": "unreadable", "note": reply.note or f"HTTP {reply.status}"}


def llms_txt(result: dict) -> dict:
    if result["status"] != "ok":
        return _carried(result)
    if "html" in result["content_type"]:
        return {"status": "ok", "present": False, "note": "answers with an HTML page, not an llms.txt"}
    links = re.findall(r"\]\((https?://[^)\s]+)\)", result["body"].decode("utf-8", "replace"))
    return {"status": "ok", "present": True, "links": len(links)}


def spec(result: dict) -> dict:
    if result["status"] != "ok":
        return _carried(result)
    try:
        document = json.loads(result["body"])
    except ValueError:
        return {"status": "ok", "present": False, "note": "the API reference is a page, not a spec file"}
    if not isinstance(document, dict) or not ({"openapi", "swagger"} & document.keys()):
        return {"status": "ok", "present": False, "note": "JSON, but not an OpenAPI or Swagger document"}
    return {"status": "ok", "present": True, "lint": lint(document),
            "format": f"openapi {document.get('openapi') or document.get('swagger')}"}


def lint(document: dict) -> dict:
    operations = [op for item in (document.get("paths") or {}).values()
                  for method, op in item.items() if isinstance(op, dict) and method != "parameters"]
    ids = [op.get("operationId") for op in operations if op.get("operationId")]
    return {
        "operations": len(operations),
        "no_title_or_version": not ((document.get("info") or {}).get("title") and (document.get("info") or {}).get("version")),
        "no_servers": "openapi" in document and not document.get("servers"),
        "without_description": sum(not (op.get("summary") or op.get("description")) for op in operations),
        "without_error_response": sum(not any(str(code).startswith("4") for code in op.get("responses") or {})
                                      for op in operations),
        "duplicate_operation_ids": len(ids) - len(set(ids)),
    }


def status(client: Client, result: dict) -> dict:
    if result["status"] != "ok":
        return _carried(result)
    parts = urlsplit(result["redirected_to"] or result["url"])
    reply = client.request("GET", f"{parts.scheme}://{parts.netloc}{STATUS_API}")
    if reply.status not in (200, 404, 410):
        return {**_unreadable(reply), "api": STATUS_API}
    found = {"status": "ok", "api": STATUS_API, "api_http": reply.status, "api_seconds": round(reply.seconds, 2)}
    try:
        components = json.loads(reply.body).get("components") if reply.status == 200 else None
    except (ValueError, AttributeError):
        components = None
    found.update(machine_readable=isinstance(components, list),
                 components=len(components) if isinstance(components, list) else None)
    return found


def probe(client: Client, call: dict | None) -> dict:
    if call is None:
        return {"status": "not found", "note": "no API endpoint in the manifest to probe"}
    url = re.sub(r"\{[^}]+\}", "invalid", call["url"])
    if call["format"] == "json":
        body, headers = b"{}", {"Content-Type": "application/json"}
    else:
        body, headers = b"", {"Content-Type": "application/x-www-form-urlencoded"}
    exempt = url in ROBOTS_EXEMPT_PROBES and client.refusal(url) == DISALLOWED
    send = client._send if exempt else client.request
    found = _probed(send(call["method"], url, body if call["method"] != "GET" else None, headers), url, call)
    return {**found, "robots_exemption": True} if exempt else found


def _probed(reply: Response, url: str, call: dict) -> dict:
    text = reply.body.decode("utf-8", "replace")
    if reply.status is None:
        return _unreadable(reply)
    if reply.status in (404, 410):
        return {"status": "not found", "url": url, "http": reply.status}
    if reply.status in REDIRECTS:
        return {"status": "unreadable", "url": url, "http": reply.status, "note": "a redirect, not followed for a POST"}
    return {"status": "ok", "url": url, "http": reply.status, "json": _is_json(text),
            "error": text[:300],
            "names_a_field": any(re.search(rf"(?<![\w-]){re.escape(field)}(?![\w-])", text, re.I) for field in call["fields"]),
            "names_the_rule": bool(NAMES_RULE.search(text)), "gives_an_example": bool(GIVES_EXAMPLE.search(text))}


def registries(client: Client, provider: dict) -> dict:
    found = {}
    term = provider["search"]
    reply = client.request("GET", MCP_REGISTRY + urlencode({"search": term}))
    servers = _json(reply).get("servers") if reply.status == 200 else None
    if isinstance(servers, list):
        names = {str((s.get("server") or s).get("name") or "") for s in servers if isinstance(s, dict)}
        found["mcp"] = {"status": "ok",
                        "servers": sorted(n for n in names if term.lower() in re.split(r"[^a-z0-9]+", n.lower()))}
    elif reply.status == 200:
        found["mcp"] = {"status": "unreadable", "note": "no server list in the reply"}
    else:
        found["mcp"] = _unreadable(reply)
    github = (provider.get("github") or {}).get("account")
    if not github:
        found["github"] = {"status": "not found", "note": "the manifest names no GitHub account"}
    else:
        reply = client.request("GET", "https://api.github.com/users/" + quote(github))
        found["github"] = ({"status": "ok", "account": github, "public_repos": _json(reply).get("public_repos")}
                           if reply.status == 200 else
                           {"status": "not found", "account": github} if reply.status == 404 else
                           {**_unreadable(reply), "account": github})
    found["npm"] = [_npm(client, package["name"]) for package in provider.get("npm", [])]
    return found


def _npm(client: Client, name: str) -> dict:
    reply = client.request("GET", "https://registry.npmjs.org/" + quote(name, safe="@"))
    if reply.status == 404:
        return {"name": name, "status": "not found"}
    if reply.status != 200:
        return {"name": name, **_unreadable(reply)}
    return {"name": name, "status": "ok", "latest": (_json(reply).get("dist-tags") or {}).get("latest")}


def _json(reply: Response) -> dict:
    try:
        data = json.loads(reply.body)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def check(client: Client, provider: dict, previous: dict | None) -> dict:
    before = (previous or {}).get("urls", {})
    urls = {slot: page(client, provider["urls"][slot], before.get(slot)) for slot in manifest.URL_SLOTS}
    checks = {"llms_txt": llms_txt(urls["llms_txt"]), "spec": spec(urls["api_reference"]),
              "status": status(client, urls["status"]), "probe": probe(client, provider.get("probe")),
              **registries(client, provider)}
    for result in urls.values():
        result.pop("body", None)
    return {"urls": urls, "checks": checks}


def tests() -> dict:
    return {test: {"undecided": f"needs {informed_by}, beyond no-account checks"} if isinstance(informed_by, str)
            else {"informed_by": informed_by} for test, informed_by in TESTS.items()}


def latest_before(day: str, root: Path = RESULTS) -> dict:
    latest = {}
    for path in sorted(path for path in root.glob("*.json") if path.stem < day):
        latest.update(json.loads(path.read_text())["providers"])
    return latest


def run(client: Client, providers: list[dict], day: str, previous: dict) -> dict:
    return {"date": day, "user_agent": USER_AGENT, "tests": tests(),
            "providers": {p["id"]: check(client, p, previous.get(p["id"])) for p in providers
                          if any(market["takes"] for market in p["markets"].values())}}


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-grade-check",
                                     description="Run the no-account checks against every payments manifest.")
    parser.add_argument("--providers", help="comma-separated provider ids (default: every provider)")
    args = parser.parse_args()
    providers = manifest.load()
    if args.providers:
        wanted = set(args.providers.split(","))
        unknown = wanted - {p["id"] for p in providers}
        if unknown:
            parser.error(f"unknown provider ids: {', '.join(sorted(unknown))}")
        providers = [p for p in providers if p["id"] in wanted]
    day = date.today().isoformat()
    client = Client()
    results = run(client, providers, day, latest_before(day))
    if not client.answered:
        sys.exit("no host answered a single request, so nothing was written")
    checked = len(results["providers"])
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{day}.json"
    if args.providers and out.exists():
        results["providers"] = {**json.loads(out.read_text())["providers"], **results["providers"]}
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"{checked} providers checked, written to {out.relative_to(RESULTS.parents[2])}")
