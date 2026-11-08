import copy
import json

import pytest

from agent_grade import check

from test_manifest import TAKER

OPENAPI = {"openapi": "3.0.3", "info": {"title": "Pay", "version": "1"}, "servers": [{"url": "https://api.example.com"}],
           "paths": {"/payments": {"post": {"operationId": "create", "summary": "Create",
                                            "responses": {"201": {}, "400": {}}},
                                   "get": {"operationId": "create", "responses": {"200": {}}}}}}


class Web:
    def __init__(self, pages: dict):
        self.pages, self.sent, self.now = pages, [], 0.0

    def send(self, method, url, body, headers):
        self.sent.append((method, url, body))
        status, content_type, payload = self.pages.get(url, (404, "text/plain", b""))
        if status is None:
            return check.Response(None, url, "", b"", 0.25, payload)
        if status in check.REDIRECTS:
            return check.Response(status, url, "", b"", 0.25, location=payload)
        return check.Response(status, url, content_type, payload, 0.25)

    def sleep(self, seconds):
        self.now += seconds

    def clock(self):
        return self.now

    def client(self):
        return check.Client(self.send, self.sleep, self.clock)


@pytest.mark.parametrize("path, allowed", [
    ("/docs/pay", True),
    ("/pypi/liqpay/json", False),
    ("/search?q=x", False),
    ("/private/", False),
    ("/private/open", True),
    ("/end", False),
    ("/end/more", True),
])
def test_robots_rules_follow_wildcards_anchors_and_the_longest_match(path, allowed):
    rules = check.robots_rules("\ufeffUser-agent: *\nDisallow: /pypi/*/json\nDisallow: /search*\n"
                               "Disallow: /private/\nAllow: /private/open\nDisallow: /end$\n")
    assert check.robots_allow(rules, path) is allowed


def test_a_group_naming_our_product_wins_over_the_wildcard_group():
    rules = check.robots_rules("User-agent: *\nDisallow: /\n\nUser-agent: Agent-Grade\nDisallow: /admin\n")
    assert check.robots_allow(rules, "/docs") and not check.robots_allow(rules, "/admin")


@pytest.mark.parametrize("agent", ["", "a", "agent", "Agent-Grader"])
def test_only_a_group_naming_our_exact_product_token_overrides_the_wildcard(agent):
    rules = check.robots_rules(f"User-agent: {agent}\nAllow: /\n\nUser-agent: *\nDisallow: /\n")
    assert not check.robots_allow(rules, "/docs")


def test_a_redirect_is_followed_through_the_client_with_its_own_robots_check():
    web = Web({"https://a.example/old": (301, "", "https://b.example/new"),
               "https://b.example/robots.txt": (200, "text/plain", b"User-agent: *\nDisallow: /new\n")})
    reply = web.client().request("GET", "https://a.example/old")
    assert reply.note == "robots.txt disallows it for our HTTP client"
    assert "https://b.example/new" not in [url for _, url, _ in web.sent]


def test_a_redirected_page_records_where_it_landed_after_the_gap():
    web = Web({"https://a.example/old": (302, "", "/new"), "https://a.example/new": (200, "text/plain", b"x")})
    result = check.page(web.client(), {"url": "https://a.example/old", "status": "ok"}, None)
    assert (result["status"], result["redirected_to"]) == ("ok", "https://a.example/new")
    assert web.now == pytest.approx(2 * check.GAP_SECONDS)


def test_a_post_is_never_redirected():
    web = Web({"https://api.example/pay": (307, "", "https://elsewhere.example/pay")})
    reply = web.client().request("POST", "https://api.example/pay", b"{}")
    assert reply.status == 307 and web.sent[-1][1] == "https://api.example/pay"


def test_an_unreachable_robots_txt_refuses_the_whole_site_and_says_why():
    web = Web({"https://bank.example/robots.txt": (None, "", "SSLError: handshake failed")})
    reply = web.client().request("GET", "https://bank.example/ecommerce")
    assert reply.status is None and reply.note.startswith("robots.txt unreachable (SSLError")
    assert [url for _, url, _ in web.sent] == ["https://bank.example/robots.txt"]


def test_requests_to_one_host_are_spaced_by_the_gap():
    web = Web({"https://a.example/1": (200, "text/plain", b"x"), "https://a.example/2": (200, "text/plain", b"y")})
    client = web.client()
    client.request("GET", "https://a.example/1")
    client.request("GET", "https://a.example/2")
    assert web.now == pytest.approx(2 * check.GAP_SECONDS)


def test_a_page_is_hashed_on_its_visible_text_and_compared_with_the_last_run():
    web = Web({"https://a.example/docs": (200, "text/html", b"<p>Pay <script>x=1</script>docs</p>")})
    first = check.page(web.client(), {"url": "https://a.example/docs", "status": "ok"}, None)
    again = check.page(web.client(), {"url": "https://a.example/docs", "status": "ok"}, {"sha256": first["sha256"]})
    assert first["changed"] is None and again["changed"] is False
    assert first["sha256"] == check.hashlib.sha256(b"Pay docs").hexdigest()


@pytest.mark.parametrize("reply, status, note", [
    ((404, "text/html", b""), "not found", None),
    ((403, "text/html", b""), "unreadable", "HTTP 403"),
    ((None, "", "SSLCertVerificationError: x"), "unreadable", "SSLCertVerificationError: x"),
])
def test_a_page_that_does_not_load_gets_the_manifest_status_words(reply, status, note):
    web = Web({"https://a.example/x": reply})
    result = check.page(web.client(), {"url": "https://a.example/x", "status": "ok"}, None)
    assert (result["status"], result.get("note")) == (status, note)


def test_a_slot_without_a_url_keeps_its_manifest_status_and_is_not_fetched():
    web = Web({})
    assert check.page(web.client(), {"url": None, "status": "unreadable", "note": "403"}, None) == {
        "status": "unreadable", "note": "403"}
    assert web.sent == []


def test_an_llms_txt_answered_with_html_is_not_an_llms_txt():
    assert check.llms_txt({"status": "ok", "content_type": "text/html", "body": b"<html>"})["present"] is False
    found = check.llms_txt({"status": "ok", "content_type": "text/plain",
                            "body": b"# Pay\n- [Accept a card](https://a.example/card)\n- [Refund](https://a.example/r)"})
    assert found == {"status": "ok", "present": True, "links": 2}


@pytest.mark.parametrize("read", [check.llms_txt, check.spec])
def test_a_check_over_an_unread_page_keeps_its_status_word_and_reason(read):
    assert read({"status": "unreadable", "note": "HTTP 403"}) == {"status": "unreadable", "note": "HTTP 403"}


def test_an_openapi_spec_is_linted_by_rules():
    found = check.spec({"status": "ok", "body": json.dumps(OPENAPI).encode()})
    assert found["present"] and found["lint"] == {"operations": 2, "no_title_or_version": False, "no_servers": False,
                                                  "without_description": 1, "without_error_response": 1,
                                                  "duplicate_operation_ids": 1}
    assert check.spec({"status": "ok", "body": b"<html>docs</html>"})["present"] is False


def test_a_status_page_is_machine_readable_when_its_api_lists_components():
    web = Web({"https://status.example/api/v2/summary.json": (200, "application/json",
                                                              b'{"components": [{"name": "API"}, {"name": "Checkout"}]}')})
    found = check.status(web.client(), {"status": "ok", "url": "https://status.example/", "redirected_to": None,
                                        "seconds": 0.4})
    assert found["machine_readable"] and found["components"] == 2


def test_a_bad_unauthenticated_call_is_judged_by_what_its_error_names():
    web = Web({"https://api.example/invalid/pay": (401, "application/json",
                                           b'{"error": "X-Token header is required, e.g. X-Token: abc"}')})
    call = {"method": "POST", "url": "https://api.example/{key}/pay", "format": "json", "fields": ["X-Token", "amount"]}
    found = check.probe(web.client(), call)
    assert (found["names_a_field"], found["names_the_rule"], found["gives_an_example"], found["json"]) == (
        True, True, True, True)
    assert web.sent[-1] == ("POST", "https://api.example/invalid/pay", b"{}")


def test_registries_report_mcp_servers_github_and_packages():
    web = Web({
        check.MCP_REGISTRY + "search=pay": (200, "application/json",
                                                     b'{"servers": [{"server": {"name": "io.example/pay"}}, '
                                                     b'{"server": {"name": "io.example/pay"}}, '
                                                     b'{"server": {"name": "com.paypal/payments"}}]}'),
        "https://api.github.com/users/example": (200, "application/json", b'{"public_repos": 7}'),
        "https://registry.npmjs.org/@example%2Fpay": (200, "application/json", b'{"dist-tags": {"latest": "2.1.0"}}'),
    })
    found = check.registries(web.client(), {
        "name": "Example Pay", "search": "pay", "github": {"account": "example", "source": "https://example.com"},
        "npm": [{"name": "@example/pay", "source": "https://example.com"}, {"name": "gone", "source": "https://example.com"}]})
    assert found["mcp"] == {"status": "ok", "servers": ["io.example/pay"]}
    assert found["github"] == {"status": "ok", "account": "example", "public_repos": 7}
    assert found["npm"] == [{"name": "@example/pay", "status": "ok", "latest": "2.1.0"},
                            {"name": "gone", "status": "not found"}]


def test_every_requirement_test_is_either_informed_or_marked_undecided():
    found = check.tests()
    assert found.keys() == check.TESTS.keys()
    assert found["R2 repeating a write with the same key is a no-op"] == {
        "undecided": "needs an account, beyond no-account checks"}
    assert found["R4 an MCP server in the public registries"] == {"informed_by": ["mcp"]}


def test_the_same_replies_give_the_same_results_and_refusers_are_skipped():
    refuser = {"id": "global", "name": "Global", "kind": "gateway", "markets": {"ME": {"takes": False}}}
    web = Web({"https://example.com/docs": (200, "text/html", b"<p>docs</p>")})
    first = check.run(web.client(), [TAKER, refuser], "2026-11-07", {})
    again = check.run(web.client(), [copy.deepcopy(TAKER), refuser], "2026-11-07", {})
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert list(first["providers"]) == ["example-pay"]


def test_a_field_counts_as_named_only_as_a_whole_word():
    web = Web({"https://api.example/pay": (200, "application/json", b'{"request_id": "x1", "error": "failure"}')})
    call = {"method": "POST", "url": "https://api.example/pay", "format": "json", "fields": ["request"]}
    assert check.probe(web.client(), call)["names_a_field"] is False


def test_a_robots_txt_behind_a_redirect_is_followed_and_its_rules_obeyed():
    web = Web({"https://a.example/robots.txt": (301, "", "https://cdn.example/robots.txt"),
               "https://cdn.example/robots.txt": (200, "text/plain", b"User-agent: *\nDisallow: /\n")})
    reply = web.client().request("GET", "https://a.example/docs")
    assert reply.note == "robots.txt disallows it for our HTTP client"
    assert "https://a.example/docs" not in [url for _, url, _ in web.sent]


@pytest.mark.parametrize("location, hop", [
    ("/next", "https://a.example/next"),
    ("https://b.example/plaćanje", "https://b.example/pla%C4%87anje"),
    ("file:///etc/passwd", None),
    ("data:text/plain,x", None),
    ("https://[::1/x", None),
])
def test_only_an_http_or_https_location_is_a_hop(location, hop):
    assert check.next_hop("https://a.example/old", location) == hop


def test_a_redirect_to_an_unusable_location_stops_with_a_reason():
    web = Web({"https://a.example/old": (302, "", "file:///etc/passwd")})
    reply = web.client().request("GET", "https://a.example/old")
    assert reply.status is None and "not an http(s) URL" in reply.note
    assert len(web.sent) == 2


def test_a_status_api_that_refuses_is_unreadable_not_a_missing_feature():
    web = Web({"https://status.example/api/v2/summary.json": (503, "text/html", b"busy")})
    found = check.status(web.client(), {"status": "ok", "url": "https://status.example/", "redirected_to": None})
    assert found == {"status": "unreadable", "note": "HTTP 503", "api": check.STATUS_API}


def test_a_probe_answered_with_a_redirect_is_not_judged_as_an_error():
    web = Web({"https://api.example/pay": (307, "", "https://elsewhere.example/pay")})
    call = {"method": "POST", "url": "https://api.example/pay", "format": "json", "fields": ["amount"]}
    assert check.probe(web.client(), call)["status"] == "unreadable"


def offline(monkeypatch):
    real = check.Client
    refused = lambda method, url, body, headers: check.Response(None, url, "", b"", 0.0, "ConnectionRefusedError")
    monkeypatch.setattr(check, "Client", lambda: real(refused, lambda seconds: None))
    monkeypatch.setattr(check.manifest, "load", lambda: [TAKER])


def test_a_run_where_no_host_answered_fails_and_writes_nothing(monkeypatch, tmp_path):
    offline(monkeypatch)
    monkeypatch.setattr(check, "RESULTS", tmp_path)
    monkeypatch.setattr("sys.argv", ["agent-grade-check"])
    with pytest.raises(SystemExit, match="no host answered"):
        check.main()
    assert list(tmp_path.iterdir()) == []
