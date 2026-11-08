import copy
import json
import os

import anthropic
import pytest

from agent_grade import price

from test_check import Web, offline
from test_manifest import TAKER

ANSWER = {"fits_one_sentence": True, "sentence": "1.3% per card payment, no monthly fee.", "why_not": None,
          "excerpts": ["1.3% за операцію"]}
USAGE = {"input_tokens": 1000, "output_tokens": 100, "cache_creation_input_tokens": 700,
         "cache_read_input_tokens": 900}


def pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


class Api:
    def __init__(self, answer=ANSWER, stop_reason="end_turn", error=None):
        self.calls, self.answer, self.stop_reason, self.error = [], answer, stop_reason, error
        self.messages = self

    def create(self, **params):
        self.calls.append(params)
        if self.error:
            raise self.error
        reply = {"model": price.MODEL, "stop_reason": self.stop_reason, "usage": USAGE,
                 "content": [{"type": "text", "text": json.dumps(self.answer)}]}
        return type("Message", (), {"to_dict": lambda _: reply})()


def provider(url="https://example.com/pricing", status="ok"):
    taker = copy.deepcopy(TAKER)
    taker["urls"]["pricing"] = {"url": url, "status": status, "checked": "2026-11-07"}
    return taker


def test_a_pdf_pricing_page_is_read_as_text():
    assert price.page_text(pdf("E-COMMERCE 4.00%"), "application/pdf") == "E-COMMERCE 4.00%"


def test_a_page_with_no_number_is_decided_by_rule_without_the_model(tmp_path):
    web, api = Web({"https://example.com/pricing": (200, "text/html", b"<p>Contact our sales team</p>")}), Api()
    found = price.price(web.client(), api, provider(), tmp_path)
    assert (found["decided_by"], found["fits_one_sentence"]) == ("rule", False) and api.calls == []


@pytest.mark.parametrize("amount", ["naknada 10 eura", "300 гривень за місяць", "EUR 10 per month", "50¢ a payment"])
def test_any_page_with_a_number_goes_to_the_model(amount, tmp_path):
    web, api = Web({"https://example.com/pricing": (200, "text/html", f"<p>{amount}</p>".encode())}), Api()
    found = price.price(web.client(), api, provider(), tmp_path)
    assert found["decided_by"] == "model" and len(api.calls) == 1


@pytest.mark.parametrize("body, content_type", [(b"<html><script>app()</script></html>", "text/html"),
                                                (b"%PDF-1.4 broken", "application/pdf")])
def test_a_page_with_no_readable_text_is_unreadable_not_scored(body, content_type, tmp_path):
    web, api = Web({"https://example.com/pricing": (200, content_type, body)}), Api()
    found = price.price(web.client(), api, provider(), tmp_path)
    assert (found["status"], found["fits_one_sentence"], "decided_by" in found) == ("unreadable", None, False)
    assert api.calls == []


def test_a_model_answer_lifts_its_score_to_the_entry(tmp_path):
    web = Web({"https://example.com/pricing": (200, "text/html", "<p>1.3% за операцію</p>".encode())})
    found = price.price(web.client(), Api(), provider(), tmp_path)
    assert (found["status"], found["fits_one_sentence"]) == ("ok", True)


def test_an_answer_without_excerpts_is_not_verified():
    assert price.verified([], "1.3% per payment") is False


def test_a_page_with_amounts_is_judged_once_then_served_from_the_cache(tmp_path):
    web, api = Web({"https://example.com/pricing": (200, "text/html", "<p>1.3% за операцію</p>".encode())}), Api()
    first = price.price(web.client(), api, provider(), tmp_path)["judgment"]
    again = price.price(web.client(), api, provider(), tmp_path)["judgment"]
    assert len(api.calls) == 1 and (first["cached"], again["cached"]) == (False, True)
    assert first["sentence"] == ANSWER["sentence"] and first["excerpts_found_on_page"] is True
    assert api.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_an_excerpt_the_page_does_not_hold_is_flagged(tmp_path):
    web = Web({"https://example.com/pricing": (200, "text/html", b"<p>2% per payment</p>")})
    found = price.price(web.client(), Api(), provider(), tmp_path)["judgment"]
    assert found["excerpts_found_on_page"] is False


def test_a_refused_or_cut_answer_is_unreadable_and_not_cached(tmp_path):
    web = Web({"https://example.com/pricing": (200, "text/html", b"<p>2% per payment</p>")})
    found = price.price(web.client(), Api(stop_reason="max_tokens"), provider(), tmp_path)
    assert (found["status"], found["fits_one_sentence"]) == ("unreadable", None)
    assert found["judgment"]["status"] == "unreadable" and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", ["not found", "unreadable"])
def test_a_pricing_slot_that_was_not_read_keeps_its_status_and_scores_no_sentence(status, tmp_path):
    found = price.price(Web({}).client(), Api(), provider(url=None, status=status), tmp_path)
    assert (found["status"], found["fits_one_sentence"]) == (status, None)


def test_cost_counts_cache_writes_and_reads_at_their_own_rates():
    assert price.cost(USAGE) == pytest.approx(1000 * 0.10e-6 + 100 * 0.50e-6 + 700 * 0.125e-6 + 900 * 0.01e-6)


def test_the_run_reports_only_new_spend(tmp_path):
    web = Web({"https://example.com/pricing": (200, "text/html", b"<p>2% per payment</p>")})
    first = price.run(web.client(), Api(), [provider()], "2026-11-07", tmp_path)
    again = price.run(web.client(), Api(), [provider()], "2026-11-07", tmp_path)
    assert first["cost_usd_this_run"] > 0 and again["cost_usd_this_run"] == 0


def test_a_failed_api_call_leaves_the_price_not_judged_and_uncached(tmp_path):
    web = Web({"https://example.com/pricing": (200, "text/html", b"<p>2% per payment</p>")})
    error = anthropic.APIError("Your credit balance is too low", None, body=None)
    found = price.price(web.client(), Api(error=error), provider(), tmp_path)
    assert (found["status"], found["fits_one_sentence"]) == ("not judged", None) and list(tmp_path.iterdir()) == []


def test_a_run_where_no_host_answered_fails_and_writes_nothing(monkeypatch, tmp_path):
    offline(monkeypatch)
    monkeypatch.setattr(price, "PRICES", tmp_path / "prices")
    monkeypatch.setattr(price, "api_client", lambda key: Api())
    (tmp_path / "key").write_text("test-key")
    monkeypatch.setattr("sys.argv", ["agent-grade-price", "--key-file", str(tmp_path / "key")])
    with pytest.raises(SystemExit, match="no host answered"):
        price.main()
    assert list(tmp_path.iterdir()) == [tmp_path / "key"]


def test_the_api_client_ignores_proxy_and_header_variables_and_restores_them(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Probe: 1")
    api = price.api_client("test-key")
    assert str(api.base_url) == "https://api.anthropic.com" and "X-Probe" not in api.default_headers
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:9"
