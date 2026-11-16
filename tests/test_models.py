import copy
import json
from types import SimpleNamespace

import anthropic
import pytest

from agent_grade import manifest, models

from test_check import Web
from test_manifest import TAKER

USAGE = {"input_tokens": 1000, "output_tokens": 200}
LABELS = {"stripe": {"label": "recommended", "quote": "Stripe"}, "paddle": {"label": "recommended", "quote": "Paddle"},
          "paypal": {"label": "rejected", "quote": "not PayPal"}}
PAGE = "<p>" + "POST https://api.example.com/payments/{merchant} with amount and currency. " * 20 + "</p>"
REQUEST = {"method": "post", "url": "https://api.example.com/payments/m-1", "headers": [{"name": "X-Key", "value": "KEY"}],
           "body": '{"amount": 10000, "currency": "EUR"}', "gaps": []}
VERDICT = {"correct": True, "errors": [], "excerpts": ["with amount and currency"]}


class Api:
    def __init__(self, replies=(), tokens=1000, stop_reason="end_turn", error=None, leave_out=()):
        self.replies, self.tokens, self.stop_reason, self.error = list(replies), tokens, stop_reason, error
        self.leave_out = set(leave_out)
        self.calls, self.counted = [], []
        self.messages = self

    def count_tokens(self, **request):
        self.counted.append(request)
        return SimpleNamespace(input_tokens=self.tokens)

    def create(self, **request):
        self.calls.append(request)
        if self.error:
            raise self.error
        if request["model"] == models.R4_LABEL_MODEL and "format" in request["output_config"]:
            asked = request["output_config"]["format"]["schema"]["required"]
            text = json.dumps({i: LABELS[i] for i in asked if i not in self.leave_out})
        else:
            text = self.replies.pop(0) if self.replies else "Stripe and Paddle; not PayPal."
        reply = {"model": request["model"], "stop_reason": self.stop_reason, "usage": USAGE,
                 "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}]}
        return SimpleNamespace(to_dict=lambda: reply)


def provider(probe=True, docs="https://example.com/docs"):
    taker = copy.deepcopy(TAKER)
    taker["probe"] = {"method": "POST", "url": "https://api.example.com/payments/{merchant}", "format": "json",
                      "fields": ["amount", "currency"], "source": "https://example.com/docs/pay"} if probe else None
    taker["urls"]["docs"] = {"url": docs, "status": "ok"} if docs else {"url": None, "status": "not found"}
    return taker


def test_a_call_that_could_pass_the_cap_is_not_sent():
    api, budget = Api(tokens=10_000), models.Budget(cap=0.01)
    found = models.ask(api, budget, models.params("claude-opus-5-5", "s", "u", 4000), dry_run=False)
    assert found["status"] == "not measured" and "cap" in found["note"] and api.calls == []


def test_spend_is_counted_from_usage_at_the_model_rate():
    budget = models.Budget()
    models.ask(Api(), budget, models.params("claude-sonnet-5-5", "s", "u", 4000), dry_run=False)
    assert budget.spent == pytest.approx((1000 * 2.0 + 200 * 10.0) / 1e6)


def test_a_dry_run_counts_tokens_and_calls_no_model():
    api = Api()
    found = models.ask(api, models.Budget(), models.params("claude-fable-5-2", "s", "u", 4000), dry_run=True)
    assert api.calls == [] and found == {"status": "dry run", "input_tokens": 1000,
                                         "worst_usd": round((1000 * 8 + 4000 * 40) / 1e6, 4)}


@pytest.mark.parametrize("api, note", [
    (Api(stop_reason="refusal"), "the model stopped with refusal"),
    (Api(error=anthropic.APIConnectionError(request=None)), "the API call failed: APIConnectionError"),
])
def test_a_refused_or_failed_call_is_not_measured(api, note):
    found = models.ask(api, models.Budget(), models.params("claude-opus-5-5", "s", "u", 4000), dry_run=False)
    assert (found["status"], found["note"], len(api.calls)) == ("not measured", note, 1)


def test_a_provider_without_public_docs_is_not_measured_and_nothing_is_fetched():
    web, api = Web({}), Api()
    found = models.first_try(web.client(), api, models.Budget(), provider(probe=False, docs=None), dry_run=False)
    assert found == {"status": "not measured", "note": "no public docs"} and web.sent == [] and api.calls == []


def test_a_page_with_little_readable_text_gets_no_model_call():
    web = Web({"https://example.com/docs/pay": (200, "text/html", b"<div id=app></div><script>x</script>")})
    api = Api()
    found = models.first_try(web.client(), api, models.Budget(), provider(), dry_run=False)
    assert found["status"] == "not measured" and "likely a page built by script" in found["note"] and api.calls == []


def test_a_first_try_is_written_from_the_task_page_then_judged_against_it():
    web = Web({"https://example.com/docs/pay": (200, "text/html", PAGE.encode())})
    api = Api([json.dumps(REQUEST), json.dumps(VERDICT)])
    found = models.first_try(web.client(), api, models.Budget(), provider(), dry_run=False)
    assert (found["status"], found["correct"], found["excerpts_found_on_page"]) == ("ok", True, True)
    assert found["attempt"]["json"] == REQUEST and len(found["page_sha256"]) == 64
    assert [call["model"] for call in api.calls] == [models.R1_MODEL, models.R1_JUDGE]
    assert "https://example.com/docs/pay" in api.calls[0]["messages"][0]["content"]


def test_without_a_task_page_the_docs_entry_page_is_used_and_said():
    web = Web({"https://example.com/docs": (200, "text/html", PAGE.encode())})
    found = models.first_try(web.client(), Api(), models.Budget(), provider(probe=False), dry_run=True)
    assert (found["page"], found["status"]) == ("https://example.com/docs", "dry run")
    assert "docs entry page" in found["page_note"]


def test_a_dry_run_estimates_the_judge_call_too():
    web, api = Web({"https://example.com/docs/pay": (200, "text/html", PAGE.encode())}), Api()
    found = models.first_try(web.client(), api, models.Budget(), provider(), dry_run=True)
    assert (found["attempt"]["status"], found["verdict"]["status"]) == ("dry run", "dry run")
    assert all("output_config" in request for request in api.counted) and api.calls == []


def test_an_answer_that_is_not_the_json_asked_for_is_kept_as_text():
    found = models.ask(Api(["not json"]), models.Budget(), models.params("claude-sonnet-5-5", "s", "u", 4000, models.R1_VERDICT),
                       dry_run=False)
    assert (found["status"], found["text"]) == ("not measured", "not json")


def test_a_provider_is_named_by_its_search_term_alias_or_id_as_a_whole_word():
    providers = [{"id": "stripe"}, {"id": "plata-by-mono", "search": "monobank"}, {"id": "ckb-otp-banka", "search": "ckb"}]
    assert models.named("Use Stripe, or OTP Banka; plata by mono too. Not stripey.", providers) == [
        "ckb-otp-banka", "plata-by-mono", "stripe"]
    assert models.named("monobanking, an OTP code, OTP Bank Ukraine", providers) == []
    assert models.named("They are all secure, a way for pay-outs.", manifest.load()) == []
    assert models.named("Concord Pay", manifest.load()) == ["concordpay"]


def test_a_run_over_every_manifest_measures_each_taker_and_asks_every_model_in_every_market(tmp_path):
    api, out = Api(), tmp_path / "models" / "2026-11-15.json"
    results = models.run(Web({}).client(), api, manifest.load(), "2026-11-15", out)
    takers = [p["id"] for p in manifest.load() if any(m["takes"] for m in p["markets"].values())]
    assert sorted(results["r1"]["providers"]) == sorted(takers)
    answers = results["r4"]["answers"]
    assert len(answers) == len(models.R4_PROMPTS) * len(models.R4_MODELS) * models.R4_SAMPLES
    assert all(answer["named"] == ["paddle", "paypal", "stripe"] for answer in answers.values())
    assert sum("system" not in call for call in api.calls) == len(answers)
    assert results["r4"]["mentioned"]["ME"] == {"paddle": 12, "paypal": 12, "stripe": 12}
    assert results["r4"]["recommended"] == {"ME": {"paddle": 12, "stripe": 12}, "UA": {"paddle": 12, "stripe": 12}}
    assert results["r4"]["labelled"] == {"ME": 12, "UA": 12}
    assert json.loads(out.read_text()) == results


def test_a_label_names_its_quote_and_whether_the_answer_holds_it():
    api = Api()
    found = models.labelled(api, models.Budget(), manifest.load(), "UA", "Stripe and Paddle; not PayPal.", dry_run=False)
    assert found["labels"]["paypal"] == {"label": "rejected", "quote": "not PayPal", "quote_found": True}
    assert api.calls[0]["output_config"]["format"]["schema"]["required"] == ["paddle", "paypal", "stripe"]


def test_labels_that_leave_out_a_mentioned_provider_are_not_kept_and_are_asked_again(tmp_path):
    out = tmp_path / "2026-11-15.json"
    partial = models.run(Web({}).client(), Api(leave_out={"stripe"}), manifest.load(), "2026-11-15", out)
    assert partial["r4"]["labelled"] == {"ME": 0, "UA": 0} and partial["r4"]["recommended"] == {"ME": {}, "UA": {}}
    assert partial["r4"]["mentioned"]["ME"] == {"paddle": 12, "paypal": 12, "stripe": 12}
    api = Api()
    again = models.run(Web({}).client(), api, manifest.load(), "2026-11-15", out)
    assert len(api.calls) == 24 and again["r4"]["labelled"] == {"ME": 12, "UA": 12}


def test_an_answer_that_names_no_provider_needs_no_labelling_call():
    api = Api()
    assert models.labelled(api, models.Budget(), manifest.load(), "ME", "Ask your bank.", dry_run=False) == {
        "named": [], "labels": {}}
    assert api.calls == []


def test_a_stop_before_the_end_keeps_the_paid_text():
    found = models.ask(Api(stop_reason="max_tokens"), models.Budget(), models.params("claude-opus-5-5", "", "u", 10),
                       dry_run=False)
    assert (found["status"], found["text"]) == ("not measured", "Stripe and Paddle; not PayPal.")


def test_a_rerun_on_the_same_date_keeps_what_was_measured_and_the_spend(tmp_path):
    out = tmp_path / "2026-11-15.json"
    first = models.run(Web({}).client(), Api(), manifest.load(), "2026-11-15", out)
    api = Api()
    again = models.run(Web({}).client(), api, manifest.load(), "2026-11-15", out)
    assert api.calls == [] and again == first and first["spent_usd"] > 0


def test_a_dry_run_leaves_a_stored_run_file_as_it_was(tmp_path):
    out = tmp_path / "2026-11-15.json"
    models.run(Web({}).client(), Api(), manifest.load(), "2026-11-15", out)
    stored = out.read_bytes()
    api = Api()
    models.run(Web({}).client(), api, manifest.load(), "2026-11-15", out, dry_run=True)
    assert out.read_bytes() == stored and api.calls == []


def test_a_named_date_must_already_have_a_run_file(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("k")
    monkeypatch.setattr(models, "RUNS", tmp_path)
    monkeypatch.setattr("sys.argv", ["agent-grade-models", "--key-file", str(key), "--date", "2026-11-14"])
    with pytest.raises(SystemExit):
        models.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["key"]


def test_no_r4_prompt_names_a_provider():
    words = {p.get("search") or p["id"] for p in manifest.load()}
    assert not any(word in prompt.lower() for prompt in models.R4_PROMPTS.values() for word in words)
