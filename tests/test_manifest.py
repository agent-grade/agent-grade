import copy
import json

import pytest

from agent_grade import manifest

EVIDENCE = {"source": "https://example.com/terms", "quote": "Za preduzetnike i pravna lica.",
            "english": "For sole traders and companies."}
TAKER = {
    "id": "example-pay",
    "name": "Example Pay",
    "kind": "gateway",
    "search": "example",
    "markets": {"ME": {"takes": True, "why": "The banks' card processor.", "evidence": EVIDENCE,
                       "sole_trader": {"answer": "yes", **EVIDENCE}, "company": {"answer": "not found"}}},
    "urls": {slot: {"url": "https://example.com/" + slot, "status": "ok", "checked": "2026-11-07"}
             for slot in manifest.URL_SLOTS},
}
REFUSER = {"id": "example-global", "name": "Example Global", "kind": "gateway",
           "markets": {"ME": {"takes": False, "why": "Not on its supported-countries list.", "evidence": EVIDENCE}}}


def test_a_complete_manifest_and_a_refusal_without_urls_both_pass():
    assert manifest.check(TAKER) == []
    assert manifest.check(REFUSER) == []


def test_a_url_can_be_not_found_or_unreadable_with_its_reason():
    taker = copy.deepcopy(TAKER)
    taker["urls"]["llms_txt"] = {"url": None, "status": "not found", "checked": "2026-11-07"}
    taker["urls"]["pricing"] = {"url": None, "status": "unreadable", "checked": "2026-11-07", "note": "403"}
    taker["urls"]["signup"] = {"url": "https://example.com/join", "status": "unreadable", "checked": "2026-11-07",
                               "note": "robots.txt disallows it"}
    assert manifest.check(taker) == []


@pytest.mark.parametrize("damage, problem", [
    (lambda m: m["urls"].pop("status"), "urls.status: missing"),
    (lambda m: m["urls"]["docs"].pop("checked"), "urls.docs: checked is not an ISO date"),
    (lambda m: m["urls"]["docs"].update(url="http://example.com"), "urls.docs: url is neither"),
    (lambda m: m["urls"]["docs"].update(url=None), "urls.docs: an ok slot has no url"),
    (lambda m: m["urls"]["docs"].update(status="not found"), "urls.docs: a not-found slot has a url"),
    (lambda m: m["urls"]["docs"].update(status="unreadable"), "urls.docs: an unreadable slot has no note"),
    (lambda m: m["urls"]["docs"].update(status="blocked"), "urls.docs: status is not one of"),
    (lambda m: m["urls"].update(docs={"status": "not found", "checked": "2026-11-07"}), "urls.docs: url is neither"),
    (lambda m: m["markets"]["ME"]["sole_trader"].pop("quote"), "ME: sole_trader.quote missing"),
    (lambda m: m["markets"]["ME"]["company"].update(answer="no"), "ME: company.answer is not one of"),
    (lambda m: m["markets"]["ME"].pop("why"), "ME: no why"),
    (lambda m: m["markets"].update(Montenegro=m["markets"]["ME"]), "Montenegro: not an ISO 3166"),
    (lambda m: m.update(kind="bank"), "kind 'bank' is not one of"),
    (lambda m: m.update(probe={"method": "PUT", "url": "https://api.example.com", "format": "json", "fields": ["amount"],
                               "source": "https://example.com", "quote": "POST /pay"}), "probe: method is not GET or POST"),
    (lambda m: m.update(github={"account": "example"}), "github needs an account and an https source"),
    (lambda m: m.update(probe={"method": "POST", "url": "https://api.example.com/pay", "format": "json",
                               "fields": ["amount"], "source": "https://example.com", "quote": "POST /pay",
                               "base": {"quote": "https://api.example.com"}}), "probe: a probe's base needs"),
    (lambda m: m.update(npm=[{"name": "pay"}]), "npm: a package needs a name and an https source"),
    (lambda m: m.pop("search"), "no search term for the registries"),
])
def test_a_damaged_manifest_names_what_is_wrong(damage, problem):
    taker = copy.deepcopy(TAKER)
    damage(taker)
    assert any(p.startswith(problem) for p in manifest.check(taker))


def test_the_loader_rejects_a_file_named_after_another_provider(tmp_path):
    (tmp_path / "other.json").write_text(json.dumps(TAKER))
    with pytest.raises(ValueError, match="differs from the file name"):
        manifest.load(tmp_path)


def test_every_manifest_in_the_repo_loads_and_validates():
    assert manifest.load()


NOTE = {"country": "ME", "checked": "2026-11-07", "summary": "Every sale needs a fiscal receipt.",
        "claims": [{"topic": "fiscal receipts", "statement": "Card sales are fiscalised.", "official": True,
                    **EVIDENCE}],
        "services": [{"name": "Example Fiscal", "source": "https://example.com/fiscal"}]}


def test_a_complete_receipts_note_passes():
    assert manifest.receipts_problems(NOTE) == []


@pytest.mark.parametrize("damage, problem", [
    (lambda n: n["claims"][0].pop("quote"), "claim 1.quote missing"),
    (lambda n: n["claims"][0].update(official="yes"), "claim 1: official is not true or false"),
    (lambda n: n["services"][0].update(source="example.com"), "service 'Example Fiscal' needs a name and an https source"),
    (lambda n: n.pop("summary"), "no summary"),
])
def test_a_damaged_receipts_note_names_what_is_wrong(damage, problem):
    note = copy.deepcopy(NOTE)
    damage(note)
    assert problem in manifest.receipts_problems(note)


def test_every_receipts_note_in_the_repo_loads_and_validates():
    assert set(manifest.load_receipts()) == {"ME", "UA"}
