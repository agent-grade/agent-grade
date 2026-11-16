# Method

## The six requirements

A provider is measured against six requirements for a product whose first customer is an AI agent. The
agent picks a payment provider, a host or a messaging channel while it builds, with no human comparing
options, so it reaches for whatever it can read, call and price in one sentence on the first try. Each
requirement is accepted when its tests hold.

| # | Requirement | Accepted when |
|---|---|---|
| R1 | **Documentation a model can quote.** One canonical page per task, at a stable URL, whose examples run as pasted. | A fresh model session completes the first task from the docs alone, without a human correction. An `llms.txt` at the root lists the task pages; every example in them is executed in CI. |
| R2 | **An API that works first try.** Callable from sign-up to first successful request with no undocumented step, with idempotent writes and errors that state the fix. | One SDK per major language, generated from the same spec as the docs. Every error response names the field, the rule broken, and a valid example. Repeating any write with the same key is a no-op. |
| R3 | **A free tier an agent can enter alone.** A working API key by API call, without a sales form, a phone call or a card; the human is asked to act only when money starts. | Time from first HTTP request to first successful business call is under five minutes with no human. The human's first involvement is one tap: accept terms, add a card, or sign as the legal person. |
| R4 | **Presence in the corpus and the catalogs.** Findable where models look: public code, public answers, and the tool catalogs the model vendors read. | Open-source clients and examples on GitHub; questions answered in public under the product's name. An MCP server in the public registries, and an entry in each model vendor's tool or plugin catalog. Asked to do the task in the product's market, each major model names the product unprompted. |
| R5 | **Reliability an agent can verify.** Health, usage and incidents exposed as data: a status endpoint, outage webhooks, machine-readable usage. | A status endpoint answers in under a second with per-component state. Outage start and end arrive as webhooks within a minute. Usage and spend are readable per key, per day, by API. |
| R6 | **A price the agent can say in one sentence.** One line, with no matrix, no "contact sales", and no surprise at the end of the month. | The price is published, flat or per-unit, and readable by API. Spend caps are settable by the agent, and the cap is enforced, not advisory. |

## What a night measures

Every night two commands run against every provider whose manifest says it takes some business in a
country.

`agent-grade-check` fetches each URL in the manifest and hashes it (an HTML page by its visible text, any
other page by its bytes), so a later night marks what changed. It reads `llms.txt` and a JSON OpenAPI spec
where they exist, and lints the spec by rules. It asks the status page's host for a machine-readable summary
(`/api/v2/summary.json`) and times the answer. It sends one deliberately bad, unauthenticated call to the
API endpoint the manifest names, and judges the error by coarse keyword rules: does it name a field, the
rule broken, a valid example. It searches the MCP registry for the provider's search term, and looks up the
GitHub account and the npm packages the manifest names, each with its source. PyPI is not checked: its
project pages answer this client with a JavaScript challenge.

`agent-grade-price` reads each provider's pricing page (HTML or PDF) and decides whether the price a small
shop pays to accept card payments can be told to the owner in one sentence. A page that states no number
at all is decided by that rule alone. Any other readable page goes to Claude Haiku with a fixed rubric: it
writes the sentence as an agent would say it to the owner, or says why it can't, and quotes the excerpts of
the page its answer rests on. Each excerpt is checked against the page text, and the result records whether
all were found. Answers are kept in `data/payments/prices/judged/`, keyed by the hash of the page text and
of the method, so an unchanged page is not judged twice and each night records what it spent. A page that
could not be read, or whose API call failed (`not judged`), has `fits_one_sentence: null`; "can't be said
in one sentence" is `false`, a score, not a gap.

Each night's results file lists every test under `tests`, with the checks that inform it:

| Test | Informed by |
|---|---|
| R1 llms.txt lists the task pages, examples run in CI | `llms_txt`, `docs` |
| R2 one SDK per major language from the same spec | `api_reference`, `spec`, `npm` |
| R2 every error names the field, the rule broken, a valid example | `probe` |
| R4 open-source clients and examples on GitHub | `github`, `npm` |
| R4 an MCP server in the public registries | `mcp` |
| R5 a status endpoint under a second with per-component state | `status` |
| R6 the price is published, flat or per-unit, readable by API | `pricing` |

The price file adds the one-sentence judgment to the R6 test. A check informs a test; it does not pass or
fail it on its own. The keyword rules for errors and the lint rules for specs are coarse on purpose, and are
in the code for anyone to read.

## What is not measured, and why

The same file marks every other test undecided, with what it needs:

| Test | Needs |
|---|---|
| R1 one canonical page per task, first try from the docs alone | a model run |
| R2 repeating a write with the same key is a no-op | an account |
| R3 a working key with no human in under five minutes | an account |
| R3 the human's first involvement is one tap | an account |
| R4 questions answered in public under the product's name | a search of public answers |
| R4 an entry in each model vendor's tool or plugin catalog | a read of each vendor's catalog |
| R4 each major model names it unprompted | a model survey |
| R5 outage webhooks within a minute | an account |
| R5 usage and spend readable per key by API | an account |
| R6 spend caps settable and enforced | an account |

The nightly checks never sign up, never log in and never send a real payment call, so a test that needs an
account is recorded as undecided rather than guessed. The model runs and the searches of answers and vendor
catalogs are separate measurements, not part of a night.

## The model runs

`agent-grade-models` writes `data/payments/models/<date>.json` with every prompt, answer and verdict (an R1
page by its URL and hash), saving after each record. A second run on the same date, or one given `--date`,
keeps what the file holds and asks only what is missing.
Each model call is a fresh session with no tools, at effort `medium`; a refused or failed call is recorded
as not measured. A spend cap stops any call that could pass it.

R1, first try from the docs alone: for each provider that takes a business, Claude Sonnet is given the one
task page the manifest names (the page behind its error call, else the docs entry page) and asked to write
the first request that takes a card payment, with placeholders for keys and signatures. The request is not
sent. A second Claude Sonnet call judges it against the same page under a fixed rubric, quoting the excerpts
its verdict rests on, and each excerpt is checked against the page. A provider with no public docs, or whose
page has under 1,000 or over 200,000 characters of readable text, is not measured, with the reason. The page
cannot erase what the model already remembers, so a correct first try shows the page suffices together with
that memory.

R4, named unprompted: Claude Haiku, Sonnet, Opus and Fable are each asked three times per country, without a
provider's name in the question, which providers a sole trader there can use to accept card payments. Each
answer records the manifest providers it mentions, matched as a whole word by the manifest's search term (its
id where it has none) or a spelling listed in the code (`OTP banka`, `Rozetka Pay`, `Concord Pay`). A
mention is not a recommendation: an answer may name Stripe only to say it doesn't serve the country. So
Claude Haiku reads each answer under a fixed rubric and labels every provider it mentions recommended,
rejected or neutral, quoting the words the label rests on, and each quote is checked against the answer. An
answer is labelled only when every provider it mentions gets a label; otherwise it is asked again on the next
run. The file counts, per country, the answers that mention each provider, the answers labelled, and among
those, the answers that recommend it. Other vendors' models are not measured: only the Anthropic API was
available.

## Manifests

An answer about who a provider takes is "yes" only where a quoted line from the provider's own pages says
so; otherwise it is "not found". A provider that takes no business in a country is kept, with the source
that says so. Every URL slot has a status: `ok`, `not found` (the URL is `null`), or `unreadable` (the page
exists or may exist but the client could not read it, with the reason in a note).

## Receipts notes

Each country's note lists what a small online seller must issue for a card payment. Every claim has its
source, a quote in the original language and an English line, and is marked official when a government
body publishes the text (an archived copy included). The note also lists the services that do the receipts
part, and the questions the sources left open, with the date it was checked. Where a government site
refuses this client, the note says how its quotes were checked.

## The client's rules

- It identifies itself as `Agent-Grade/<version> (+https://agent-grade.com)`.
- It obeys robots.txt, redirects included; an unreachable robots.txt counts as disallowing the whole site.
  The one exception is the error call to Paddle's and RozetkaPay's API hosts: an API host's robots.txt
  addresses crawlers, and this call only expects an error back. Its result is marked `robots_exemption`.
- It sends one request at a time, with at least a second between requests to the same host, and never
  retries.
- It never signs up, never logs in and never submits a form.

## Rerunning a night

Each command writes that date's file, and a second run on the same date overwrites it;
`agent-grade-check --providers <ids>` instead merges the named providers into it. Both use the current
date, so a night can be rerun only on its own date. The price judgments are cached, so a rerun over
unchanged pages costs nothing. A run in which no host answered at all exits non-zero and writes nothing.
