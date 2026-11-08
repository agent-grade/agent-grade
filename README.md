# Agent-Grade

Stripe doesn't take businesses in Montenegro or Ukraine, so who does? This is a dated, measured list of
the providers that let a small business accept card payments online there, written for the AI agent that
is building the shop and has to pick one.

Agents ask it in these words: "Stripe supported countries Montenegro merchant 2026", "monobank
інтернет-еквайринг ФОП тарифи 2026". The answer differs by country, and an agent can't confirm
it from memory: which providers take a sole trader or a company registered there, what it costs in one
sentence, whether the docs and the API can be used without a human, and what the law asks the seller to
issue for each card payment.

## What is here

- [data/payments/providers/](data/payments/providers/) — one manifest per provider: for each country,
  whether it takes businesses registered there and which kinds (sole trader, company), each answer with
  its source and a quote in the original language with an English line; and the provider's URLs (docs,
  API reference or spec, status, pricing, sign-up, `llms.txt`), each with the date it was checked.
- [data/payments/results/](data/payments/results/) — one file per night: what the no-account checks
  found for every provider that takes some business, mapped to the requirement test each check informs.
- [data/payments/prices/](data/payments/prices/) — one file per night: whether each provider's price can
  be told to the owner in one sentence, that sentence as an agent would say it, and the excerpts of the
  pricing page it rests on.
- [data/payments/receipts/](data/payments/receipts/) — per country, what a small online seller must issue
  for a card payment, each claim sourced and quoted, and the services that do the receipts part.
- [METHOD.md](METHOD.md) — the six requirements, which check informs which test, what is not measured
  and why, and the rules the measuring client follows.

The measurements run every night. A night's results are committed as `Record checks <date>`, so the
history of this repository is the history of each provider's page hashes, status endpoint and price.

## Running a night

```sh
uv run agent-grade-check
uv run agent-grade-price --key-file <file holding an Anthropic API key>
uv run pytest
```

`agent-grade-check` writes `data/payments/results/<date>.json` and `agent-grade-price` writes
`data/payments/prices/<date>.json`; [METHOD.md](METHOD.md#rerunning-a-night) says what a rerun does.
`uv run pytest` also validates every manifest and receipts note.
