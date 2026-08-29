## Agent skills

### Issue tracker

Issues and specifications are tracked in `Dharshan2004/techjam-2026-track-4-shopping-copilot` using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five canonical triage labels without aliases. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository using a root `CONTEXT.md` and system-wide ADRs under `docs/adr/`. See `docs/agents/domain.md`.

### Submission constraints

The deliverable is a runnable Python Shopping Agent, not a hosted website or UI. It must:

- implement the provided `Agent` interface, including `reset(...)` and the response method receiving the latest user message, turn number, and requested top-K;
- return a natural-language `message`, an optional `ask_attribute`, ordered catalog-valid parent-ASIN `recommendations`, and optional token `usage`;
- run locally through the official evaluator and obey its schema, dependencies, time limits, stopping rules, and invalid-output handling;
- include reproducible setup instructions and disclose external models, datasets, APIs, embeddings, and indexes; and
- keep large assets out of the repository and provide documented download instructions instead.

Treat direct execution through the official Python evaluator as the primary contract. Containers, hosted services, and user interfaces may only be optional development or demonstration aids. See `docs/submission_rules.md`.
