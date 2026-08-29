# Domain Docs

This is a single-context repository. Engineering skills should use the root domain glossary and system-wide architecture decisions when they exist.

## Before exploring

- Read `CONTEXT.md` at the repository root when it exists.
- Read ADRs under `docs/adr/` that affect the area being changed.
- If either location does not exist, proceed without treating its absence as a problem.

`CONTEXT.md` and ADRs are created lazily by domain-modeling workflows when real terminology or durable decisions are resolved.

## Expected layout

```text
/
├── CONTEXT.md
├── docs/
│   ├── agents/
│   └── adr/
└── starter/
```

## Vocabulary

Use terms exactly as defined in `CONTEXT.md` in issue titles, specifications, tests, architecture proposals, and code. Avoid introducing synonyms for established concepts.

If a necessary concept is absent from the glossary, reconsider whether it belongs to the domain. If it does, record the gap for a domain-modeling discussion.

## Architecture decisions

If proposed work conflicts with an existing ADR, surface the conflict explicitly. Do not silently replace or work around a recorded decision.
