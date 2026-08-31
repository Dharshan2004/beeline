# Connected OpenAI Planning Benchmark — Phase A

## Status

The Phase A adapter and benchmark machinery are implemented and validated both
offline and through a credentialed one-session paired smoke run. The frozen
160-session development split produces 798
canonical planning fixtures with input-corpus SHA-256
`c05e7f1b88abbc7d6cee3f5b81b7366c98daab86f63ec6da44fcf820d13df7c7`.
The loader discarded all 40 designated Locked Holdout rows before JSON
deserialization. No production default is selected or activated in Phase A.

The credentialed smoke was run after explicit approval to disclose its request
payloads to OpenAI. Each request includes the development user message,
the local Constraint State snapshot, up to four recent canonical turns, the
allowed Retrieval Routes, and catalog-derived supported attribute values. It
does not include the API key, raw user profile, complete product records, the
Locked Holdout, or private chain-of-thought. Responses use `store=false`.

## Frozen comparison contract

- Quality reference: `gpt-5.6-sol`.
- Lower-cost model: `gpt-5.6-luna`.
- API: stateless Responses API with strict JSON Schema Structured Outputs.
- Reasoning effort: `low` for both models.
- Prompt/schema: the unchanged Slice 12 `shopping-turn-planner-v2` contract.
- Transport schema: `openai-anyof-v1` converts canonical `oneOf` unions to
  OpenAI-supported `anyOf`, removes unsupported `uniqueItems`, and makes
  implicit string types explicit. The canonical prompt/schema and strict local
  validation remain unchanged.
- Tools: the same `structured`, `bm25`, and `dense` identifiers; no hosted or
  executable tools are exposed.
- Input parity: fixtures are generated once from canonical local state and then
  replayed to both models. A model proposal cannot alter a later fixture.
- Split: `public-split-v1-development-only`; holdout identifiers are rejected.
- Budget: shared $10 Phase A ceiling, $40 warning, review before $50, and an
  absolute programmatic stop at $600. A worst-case reservation is checked before
  every request. Actual token cost is recorded whenever the API supplies valid
  usage; a submitted request that times out or returns unusable usage is charged
  the full reservation pessimistically. Raising the configured limit above $50
  requires an explicit review-approval flag.

The checked-in prices are the standard text-token prices published on
2026-08-31: Sol at $4/M input and $20/M output, and Luna at $0.20/M input and
$1.20/M output. Recheck prices before a later run and version any change in the
configuration. See the official [Sol model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
and [Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna).

## Metrics and report schema

Each model report contains aggregate and per-scenario:

- exact canonical Constraint State accuracy;
- exact mutation-decision accuracy across add, reintroduce, dismiss,
  Constraint replacement, and Product Intent replacement operations;
- exact replacement subtype/target accuracy;
- canonical Retrieval Route agreement (order-insensitive), plus an individual
  selection rate for `structured`, `bm25`, and `dense`;
- provisional Clarification protocol quality;
- downstream local retrieval HitRate@10 and MRR;
- failure rate and classified failure causes;
- p50, p95, and mean request latency;
- prompt, completion, and total token use; and
- estimated cost from the versioned per-model prices, including each model's
  attributed pessimistic reservations for submitted calls without usable usage.

The stable paired-report contract is checked before output and published as
`docs/openai_model_benchmark_report.schema.json`. The corpus hash proves both
models receive the same canonical `PlanningRequest` inputs. The separately
reported transport-schema version and SHA identify the transformed schema sent
to OpenAI.

Clarification scoring is intentionally limited to schema and state-policy
validity in Phase A. Slice 13 owns the final Session Mode, expected-value policy,
prompt/schema, and Clarification fixtures. Phase B must rerun the complete
development benchmark after Slice 13 merges before selecting any default.

The provisional tolerance is predeclared in
`config/openai_phase_a_benchmark.json`: no more than two percentage points of
aggregate state-accuracy regression, five points per scenario, five points of
downstream HitRate@10 regression, and no additional failures. The report may
state whether Luna is provisionally within tolerance, but its
`default_activation_allowed` field remains false.

## Reproduction

Install the optional development-only dependencies:

```bash
.venv/bin/pip install -r requirements-openai.txt
```

Validate all 160 development sessions without credentials, network, or API cost:

```bash
.venv/bin/python -m tools.benchmark_openai_planning \
  --validate-only \
  --output benchmarks/openai_phase_a_validation.json
```

After explicitly approving the external disclosure described above, reproduce
the small deterministic paired smoke benchmark using the ignored `.env` file:

```bash
.venv/bin/python -m tools.benchmark_openai_planning \
  --sessions 1 \
  --env-file .env \
  --output benchmarks/openai_phase_a_smoke_1.json
```

Omit `--sessions` only when the budget and full development run have been
reviewed. Raw reports remain under the ignored `benchmarks/` directory. The
checked-in machine-readable Phase A status is
`docs/openai_model_benchmark_phase_a.json`.

## Current provisional result

The 2026-08-31 paired smoke evaluated four canonical browsing fixtures for each
model with identical input hash
`b3063aa33b90fa3e4b8bded234670aa69d75d26248b339c116339ec124d97fda`.
Both models completed with zero failures, 0.25 exact state accuracy, 0.50 local
HitRate@10, and 1.0 clarification-protocol validity. Sol used 4,818 tokens,
cost $0.035128, and had p50/p95 latency of 3.956833/6.607046 seconds. Luna used
4,811 tokens, cost $0.0019462, and had p50/p95 latency of
3.253087/5.694569 seconds. Total recorded spend was $0.0370742.

Luna is within the predeclared provisional tolerance on this deliberately small
smoke. This is not evidence for a production default: `default_activation_allowed`
remains false, and the complete Phase B development rerun remains gated on
Slice 13.
