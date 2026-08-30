# Session Tracing

Slice 16 traces every evaluation session through Langfuse without letting
telemetry reach the scoring path. The behavior follows
`docs/adr/0003-fail-open-langfuse-observability.md` and PRD stories 43–46.

## Where tracing lives

`starter/telemetry.py` holds the whole feature:

| Component | Responsibility |
| --- | --- |
| `Tracer` | Groups turn traces by `session_id`, buffers them in memory, and exports only on `flush()` |
| `TurnTrace` | One evaluation turn, with nested operation timings and failure causes |
| `sanitize` | Reduces every payload to structured operational evidence |
| `LangfuseSink` | Lazily built Langfuse client used at flush time |
| `NullSink` | Disabled telemetry; it never constructs a client or touches the network |
| `MemorySink` | In-process sink for tests and offline inspection |

`starter.Agent` starts a session trace in `reset(...)` and opens one turn trace
per `respond(...)` call. Neither method changes its return value when tracing is
active, disabled, or failing.

## Configuration

| Variable | Effect |
| --- | --- |
| `LANGFUSE_PUBLIC_KEY` | Required to enable export |
| `LANGFUSE_SECRET_KEY` | Required to enable export |
| `LANGFUSE_HOST` | Optional self-hosted endpoint |
| `SHOPPING_AGENT_TELEMETRY` | `0`, `false`, `off`, or `no` disables tracing even with credentials |

Both keys must be present. Without them `Tracer.from_environment()` returns a
disabled tracer whose reason is `missing_credentials`, so a clean checkout traces
nothing and needs no extra dependency. Credentials belong in the environment and
are never committed or recorded in a trace.

Export requires the optional client:

```bash
python -m pip install -r requirements-observability.txt
```

## Trace shape

One trace is emitted per turn, named `shopping-turn` and grouped by
`session_id`. Its metadata carries the turn number and the configuration
identity captured at `reset(...)`: Fusion Policy version and weights, route and
fused candidate depths, planning prompt version, whether planning is connected or
local, dense route status, and catalog size.

Each trace nests one observation per operation, in execution order:

| Observation | Recorded evidence |
| --- | --- |
| `interpretation` | Plan source, attempt count, fallback reason, reason codes, selected retrieval tools, mutation count, prompt and completion tokens |
| `state_validation` | Applied plan, revision before and after, active and dismissed counts, and one structured decision per Constraint with attribute, normalized values, match rule, classification, scope, status, confidence, and source turn |
| `retrieval` | Requested routes, route depth, and per-route candidate counts, nesting `retrieval.dense` with the route status and its failure cause |
| `fusion` | Policy version, weights, candidate depths, fused candidate count, and whether the deterministic backfill ran |
| `reranking` | Whether a connected plan requested `local_rerank` and why it was skipped |
| `clarification` | Clarification source, asked attribute, and whether a question was asked |
| `response` | Requested `top_k`, applied recommendation limit, recommendation count, message length, and reported token usage |

Every observation carries its own `latency_ms`, an `ok` or `error` status, and a
classified `failure_cause`. A refused dense route, a rejected Turn Plan, and a
rejected fallback plan are therefore distinguishable from each other and from a
model failure.

## What is never recorded

`sanitize` runs over every payload before it leaves the process. It drops:

- credentials and secrets, by exact key and by substring (`api_key`, `secret`,
  `password`, `credential`, `authorization`, `bearer`, `access_token`);
- the raw user profile, which `reset(...)` never forwards to the tracer;
- customer message text, clarification text, and `raw_phrase` provenance, which
  can quote the customer verbatim;
- planning instructions, prompts, and any private chain-of-thought, rationale, or
  reasoning field; and
- catalog records; only identifiers and counts are exported.

Planning error text is reduced to its error class by `starter.agent.reason_codes`
because a rejection message can quote customer phrasing. Strings are truncated to
200 characters, sequences to 20 items, mappings to 40 keys, and nesting to four
levels, so an unexpected payload cannot become an unbounded export.

## Fail-open guarantees

Telemetry can only add in-memory dictionary work to a turn:

- missing credentials, a disabled switch, or a failed sink construction produce a
  disabled tracer that buffers nothing;
- the Langfuse client is built at the first export, so an absent
  `langfuse` package, connection refusal, timeouts, and an unreachable host
  surface at flush time, never inside `respond(...)`; a missing package disables
  the sink with `dependency_unavailable` after one attempt instead of retrying on
  every flush;
- `Tracer.flush()` catches every exception, records the cause in
  `export_failures` and `last_failure_cause`, and returns `False`;
- the buffer is a bounded deque of 512 traces; a stalled export drops the oldest
  traces and counts them in `dropped_traces` instead of growing without limit; and
- trace construction, recording, and submission are individually guarded, so a
  telemetry defect degrades to `last_failure_cause` rather than an agent error.

`tests/test_telemetry.py` asserts response equality between a traced agent and a
disabled agent across missing credentials, connection refusal, timeout, queue
failure, export failure, and a network-disabled run.

## Flush boundary

Turns only append to the in-memory buffer. Export happens when:

- `Agent.flush_telemetry()` is called, for example after an evaluator run; or
- the process exits, through the `atexit` hook an enabled tracer registers.

Nothing is exported inside the latency-critical response path. `python3 -m
evaluator.local_evaluator` is therefore unchanged and still flushes at process
completion.

## Inspecting a run

```bash
python3 - <<'PY'
from starter.agent import Agent
from starter.telemetry import MemorySink, Tracer

sink = MemorySink()
agent = Agent("data/catalog.jsonl", tracer=Tracer(sink, register_atexit=False))
agent.reset("demo", {})
agent.respond("demo", "comfortable house shoes for cold floors", 1, 10)
agent.flush_telemetry()
print(sink.traces[0]["metadata"]["configuration"])
for observation in sink.traces[0]["observations"]:
    print(observation["name"], observation["status"], observation["latency_ms"])
PY
```

`Agent.get_telemetry_metrics()` reports the same evidence for a live run:
whether tracing is enabled, buffered, submitted, exported, and dropped trace
counts, export failures, flush count, last failure cause, and the sink status.
