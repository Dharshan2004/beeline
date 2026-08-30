# Reranker and Deep Candidate Pool Benchmark

Slice 07 freezes `cross-encoder/ms-marco-MiniLM-L-6-v2` at Candidate Pool
depth **50** for Slice 08. The decision was made on all 160 development
sessions. The locked 40-session holdout was not loaded, including for timing.

The selected replay result improves HitRate@10 from 0.543750 to 0.600000 and
TechnicalScore from 0.467096 to 0.507240. Its measured rerank latency is
435.5 ms p50 and 548.9 ms p95, and its normalized 200-session wall-clock
projection is 800.4 seconds. The next depth, 100, improves quality further but
projects to 1,366.7 seconds and therefore fails the frozen 900-second gate.

## Decision rule

A configuration is admissible only when all of these predeclared conditions
hold:

- normalized 200-session wall time is at most 900 seconds;
- measured p95 added rerank latency is at most 1.5 seconds per turn;
- HitRate@10 does not regress from the fused-30 baseline; and
- replay TechnicalScore strictly improves over that baseline.

The deepest admissible pool wins. At that depth, candidates are ordered by
TechnicalScore, MRR, lower p95 latency, and smaller package. If none qualifies,
one predeclared smaller-model contingency may run without relaxing the gates;
otherwise the system stays at fused-30 with no reranker.

This depth-first rule preserves candidate reachability for later Fusion Policy
work. It is intentionally different from choosing the single highest replay
score regardless of runtime.

## Candidate Pool contract

Structured, BM25, and dense retrieval independently admit at most 100
candidates. Their deduplicated union is therefore bounded at 300. Tied route
scores are capped deterministically by canonical parent-ASIN order. The
benchmark records independently generated exact pools at depths 30, 50, 100,
150, 200, 250, and 300 and verifies that each shallower pool is a prefix of the
deepest pool before reusing cross-encoder scores.

The observed union contained 51–299 products per turn, with mean 227.29 and
median 229. Session-level reachability was:

| Depth | Pool recall |
| ---: | ---: |
| 30 | 0.72500 |
| 50 | 0.77500 |
| 100 | 0.80000 |
| 150 | 0.83750 |
| 200 | 0.88125 |
| 250 | 0.89375 |
| 300 | 0.90000 |

Freezing depth 50 gives up 0.125 session-level pool recall relative to the
observed depth-300 ceiling. That loss is explicit evidence for later pool-aware
fusion work, not a claim that the deeper pool lacks value.

## Method

The benchmark uses three stages so models never see different trajectories or
labels.

1. **Cache:** replay the planning-aware shipped Agent once through the official
   evaluator on all 160 development sessions. Record the exact pools, fused-30
   response pool, reranker query, target, scenario, turn, and conversion
   eligibility. Intent Override targets are ineligible before the replacement
   Product Intent commits.
2. **Score:** in a fresh CPU-only process for each model, score every cached pool
   with fixed padding, batch size 32, sequence length 128, and eight torch
   threads. No conversation or retrieval step is rerun between candidates.
3. **Summarize:** merge the reports, enforce both runtime and quality gates, and
   project the measured 160-session baseline and added per-turn work linearly to
   200 sessions.

The cache covers 1,009 scored turns. Its dense route remained available for all
1,009 queries and returned 100 candidates on the final query. Cache construction
fails if dense readiness is not `available`, if fused-30 differs from the exact
depth-30 pool, or if a scoring pool violates the prefix invariant.

The replay trajectory is frozen, so post-rerank metrics compare configurations
fairly but cannot model a reranker's effect on later customer answers. Slice 08
must validate the chosen configuration end to end through the official Agent.

## Execution controls and model provenance

The run used Python 3.11.9 on an Apple M2 MacBook Air with 8 CPU cores and 16 GB
memory. Every stage forces `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`; cache construction additionally rejects socket
connections. Missing or mismatched local model provenance is a hard failure.

| Model | Immutable revision | Package | Peak RSS |
| --- | --- | ---: | ---: |
| `ms-marco-TinyBERT-L-2-v2` | `81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc` | 18,504,352 B | 722,173,952 B |
| `ms-marco-MiniLM-L-2-v2` | `1b5cd67b15209f24824c50370e0397743aa9b787` | 63,423,160 B | 863,977,472 B |
| `ms-marco-MiniLM-L-6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | 91,821,988 B | 775,372,800 B |

The dense-enabled fused-30 cache took 191.71 seconds for 160 development
sessions, or 239.6 seconds when normalized to 200. The 200-session value is a
projection, not a run over development plus holdout.

## Results

The table shows the baseline and the rows that determine the decision. The
machine-readable artifact contains all 22 baseline/model/depth rows.

| Configuration | Depth | Pool recall | HR@10 | MRR | TechnicalScore | p95 ms | Projected wall s | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fused, no reranker | 30 | 0.72500 | 0.54375 | 0.368237 | 0.467096 | 0.0 | 239.6 | baseline |
| TinyBERT-L2 | 50 | 0.77500 | 0.56250 | 0.308276 | 0.463608 | 49.5 | 289.1 | quality floor failed |
| MiniLM-L2 | 30 | 0.72500 | 0.57500 | 0.324950 | 0.476860 | 116.2 | 355.0 | admissible, shallower |
| MiniLM-L2 | 50 | 0.77500 | 0.56250 | 0.309328 | 0.463298 | 198.4 | 434.9 | quality floor failed |
| MiniLM-L6 | 30 | 0.72500 | 0.60625 | 0.392250 | 0.516425 | 318.6 | 569.6 | admissible, shallower |
| **MiniLM-L6** | **50** | **0.77500** | **0.60000** | **0.376215** | **0.507240** | **548.9** | **800.4** | **selected** |
| MiniLM-L6 | 100 | 0.80000 | 0.61875 | 0.373383 | 0.519015 | 1092.6 | 1366.7 | wall gate failed |
| MiniLM-L6 | 150 | 0.83750 | 0.63125 | 0.378269 | 0.528231 | 1625.9 | 1919.4 | both runtime gates failed |

TinyBERT fits the runtime envelope at every tested depth but never beats the
baseline TechnicalScore. MiniLM-L2 qualifies only at depth 30. MiniLM-L6 has the
strongest quality and qualifies at depths 30 and 50; depth 50 wins because the
frozen rule prioritizes the deepest admissible pool.

Slice 07 records the decision but does not activate live reranking. Slice 08 is
responsible for running this depth-50 model in the persistent cancellable worker,
enforcing the absolute 1.5-second deadline, and preserving the fused ordering on
startup failure, crash, malformed output, or timeout.

## Reproduction

Fetch the exact revisions outside the scoring path:

```bash
python -m tools.fetch_model --identity cross-encoder/ms-marco-TinyBERT-L-2-v2 \
  --destination models/cross-encoder__ms-marco-TinyBERT-L-2-v2 \
  --revision 81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc
python -m tools.fetch_model --identity cross-encoder/ms-marco-MiniLM-L-2-v2 \
  --destination models/cross-encoder__ms-marco-MiniLM-L-2-v2 \
  --revision 1b5cd67b15209f24824c50370e0397743aa9b787
python -m tools.fetch_model --identity cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --destination models/cross-encoder__ms-marco-MiniLM-L-6-v2 \
  --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

Then run all 160 development sessions and all declared depths:

```bash
.venv/bin/python -m tools.benchmark_reranker cache \
  --output benchmarks/rerank_cache.jsonl

.venv/bin/python -m tools.benchmark_reranker score \
  --identity cross-encoder/ms-marco-TinyBERT-L-2-v2 \
  --output benchmarks/rerank_TinyBERT-L-2.json
.venv/bin/python -m tools.benchmark_reranker score \
  --identity cross-encoder/ms-marco-MiniLM-L-2-v2 \
  --output benchmarks/rerank_MiniLM-L-2.json
.venv/bin/python -m tools.benchmark_reranker score \
  --identity cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --output benchmarks/rerank_MiniLM-L-6.json

.venv/bin/python -m tools.benchmark_reranker summarize \
  benchmarks/rerank_TinyBERT-L-2.json \
  benchmarks/rerank_MiniLM-L-2.json \
  benchmarks/rerank_MiniLM-L-6.json \
  --output docs/reranker_benchmark.json
```

Raw caches and per-model reports remain ignored under `benchmarks/`. The full
selection evidence is checked in as `docs/reranker_benchmark.json`.
