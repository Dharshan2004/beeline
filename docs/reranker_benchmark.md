# Reranker and Deep Candidate Pool Benchmark

Slice 07 is a decision gate. It selects two things that later slices are not
allowed to revisit casually:

1. which compact cross-encoder the packaged agent bundles, and
2. the deepest Candidate Pool that cross-encoder may rerank on a scored turn.

Both are chosen against a documented quality-versus-runtime rule, on the
development split only. The locked 40-session holdout is not opened here.

## Why the depth question exists

Slice 06 fuses three Retrieval Routes and truncates to 30 candidates. The
candidate-boundary analysis in the engineering journal reported that the fused
top 30 makes the Target Product reachable in roughly 0.77 of sessions, while the
deeper base-route union of roughly 100–200 candidates reaches roughly 0.93.
Reranking cannot recover a product fusion already discarded, so the pool depth
sets a hard ceiling on everything downstream.

The counterweight is runtime. A cross-encoder scores every (query, product) pair
independently, so its cost is linear in pool depth and is paid on every turn of
every session. The dense-enabled no-reranker run already costs a few minutes for
200 sessions before any reranking.

## What is compared

Three bundled cross-encoders, all CPU-only and all loaded from disk:

| Identity | Layers | Hidden | Packaged size |
| --- | ---: | ---: | ---: |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | 2 | 128 | 35 MB |
| `cross-encoder/ms-marco-MiniLM-L-2-v2` | 2 | 384 | 121 MB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 6 | 384 | 175 MB |

against candidate depths 30, 50, 100, 150, and 200, plus the shipped fused-30
baseline with no reranking at all.

Each route contributes at most 100 candidates, so the base-route union is at most
200 products. Depth 200 is therefore the whole union rather than an arbitrary
cutoff, and pool recall at 200 is the recall ceiling that any shallower depth
gives up.

## Method: one cache, every configuration

A benchmark that re-runs the conversation per configuration would compare
different Candidate Pools and different conversations at the same time. This
benchmark instead runs in three stages.

**Stage 1 — cache.** The shipped fused-30 configuration is replayed once through
the official evaluator on the 160 development sessions. For every scored turn the
Agent records the reranker query, the fused ordering over the base-route union at
depth 200, the fused top 30 actually used for the response, and the Target
Product label. Tracing does not change the response, so the cached trajectory is
exactly the trajectory the shipped agent produced.

**Stage 2 — score.** Each model scores the cached deep pools in a fresh process.
A deeper pool is a prefix extension of a shallower one, so scoring the pool in
segments that end on each depth boundary yields both the ordering and the
measured latency for every depth from one pass. Every model and every depth sees
byte-identical Candidate Pools and labels.

**Stage 3 — summarize.** Per-model reports are merged, projected onto one full
200-session evaluator run, and passed through the selection rule.

```bash
python -m tools.benchmark_reranker cache \
    --output benchmarks/rerank_cache.jsonl

for identity in ms-marco-TinyBERT-L-2-v2 ms-marco-MiniLM-L-2-v2 ms-marco-MiniLM-L-6-v2; do
  python -m tools.benchmark_reranker score \
      --identity "cross-encoder/${identity}" \
      --sessions 40 \
      --output "benchmarks/rerank_${identity}.json"
done

python -m tools.benchmark_reranker summarize benchmarks/rerank_ms-marco-*.json \
    --output docs/reranker_benchmark.json
```

Fetch the three cross-encoders once, before benchmarking, with the same
development-only script used for the embedding model:

```bash
python -m tools.fetch_model \
    --identity cross-encoder/ms-marco-MiniLM-L-2-v2 \
    --destination models/cross-encoder__ms-marco-MiniLM-L-2-v2
```

## Execution environment

Every stage sets `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1` before torch or transformers are imported, so the
benchmark is CPU-only and a missing local model is a hard failure rather than a
silent download. Sequences are padded to a fixed length rather than to the
longest item in a batch, so a product's score does not depend on which products
shared its batch.

## Metrics and what they mean

Quality is reported at two separable levels, because they fail differently.

- **Pool recall** is the probability the Target Product is inside the pool at a
  given depth. It is the ceiling; no reranker can exceed it.
- **Replay Hit Rate@10 and MRR** score the cached replay the way the evaluator
  scores a live run: a session converts on the earliest cached turn whose
  reranked top ten contains the Target Product.
- **Recall-to-hit conversion** is Hit Rate@10 divided by pool recall: of the
  sessions where the product was reachable, the fraction actually surfaced.

Replay metrics are an offline proxy. The trajectory is frozen by the cache, so
reranking cannot change which questions the simulated customer answers next.
The end-to-end figures that count are produced by Slice 08 running the real
evaluator; the replay numbers exist to compare configurations cheaply and fairly.

Runtime is reported as p50, p95, and mean added latency per turn, projected added
seconds and projected wall clock for a full 200-session run, peak resident set
size, and packaged model size.

## Selection rule

The competition specification publishes no numeric wall-clock limit. It states
only that timeouts may count as a miss and that latency is a reported feasibility
measure. The rule below is therefore a self-imposed budget, fixed before the
results were read, and frozen for Slices 08–17:

> A configuration is admissible when its projected wall clock for one full
> 200-session evaluator run is at most **900 seconds** and its p95 added per-turn
> latency is at most **1.5 seconds**. Among admissible configurations, select the
> highest replay Hit Rate@10; break ties by lower projected wall clock, then by
> smaller packaged model.

900 seconds keeps a complete public-set run under fifteen minutes, so the release
gate stays cheap enough to run repeatedly, and it leaves roughly 3.5x headroom
over the dense-enabled no-reranker baseline. The per-turn bound is the part that
protects against a timeout being scored as a miss, and it is enforced at runtime
by the reranker's per-turn deadline rather than only in this report.

If no reranked configuration fits, the report freezes the largest feasible
deterministic depth and states the pool recall given up by that truncation.

## Preliminary sizing measurement

Before the full benchmark, per-turn scoring cost was measured directly on this
CPU with a fixed query and a representative product rendering, eight torch
threads, batch size 32, and sequence length 128. It is a micro-benchmark on
synthetic input, not the benchmark result, and it exists to size the run:

| Model | Depth 100 | Pairs/s | Depth 200 (projected) |
| --- | ---: | ---: | ---: |
| `ms-marco-TinyBERT-L-2-v2` | 118 ms | 847 | 236 ms |
| `ms-marco-MiniLM-L-2-v2` | 494 ms | 203 | 987 ms |
| `ms-marco-MiniLM-L-6-v2` | 1395 ms | 72 | 2790 ms |

Charged against roughly 1,300 scored turns in a 200-session run, that projects
to +307 s for TinyBERT at depth 200, +642 s for MiniLM-L-2 at depth 100, and
+3,600 s for MiniLM-L-6 at depth 200. Deep reranking with the largest model is
therefore already implausible, and the benchmark's job is to find where quality
stops paying for the depth it costs.

An earlier version of this measurement was taken while seven dense-index builds
were saturating the machine and reported figures 4x to 17x worse. It is recorded
here because it is exactly the kind of number that would have frozen the wrong
depth: a contended benchmark host silently rules out configurations that are
comfortably affordable on an idle one.

## Results

**Not yet run.** The cache stage requires a dense-enabled agent, and the dense
artifact must be rebuilt for the pinned `all-MiniLM-L6-v2` embedder before the
baseline in stage 1 measures the intended three-route system. The artifact is
deliberately untracked, so it is a local build step, not a repository asset:

```bash
python -m retrieval.build_dense_index --catalog data/catalog.jsonl --verify-load
```

Verify `Agent.get_dense_route_metrics()` reports `status: available` with the
readiness preflight in `README.md` before running stage 1. A run whose dense
route silently disabled itself measures a two-route system and must not be
reported here.

Until this section carries measured numbers, the reranker identity and depth
constants in `retrieval/reranker.py` and `starter/agent.py` are provisional
defaults chosen from the sizing measurement above, not a selection.

## Live integration (Slice 08)

Slice 08 places the selected cross-encoder after fixed fusion in the live Agent
and applies it to the deep pool rather than a pre-truncated fused top 30:

- fusion produces exactly `RERANK_CANDIDATE_DEPTH` candidates, so there is no
  fused-30 truncation before reranking;
- the reranker may only reorder that set. If it returns a different candidate
  set, the Agent refuses the result and keeps the fused ordering;
- a missing model, a load error, a scoring error, or an expired per-turn
  deadline all return the fused ordering over the same candidates;
- the returned ten are catalog-valid and unique in every case.

`Agent.get_reranker_metrics()` exposes status, disabled reason, load time, turn
count, fallback and deadline counts, and last turn duration, without adding
fields to the official response schema.

## Reproduction

The machine-readable summary of the run reported above is checked in as
`docs/reranker_benchmark.json`. Raw caches and per-model reports are written to
`benchmarks/`, which is deliberately untracked.
