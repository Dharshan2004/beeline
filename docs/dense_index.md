# Dense Index Artifact

The dense Retrieval Route searches a versioned artifact that is built ahead of
time and loaded once at agent startup. No turn pays indexing cost, no model is
downloaded on the scoring path, and no vector service has to be running.

The live Shopping Agent loads the Slice 04 artifact once at startup and queries
it at depth 100 on every turn. Dense candidates keep their route order after
unknown identifiers, duplicates, and Hard Constraint mismatches are removed;
deterministic lexical retrieval fills any recommendation positions left open.

The route is optional for validity. Missing dependencies or assets, manifest
mismatches, load failures, and query failures set its status to `disabled` and
leave the standard-library lexical route in control. The failure reason remains
inspectable through `Agent.get_dense_route_metrics()` and never appears as an
extra field in the schema-constrained turn response.

## Live route measurements

`Agent.get_dense_route_metrics()` returns the route status and disabled reason,
startup load time, completed query count, most recent query time, and raw dense
candidate count. These operational values let development and evaluation tooling
measure the route without changing the official response schema. Times use the
process monotonic performance clock and are reported in seconds.

## Required preflight before a scored run

The dense route is deliberately fail-open: an unavailable dependency or asset
does not invalidate the Agent response, but it does mean the evaluation is no
longer measuring the intended three-route system. Run this preflight with the
exact Python interpreter that will launch the evaluator:

```bash
.venv/bin/python - <<'PY'
from starter.agent import Agent

agent = Agent("data/catalog.jsonl")
metrics = agent.get_dense_route_metrics()
assert metrics["status"] == "available", metrics
agent.reset("dense-preflight", {})
agent.respond("dense-preflight", "comfortable house shoes for cold floors", 1, 10)
metrics = agent.get_dense_route_metrics()
assert metrics["status"] == "available", metrics
assert metrics["query_count"] == 1, metrics
assert metrics["last_candidate_count"] > 0, metrics
print(metrics)
PY
```

All assertions must pass. `disabled_reason` identifies the first startup or
query failure. Common causes are using system Python instead of `.venv`, missing
packages from `requirements-dense.txt`, missing model/index directories, or a
catalog/model checksum mismatch. Do not accept an evaluator result as
dense-enabled unless this preflight passes in the same environment.

## Build it

```bash
pip install -r requirements-dense.txt
python -m tools.fetch_model                     # once; writes models/sentence-transformers__all-MiniLM-L6-v2
python -m retrieval.build_dense_index \
    --catalog data/catalog.jsonl \
    --artifact-dir artifacts/dense \
    --verify-load
```

`tools.fetch_model` is the only step that touches the network, and it is a
development and image-build step. `retrieval.build_dense_index` sets
`HF_HUB_OFFLINE` and loads with `local_files_only=True`, so a missing or
corrupted model directory is a clear `ModelUnavailable` error rather than a
silent download.

## What the artifact contains

```text
artifacts/dense/
├── manifest.json   catalog/model/index checksums, dimensions, build config, metrics
├── ids.json        checksummed row index → parent_asin map
└── qdrant/         checksummed embedded Local Mode storage, not a service
```

## Determinism

A rebuild from the same catalog, model, and configuration produces bit-identical
vectors. Three choices make that true, and each is worth keeping:

- **Fixed-length padding.** Every text is padded to `max_sequence_length`
  (256) rather than to the longest item in its batch, so a product's vector does
  not depend on which products shared its batch. This is the main cost driver in
  the table below and the main reason a rebuild is reproducible.
- **Sorted catalog order.** Products are keyed by `parent_asin` and sorted, so a
  re-exported catalog with different line order rebuilds to the same index.
- **Normalized product text.** `retrieval.product_text` sorts dictionary keys and
  collapses whitespace, so a row that differs only in key order or formatting
  embeds identically.

Thread count was measured and does not affect results: builds at 1, 4, 8, and 16
threads produce the same `embedding_checksum`. `tests/test_dense_index.py` asserts
determinism across batch size, thread count, and catalog row order.

Determinism is asserted on the embedding matrix (`embedding_checksum`), not on
the bytes of the `qdrant/` directory, because embedded Qdrant is free to lay its
storage out differently between otherwise identical builds.

## Mismatch detection

`DenseIndex` verifies the manifest before it serves anything, so an incompatible
artifact is a startup failure rather than a quiet quality regression mid-session.
It raises `DenseIndexMismatch` naming every problem it found when:

- the catalog file has changed (checked by `file_sha256` first, falling back to
  `content_sha256` so a merely reformatted catalog is not rejected);
- the embedding model directory, downloaded revision, or weight bytes have changed
  or are absent;
- the identifier map or persisted Qdrant directory fails its recorded checksum;
- the Qdrant collection's vector dimensions or point count disagree with the
  manifest, or the identifier count disagrees with the catalog product count;
- `artifact_version` is not the version this build of the agent reads;
- the manifest is missing entirely.

Builds are created in a sibling staging directory. The published artifact is not
mutated until the staged index, identifier map, checksums, and manifest are all
complete. Publication swaps the complete directory into place and retains the
previous artifact until that succeeds, so a failed rebuild cannot leave an old
manifest describing partially replaced storage. These integrity fields require
artifact version 2; rebuild version 1 artifacts before loading them.

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It produces 384-dimensional,
L2-normalized vectors using attention-mask-aware mean pooling. MiniLM is selected
because it has higher HitRate@10 in the candidate comparison, embeds passages
about 1.9 times as fast, answers queries about 2.1 times as fast, and packages
about 32% smaller than BGE. BGE's higher MRR is recorded below rather than hidden.
`--model-identity` and `--model-dir` keep the choice configurable, and the
manifest records identity, revision, and a fingerprint covering the weights,
fetch metadata, configuration, and tokenizer files.

MiniLM uses the same unprefixed representation for passages and queries. The
empty query prefix and `mean` pooling choice are recorded in the manifest so the
live Retrieval Route in Slice 05 cannot drift from the index build convention.

### Candidate comparison

Measured in separate fresh processes on macOS arm64 with Python 3.11, 8 Torch
threads, batch size 32, and a maximum sequence length of 256. The deterministic
proxy contains all 200 public-set Target Products plus 1,800 catalog distractors
selected with seed `20260829`. Queries come from the public evaluator's initial
message generator. BGE uses its query prefix and CLS pooling; MiniLM uses no
prefix and attention-mask-aware mean pooling. The benchmark is reproducible with
`python -m tools.benchmark_embedding_candidate` and never downloads at runtime.

| Candidate | HitRate@10 | MRR | Passages/s | Query mean | Peak RSS | Model directory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BAAI/bge-small-en-v1.5 (`5c38ec7c…`) | 0.520 | **0.297381** | 36.37 | 31.09 ms | 881 MiB | 128 MiB |
| sentence-transformers/all-MiniLM-L6-v2 (`1110a243…`) | **0.545** | 0.282442 | **69.55** | **15.06 ms** | **868 MiB** | **87 MiB** |

This proxy is selection evidence, not a substitute for the official full-catalog
route and fusion evaluation. It deliberately reports the mixed result rather
than claiming one candidate wins every quality and resource measure.

## Measured scale

Measured on macOS arm64, Python 3.11, 8 Torch threads, batch size 64, no GPU.
The artifact uses schema version 2 and the pinned MiniLM revision above.

| Measure | 50,000 products |
| --- | ---: |
| Build time (total) | 669.8 s (11.2 min) |
| — of which embedding | 629.2 s (10.5 min) |
| Embedding throughput | 79.47 products/s |
| Artifact size | 198 MB (207,603,436 bytes) |
| Load time | 1.89 s |
| Peak process memory (build) | 1.96 GB |
| Peak process memory (load) | 1.25 GB |
| Dense search, depth 100 | 36.9 ms median, 39.9 ms p95 |

Load time and load memory are measured in a **fresh process**. Do not read them
off a `--verify-load` run: `peak_rss_bytes` is a process-lifetime peak, so a
verify that follows a build in the same process reports the build's peak, not
the load's.

## Qdrant Local Mode at 50,000 points

Qdrant emits a warning on every build and load:

> Local mode is not recommended for collections with more than 20,000 points.
> Current collection contains 50000 points.

ADR 0004 requires Local Mode, and the frozen catalog is 50,000 products, so this
warning is aimed at exactly our configuration and should not be dismissed
silently. It was measured rather than assumed:

- Dense search at depth 100 runs in 36.9 ms median, 39.9 ms p95 over 100
  post-warmup searches. A ten-turn session spends under half a second in this route.
- Loading costs 1.89 s and 1.25 GB, paid once at startup, never per turn.

Both are acceptable, so Local Mode stands and no hosted vector service is
introduced. Local Mode brute-forces the scan, so search cost grows linearly with
catalog size: these figures hold for a 50,000-product catalog and would need
re-measuring if the catalog grew substantially. The 1.25 GB resident cost is the
figure to carry into the Docker sizing work in Slice 17.

Built from the organizer's frozen `data/catalog.jsonl`: 50,000 products,
sha256 `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`, all
200 `data/public_set.jsonl` ground-truth targets present.

The dominant cost is fixed-length padding to 256 tokens, which is what buys
reproducibility. The complete MiniLM build takes 11.2 minutes on the measured
machine and is paid once, offline; no evaluation turn pays any part of
it. If build time ever becomes the constraint, `--max-sequence-length` is the
knob, and changing it invalidates the artifact by design.
