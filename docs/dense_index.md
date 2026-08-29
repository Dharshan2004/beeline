# Dense Index Artifact

The dense Retrieval Route searches a versioned artifact that is built ahead of
time and loaded once at agent startup. No turn pays indexing cost, no model is
downloaded on the scoring path, and no vector service has to be running.

Slice 04 delivers the artifact, its manifest, and its verification. Wiring the
route into live recommendations is Slice 05 (issue #6).

## Build it

```bash
pip install -r requirements-dense.txt
python -m tools.fetch_model                     # once; writes models/BAAI__bge-small-en-v1.5
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
├── manifest.json   catalog checksums, model identity, dimensions, build config, metrics
├── ids.json        row index → parent_asin, in sorted parent_asin order
└── qdrant/         embedded Qdrant Local Mode storage (a directory, not a service)
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
- the embedding model directory has changed or is absent;
- the recorded vector dimensions disagree with the artifact;
- `artifact_version` is not the version this build of the agent reads;
- the manifest is missing entirely, which is also what an interrupted build
  leaves behind — the manifest is written last, on purpose.

## Embedding model

`BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
384 dimensions, CLS pooling, L2-normalized, 127 MB of weights. Chosen over
`all-MiniLM-L6-v2` for retrieval quality at the same dimensionality and a
comparable footprint; HitRate@10 is half the TechnicalScore, and the size
difference (127 MB against roughly 90 MB) is small against the image budget.
`--model-identity` and `--model-dir` keep the choice configurable, and the
manifest records identity, revision, and a fingerprint of the config and
tokenizer files.

BGE is asymmetric: passages are embedded bare, queries carry the prefix
`Represent this sentence for searching relevant passages: `. The prefix is
recorded in the manifest so the live Retrieval Route in Slice 05 cannot drift
from the convention the index was built under.

## Measured scale

Measured on Windows 11, Python 3.11, 16-core CPU, 8 torch threads, batch size 64,
no GPU.

| Measure | 50,000 products |
| --- | ---: |
| Build time (total) | 2,955.8 s (49.3 min) |
| — of which embedding | 2,464.7 s (41.1 min) |
| Embedding throughput | 20.3 products/s |
| Artifact size | 198 MB (207,603,323 bytes) |
| Load time | 5.0 s |
| Peak process memory (build) | 2.45 GB |
| Peak process memory (load) | 1.22 GB |
| Dense search, depth 100 | 72 ms median, 87 ms p95 |

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

- Dense search at depth 100 runs in 72 ms median, 87 ms p95. A ten-turn session
  spends under a second in this route.
- Loading costs 5.0 s and 1.22 GB, paid once at startup, never per turn.

Both are acceptable, so Local Mode stands and no hosted vector service is
introduced. Local Mode brute-forces the scan, so search cost grows linearly with
catalog size: these figures hold for a 50,000-product catalog and would need
re-measuring if the catalog grew substantially. The 1.22 GB resident cost is the
figure to carry into the Docker sizing work in Slice 17.

Built from the organizer's frozen `data/catalog.jsonl`: 50,000 products,
sha256 `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`, all
200 `data/public_set.jsonl` ground-truth targets present.

The dominant cost is fixed-length padding to 256 tokens, which is what buys
reproducibility. It is paid once, offline; no evaluation turn pays any part of
it. If build time ever becomes the constraint, `--max-sequence-length` is the
knob, and changing it invalidates the artifact by design.
