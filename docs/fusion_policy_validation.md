# Fusion Policy Validation and Freeze

Slice 11 validates the Slice 10 candidate region, freezes the complete scoring
configuration, and activates `pool-aware-global-v2`. The Locked Holdout remains
unread; all selection and live verification use the 160 allowed development
sessions.

## Reproduction

With the ignored Slice 9 artifact present, reproduce fold validation and the
freeze record with:

```bash
.venv/bin/python -m tools.validate_fusion_policy \
  benchmarks/fusion_training.jsonl \
  --output docs/fusion_policy_freeze.json
```

Reproduce the official live Agent/evaluator benchmark with:

```bash
.venv/bin/python -m tools.evaluate_live_reranker \
  --output docs/fusion_policy_live_evaluation.json
```

The freeze report SHA-256 is
`89af5ea0ea1359d233f0982fc6775b3cb65d7fc44f9a736353b005b97ded3db1`.
It binds the training report, fusion artifact and identities, live runtime
report, catalog, dense index, embedding model, reranker, planner prompt, and
all selected depths and weights. The live report SHA-256 is
`bee8bf6d36d24d6979e07e1b14f76596688c8d4b4a3bbe7a3346d2fcd4a3e4fc`.

## Fold design and gates

The 160 development sessions are deterministically assigned by scenario and
session ID to four disjoint folds. Every fold contains exactly 40 sessions:
2 Boundary, 16 Browsing, 16 Buying, and 6 Intent Override. Candidate evaluation
uses only cached route and reranker scores.

The validator examines all 91 Slice 10 local-grid points. A point is admissible
only when it:

- stays at the best observed 0.825 session pool recall (zero tolerance);
- improves the current depth-50 TechnicalScore;
- loses no more than five percentage points of HitRate@10 in any scenario; and
- belongs to a plateau with another admissible point one simplex-grid move away.

Nine candidates are admissible. Stable selection prioritizes local plateau
support, then worst-fold pool recall, lower fold variance, worst-fold
TechnicalScore, and full-development ranking quality.

## Frozen result

| Route | Slice 9 baseline | Slice 10 edge optimum | Frozen plateau point |
| --- | ---: | ---: | ---: |
| Structured | 0.15 | 0.00 | 0.02 |
| BM25 | 0.55 | 0.68 | 0.64 |
| Dense | 0.30 | 0.32 | 0.34 |

The frozen point has five adjacent admissible neighbors. Across folds, session
pool recall is 0.750, 0.825, 0.850, and 0.875 (mean 0.825; population standard
deviation 0.046771). Its worst-fold TechnicalScore is 0.472530 and mean is
0.551831.

On all development sessions it records:

| Metric | Current depth 50 | Frozen policy |
| --- | ---: | ---: |
| Session pool recall | 0.775000 | 0.825000 |
| HitRate@10 | 0.637500 | 0.656250 |
| MRR | 0.405035 | 0.406937 |
| TechnicalScore | 0.539511 | 0.551831 |
| Recall-to-hit conversion | 0.822581 | 0.795455 |

The policy closes 57.1429% of the measured fused-30-to-full-union pool-recall
gap (0.725 → 0.825 toward the 0.900 ceiling), and 40% of the current-depth-50
gap. Boundary, Browsing, and Buying HitRate@10 do not regress; Intent Override
improves by 12.5 percentage points.

Intent Override remains the clearest conversion opportunity: pool recall is
0.666667, HitRate@10 is 0.500000, and recall-to-hit conversion is 0.75, leaving
a 0.166667 reachability-to-hit gap.

## Live activation evidence

The complete live run exactly reproduces cached quality metrics and pool recall.
It completes 892 reranked turns with no failure or fallback. Rerank p95 latency
is 0.348351 seconds and the normalized 200-session wall projection is 568.298
seconds, below the fixed 1.5-second and 900-second gates.

The frozen configuration records route depth 100, rerank depth 50, per-route
min-max normalization, the MiniLM embedding and reranker revisions and artifact
checksums, catalog and Qdrant checksums, the 1.5-second timeout, and planner
planning contract version plus prompt/schema checksum
`3006accb659a14cb5feaf6fa6e03b13e9044bd7ffade16e0a73fc663554c35bd`,
and Explicit Replacement Evidence classifier version plus checksum
`b2fb42142bd8cf5e8fd7a20881a5fd8d89f0c915c5109566c74037f76fc95d4d`.
