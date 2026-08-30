# Fusion Policy Weight Training

Slice 10 trains one global set of non-negative route weights that sums to one.
It does not change the live Agent yet. Slice 11 must validate the selected
region across scenario folds before the configuration is frozen and activated.

## Reproduction

Regenerate the ignored Slice 9 artifact first, then run:

```bash
.venv/bin/python -m tools.train_fusion_policy \
  benchmarks/fusion_training.jsonl \
  --output docs/fusion_policy_training.json
```

The command loads the artifact through its strict checksum and identity
validator. It performs no retrieval, embedding, or reranker inference. Every
candidate pool is reconstructed from cached normalized route scores and every
Top-K is reconstructed from cached cross-encoder scores.

The checked-in report is deterministic. Its SHA-256 is
`f62a554d3eaad9e263714416cbe76da4671179751ce9e961cee5a84a9683ccfe` and
it binds the search to artifact
`712378b277e1bb3fa7543828f101312725dab727f0bf1c59756ed2d253c4cd31`.
The locked holdout is not loaded.

## Search and selection

The coarse search evaluates all 66 points on the three-route simplex at 0.10
increments. The local search evaluates 91 points at 0.02 increments within 0.10
per route of the coarse winner. Both stages use Candidate Pool depth 50.

Selection uses a predeclared zero tolerance (exact pool-recall ties only) and is
lexicographic: session-level pool recall first, then official
TechnicalScore, HitRate@10, MRR, turn-level pool recall, distance from the
current weights, and a deterministic weight tuple. This prevents a high final
rank score from hiding a weak retrieval pool.

## Result

The selected development candidate is:

| Route | Current | Selected |
| --- | ---: | ---: |
| Structured | 0.15 | 0.00 |
| BM25 | 0.55 | 0.68 |
| Dense | 0.30 | 0.32 |

At depth 50, the selected candidate improves session pool recall from 0.7750
to 0.8250, HitRate@10 from 0.6375 to 0.65625, MRR from 0.405035 to
0.407999, and TechnicalScore from 0.539511 to 0.552275. Its recall-to-hit
conversion is 0.795455.

The current-weight fused-30/no-reranker control reaches 0.7250 pool recall,
0.54375 HitRate@10, and 0.467096 TechnicalScore. The selected weights at depth
30 reach 0.7375 pool recall and 0.539881
TechnicalScore. The complete route union reaches 0.9000 pool recall but falls
to 0.529107 TechnicalScore, showing that deeper pools create more reranker
confusion and are not automatically better. The full union contains 51–299
products per turn, with mean 232.20815.

Scenario-level HitRate@10 is unchanged for Boundary (0.625), Browsing
(0.78125), and Buying (0.59375), while Intent Override improves from 0.375 to
0.500. The selected candidate still needs Slice 11 fold validation: its zero
structured weight lies on the simplex boundary, and Browsing MRR decreases from
0.482688 to 0.461886 even though aggregate metrics improve.

The report also retains depth-50 single-route and fixed-RRF controls. BM25 alone
is the strongest control at 0.7875 pool recall and 0.547608 TechnicalScore; RRF,
dense-only, and structured-only are all weaker than the selected weighted
candidate.
