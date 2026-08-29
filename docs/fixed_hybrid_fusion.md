# Fixed Hybrid Fusion

Slice 06 combines three independent Retrieval Routes on every default Shopping
Agent turn:

- `structured` scores exact active Constraint matches and preserves eligible
  backfill candidates so Soft Preferences cannot narrow the Candidate Pool;
- `bm25` scores the current message plus active Constraint evidence through the
  embedded SQLite FTS5 index; and
- `dense` uses the versioned local MiniLM/Qdrant route from Slice 05.

Hard Constraints are applied as an eligibility rule to every route before
fusion. Unknown catalog identifiers and duplicate dense identifiers are removed
at the same boundary.

## Fixed policy

The default policy is `fixed-hybrid-v1`:

| Setting | Value |
| --- | ---: |
| Structured weight | 0.15 |
| BM25 weight | 0.55 |
| Dense weight | 0.30 |
| Candidate depth per route | 100 |
| Fused Candidate Pool depth | 30 |
| Evaluator recommendation depth | 10 |

For each turn, finite scores within each route are min-max normalized. A
non-empty constant-score route assigns `1.0` to all of its candidates; an empty
route contributes nothing. Missing candidates receive no contribution from that
route. Weighted scores are summed and ties are resolved by `parent_asin`, making
the ordering deterministic for fixed route outputs.

These weights are a transparent fixed starting point, not a learned or tuned
claim. Later slices train and validate the Fusion Policy on the allowed
development split.

## Reproducible baselines

The evaluator uses the default fixed policy when no switch is supplied:

```bash
python3 -m tools.evaluate_retrieval --policy fixed
```

The same command and output schema can run fixed Reciprocal Rank Fusion or any
single route:

```bash
python3 -m tools.evaluate_retrieval --policy rrf
python3 -m tools.evaluate_retrieval --policy structured
python3 -m tools.evaluate_retrieval --policy bm25
python3 -m tools.evaluate_retrieval --policy dense
```

Each result records the selected baseline plus the policy version, route depth,
fused depth, and weights where applicable. Dense asset failure remains fail-open:
the fixed policy deterministically fuses the remaining routes and still returns
a valid Agent response.

## Slice 05 comparison

Both runs used the same 200-session public dataset, catalog, Python 3.11 process,
and evaluator scoring code. The Slice 05 baseline was captured at commit
`61221dd`; Slice 06 used `fixed-hybrid-v1`.

| Metric | Slice 05 | Slice 06 | Absolute change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.560000 | 0.530000 | -0.030000 |
| MRR | 0.383780 | 0.370480 | -0.013300 |
| Mean turns to conversion | 6.725 | 6.975 | +0.250 |
| Efficiency | 0.427500 | 0.402500 | -0.025000 |
| TechnicalScore | 0.480634 | 0.456644 | -0.023990 |
| Evaluator wall time | 139.00 s | 163.70 s | +24.70 s |

Scenario Hit Rate@10 changed from `0.600000` to `0.600000` for Boundary,
`0.650000` to `0.625000` for Browsing, `0.512500` to `0.475000` for Buying,
and `0.433333` to `0.400000` for Intent Override.

This is an honest regression, not evidence that hybrid retrieval is weaker in
the intended packaged configuration. `torch`, `transformers`, and
`qdrant-client` were unavailable to the benchmark interpreter, so the dense
route disabled itself in both runs. The comparison therefore measures Slice 05's
lexical fallback against Slice 06's structured/BM25 missing-route fusion. The
first experimental `0.40/0.30/0.30` weighting was substantially worse
(Hit Rate@10 `0.125`, TechnicalScore `0.109919`, 279.17 s); reducing broad
structured dominance and indexing exact-value eligibility produced the reported
result. No claim is made that these fixed weights are optimal. Learned policy
selection and regression guardrails remain work for Slices 10 and 11.

Reproduction commands:

```bash
/usr/bin/time -p python3 -m tools.evaluate_retrieval \
  --policy fixed \
  --output results.json
python3 -m unittest discover -v
```
