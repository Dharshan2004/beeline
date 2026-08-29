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
non-empty constant nonzero route assigns `1.0` to all candidates, while an
all-zero route keeps `0.0` because it carries no positive evidence. An empty
route contributes nothing. Missing candidates receive no contribution from that
route.

The structured route truncates to the candidate depth by descending score but
never splits a tied score group: every product tying with the score at the
cutoff is kept, so route membership cannot depend on ASIN ordering. On the
public catalog, exact-match evidence commonly ties across more than 100
products; cutting a tie group alphabetically silently dropped eligible
products, including evaluator Target Products.

BM25 and dense candidates establish stable base membership in the 30-product
Candidate Pool. Structured evidence can reorder those products and add exact
matches when fewer than 30 base candidates exist, but cannot evict an admitted
base product. Weighted-score ties are resolved by `parent_asin`, making the
ordering deterministic for fixed route outputs.

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
| MRR | 0.383780 | 0.370536 | -0.013244 |
| Mean turns to conversion | 6.725 | 6.975 | +0.250 |
| Efficiency | 0.427500 | 0.402500 | -0.025000 |
| TechnicalScore | 0.480634 | 0.456661 | -0.023973 |
| Evaluator wall time | 139.00 s | 164.62 s | +25.62 s |

Scenario Hit Rate@10 changed from `0.600000` to `0.600000` for Boundary,
`0.650000` to `0.625000` for Browsing, `0.512500` to `0.475000` for Buying,
and `0.433333` to `0.400000` for Intent Override.

The regression above was diagnosed and repaired. Its cause was structured-route
truncation splitting tied score groups: for example, on `public_0132` turn 1,
145 hard-eligible products tied at structured score 6.0, the route kept 100 by
ASIN order, and the Target Product (BM25 rank 3) received no structured
contribution while 100 alphabetically earlier products each received the flat
0.15 weight, pushing the target to fused rank 22. With tie groups preserved,
the same benchmark environment recovers the Slice 05 result exactly,
session for session:

| Metric | Slice 05 | Slice 06 (tie split) | Slice 06 (tie preserved) |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.560000 | 0.530000 | 0.560000 |
| MRR | 0.383780 | 0.370536 | 0.383780 |
| Mean turns to conversion | 6.725 | 6.975 | 6.725 |
| TechnicalScore | 0.480634 | 0.456661 | 0.480634 |

The original tie-splitting regression is preserved below for the record.

This was an honest regression, not evidence that hybrid retrieval is weaker in
the intended packaged configuration. `torch`, `transformers`, and
`qdrant-client` were unavailable to the benchmark interpreter, so the dense
route disabled itself in both runs. The comparison therefore measures Slice 05's
lexical fallback against Slice 06's structured/BM25 missing-route fusion. The
first experimental `0.40/0.30/0.30` weighting was substantially worse
(Hit Rate@10 `0.125`, TechnicalScore `0.109919`, 279.17 s); reducing broad
structured dominance and indexing exact-value eligibility produced the reported
result. No claim is made that these fixed weights are optimal. Learned policy
selection and regression guardrails remain work for Slices 10 and 11.

## Dense-enabled validation

After installing `requirements-dense.txt` into the project `.venv`, the same
fixed-policy evaluation loaded the 50,000-point embedded Qdrant collection and
queried it on every evaluator turn. The Agent reported the route as available,
with 100 candidates returned by a smoke query, and the complete test suite ran
without dense-runtime skips.

| Metric | Tie-preserved fallback | Dense-enabled Slice 06 | Absolute change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.560000 | 0.580000 | +0.020000 |
| MRR | 0.383780 | 0.385437 | +0.001657 |
| Mean turns to conversion | 6.725 | 6.545 | -0.180 |
| Efficiency | 0.427500 | 0.445500 | +0.018000 |
| TechnicalScore | 0.480634 | 0.494731 | +0.014097 |
| Evaluator wall time | 139.00 s | 248.45 s | +109.45 s |

This establishes that the intended three-route path is active and improves the
fallback result. It is not an apples-to-apples dense-enabled comparison with
the Slice 05 commit; that historical comparison requires running `61221dd`
with the same environment and assets.

Reproduction commands:

```bash
/usr/bin/time -p .venv/bin/python -m tools.evaluate_retrieval \
  --policy fixed \
  --output results.json
.venv/bin/python -m unittest discover -v
```
