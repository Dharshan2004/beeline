# Engineering Journal: Conversational Shopping Agent

This living document records the project's iterations, evidence, decisions, failures, and lessons. It is the source material for the final demo, presentation, technical report, judging Q&A, and engineering interviews.

Record decision rationale and observable evidence here, not private chain-of-thought. Claims should be reproducible from a commit, command, test, metric file, or linked issue.

## How to use this journal

Add an entry whenever the team changes behavior, learns something from an experiment, accepts an architectural trade-off, or discovers a limitation.

Every entry should answer:

1. What problem or hypothesis were we addressing?
2. What changed?
3. What evidence did we collect?
4. What did we learn?
5. What remains uncertain or risky?
6. What is the next decision or experiment?

Do not overwrite disappointing results. A credible progression is more persuasive than a perfect-looking retrospective.

## Executive narrative

The project began with a deterministic BM25 baseline that treated each customer message largely as an isolated query. The first iterations established a reliable offline Agent contract, then introduced session-local Constraint State so customer requirements could persist and affect recommendations. Intent Override and Boundary Response behavior exposed a deeper architectural requirement: semantic interpretation may be proposed by an LLM, but state transitions and retrieval semantics must remain deterministic, validated, atomic, and reproducible.

The current direction is therefore a validated hybrid Shopping Agent:

```text
Customer message
    -> connected LLM or deterministic fallback proposes a Turn Plan
    -> deterministic validation applies the complete plan atomically
    -> active Hard Constraints determine eligibility
    -> Soft Preferences add evidence and influence ordering without excluding
    -> structured, BM25, and dense routes produce independent candidates
    -> versioned fusion combines route evidence
    -> a local reranker orders the deepest candidate pool allowed by the latency budget
    -> catalog-valid recommendations are returned
```

This architecture aims to combine natural conversational understanding with evaluator-safe behavior when model output is malformed or the network is unavailable. The latest retrieval diagnosis also changed the optimization order: first keep the Target Product reachable in the rerank pool, then optimize its final rank.

## Scorecard

### Reproducible weak baseline

Source: `docs/baseline_results.json`

| Metric | Baseline |
| --- | ---: |
| Public sessions | 200 |
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| Mean turns to conversion | 9.81 |
| Efficiency | 0.119 |
| Technical Score | 0.10671 |

### Historical result after Slices 2 and 3

Command: `python3 -m evaluator.local_evaluator`

| Metric | Current | Absolute change | Relative result |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.530 | +0.405 | 4.24x baseline |
| MRR | 0.373433 | +0.305399 | 5.49x baseline |
| Mean turns to conversion | 7.02 | -2.79 turns | 28.4% fewer turns |
| Efficiency | 0.398 | +0.279 | 3.34x baseline |
| Technical Score | 0.45663 | +0.34992 | 4.28x baseline |

Scenario results:

| Scenario | Sessions | Hit Rate@10 | MRR | Mean turns |
| --- | ---: | ---: | ---: | ---: |
| Boundary | 10 | 0.600000 | 0.460000 | 7.200000 |
| Browsing | 80 | 0.637500 | 0.460620 | 6.387500 |
| Buying | 80 | 0.475000 | 0.314420 | 6.975000 |
| Intent Override | 30 | 0.366667 | 0.269444 | 8.766667 |

Interpretation:

- The combined system is substantially stronger than the weak baseline.
- The result does not isolate the causal contribution of Slice 2 versus Slice 3; a per-commit evaluation is required before making slice-specific performance claims.
- Intent Override is currently the weakest scenario and remains the highest-priority correctness and quality risk.
- At that point, Hit Rate@10 of 0.53 remained below the PRD's initial hybrid target of 0.60.

### Historical result after Engineering Iteration 4

Status: **Closed** on 2026-08-29 at commit `ff83564` (`feat: make constraint transitions atomic`).

Command: `python3 -m evaluator.local_evaluator`

| Metric | Current | Change from Slices 2 and 3 | Relative change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.560 | +0.030 | +5.66% |
| MRR | 0.383780 | +0.010347 | +2.77% |
| Mean turns to conversion | 6.725 | -0.295 turns | 4.20% fewer turns |
| Efficiency | 0.4275 | +0.0295 | +7.41% |
| Technical Score | 0.480634 | +0.024004 | +5.26% |

Scenario results:

| Scenario | Sessions | Hit Rate@10 | MRR | Mean turns |
| --- | ---: | ---: | ---: | ---: |
| Boundary | 10 | 0.600000 | 0.371111 | 6.900000 |
| Browsing | 80 | 0.650000 | 0.462703 | 6.300000 |
| Buying | 80 | 0.512500 | 0.332649 | 6.512500 |
| Intent Override | 30 | 0.433333 | 0.313889 | 8.366667 |

Evidence:

- All 45 unit and evaluator tests passed with `python3 -m unittest discover -v`.
- Intent Override improved most in Hit Rate@10, rising from 0.366667 to 0.433333.
- Buying Hit Rate@10 rose from 0.475000 to 0.512500.
- Boundary Hit Rate@10 remained 0.600000, while MRR declined from 0.460000 to 0.371111; this ranking regression remains an investigation target.
- This journal iteration is distinct from implementation-plan Slice 04 / GitHub issue #5, which is the versioned dense-index deliverable.

### Current verified three-route result

Source: `results.json` and `docs/fixed_hybrid_fusion.md`.

Environment: project `.venv` with the bundled MiniLM model and 50,000-product Qdrant Local Mode artifact available. This distinction matters because the dense route intentionally fails open when dependencies or assets are absent.

Command:

```bash
/usr/bin/time -p .venv/bin/python -m tools.evaluate_retrieval \
  --policy fixed \
  --output results.json
```

| Metric | Dense-disabled, tie-preserved | Dense-enabled | Absolute change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.560000 | 0.580000 | +0.020000 |
| MRR | 0.383780 | 0.385437 | +0.001657 |
| Mean turns to conversion | 6.725 | 6.545 | -0.180 turns |
| Efficiency | 0.427500 | 0.445500 | +0.018000 |
| Technical Score | 0.480634 | 0.494731 | +0.014097 |
| Evaluator wall time | 139.00 s | 248.45 s | +109.45 s |

Dense-enabled scenario results:

| Scenario | Sessions | Hit Rate@10 | MRR | Mean turns |
| --- | ---: | ---: | ---: | ---: |
| Boundary | 10 | 0.600000 | 0.475000 | 6.800000 |
| Browsing | 80 | 0.662500 | 0.467187 | 6.175000 |
| Buying | 80 | 0.550000 | 0.326577 | 6.200000 |
| Intent Override | 30 | 0.433333 | 0.294537 | 8.366667 |

Interpretation:

- All three intended routes are active and improve Hit Rate@10 and Technical Score over the dense-disabled fallback.
- The gain costs approximately 109 seconds over 200 sessions, leaving reranker latency as a first-class constraint rather than an afterthought.
- Intent Override remains the weakest scenario by Hit Rate@10 and mean turns.
- This is not an apples-to-apples dense-enabled comparison with historical Slice 05; that commit still needs to be run with the same environment and artifacts.

## Iteration history

### Iteration 0: weak BM25 baseline

**Goal:** Establish a reproducible starting point using the supplied local evaluator.

**System shape:** Stateless lexical retrieval with the required Agent response contract.

**Evidence:** Hit Rate@10 0.125, MRR 0.068034, MTTC 9.81, Technical Score 0.10671 on 200 public sessions.

**Lesson:** Lexical matching alone is insufficient for multi-turn shopping. It does not reliably preserve customer requirements, interpret overrides, or respect Boundary Responses.

### Iteration 1: guarantee valid offline turns

**Goal:** Ensure every evaluator turn remains schema-valid without model or network access.

**Change:** Established deterministic response construction, session isolation, catalog-valid recommendation normalization, and zero non-negative reported model usage for offline operation.

**Why it matters:** Optional intelligence cannot be allowed to break the official Agent contract. This became the safety floor for every later iteration.

### Iteration 2: apply customer constraints end to end

**Goal:** Preserve a supported customer requirement across turns and use it during retrieval.

**Change:** Added inspectable Constraint State with attribute, raw phrase, normalized value, hard-or-soft classification, source turn, confidence, and status. Hard Constraints filter incompatible products; Soft Preferences influence ranking.

**Evidence:** Unit and small-catalog evaluator tests cover provenance, persistence, hard filtering, soft ordering, unsupported values, and ranking improvement.

**Review finding:** Soft Preferences were also inserted into the FTS query. When enough soft-matching products exist, this can prevent otherwise relevant nonmatching products from entering the Candidate Pool. A Soft Preference must influence evidence and ordering, never eligibility.

**Lesson:** Correct state representation is not enough; retrieval must preserve the semantic distinction between hard and soft requirements.

### Iteration 3: Intent Overrides and Boundary Responses

**Goal:** Preserve history while allowing later customer instructions to replace or dismiss earlier requirements.

**Change:** Added superseded and dismissed statuses, repeated same-attribute replacement, dismissed-attribute tracking, and Clarification avoidance for dismissed attributes.

**Evidence:** Tests cover repeated category replacement, preference replacement, Boundary dismissal and reintroduction, and evaluator-level Intent Override and Boundary sessions.

**Review findings:**

- Broad overrides use recency and hard-or-soft heuristics because constraints have no explicit Product Intent association.
- A Boundary Response and a new requirement in the same message cannot both be applied because current turn handling chooses one branch.
- Constraint extraction returns only one Constraint, so mixed messages such as “I prefer blue, but must have cotton” lose information.
- Tests do not yet cover compound transitions, full obsolete Product Intent replacement, atomic rollback, or enough soft matches to expose Candidate Pool narrowing.

**Lesson:** These are state-contract problems, not merely language-understanding problems. An LLM may propose better interpretations, but it cannot make partial or ambiguous deterministic mutations safe.

### Iteration 4: architectural correction before LLM planning

**Status:** Closed on 2026-08-29. Implemented and merged to `main` at `ff83564`.

**Goal:** Establish a safe boundary for both future LLM plans and deterministic fallback plans.

**Decisions reached:**

1. A **Product Intent Constraint** applies only to the current Product Intent and is retired by a broad Intent Override.
2. A **Session Constraint** survives product changes only when the customer explicitly established cross-intent scope.
3. Broad replacement is an explicit Turn Plan transition; a different category mention does not automatically replace the Product Intent.
4. A Turn Plan is validated against an unchanged snapshot and committed atomically. One invalid transition rejects the complete plan.
5. Mutation order never resolves contradictions. Replacement, dismissal, and reintroduction are explicit semantic transitions.
6. Adding a Soft Preference may add or reorder candidates but must never remove a base candidate.
7. A Constraint may contain multiple values with an explicit any-values or all-values relationship.
8. Later same-attribute values extend or replace an existing Constraint only when the customer's language makes that relationship explicit.
9. Constraint State uses a monotonically increasing revision; stale or replayed Turn Plans are rejected without mutation.
10. Deterministic fallback may interpret the supported, high-confidence portions of a message, but the resulting Turn Plan still validates and commits as one unit.

**Documentation:** See `CONTEXT.md`, `docs/adr/0001-validated-hybrid-agent-control.md`, `docs/adr/0002-evaluator-aware-hybrid-retrieval.md`, and `docs/adr/0006-apply-turn-plans-atomically.md`.

### 2026-08-29 — Iteration 5: select and build the versioned dense index

**Related issue/commits:** GitHub issue #5; `3129219`, `0718b26`, and merge `895eac0`.

**Problem or hypothesis:** Literal retrieval misses paraphrases, but a dense route is only submission-safe if its model and index are reproducible, bundled, integrity-checked, and usable without a hosted service or runtime download.

**What we tried:**

- Benchmarked `BAAI/bge-small-en-v1.5` against `sentence-transformers/all-MiniLM-L6-v2` on a deterministic proxy containing all 200 public Target Products and 1,800 seeded distractors.
- Tested different batch sizes, catalog row orders, and Torch thread counts to find sources of embedding nondeterminism.
- Built and verified an embedded Qdrant Local Mode artifact for all 50,000 catalog products.
- Added manifest, model, catalog, identifier-map, and storage checksums plus staged publication so a failed rebuild cannot corrupt the last valid artifact.

**Evidence:**

| Candidate | HitRate@10 | MRR | Passages/s | Query mean | Peak RSS | Model size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE small | 0.520 | **0.297381** | 36.37 | 31.09 ms | 881 MiB | 128 MiB |
| MiniLM L6 v2 | **0.545** | 0.282442 | **69.55** | **15.06 ms** | **868 MiB** | **87 MiB** |

Full artifact measurements: 669.8-second build, 198 MB artifact, 1.89-second fresh-process load, 1.96 GB peak build memory, 1.25 GB peak load memory, and depth-100 dense search at 36.9 ms median / 39.9 ms p95.

**What failed or surprised us:** BGE had better MRR, so model selection was not a clean quality sweep. Dynamic batch padding also made vectors depend on batch composition; fixed padding to 256 tokens restored reproducibility at the cost of build speed. Qdrant warns that Local Mode is not recommended above 20,000 points, but measurements at 50,000 points remained acceptable for this evaluator.

**Decision and rationale:** Select MiniLM because it won HitRate@10, throughput, query latency, memory, and package size, while recording rather than hiding BGE's MRR advantage. Keep Qdrant embedded because measured query latency is acceptable and a hosted vector service is forbidden as a required scoring dependency.

**Known limitations:** The proxy is selection evidence, not the official end-to-end evaluator. Fixed-length padding makes the one-time build take about 11 minutes. The 1.25 GB load footprint must be carried into packaging work.

**Next experiment:** Exercise the artifact through the real Agent route and verify both paraphrase recovery and fail-open behavior.

### 2026-08-29 — Iteration 6: connect dense retrieval to the live Agent

**Related issue/commits:** GitHub issue #6; `61221dd` and `6720891`.

**Problem or hypothesis:** A good offline index has no value unless the official `Agent.reset` / `Agent.respond` path queries it safely and preserves valid output when dense dependencies or assets are unavailable.

**Change:** Added a depth-100 embedded dense Retrieval Route, catalog/constraint filtering, lexical backfill, load-once lifecycle, and inspectable route metrics. Missing packages, assets, manifest compatibility, model load, or query execution disable only the route.

**Evidence:** Agent-level tests cover ordered valid candidates, paraphrase recovery that literal matching misses, one-time loading, hard-constraint filtering, duplicate/unknown identifier removal, missing assets, incompatible assets, and query failure. A readiness preflight verifies `status == available`, one completed query, and a positive candidate count under the exact evaluator interpreter.

**What failed or surprised us:** The system Python environment did not contain `torch`, `transformers`, or `qdrant-client`, so the route correctly disabled itself and historical Slice 04 and Slice 05 evaluations were bit-identical. A schema-valid evaluation is therefore not proof that dense retrieval actually ran.

**Decision and rationale:** Dense remains fail-open for contract validity, but every scored dense claim must include the readiness preflight from the same interpreter. Use `.venv/bin/python`, not an ambiguous system `python3`, for dense benchmarks.

**Known limitations:** Slice 05 originally let dense ordering preempt lexical ordering rather than combine independent evidence. That motivated fixed hybrid fusion.

**Next experiment:** Fuse structured, BM25, and dense evidence transparently and compare with single-route baselines.

### 2026-08-29 — Iteration 7: introduce and tune the fixed hybrid baseline

**Related issue/commits:** GitHub issue #7; `5c2e258`, `403bdc1`, and `259f64d`.

**Problem or hypothesis:** Independent routes should complement one another, but fixed fusion first needs transparent normalization, deterministic missing-route behavior, reproducible baselines, and a Candidate Pool whose membership is not accidentally narrowed by broad structured evidence.

**What we tried:**

1. The first fixed weighting used `structured=0.40`, `bm25=0.30`, `dense=0.30`. It scored only 0.125 Hit Rate@10 and 0.109919 Technical Score and took 279.17 seconds. Broad, frequently tied structured evidence dominated useful lexical ranking.
2. Revised weights to `structured=0.15`, `bm25=0.55`, `dense=0.30`, with route depth 100 and fused depth 30.
3. Corrected normalization and constant/all-zero/missing-route semantics after review.
4. Added a separate evaluation wrapper for fixed fusion, fixed RRF, and each single route without modifying the official evaluator.
5. Preserved BM25/dense base-pool membership so structured evidence can add or reorder candidates but cannot evict already admitted base candidates.

**Evidence:** The initial 0.40/0.30/0.30 result and the revised Slice 05 comparison are preserved in `docs/fixed_hybrid_fusion.md`. Tests cover exact weighted order, missing and constant routes, hard eligibility, deterministic ties, single-route baselines, fixed RRF, and monotonic base Candidate Pool membership.

**What failed or surprised us:** A plausible-looking equal-ish weighting nearly returned the project to its weak baseline. Fixed fusion also initially regressed Hit Rate@10 from 0.560 to 0.530 in the dense-disabled environment, revealing that membership behavior matters at least as much as score arithmetic.

**Decision and rationale:** Keep `fixed-hybrid-v1` only as a transparent starting point. Learned fusion must be trained with scenario guardrails, and broad structured evidence must never receive arbitrary power to remove lexical or dense candidates.

**Known limitations:** The fused top-30 boundary was chosen before measuring its reachable-hit ceiling. The fixed weights are not an optimality claim.

**Next experiment:** Diagnose the 0.560 to 0.530 regression session by session before attempting learned fusion.

### 2026-08-29 — Iteration 8: repair structured-route tie truncation in fixed fusion

**Related issue/commit:** GitHub issue #7 (Slice 06 fixed hybrid fusion); fix applied on `main` after `259f64d`.

**Problem or hypothesis:** Slice 06 regressed against Slice 05 on the public set (Hit Rate@10 0.560 to 0.530, TechnicalScore 0.480634 to 0.456661) even though both runs used the same dense-disabled environment. Per-session differencing showed 7 sessions lost their hit entirely and 11 more got worse ranks.

**Change:** `CatalogRetrieval.hybrid_route_scores` no longer splits a tied structured score group when truncating to the route depth. Every product tying with the cutoff score is kept.

**Evidence:**

- Root cause instrumented on `public_0132` turn 1: 145 hard-eligible products tied at structured score 6.0; the route kept 100 by ASIN order; the Target Product (BM25 rank 3, fused 0.55 × 0.818 = 0.45) fell to fused rank 22 behind products that won a flat 0.15 structured bonus through the alphabetical cut. Every lost session showed the same signature.
- Regression tests: `test_structured_route_keeps_every_member_of_a_truncated_tie_group` and `test_structured_membership_is_invariant_to_asin_spelling` in `tests/test_retrieval.py`, written first and observed red before the fix.

| Metric | Slice 05 | Slice 06 before | Slice 06 after |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.560 | 0.530 | 0.560 |
| MRR | 0.383780 | 0.370536 | 0.383780 |
| MTTC | 6.725 | 6.975 | 6.725 |
| Technical Score | 0.480634 | 0.456661 | 0.480634 |

**Scenario effects:** All four scenarios returned exactly to their Slice 05 values; the fixed run is session-for-session identical to Slice 05 in the dense-disabled environment.

**What failed or surprised us:** Slice 04 (`895eac0`) and Slice 05 (`61221dd`) evaluate bit-identically in this environment because the dense route disables itself without `torch`/`qdrant-client`; any observed Slice 04 versus Slice 05 gap must come from a dense-enabled environment, where Slice 05 lets dense candidates preempt lexical ordering wholesale rather than fuse.

**Decision and rationale:** A route may truncate by score, never by identifier: route membership feeds fused evidence, so an ASIN-order cut made ranking arbitrary. Constant tied evidence now behaves as unbiased membership evidence for every product that earned it.

**Known limitations:** Structured evidence still saturates into large tie groups because `_soft_constraint_score` counts matched values; the route currently orders ties only through fusion with BM25. In the dense-disabled environment the fixed policy now exactly matches the Slice 05 lexical result rather than beating it; any fusion gain must be demonstrated with the dense route enabled.

**Next experiment:** Re-run the comparison with dense assets and dependencies installed to measure fixed fusion against Slice 05's dense-first ordering, and add a scenario-level regression guardrail before training Fusion Policy weights (Slices 10 and 11).

### 2026-08-29 — Iteration 9: run the intended dense-enabled three-route system

**Related issue/commit:** GitHub issue #7; `e589d0c`; evidence in `results.json` and `docs/fixed_hybrid_fusion.md`.

**Problem or hypothesis:** After repairing candidate membership, determine whether the intended three-route configuration actually improves quality when dense assets are available, and measure the operational cost.

**Change:** Installed `requirements-dense.txt` into `.venv`, loaded the bundled MiniLM model and 50,000-point Qdrant artifact, passed the dense readiness preflight, and ran the same fixed-policy evaluation across all 200 public sessions.

**Evidence:** Hit Rate@10 improved from 0.560 to 0.580, Technical Score from 0.480634 to 0.494731, and mean turns from 6.725 to 6.545. Wall time increased from 139.00 to 248.45 seconds. The complete test suite ran without dense-runtime skips.

**Scenario effects:** Browsing reached 0.6625 Hit Rate@10, Buying 0.55, Boundary 0.60, and Intent Override 0.433333. Intent Override did not improve in Hit Rate@10 over the atomic-state result and remained the slowest scenario at 8.366667 mean turns.

**What failed or surprised us:** Dense retrieval produced a real aggregate gain, but MRR improved only 0.001657 and runtime grew roughly 78%. More candidate-generation quality does not automatically become top-ten ranking quality.

**Decision and rationale:** Keep dense retrieval enabled for the next experiments, but make full-run latency a release metric and benchmark rerank depth explicitly.

**Known limitations:** This run still truncates the fused Candidate Pool to 30 before any cross-encoder, and it does not isolate dense-enabled Slice 05 against dense-enabled Slice 06.

**Next experiment:** Measure target reachability at each candidate boundary and distinguish retrieval recall from final-hit conversion.

### 2026-08-29 — Iteration 10: diagnose the candidate-pool ceiling and revise Slices 7–12

**Related issues:** #8, #9, #11, #12, and #13. Local plan: `docs/IMPLEMENTATION_PLAN.md`.

**Problem or hypothesis:** A reranker cannot recover a Target Product that fusion already removed. Determine whether the current fused top-30 or final ordering is the dominant ceiling, and identify the scenario with the largest reachable-target-to-hit loss.

**Evidence received from the candidate-boundary analysis:**

- The fused pool of 30 makes the Target Product reachable in approximately 0.77 of sessions.
- The deeper Candidate Pool of roughly 100–200 candidates makes it reachable in approximately 0.93 of sessions.
- Intent Override has the largest gap between post-override target recall and a valid post-override hit.
- The latest dense-enabled end-to-end run already costs 248.45 seconds for 200 sessions before adding a cross-encoder.

These reachability figures currently come from the latest analysis/plan review and still need a checked-in reproduction command and machine-readable artifact before they are treated as release claims.

**What we changed in the plan:**

1. Slice 7 now benchmarks both compact cross-encoders and practical deep-pool depths, reporting per-turn latency, full-run time, memory, package size, pool recall, final HitRate@10, MRR, and recall-to-hit conversion.
2. Slice 8 reranks the Deep Candidate Pool up to the Slice 7 frozen latency budget rather than hard-coding the fused top 30.
3. Slice 10 trains Fusion Policy weights to reward the Target Product entering the rerank pool before refining final ordering.
4. Slice 11 validates pool recall separately from post-rerank ranking, compares the fused-30 and full-union ceilings, and freezes rerank depth with the weights and models.
5. Slice 12 is narrowed to validated LLM planning for Intent Overrides and measures only post-override eligibility and conversion; evaluator hits before the override do not count.

**Lesson:** Ranking quality has two separate failure modes: the correct product may be absent from the pool, or it may be present but ordered poorly. Optimizing only the second problem leaves the 0.77 membership ceiling untouched.

**Decision and rationale:** Treat Slice 7 as a decision gate. Select the deepest rerank pool that fits the evaluator time limit, maximize pool inclusion at that depth, then optimize final ordering. Keep Intent Override as the dedicated planning scenario because its post-override conversion gap is largest and the evaluator resets conversion eligibility after the override.

**Known limitations:** A cross-encoder over 100–200 candidates per turn may exceed the evaluator budget. The 0.77 and 0.93 measurements need reproducible checked-in evidence, and the feasible depth may be smaller than the full union.

**Next experiment:** Produce cached per-turn Deep Candidate Pools and labels, reproduce recall@30 and full-pool recall, then benchmark cross-encoders at several candidate depths against the 248.45-second no-reranker baseline.

### 2026-08-30 — Iteration 11: carry validated planning through the retrieval experiments

**Related issues/commit:** #8–#13; validated planning commit `0f3130e` is already in `main` and in the ancestry of the Slice 7 and Slice 8 branches.

**Problem or hypothesis:** Slices 7–11 were originally planned as a retrieval sequence parallel to Slice 12. Because validated LLM planning landed first, every reranker benchmark, replay dataset, Fusion Policy experiment, and validation run now executes through the planning-aware Agent. Treating those experiments as stateless ranking work would allow pre-override recommendations, obsolete Product Intent Constraints, or a different set of selected Retrieval Routes to corrupt the measured result.

**Evidence:** The Agent validates and commits the Turn Plan before it chooses Retrieval Routes, constructs route scores, applies fusion, and returns recommendations. The official evaluator counts an Intent Override conversion only after the override is applied. The initial Slice 7 replay implementation cached the Target Product and ranking but not conversion eligibility, so its offline scorer could count a pre-override target that the evaluator correctly rejects.

**Cross-slice consequences:**

1. **Slice 7 — reranker decision gate.** Generate the benchmark trajectory with deterministic offline planning, record hit eligibility on every turn, and compute Intent Override HitRate@10, MRR, efficiency, and TechnicalScore only from eligible post-override recommendations. A reranker comparison is invalid if planning or route selection varies between candidate configurations.
2. **Slice 8 — live deep reranking.** Keep reranking downstream of the committed Turn Plan and Fusion Policy. After an Intent Override, the reranker must receive the new-intent Candidate Pool immediately; timeout or model failure must preserve that post-override fused ordering and may never restore obsolete constraints.
3. **Slice 9 — replayable fusion-training dataset.** Record the planning configuration, Turn Plan source, selected Retrieval Routes, Constraint State revision, per-turn conversion eligibility, exact depth-specific Candidate Pools, scenario type, and post-override query/target evidence. Generate the dataset with deterministic offline planning so repeated runs do not vary with provider output.
4. **Slice 10 — pool-aware Fusion Policy training.** Ignore pre-override target inclusion. For Intent Override sessions, optimize pool recall only after the new Product Intent commits, and report the scenario separately rather than allowing aggregate gains to hide a post-override regression.
5. **Slice 11 — validation and freeze.** Validate the complete versioned planning/retrieval trajectory: planner and fallback versions, selected tools, route limits, Fusion Policy, reranker identity, depth, and timeout. Apply scenario regression guardrails to post-override Intent Override metrics and keep the locked holdout unopened until Slice 18.

**Decision and rationale:** Slice 12 does not move behind the retrieval work and is not bypassed during benchmarking. Its validated planning boundary is now part of the experimental contract for Slices 7–11. Development reranker and fusion experiments use `planning_provider=None` to obtain deterministic local Turn Plans; connected-provider quality and cost remain the responsibility of Slices 14–15. This isolates retrieval decisions while preserving the same state-transition and conversion semantics used by the official Agent.

**Known limitations:** Slice 7 measures the local/offline Agent budget, not the eventual connected configuration's complete latency. Later connected-model and release gates must remeasure end-to-end latency with planning enabled. Existing candidate-boundary figures produced before eligibility-aware replay remain hypotheses until regenerated.

**Next experiment:** Repair Slice 7 replay eligibility and exact Candidate Pool tracing, run the 160-session development benchmark without opening the holdout, and preserve the selected configuration and machine-readable evidence before Slice 8 activates live reranking.

### 2026-08-30 — Iteration 12: freeze the local reranker at MiniLM-L6 depth 50

**Related issue/artifact:** #8; `docs/reranker_benchmark.json`.

**Problem or hypothesis:** Select the deepest three-route Candidate Pool that a compact CPU cross-encoder can rerank without sacrificing fused-30 quality or exceeding the predeclared runtime gates.

**Change:** Repaired route admission and strict caps, cached exact pools at depths 30/50/100/150/200/250/300 across all 160 development sessions, preserved post-override conversion eligibility, and benchmarked three immutable local cross-encoder revisions on the same 1,009 cached turns. Frozen split identifiers discarded the locked 40-session JSONL rows before their payloads were deserialized or inspected.

**Evidence:** The fused-30 baseline produced HitRate@10 0.543750, MRR 0.368237, TechnicalScore 0.467096, and a normalized 200-session wall projection of 239.8 seconds. `ms-marco-MiniLM-L-6-v2` at depth 50 produced HitRate@10 0.600000, MRR 0.376215, TechnicalScore 0.507240, p95 rerank latency 548.9 ms, and an 800.6-second projection. Its depth-100 row improved HitRate@10 to 0.618750 and TechnicalScore to 0.519015 but projected to 1,366.9 seconds, failing the 900-second gate. TinyBERT never improved TechnicalScore; MiniLM-L2 qualified only at depth 30.

**Decision and rationale:** Freeze `cross-encoder/ms-marco-MiniLM-L-6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` at Candidate Pool depth 50. It is the deepest configuration that passes both runtime gates and both quality floors. Depth 50 has 0.775 session pool recall versus the observed depth-300 ceiling of 0.900, so truncation gives up 0.125 reachability for runtime feasibility.

**Known limitation:** These quality figures are replay metrics on a fixed deterministic trajectory. Slice 8 must validate live end-to-end behavior and implement the persistent cancellable worker with an absolute 1.5-second deadline; Slice 7 does not activate reranking.

**Next experiment:** Integrate the frozen model and depth downstream of the committed Turn Plan and Fusion Policy, then run the official development evaluator with timeout and fail-open tests.

## Current architectural invariants

### State

- Constraint State is authoritative and isolated by `session_id`.
- Historical evidence is preserved; transitions change applicability rather than deleting records.
- Product Intent Constraints and Session Constraints have distinct lifetimes.
- One customer turn may propose multiple state transitions.
- The complete Turn Plan validates against one state revision and commits once.
- Invalid, stale, replayed, or contradictory plans change nothing.
- Connected and offline interpreters must use the same Turn Plan and validation boundary.
- A valid Turn Plan commits before Retrieval Routes, fusion, or reranking execute for that turn.
- Intent Override conversion eligibility begins only after the replacement Product Intent commits; pre-override recommendations never count as hits in live or replay evaluation.

### Retrieval

- Hard Constraints determine product eligibility.
- Soft Preferences contribute evidence and ordering but never narrow the base Candidate Pool.
- For fixed message evidence and active Hard Constraints, adding Soft Preferences must be monotonic with respect to candidate membership.
- Superseded or dismissed Product Intent Constraints do not affect retrieval.

### Reliability

- Model and network availability are optional for response validity.
- A malformed connected plan receives at most one bounded correction attempt before deterministic takeover.
- Fallback begins from the same unchanged state snapshot after a rejected model plan.
- Recommendations remain ordered, unique, catalog-valid, and bounded by `top_k`.
- Development retrieval experiments use deterministic offline planning unless connected planning is the variable explicitly under test.
- Replay evidence records enough planning and state identity to reproduce the route set, Candidate Pools, and conversion eligibility of every scored turn.

## Open questions and risks

- Final module boundary between turn interpretation, Constraint State validation, retrieval, and Agent orchestration.
- How the deterministic interpreter identifies high-confidence Product Intent replacement without overreaching.
- Whether Session Constraints should participate in every future Product Intent automatically or require attribute-specific compatibility checks.
- How Clarifications should behave when a same-attribute addition versus replacement remains ambiguous.
- Intent Override quality remains weak at 0.433333 Hit Rate@10 in the dense-enabled run; pre-override hits are irrelevant after the evaluator resets conversion eligibility.
- The fused top-30 pool appears to cap reachability near 0.77 versus approximately 0.93 for the Deep Candidate Pool; these figures need eligibility-aware checked-in reproduction evidence.
- Cross-encoding practical depths up to the 300-candidate three-route cap may exceed the evaluator time limit on top of the current 248.45-second dense-enabled run.
- Fusion training can overfit final ordering while failing to improve pool membership unless pool recall at the frozen rerank depth is an explicit objective.
- Public evaluator sessions are not sufficient evidence for robust compound-turn behavior. Adversarial tests are required.

## Next experiments and acceptance evidence

### Atomic Turn Plan

- Apply two valid additions in one turn.
- Apply dismissal of one attribute and addition of another in one turn.
- Include one valid and one invalid transition; assert zero state changes.
- Replay an already committed plan; assert no duplicate history.
- Submit a stale expected revision; assert no state change.
- Reintroduce a dismissed attribute explicitly and preserve dismissal history.
- Repeat Product Intent transitions A -> B -> C; assert only C is active and A/B remain inspectable.

### Product Intent grouping

- Replace shoes with slippers and retire all shoe-specific constraints.
- Preserve an explicitly Session-scoped budget across the replacement.
- Do not preserve an ordinary Product Intent color/material requirement unless explicitly restated.
- Mention another category additively; assert no unintended replacement.
- Submit an ambiguous category mention; preserve current state and Clarify.

### Hard and soft retrieval semantics

- Create more than `top_k` soft-matching products plus a base-relevant nonmatch; assert the nonmatch remains in the pre-ranking Candidate Pool.
- Assert a hard mismatch is excluded even with excellent lexical and soft evidence.
- Assert no soft matches still returns the base pool.
- Assert a soft match outranks an otherwise tied nonmatch.
- Property test: adding any Soft Preference never removes a base candidate.

### Connected-planner boundary

- Malformed schema.
- Unknown attribute or value.
- Unknown Retrieval Route or tool.
- Conflicting transitions.
- Timeout and provider unavailability.
- One invalid plan, one invalid retry, then deterministic fallback from unchanged state.

### Deep Candidate Pool and reranker budget

- Cache the per-turn structured/BM25/dense Route Candidate Sets, exact depth-specific Candidate Pools, target label, state revision, selected tools, and conversion eligibility.
- Reproduce aggregate and per-scenario pool recall at exact depths 30, 50, 100, 150, 200, 250, and 300.
- Benchmark at least two compact cross-encoders on identical cached pools.
- Measure the complete 160-session development run; report p50/p95 rerank latency, peak memory, package size, and a clearly labelled 200-session runtime projection without executing the locked holdout.
- Freeze the deepest pool that fits the evaluator time budget and record any recall ceiling sacrificed by truncation.
- Report pool recall, post-rerank HitRate@10, MRR, efficiency, TechnicalScore, and recall-to-hit conversion separately, using post-override-only eligibility for Intent Override sessions.

### Pool-aware fusion

- Train non-negative weights with target inclusion at the frozen rerank depth as the primary signal.
- Compare learned membership with fused-30, full-union, fixed RRF, and every single route.
- Use the official Technical Score and final rank metrics to refine policies within the accepted pool-recall range.
- Preserve per-scenario regression guardrails and compute Intent Override training and validation signals only after the replacement Product Intent commits.

### Post-override planning

- Ignore recommendations made before the evaluator applies the override.
- Measure whether the new target enters the pool immediately after override and whether reranking converts it to a valid top-ten hit.
- Test successful, ambiguous, rejected, and repeated A -> B -> C override plans through the same atomic validator.

## Demo storyline

Use one short conversation that visibly exercises the differentiator:

1. Customer requests a product with one Hard Constraint and one Soft Preference.
2. Agent recommends immediately while asking a useful Clarification.
3. Customer gives a compound response that dismisses one attribute and adds another requirement.
4. Customer performs a broad Intent Override to another product category.
5. Agent retires Product Intent Constraints, preserves an explicitly Session-scoped requirement, and immediately recommends against the new Product Intent.
6. Repeat the demo with network/model access disabled to show identical contract validity through deterministic fallback.

What to display:

- Customer-facing recommendations.
- A compact Constraint State timeline showing active, superseded, and dismissed entries.
- Product Intent and state revision changes.
- Retrieval route contributions without private chain-of-thought.
- Before-and-after ranking of the Target Product.
- Token usage, latency, fallback reason, and configuration version.

## Presentation outline

### 1. Problem

Shopping conversations are stateful. Customers refine, contradict, dismiss, and replace requirements; stateless keyword search cannot reliably follow those transitions.

### 2. Baseline evidence

Show the reproducible baseline metrics and one failure trace.

### 3. Architectural insight

Natural-language interpretation and state correctness are different responsibilities. The LLM proposes meaning; deterministic code owns validity, history, retrieval eligibility, and fallback.

### 4. System

Show the Turn Plan boundary, atomic validator, Constraint State, independent Retrieval Routes, Fusion Policy, and response contract.

### 5. Iteration evidence

Show how each slice addressed a concrete failure, including findings that caused architectural correction. Use the score progression only where commits were evaluated independently.

### 6. Reliability

Demonstrate malformed-plan rejection and network-disabled fallback.

### 7. Results and limitations

Report overall and per-scenario metrics. State clearly that Intent Override remains the weakest scenario and explain the next experiment.

### 8. Why this matters beyond the competition

The design demonstrates production-oriented AI engineering: typed model boundaries, deterministic invariants, graceful degradation, evaluation discipline, provenance, and honest measurement.

## Judge and interviewer Q&A bank

### Why use an LLM if deterministic fallback is required?

The LLM handles semantic interpretation and flexible language. Deterministic code handles invariants that must remain true regardless of model quality: atomic state transitions, catalog validity, hard-versus-soft eligibility, cost limits, and offline operation.

### Why not let the LLM directly update state?

Model output can be malformed, incomplete, contradictory, or stale. A typed Turn Plan and atomic validator make those failures observable and prevent partially applied customer intent.

### What is the main technical differentiator?

The LLM and deterministic fallback share one validated Turn Plan boundary. Connected intelligence improves interpretation without becoming the sole copy of state or a dependency for valid scoring.

### How do you handle Intent Overrides safely?

Constraints belong either to the current Product Intent or explicitly to the session. A broad replacement retires the complete old Product Intent while preserving history and carrying only explicit Session Constraints.

### Why can Soft Preferences not filter products?

A preference improves relevance but is not a requirement. Allowing it to determine eligibility violates its domain meaning and can hide the Target Product even when all Hard Constraints are satisfied.

### How do you know the system improved?

The current verified dense-enabled three-route system raised Hit Rate@10 from 0.125 to 0.58 and Technical Score from 0.10671 to 0.494731 on the same 200-session public evaluator. The journal preserves intermediate regressions and avoids attributing gains to an individual component without a controlled comparison.

### What is currently weakest?

Intent Override, at 0.433333 Hit Rate@10 and 8.366667 mean turns in the dense-enabled run. It also has the largest observed gap between a reachable post-override target and a qualifying hit, so Slice 12 is now focused specifically on that conversion problem.

### What happens when the model is unavailable?

The deterministic interpreter proposes a smaller, high-confidence Turn Plan, passes through the same validator, and runs local retrieval. The response contract remains valid with zero model-token usage.

### Are you overfitting to the public evaluator?

Evaluator-aware structured evidence is one transparent Retrieval Route, not the whole system. Independent sparse and dense semantic routes, scenario-level guardrails, frozen configuration, and a locked holdout are intended to limit narrow fitting.

### What engineering judgment does this project demonstrate?

The team separated probabilistic interpretation from deterministic correctness, kept failure paths first-class, used explicit domain language, tested at public seams, measured scenario regressions, and changed architecture when green unit tests failed to guarantee the intended semantics.

## Evidence checklist for final submission

- [ ] Record commit SHA and configuration version for every reported run.
- [ ] Save aggregate and per-scenario metrics.
- [ ] Record latency distribution and peak memory.
- [ ] Record connected model name, prompt version, token usage, and estimated cost.
- [ ] Preserve a network-disabled evaluation result.
- [ ] Preserve malformed-plan and atomic-rollback test output.
- [ ] Capture one multi-turn Intent Override trace for the demo.
- [ ] Compare each Retrieval Route and fixed fusion against the selected Fusion Policy.
- [ ] Record known limitations and failed experiments.
- [ ] Freeze configuration before opening the locked holdout.
- [ ] Record individual team contributions as work is completed.

### 2026-08-31 — Iteration 13: revisable Session Mode and useful Clarifications

**Owner(s):** dylothx

**Related issue/commit:** GitHub issue #14 (Slice 13)

**Problem or hypothesis:** A fixed question order repeated attributes that had
already produced no information, and the agent did not expose or revise its view
of whether a customer was buying, browsing, or uncertain.

**Change:** Added versioned session policy `shopping-session-policy-v1` and
planning contract `shopping-turn-planner-v3`. Session Mode is revised and logged
on every turn. Clarifications are limited to allowed catalog-supported attributes
with multiple possible answers, excluding active, dismissed, and already asked
attributes for the current Product Intent. Safe aggregate profile tags can reorder
only eligible questions and never become Constraints. Ranked recommendations
remain present whenever a Clarification is returned.

**Commands and environment:** Python 3.12 on Windows; `python -m compileall -q
starter tests`; `python -m unittest -v tests.test_session_policy
tests.test_planning tests.test_agent tests.test_turn_interpreter`; `python -m
unittest discover -s tests -p "test_*.py"`.

**Evidence:** The focused contract suite ran 72 tests successfully with two
optional-model skips. The full suite ran 185 tests and retained the same six
baseline environment/data errors observed before Slice 13: five Unix-only
`multiprocessing` `fork` tests on Windows and one frozen public-set checksum
mismatch. Seven new Slice 13 tests cover mode transitions, recommendation-bearing
Clarifications, repeated-question rejection, Boundary behavior, all four scenario
fixtures, and profile/current-turn precedence.

**Scenario effects:** Buying favors direct narrowing attributes; Browsing favors
broad use-case exploration; Intent Override starts a new asked-attribute history
for the successor Product Intent; Boundary dismissals remain excluded across
Product Intent changes.

**What failed or surprised us:** The existing full suite is not fully portable to
Windows because its local reranker worker tests request the unavailable `fork`
start method. The checked-in public set also does not match the frozen split
checksum in this workspace; neither condition was introduced by Slice 13.

**Decision and rationale:** Keep mode outside Constraint State revisioning so a
mode-only turn remains observable without pretending that authoritative customer
constraints changed. Use deterministic usefulness validation around connected
Clarifications, following the same bounded retry and fallback behavior as other
invalid plan fields.

**Known limitations:** Catalog diversity is an inexpensive global expected-value
proxy; it does not yet measure entropy over the exact per-turn Candidate Pool.

**Next experiment:** Slice 14 should compare connected models on mode accuracy,
Clarification acceptance/retry rate, latency, token use, and downstream conversion.

## Team contribution log

Add entries continuously; do not reconstruct this section at submission time.

| Date | Person | Contribution | Evidence | Outcome |
| --- | --- | --- | --- | --- |
| YYYY-MM-DD | Name | Concise technical contribution | Commit, issue, test, or document | Observable result |
| 2026-08-31 | dylothx | Implemented Slice 13 Session Mode and Clarifications | Issue #14; `tests.test_session_policy` | Seven deterministic acceptance tests pass |

## Iteration entry template

Copy this block for every meaningful iteration:

```markdown
### YYYY-MM-DD — Iteration N: concise title

**Owner(s):**

**Related issue/commit:**

**Problem or hypothesis:**

**Change:**

**Expected result:**

**Commands and environment:**

**Evidence:**

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | | | |
| MRR | | | |
| MTTC | | | |
| Technical Score | | | |

**Scenario effects:**

**What failed or surprised us:**

**Decision and rationale:**

**Known limitations:**

**Next experiment:**
```

## Maintenance rule

Update this journal in the same pull request as each meaningful behavioral change. A change is not presentation-ready until its hypothesis, evidence, limitation, and next step are recorded here.
