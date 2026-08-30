# Implementation Plan: LLM-Centered Hybrid Shopping Agent

Parent specification: GitHub issue #1, **PRD: LLM-Centered Hybrid Shopping Agent**.

The build is divided into thin, demonstrable vertical slices. Slices 1–17 are suitable for autonomous implementation. Slice 18 requires human review because it opens the locked holdout once and finalizes competition claims and team contributions.

| Slice | GitHub | Deliverable | Type | Blocked by | PRD stories |
| ---: | ---: | --- | --- | --- | --- |
| 1 | #2 | Guarantee a valid offline turn | AFK | None | 3, 16–24 |
| 2 | #3 | Apply one customer constraint | AFK | 1 | 1, 4, 7 |
| 3 | #4 | Handle overrides and Boundary Responses | AFK | 2 | 5, 6, 11, 12, 20 |
| 4 | #5 | Build a versioned dense index | AFK | 1 | 30, 40, 41 |
| 5 | #6 | Use embedded dense retrieval | AFK | 4 | 2, 23–25 |
| 6 | #7 | Introduce fixed hybrid fusion | AFK | 2, 5 | 13, 25, 27 |
| 7 | #8 | Benchmark rerankers and feasible deep-pool depths | AFK | 6 | 30, 31 |
| 8 | #9 | Rerank the selected deep candidate pool | AFK | 7 | 13, 30 |
| 9 | #10 | Build the replayable fusion-training dataset | AFK | 8 | 29, 31, 42 |
| 10 | #11 | Train Fusion Policy weights for pool recall | AFK | 9 | 26, 29, 42 |
| 11 | #12 | Validate and freeze pool-aware fusion | AFK | 10 | 27–31, 39, 51 |
| 12 | #13 | Add validated LLM planning for Intent Overrides | AFK | 3, 6 | 5, 6, 12, 20, 32–34, 38, 47, 48 |
| 13 | #14 | Add Session Mode and Clarifications | AFK | 3, 12 | 3, 8–11, 15 |
| 14 | #15 | Benchmark connected OpenAI models | AFK | 12, 13 | 35, 36 |
| 15 | #16 | Add conditional second calls and budget enforcement | AFK | 8, 13, 14 | 35–37 |
| 16 | #17 | Trace complete sessions with Langfuse | AFK | 12 | 43–46 |
| 17 | #18 | Package and reproduce the complete agent | AFK | 11, 15, 16 | 23, 24, 39–41, 48, 52 |
| 18 | #19 | Run the locked holdout and prepare submission | HITL | 17 | 28–31, 49–52 |

## Dependency map

```text
1 ─┬─→ 2 → 3 ───────────────┐
   └─→ 4 → 5 → 6 ─┬─→ 7 → 8 ├─→ 9 → 10 → 11 ─┐
                   └─→ 12 ───┼─→ 13 → 14 → 15 ├─→ 17 → 18
                             └─→ 16 ───────────┘
```

Slice 7 is a decision gate: it selects both the local reranker and the deepest three-route-union pool that fits the evaluator's runtime limits. Each Retrieval Route has a strict deterministic cap of 100 candidates, so the deduplicated union is bounded at 300; exact pools are benchmarked independently at depths 30, 50, 100, 150, 200, 250, and 300, with actual union sizes reported. Runtime evidence measures the complete 160-session development run and labels a normalized 200-session figure as a projection; the locked 40 sessions are not executed even for timing before Slice 18. A candidate configuration is admissible only when that projection is at most 900 seconds and measured p95 added rerank latency is at most 1.5 seconds per turn. Configurations that regress HitRate@10 or do not improve replay TechnicalScore over fused-30 are rejected; Slice 7 then freezes the greatest remaining depth and selects the highest-TechnicalScore model at that depth, with MRR, p95 latency, and packaged size as tie-breakers. If no declared model qualifies, one predeclared smaller-model contingency round uses the same frozen gate; if that also fails, Slice 7 records fused-30 with no local cross-encoder and the limits are not relaxed. Slice 8 reranks the selected deep pool rather than a pre-truncated fused top 30, but it is not activated when Slice 7 selects no reranker. Cross-encoder inference runs in one persistent cancellable local worker: a 1.5-second timeout terminates the worker, returns the fused ordering, and disables reranking for the rest of the evaluator run, while startup failure, crash, or malformed output follows the same fallback. Fusion training is deliberately split into four stages: produce replayable training data, optimize constrained route weights for target inclusion at the frozen rerank depth before final ordering, validate and freeze the selected policy without touching the holdout, and open the locked holdout only during the final human-reviewed release slice. Slice 12 has already established validated planning for post-override behavior, and Slices 7–11 preserve that planning and conversion contract throughout their experiments.

Measured Slice 7 outcome: freeze `cross-encoder/ms-marco-MiniLM-L-6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` at depth 50. It is the deepest quality-improving configuration below both runtime gates; depth 100 exceeds the 900-second normalized wall projection.

## Planning-aware retrieval experiments

Validated planning from Slice 12 is already present on `main` and therefore forms part of the Agent used by Slices 7–11. It does not change their dependency order, but it changes their evidence contract:

- Slice 7 uses deterministic offline planning and records per-turn conversion eligibility so pre-override recommendations never count.
- Slice 8 reranks only after the Turn Plan commits and must preserve the new-intent fused ordering on every fallback.
- Slice 9 records planning configuration, selected Retrieval Routes, state revision, conversion eligibility, and exact depth-specific Candidate Pools in its replay dataset.
- Slice 10 trains Intent Override pool-recall signals only from eligible post-override turns.
- Slice 11 freezes and validates planner, route, fusion, reranker, depth, and timeout identities together, with post-override scenario guardrails.

Connected-provider behavior is not a variable in Slices 7–11: their datasets and comparisons use `planning_provider=None`. Slices 14–15 separately measure connected planning quality, latency, token use, and cost, and the final release gate remeasures the complete connected configuration.

## Slice 12 repair contract

The validated planning boundary uses one versioned deterministic Explicit Replacement Evidence classifier in both the offline interpreter and connected-plan validator. Attribute-level correction replaces only one unambiguous Constraint and preserves the rest of the Product Intent. Product Intent replacement requires product-type replacement language or explicit withdrawal of the whole prior intent, plus a distinct supported successor in the same atomic Turn Plan. Ambiguous mentions are rejected, retried once through the existing bounded planning loop, and then handled by deterministic takeover without partial state change.

The model-selectable tool contract contains only Candidate Pool-producing Retrieval Routes: structured, BM25, and dense. Local Reranking remains the frozen mandatory post-fusion policy when available and cannot be selected, disabled, or bypassed by a Turn Plan. Slice 12 is provider-neutral and uses deterministic fake providers for connected success, ambiguity, repeated replacement, rejection, retry, and offline-takeover tests; the concrete OpenAI adapter remains part of Slice 14's measured model comparison.

These corrections create planning contract v2 and a separately identified replacement-evidence classifier. They invalidate the development-only Slice 9 replay artifact and the dependent Slice 10–11 reports. Regeneration may activate v2 only when the complete artifact chain passes its existing provenance, scenario guardrail, runtime, and regression gates; otherwise the v1 frozen build remains active. The locked holdout is not opened during regeneration.
