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
| 7 | #8 | Benchmark local rerankers | AFK | 6 | 30, 31 |
| 8 | #9 | Integrate the selected reranker | AFK | 7 | 13, 30 |
| 9 | #10 | Build the replayable fusion-training dataset | AFK | 8 | 29, 31, 42 |
| 10 | #11 | Train Fusion Policy weights | AFK | 9 | 26, 29, 42 |
| 11 | #12 | Validate and freeze the Fusion Policy | AFK | 10 | 27–31, 39, 51 |
| 12 | #13 | Add the validated LLM planning loop | AFK | 3, 6 | 32–34, 38, 47, 48 |
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

Fusion training is deliberately split into four stages: produce replayable training data, optimize constrained route weights, validate and freeze the selected policy without touching the holdout, and open the locked holdout only during the final human-reviewed release slice.
