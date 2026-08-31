# Benchmark Target Findings

Date: 2026-08-31

## Conclusion

The initial hybrid retrieval target has been achieved on the 160-session
development split. The full submission targets remain possible but ambitious,
especially at the frozen Candidate Pool depth of 50. Further progress must come
from improving pool-to-hit conversion, final ranking, and conversation
efficiency rather than from minor Fusion Policy weight adjustments alone.

## Target comparison

The PRD defines an initial HitRate@10 target of at least 0.60 and submission
targets of at least 0.80 HitRate@10, at least 0.65 MRR, mean turns to conversion
no greater than 4, and zero uncaught exceptions or invalid responses.

| Measure | Initial target | Submission target | Current development result | Status |
|---|---:|---:|---:|---|
| HitRate@10 | 0.60 | 0.80 | 0.656250 | Initial target achieved |
| MRR | Not separately set | 0.65 | 0.406937 | Submission target not yet achieved |
| Mean turns to conversion | Not separately set | 4.00 or lower | 5.918750 | Submission target not yet achieved |
| Uncaught exceptions or invalid responses | 0 | 0 | 0 | Achieved in the measured run |

The current frozen policy uses weights `structured=0.02`, `bm25=0.64`, and
`dense=0.34`, followed by the frozen local reranker at Candidate Pool depth 50.
It records a TechnicalScore of 0.551831 and pool recall of 0.825 on the same
development split.

## Feasibility analysis

### HitRate@10

The initial 0.60 HitRate@10 target is exceeded by 0.056250, or 5.625 percentage
points. The 0.80 submission target requires another 0.143750, equivalent to 23
additional successful sessions out of 160.

At depth 50, pool recall is 0.825. Because a target cannot become a top-ten hit
when it is absent from the Candidate Pool, 0.825 is the observed upper bound on
HitRate@10 for this development trajectory. Reaching 0.80 would therefore
require converting approximately 96.97% of pool-reachable sessions into hits:

```text
required conversion = 0.80 / 0.825 = 0.9697
current conversion  = 0.65625 / 0.825 = 0.7955
```

This leaves only 0.025 absolute HitRate headroom below the observed pool ceiling.
The target is possible in principle, but high-risk under the current depth-50
configuration. Improving Candidate Pool recall or conversion quality would make
the target materially safer.

### MRR

The current MRR of 0.406937 is 0.243063 below the 0.65 submission target. The
current MRR divided by HitRate@10 is approximately 0.620, indicating that the
successful sessions already tend to rank the target reasonably highly. Reaching
0.65 MRR alongside 0.80 HitRate@10 would require a conditional reciprocal rank
of approximately 0.8125, meaning that a large majority of successful targets
would need to appear at rank 1 or 2.

This target is unlikely to be reached through small Fusion Policy changes. It
requires stronger final ordering, better queries and intent interpretation, or
both.

### Mean turns to conversion

Mean turns to conversion must improve from 5.918750 to 4.00 or lower, a reduction
of approximately 1.92 turns. This is primarily a conversational efficiency
problem. Session-mode selection, higher-value clarifications, connected planning,
and conditional model calls in later slices are the most plausible sources of
improvement.

## Recommended focus

1. Preserve the frozen Slice 11 configuration as the reproducible retrieval
   baseline rather than repeatedly tuning nearby weights.
2. Use Slice 13 to improve clarification value and reduce unnecessary turns.
3. Use Slice 14 to measure whether connected planning improves Intent Override
   handling, pool inclusion, and query quality enough to justify its latency and
   cost.
4. Use Slice 15 to apply connected calls only where they are likely to improve
   conversion or rank.
5. Diagnose the development sessions where the target is in the depth-50 pool
   but does not become a top-ten hit; these are the most direct opportunities for
   closing the 0.168750 pool-to-hit gap.
6. Treat a deeper Candidate Pool as an explicit quality-versus-runtime decision,
   not as an automatic change.

## Evidence limitations

These figures come from the deterministic 160-session development split used by
Slices 8 through 11. They demonstrate that the initial target is achieved and
support planning the next experiments, but they are not untouched holdout or
private-evaluator evidence. Final claims must be based on a frozen release
configuration evaluated without further tuning.

Primary evidence:

- `docs/PRD.md`
- `docs/fusion_policy_validation.md`
- `docs/fusion_policy_live_evaluation.json`
- `docs/fusion_policy_freeze.json`
