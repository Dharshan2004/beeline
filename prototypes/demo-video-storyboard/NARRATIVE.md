# Demo narrative: engineering judgment over score at any cost

## Core claim

We built a fast, reproducible conversational shopping agent by separating probabilistic language interpretation from deterministic state correctness. We measured every major change, rejected improvements that violated runtime or robustness gates, and kept valid local behavior as the foundation even when optional semantic ranking is connected.

## Three-minute cut

### 0:00–0:12 — The challenge

**Visual:** “50,000 products. One hidden purchase. Ten turns maximum.”

**Voiceover:**

> Fifty thousand products, one hidden purchase, and only ten turns to find it.

### 0:12–0:30 — Where we started

**Visual:** Official baseline `0.1067`; a three-turn conversation where the first two disclosures fade away.

**Voiceover:**

> The official baseline treated every message like a new search. It forgot earlier disclosures, could not safely replace stale intent, and hit only twelve and a half percent of targets. Its TechnicalScore was 0.1067.

### 0:30–0:50 — Architectural insight

**Visual:** `Turn Plan → atomic validator → Constraint State → hybrid retrieval → rank`.

**Voiceover:**

> Our key insight was that language interpretation and state correctness are different responsibilities. A model—or the local interpreter—may propose meaning, but deterministic code validates the complete Turn Plan atomically. Either every transition commits against one state revision, or none of it does.

### 0:50–1:10 — How we made decisions

**Visual:** Depth-50 and depth-100 rerankers against the predeclared latency gates.

**Voiceover:**

> We did not select components by score alone. We wrote runtime gates before benchmarking. Depth one hundred scored slightly higher, but projected to more than 1,366 seconds for the complete run, so we rejected it. Depth fifty met both quality and latency constraints and became the frozen boundary.

### 1:10–1:28 — What actually moved the score

**Visual:** Development score moving from `0.5518` to `0.7392`, next to the three customer-facing improvements and `0.707 s` p95 latency.

**Voiceover:**

> The largest useful jump came from three general shopping improvements: remembering the conversation, asking the question with the highest information value, and understanding stated budgets. Together they moved the 160-session development score from 0.5518 to 0.7392 at 0.707 seconds p95.

### 1:28–2:08 — Live Intent Override demo

**Visual:** Actual captured conversation on the left; compact state revisions and candidate counts on the right.

The chosen session should show:

1. an open-ended need;
2. one information-value clarification;
3. accumulated evidence influencing retrieval;
4. a broad Product Intent replacement;
5. old Product Intent constraints retiring atomically;
6. an explicit Session Constraint surviving;
7. the target entering the final top ten.

**Voiceover:**

> Here the shopper begins with an ambiguous need. The agent asks the question that best divides the current candidate pool, then carries that answer into retrieval. When the shopper changes product category, the old Product Intent retires atomically while explicit session-level evidence survives. The new retrieval starts from valid state—not a pile of contradictory prompt text.

### 2:08–2:28 — Optional connected intelligence

**Visual:** Paired-60 Pareto chart: local `0.7222 / 0.72 s p95`; mini-ranked `0.7569 / 4.59 s p95`.

**Voiceover:**

> Optional semantic ranking improves the paired score from 0.7222 to 0.7569, with 4.59-second p95 latency. But it never owns correctness: a timeout, malformed response, or exhausted budget returns the valid local ordering unchanged.

### 2:28–2:52 — Evidence

**Visual:** Start with a short uncut evaluator capture, then transition to exact/paraphrased/novel bars.

**Voiceover:**

> One successful conversation is a demo. Robustness is the evidence. The shipped local system scores 0.7556 across all two hundred released sessions, 0.7250 when every message is paraphrased, and 0.7182 on sessions with targets absent from the public labels. All 218 tests pass.

### 2:52–3:00 — Close

**Visual:** `0.7556 local · full 200`, “High quality. Seconds per turn. Valid without the network.”

**Voiceover:**

> This is not the highest score at any cost. It is a fast, reproducible shopping agent whose correctness does not depend on the network.

## Evidence provenance

- Official weak baseline: `docs/baseline_results.json`
- Architecture and iteration history: `docs/ENGINEERING_JOURNAL.md`
- Atomic plan boundary: `docs/adr/0006-apply-turn-plans-atomically.md`
- Reranker decision gates: `docs/reranker_benchmark.md`
- Shopping-behavior optimizations and paired connected result: `docs/honest_optimizations.md`
- Robustness release gates: `docs/ENGINEERING_JOURNAL.md`, late 2026-08-31 entry

## Claims to avoid

- Do not compare against another team or quote an unverified competitor latency.
- Do not call the paired-60 connected score a full-200 result.
- Do not mention abandoned benchmark experiments; keep the narrative on the shipped system and reproducible evidence.
- Do not call the public 200 sessions an untouched holdout.
- Do not imply optional LLM ranking is required for valid output.
