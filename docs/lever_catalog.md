# Honest Improvement Lever Catalog

Date: 2026-08-31. Synthesized from (a) per-session failure mining over the
checked benchmark reports and (b) a technique survey of current conversational
product-search practice. Every lever must satisfy the anti-reward-hacking
rules in `docs/honest_optimizations.md`: general principle, generalization
argument, runtime inputs limited to user meaning + catalog, and an adversarial
test. Baselines: local-only dev160 0.739152; paired dev60 local 0.7222 vs
gpt-5.4-mini-ranked 0.7569.

## What the failure data says

- 37% of hits land at rank 4–10 (conditional MRR 0.578) — **reordering the
  existing pool is the largest single lever** (+0.116 TS if every hit reached
  rank 1; +0.017 even guaranteeing top-3).
- 13/160 dev misses: ~5 sit in crowds of 29–155 near-identical products
  (soft honest ceiling ≈ 0.95–0.97 HR@10); ~6–8 are recoverable query-
  formulation failures (slogan/novelty titles with ≤1 near-substitute that an
  exact-phrase match would isolate).
- 26 hits convert only at turn ≥8 (mostly intent_override/buying) — worth
  ≈ +0.018 TS if converted by turn 4; partially capped by the override turn.
- Paraphrase condition costs −0.054 TS vs exact — lexical over-reliance
  (BM25 weight 0.64) is the robustness gap.
- The new stack broke exactly 1 of 160 old sessions while fixing 43 — the
  honest additions are near-strictly additive.
- The LLM ranker's 11 paired losses are all rank slips on sessions already
  hitting top-3 locally — variance, not systematic harm.

## Prioritized levers

| # | Lever | Expected TS | Cost | Status |
|---|---|---:|---|---|
| 1 | Local+LLM rank blending (variance guard) | +0.01 over LLM | ~1h | shipped (0.4 local / 0.6 LLM) |
| 2 | Exact-phrase n-gram route | +0.02–0.03 | ~2h | **rejected: measured 0.714 vs 0.739** — appended phrase hits enlarge the rerank pool and the cross-encoder promotes noisy matches over the target |
| 3 | Constraint/budget-aware final boost | +0.01–0.02 MRR-side | ~2h | **rejected: batch with it measured 0.715** — generic soft values ("cotton") mass-promote near-duplicates past the target |
| 4 | LLM conversational query rewrite | +0.02–0.04 | ~3h | deferred (post-deadline) |
| 5 | bge-small-en-v1.5 embedding swap + index rebuild | +0.01–0.03 | ~3h | deferred (post-deadline) |
| 6 | Fusion weight retrain on new trajectories | +0.005–0.02 | ~2h | deferred (post-deadline) |
| 7 | Belief-weighted (entropy) question selection | MTTC −0.3–0.6 turns | ~4h | deferred (post-deadline) |
| 8 | Listwise prompt hardening (full permutation, best-first input) | small MRR | ~1h | shipped |

Negative results are retained deliberately: both rejections were measured on
the full development split and reverted the same hour
(`benchmarks/honest_levers_dev160.json`, `benchmarks/honest_phrase_only_dev160.json`).

Rejected by measurement or survey: deeper rerank pool (depth 80 measured
0.705 < 0.739), reranker model swap (ms-marco-MiniLM-L-6 is already the CPU
latency frontier), RRF (tuned weights beat it), HyDE and doc2query (cost out
of hackathon scope).

## Lever documentation

**1. Rank blending.** Principle: combine two imperfect rankers by rank fusion
instead of letting the later one overwrite the earlier. Generalizes: standard
ensemble ranking. Inputs: the local ordering and the LLM ordering only.
Adversarial test: paired per-session comparison must show fewer regressions
with no wording dependence.

**2. Exact-phrase route.** Principle: distinctive multi-word phrases a
customer quotes ("live love lacrosse") are high-precision retrieval signals;
match them as FTS5 phrase queries over titles/features alongside the bag-of-
words route. Generalizes: phrase matching is core web/e-commerce search.
Inputs: customer message n-grams + catalog text. Adversarial test: phrases
come only from customer wording; paraphrase condition must not collapse
(bag-of-words routes still carry it), novel targets benefit equally.

**3. Constraint-aware boost.** Principle: products satisfying more of the
customer's stated attributes and price range should outrank equal-text-match
products. Generalizes: faceted relevance. Inputs: constraint state, budget
range, catalog fields. Adversarial test: unit tests on synthetic catalogs;
gains must appear across scenarios, not one template.

**4. Query rewrite.** Principle: restate the accumulated dialog as one
standalone search query (conversational query rewriting, strong published
gains on session-search benchmarks). Generalizes: trained on nothing local.
Inputs: the dialog itself. Adversarial test: by construction wording-robust;
paraphrase condition should *improve* relative to lexical accumulation.

**5. Embedding swap.** Principle: stronger sentence embeddings retrieve
better; bge-small-en-v1.5 outperforms all-MiniLM-L6-v2 at the same size.
Inputs: catalog + queries. Adversarial test: same suite, all three conditions.

**6. Fusion retrain.** Principle: route weights should match the current
query distribution, which dialog accumulation changed. Uses the existing
replayable Slice 9/10 tooling on regenerated trajectories. Adversarial test:
fold validation as in Slice 11, plus the robustness conditions.

**7. Belief-weighted asks.** Principle: weight candidates by retrieval score
when computing a question's expected information gain (expected entropy
reduction over a soft belief, not worst-case elimination). Inputs: pool
scores + catalog value index. Adversarial test: unit tests; MTTC must drop
across scenarios without wording dependence.
