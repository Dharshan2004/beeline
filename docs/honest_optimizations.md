# Honest Optimization Log

Date: 2026-08-31

Reward-hacking prevention is the top acceptance criterion for this work. Every
optimization below documents the general shopping principle it implements, why
it should work outside this evaluator, exactly what it consumes at runtime,
and an adversarial test that would expose benchmark coupling. The evaluator,
public labels, and scoring are untouched; production code contains no
evaluator imports and no reconstruction of the simulator's hidden intent-card
or message-generation logic.

The removed simulator-aligned verbatim-evidence route (see
`docs/ENGINEERING_JOURNAL.md`) is the counter-example these rules exist to
prevent: it scored 0.9628 by mirroring evaluator templates and collapsed under
any paraphrase. Nothing below may depend on exact wording.

## A. Conversational evidence accumulation

- **Principle:** a customer's need is the accumulation of everything they have
  said in the session, minus what they have explicitly retracted. Retrieval
  should condition on the whole dialog, not only the latest message.
- **Why it generalizes:** in any conversational commerce setting, disclosures
  arrive incrementally ("for hiking", then "waterproof", then "under $100");
  a system that forgets earlier turns cannot converge. This is standard
  conversational query expansion / context carry-over.
- **Runtime inputs:** the customer's own prior messages for the session
  (newest-first, budget-capped for the embedding model) and the validated
  Constraint State. Terms belonging to superseded constraints are excluded, so
  Intent Overrides do not drag replaced requirements back in.
- **Implementation:** `Agent._prior_dialog_text` feeds prior messages into the
  dense query (`Agent._dense_query`) and, term-filtered, into the BM25 route
  (`CatalogRetrieval.hybrid_route_scores(dialog_text=...)`).
- **Adversarial test:** the `paraphrase` condition of
  `tools/robustness_eval.py` rewrites every message's scaffold wording; the
  accumulated-evidence gain must survive because it uses content words and
  embeddings, not phrasing. The `novel` condition checks unseen targets.

## B. Information-value clarification

- **Principle:** a useful clarification is one whose answer can rule products
  out (maximum expected information gain). Never re-ask what the customer has
  answered or dismissed.
- **Why it generalizes:** expected-value-of-information question selection is
  the standard approach for interactive product finding; it adapts to any
  catalog and any conversation state, and asking discriminative questions is
  better for real customers, not just for this simulator.
- **Runtime inputs:** the current fused Candidate Pool, the catalog's
  attribute-value membership index, active constraints, dismissed attributes,
  and the session's own previously asked attributes.
- **Implementation:** `Agent._next_ask_attribute` scores each askable
  attribute by the minimum number of pool products a definitive answer would
  eliminate (`pool − max value group`), tie-breaking by the stable question
  order; attributes without an indexed value split fall back to that order.
  Already-asked, active, and dismissed attributes are never re-asked.
- **Adversarial test:** the choice depends only on pool statistics, so any
  conversation (template or free text) with the same pool yields the same
  question. Unit tests cover selection, tie-breaks, and no-repeat behavior;
  the paraphrase condition confirms no wording dependence.

## C. Budget understanding

- **Principle:** price is a first-class shopping constraint; ordinary spending
  language ("under $50", "between 20 and 40 dollars", "around $60") should
  reorder results toward affordable items without hard-eliminating products
  whose catalog price is missing.
- **Why it generalizes:** every real storefront supports price facets;
  the parser handles common English money phrasing, requires an explicit money
  marker (so sizes and quantities never trigger), and widens approximate
  mentions to a tolerance band.
- **Runtime inputs:** the customer's messages and the catalog `price` field.
- **Implementation:** `starter.retrieval.parse_budget` plus a stable partition
  of the final ranking (`within_budget` first); the most recent spending
  statement supersedes earlier ones.
- **Adversarial test:** unit tests cover phrasing variants, currency-marker
  requirement, and range semantics; the parser has no knowledge of any
  evaluator wording ("budget around $X" is parsed by the same general rule as
  any "around $X").

## D. LLM semantic ranking (optional, connected)

- **Principle:** after multi-route retrieval, a language model reorders the
  top candidates against the accumulated dialog — the "Multi-Route Retrieval
  → LLM Semantic Ranking" pipeline the track describes. Retrieval recall is
  handled locally; the LLM contributes fine-grained semantic precision where
  the local cross-encoder plateaus.
- **Why it generalizes:** LLM reranking is a production-standard search stage;
  it reads meaning rather than wording, so it is inherently robust to
  paraphrase and novel targets.
- **Runtime inputs:** the customer's own messages (oldest first), the active
  constraint summary, and public catalog renderings (with price) of the top
  candidates. No session identifiers, no evaluator knowledge.
- **Implementation:** `starter/llm_ranker.py` (`OpenAISemanticRanker`) behind
  an explicit `Agent(semantic_ranker=...)` opt-in; strict JSON-schema output,
  `store=false`, per-call worst-case budget reservation against the shared
  development budget, no SDK auto-retries, and unconditional fail-open to the
  local ordering on timeout, malformed output, or budget exhaustion. Model
  and prices are versioned in `config/semantic_ranker.json`
  (`gpt-5.4-mini` as the low-latency default role, checked 2026-08-31).
- **Adversarial test:** offline unit tests (`tests/test_llm_ranker.py`) cover
  fail-open, permutation hygiene, and budget gating; the connected stage is
  measured with `tools/llm_rank_experiment.py` and must beat the local path on
  the development split to be retained, and its gain must persist under the
  paraphrase condition.
- **Measured result (paired, identical 60-session development subset):**
  local-only 0.722200 versus mini-ranked 0.756931 (+0.034731); per-session the
  LLM ordering was better on 19, worse on 11, and identical on 30. p95
  complete-turn latency rises from 0.72 s to 4.59 s (bounded by the 4 s
  timeout plus the local pipeline) with a 14% fail-open rate and roughly $1.1
  per 60 sessions at gpt-5.4-mini prices. Tuning notes: `minimal` reasoning
  effort is rejected by gpt-5.4-mini (every call failed open, proving the
  safety path); `low` effort with a 2000-token output ceiling, 12 candidates,
  and a 4 s timeout removed the truncation failures that a 300-token ceiling
  caused. The offline configuration remains the default; the connected
  configuration is an explicit opt-in.

## E. Second-wave levers (measured 2026-08-31 evening)

Run-to-run noise was quantified first: two identical-code dev-160 runs differ
by ±0.013 TechnicalScore (20/160 sessions flip on float-level ranking
jitter), so single-lever effects below ~0.02 are reported as within-noise.

**Shipped (batch measured 0.759252 dev-160 vs 0.739152 baseline; MRR
0.531→0.590 exceeds the noise band):**

- *Dual-query lexical evidence* — the BM25 route scores both the accumulated
  dialog query and a fresh latest-message query, keeping each product's
  better score (topic-shift recovery; standard selective-history practice in
  conversational search). Inputs: dialog + constraints.
- *Popularity-aware pool admission* — fused rank blends with a Bayesian
  popularity prior whose weight decays with active-constraint count
  (0.3/(1+n)). Grounded in our own failure data: every crowded-category miss
  target was the single most-reviewed listing in its clone crowd. The general
  principle is bestseller-prior cold ranking; inputs are catalog rating
  fields only, so it is target-agnostic and novel-target-robust by
  construction.
- *Popularity tie-breaking within rerank score bands* (top-10 only,
  HR-invariant by design).
- *Profile-hint personalization* — the anonymized preference tags join the
  dense query only while fewer than two constraints are active (spec-listed
  safe personalization; stated requirements always dominate).
- *Ask-order preference for open questions* — use_case moves ahead of
  single-token attributes in the fallback order; free-text answers feed
  retrieval more evidence per turn.

**Measured and rejected (kept in history, reverted from the scoring path):**

- Exact-phrase n-gram pool injection: 0.714 vs 0.739 — appended phrase hits
  displace targets at the reranker.
- Global soft-preference boost: batch 0.715 — generic values mass-promote
  near-duplicates.
- Compact cross-encoder query: HR −4pts — dialog context in the rerank query
  outweighs its truncation cost.
- Dialog reset on replacement mutations: regressed with the compact query;
  superseded-term filtering already handles overrides.
- Local/LLM rank blending (0.4/0.6): 0.736 vs 0.757 for the raw LLM order —
  variance is handled by the confidence-margin gate instead.
- Two-call LLM pipeline under a 3-second p95 budget: rewrite (nano, 1.2 s)
  plus rerank (mini, 1.6 s) produced 96% rerank timeouts and 0.746 — the
  latency budget and the LLM gains are incompatible; the connected
  configuration uses the single proven mini rerank at 4 s instead. The
  rewriter (`starter/llm_rewriter.py`) remains available and honest but is
  not part of the shipped configuration.

## Measurement protocol

Iteration decisions use the frozen 160-session development split. Final
reporting uses all 200 released public sessions plus the separate
`exact` / `paraphrase` / `novel` conditions of `tools/robustness_eval.py`.
A large jump in `exact` that does not survive `paraphrase` or `novel` is
treated as benchmark coupling and reverted.
