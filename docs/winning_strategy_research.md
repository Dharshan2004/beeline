# Winning-Strategy Research: TechJam 2026 Shopping Copilot

Research date: 2026-08-31

Scope: what strategy maximizes the chance of winning the TechJam 2026
"Shopping Copilot: AI Conversational Search and Recommendations" challenge,
given (a) how the automated TechnicalScore is actually computed, (b) the
current state of this repository, (c) the state of the art for each required
pillar, and (d) the fact that 65% of judging is human. Every claim is cited
with a URL (external) or `file:line` (this repo). Where evidence is thin the
claim is explicitly marked as speculation.

## Executive summary — recommended strategy

1. **The private set is mechanically identical to the public set.** The
   organizer's final-evaluation FAQ states the 800 private sessions use "the
   same input schema, Agent interface, metric formula, stopping rule,
   invalid-output handling, deterministic customer-message templates, and
   `ask_attribute` response policy" and that "no undisclosed natural-language
   paraphrases are introduced"
   ([final_evaluation_faq.md](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/final_evaluation_faq.md)).
   Only targets and users differ. Generalization risk is therefore about
   *unseen products*, not unseen wording.
2. **The 0.9628 "exact-evidence" route was withdrawn as reward hacking and is
   not in the current build.** The current working tree is the honest
   conversational build at TechnicalScore 0.739 on the 160-session dev split
   (`benchmarks/honest_abc_dev160.json`; `docs/ENGINEERING_JOURNAL.md:751-786`).
   Re-activating the evidence route would very likely reproduce ~0.96 on the
   private set, but it re-derives the organizer's hidden intent cards, and
   "private-label reconstruction" is explicitly out of scope
   (`docs/competition_specification.md:13`). Recommendation: do **not**
   re-ship it as the scoring path; keep it (documented, disabled) as an
   ablation exhibit that demonstrates evaluator insight.
3. **Always recommend AND always ask, every turn.** The interface allows both
   simultaneously, and the simulator discloses new constraints *only* when
   `ask_attribute` is set (`evaluator/local_evaluator.py:316-335`). A null ask
   wastes a turn with zero information gain (line 320-321). There is no
   ask-vs-show tradeoff in this challenge; the classic tradeoff literature
   applies only to *which* attribute to ask.
4. **MRR is the weak axis (0.531) and the highest-leverage metric work.**
   HitRate@10 is already 0.919 on dev; each MRR point is worth 0.3× in
   TechnicalScore (`evaluator/local_evaluator.py:474-475`). Focus on final
   ordering: better rerank text rendering, a stronger local cross-encoder, or
   an LLM listwise pass on the final top-10 (RankGPT-style,
   [arXiv:2304.09542](https://arxiv.org/abs/2304.09542)).
5. **Run and check in the robustness evidence — it is currently missing.**
   The journal and README claim results are "recorded in
   `benchmarks/robustness_final.json`" (`docs/ENGINEERING_JOURNAL.md:732-734`),
   but no `robustness*.json` exists in `benchmarks/`. This is the single
   cheapest credibility fix: run `tools/robustness_eval.py` (exact /
   paraphrase / novel conditions) and check in the report.
6. **Self-imposed latency gates can be relaxed for the final run.** Teams run
   the final evaluator themselves; "there is no standardized
   organizer-provided CPU, RAM, GPU … or per-response limit"
   ([final_evaluation_faq.md](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/final_evaluation_faq.md)).
   The 1.5 s rerank deadline and 900 s wall gate are internal choices — but
   note the depth-80 experiment *hurt* quality (0.705 vs 0.739,
   `docs/ENGINEERING_JOURNAL.md:723-726`), so relax gates only where measured
   gains exist.
7. **Keep connected LLM planning out of the scoring path unless it is
   re-benchmarked to a win.** The measured `gpt-5.6-luna` connected run scored
   0.161 vs 0.671 offline because the model returned `ask_attribute: null` on
   90/169 turns, starving disclosure (`docs/ENGINEERING_JOURNAL.md:833-844`).
   Any LLM in the loop must be constrained to always emit an ask.
8. **Spend the freed effort on the 65% human-judged criteria.** Innovation
   (20%), Impact (20%), Feasibility (15%), Presentation (10%) are not scored
   by the evaluator. The repo's evidence discipline (journal, ADRs, ablations,
   fail-open paths, reward-hacking gate) is itself a differentiator — package
   it: architecture diagram, live demo with state timeline, offline-fallback
   demo, honest limitations slide.
9. **Hedge the unseen-product risk with the novel-target condition.** The
   `novel` condition of `tools/robustness_eval.py:100-121` generates sessions
   whose targets never appear in public labels — this is the closest local
   proxy for the private set and should gate every release.
10. **Protect the floor:** zero exceptions, zero invalid responses, valid
    output with no network and no model (already achieved:
    `benchmarks/honest_abc_dev160.json` shows 0 exceptions, 0 tokens). A
    session that exceeds 10 turns or throws scores as a miss
    (`evaluator/local_evaluator.py:417-451`; `docs/competition_specification.md:66`).

---

## 1. What the evaluator actually scores (primary source)

The participant kit is public:
[github.com/TechJam2026/techjam-conversational-search](https://github.com/TechJam2026/techjam-conversational-search)
(participant-kit release). The local copy of the evaluator is
`evaluator/local_evaluator.py` and is the single most authoritative artifact.

### Scoring formula

`evaluator/local_evaluator.py:473-475`:

```text
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency
```

- A miss contributes MTTC = 11 (`metric_summary`, line 344) and MRR 0.
- Only the first 10 unique catalog-valid `parent_asin` values are scored
  (`normalize_recommendations`, lines 245-259).
- A hit ends the session at that turn (lines 447-450).
- Intent Override sessions cannot convert before the override message is sent
  (`override_applied` gate, lines 412, 447, 453-459).

Marginal-value arithmetic per session (N sessions): converting a miss to a
hit at turn t adds `0.5/N + (1/rank)·0.3/N + (11−t)/10·0.2/N`; each turn
saved on an existing hit adds `0.02/N`; moving rank 3 → 1 adds `0.2/N`. With
HitRate near 0.92, rank improvements now dominate turn savings.

### How the simulator emits user utterances and evidence

- The hidden **intent card** is derived from the target product's own catalog
  metadata: title, first regex-matched material/color from the searchable
  text, features/details strings, and `budget around $<price>`
  (`intent_card`, lines 202-221). `hard_constraints` = first 2 cleaned
  strings; `soft_preferences` = next 2.
- **First message** by scenario (`initial_message`, lines 304-313): Buying
  discloses one hard constraint verbatim; Browsing is vague ("still
  exploring"); Intent Override opens with a soft preference and later sends
  "Actually, ignore my earlier preference. What I need is: <hard[0]>." on
  turn 3 or 4 (`behavior_for`, lines 224-237).
- **Replies** (`customer_reply`, lines 316-335): if the agent set
  `ask_attribute`, the simulator discloses up to **2** undisclosed intent-card
  constraints whose `classify_constraint` class matches the asked attribute
  (`other` matches anything). If the agent asks nothing, the reply is
  "Those options are not quite right yet…" with **no disclosure**. Boundary
  sessions burn the first ask on a no-preference reply (lines 318-319).
- `classify_constraint` (lines 287-301) buckets constraint strings into
  budget/material/color/size/style/use_case, defaulting to **feature** — so
  many card strings are only reachable via `feature` or `other` asks. This is
  a mechanical fact worth knowing when designing the ask policy, and also the
  exact seam the withdrawn exploit lived in.

### Final evaluation of the 800 private sessions

Per the organizer FAQ
([docs/final_evaluation_faq.md](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/final_evaluation_faq.md)):
teams run the unmodified evaluator themselves on sessions released
post-deadline; deterministic templates are preserved; **no paraphrases are
introduced**; full network access is allowed (teams carry their own API
costs); no organizer hardware or latency limits. Note the tension with
`docs/submission_rules.md:59` ("organizer policy may disable network access")
— the FAQ is newer and more specific, but the safe posture remains an agent
that is *fully valid offline* (already true) with network as an optional
enhancement, explicitly documented (`docs/submission_rules.md:59-64`).

Judging weights (FAQ): Technical Execution 35%, Innovation & Problem Insight
20%, Impact & Relevance 20%, Feasibility & Practicality 15%, Presentation &
Communication 10%. TechnicalScore is only an *input* to Technical Execution
(`docs/competition_specification.md:77`).

## 2. Honest current state of this repository

- **Baseline:** 0.107 TechnicalScore (BM25, `docs/baseline_results.json`;
  `docs/ENGINEERING_JOURNAL.md:44-55`).
- **Frozen Slice 11 hybrid:** structured/BM25/dense fusion (weights 0.02 /
  0.64 / 0.34), MiniLM-L6 cross-encoder at pool depth 50 → 0.5518 on the
  160-dev split with pool recall 0.825
  (`docs/benchmark_target_findings.md:26-29`).
- **Withdrawn 0.9628 route:** a verbatim-evidence route re-derived every
  product's hidden intent card exactly as the simulator does and matched
  template disclosures; it reached TechnicalScore 0.962775 / HitRate 1.0 on
  all 200 public sessions, then was classified as reward hacking and removed
  (`docs/ENGINEERING_JOURNAL.md:788-860` and `:751-786`). The memory note
  "0.9628 locked by parity tests" refers to this removed route, not the
  current build.
- **Current working tree (uncommitted):** honest improvements A/B/C —
  dialog-evidence accumulation (`starter/agent.py:340`, `starter/agent.py:360`,
  `starter/retrieval.py:469`), information-value clarification
  (`starter/agent.py:285`), budget parsing/reordering
  (`starter/retrieval.py:73`, `starter/agent.py:427`) — measured at
  **0.739152** TechnicalScore, HitRate@10 0.919, MRR 0.531, MTTC 4.98 on the
  160-dev split (`benchmarks/honest_abc_dev160.json`;
  `docs/ENGINEERING_JOURNAL.md:711-737`; tests in
  `tests/test_honest_improvements.py`).
- **Depth experiment:** pool depth 80 scored 0.705 < depth-50's 0.739 — the
  cross-encoder, not pool recall, is the precision bottleneck
  (`docs/ENGINEERING_JOURNAL.md:723-726`, `tools/depth_experiment.py`,
  `benchmarks/honest_abc_depth80_dev160.json`).
- **Evidence gaps (fix before submission):**
  1. `benchmarks/robustness_final.json` is cited by the journal
     (`docs/ENGINEERING_JOURNAL.md:732-734`) and README diff but **does not
     exist** — the paraphrase/novel robustness claims are currently
     unsubstantiated.
  2. The A/B/C changes and journal entries are uncommitted working-tree state
     (`git status` at session start).
  3. The 40-session holdout was exposed on 2026-08-30 and no replacement
     untouched validation evidence has been declared
     (`docs/ENGINEERING_JOURNAL.md:191, 522, 906`).
- **Connected LLM planning is a measured regression** (0.161 vs 0.671
  offline; disclosure starvation via null asks,
  `docs/ENGINEERING_JOURNAL.md:833-844`).

**Overfitting risk assessment (honest):** because the organizer guarantees
identical templates, the main private-set risk for the *current honest build*
is unseen-product distribution shift (different intent cards, categories,
prices), not wording. The dialog-accumulation and info-value features consume
only content words, pool statistics, and catalog fields
(`docs/honest_optimizations.md:18-59`), so they should transfer; but this is
untested until the `novel` condition report is generated. The info-value ask
policy has one simulator-coupled subtlety: it optimizes *pool narrowing*
while disclosure actually depends on `classify_constraint` matching — an ask
that splits the pool but matches no card string yields "no additional
preference" and zero evidence. Measuring disclosure yield per asked attribute
on dev (without hardcoding the classifier) is a legitimate diagnostic;
mirroring `classify_constraint` in production would recreate the exploit.

## 3. State of the art per pillar

### 3.1 Intent routing (Buying vs Browsing)

- Conversational-recommendation surveys frame exactly this dual mode:
  system-ask/user-respond with attribute constraints vs open-ended
  exploration — Gao et al., "Advances and Challenges in Conversational
  Recommender Systems" ([arXiv:2101.09459](https://arxiv.org/abs/2101.09459));
  Jannach et al., "A Survey on Conversational Recommender Systems"
  ([arXiv:2004.00646](https://arxiv.org/abs/2004.00646)).
- For this challenge the scenario is *observable from the first message
  template* (requirement present → buying-like; "still exploring" →
  browsing-like), so a lightweight deterministic router is sufficient and
  auditable; an LLM few-shot classifier adds latency/cost without measurable
  score benefit here. Frame the dual-track requirement as: hard-constraint
  filtering track (eligibility) vs dense/diverse track (recall), which the
  repo already implements as hard-constraint eligibility + soft-preference
  evidence (`docs/ENGINEERING_JOURNAL.md:496-500`).
- Industrial precedent for candidate-generation + rerank + ensemble on
  Amazon-scale session data: Amazon KDD Cup 2023 / Amazon-M2
  ([kddcup23.github.io](https://kddcup23.github.io/),
  [arXiv:2307.09688](https://arxiv.org/abs/2307.09688)); the NVIDIA winning
  solution used multi-route candidates fused by gradient-boosted rerankers
  ([openreview.net/pdf?id=J3wj55kK5t](https://openreview.net/pdf?id=J3wj55kK5t)).

### 3.2 Hybrid retrieval

- Sparse+dense hybrids are the robust default: BM25 remains a strong
  zero-shot baseline across domains (BEIR,
  [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)); dense adds
  paraphrase recall. Reciprocal Rank Fusion is the standard training-free
  fusion (Cormack, Clarke & Buettcher, SIGIR 2009,
  [dl.acm.org/doi/10.1145/1571941.1572114](https://dl.acm.org/doi/10.1145/1571941.1572114)).
  The repo went further with trained non-negative fusion weights optimizing
  pool recall at the frozen rerank depth (`docs/fusion_policy_training.md`),
  which is defensible and more presentable than RRF.
- In-memory local embedding models (fits the no-vector-DB-cluster rule):
  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (22M params), [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5),
  [gte-small](https://huggingface.co/thenlper/gte-small); compare on MTEB
  ([arXiv:2210.07316](https://arxiv.org/abs/2210.07316)). The repo's measured
  choice (MiniLM wins HitRate/latency/memory; BGE wins MRR —
  `docs/ENGINEERING_JOURNAL.md:277-291`) is exactly the kind of evidence
  judges reward. Speculation: bge-small's MRR edge might matter more now that
  MRR is the weak axis; re-testing it under the current pipeline is a cheap
  experiment.
- Conversational query rewriting/expansion (feeding accumulated dialog into
  the query) is standard: QReCC ([arXiv:2010.04898](https://arxiv.org/abs/2010.04898));
  the repo's `_prior_dialog_text` implementation is the local realization
  (`starter/agent.py:340`).

### 3.3 LLM / cross-encoder reranking

- Pointwise cross-encoders (ms-marco MiniLM family) are the CPU-cheap
  standard; the repo froze
  `cross-encoder/ms-marco-MiniLM-L-6-v2` at depth 50 after a 3-model ×
  7-depth benchmark (`docs/ENGINEERING_JOURNAL.md:438-452`,
  `docs/reranker_benchmark.md`).
- Listwise LLM reranking: RankGPT permutation generation with sliding windows
  ([arXiv:2304.09542](https://arxiv.org/abs/2304.09542)); pairwise ranking
  prompting is more robust for smaller models
  ([arXiv:2306.17563](https://arxiv.org/abs/2306.17563)); open-source
  distilled listwise rerankers RankVicuna
  ([arXiv:2309.15088](https://arxiv.org/abs/2309.15088)) and RankZephyr
  ([arXiv:2312.02724](https://arxiv.org/abs/2312.02724)); position-bias
  mitigation via permutation self-consistency
  ([arXiv:2310.07712](https://arxiv.org/abs/2310.07712)).
- Cost/latency: a listwise LLM pass over only the final top-10/top-20 per
  turn (~1 call/turn) is the cheapest way to attack MRR; with ~800 sessions ×
  ~5 turns and small prompts this is thousands of small calls — budget and
  fail-open behavior must be designed in (the repo's bounded-fallback pattern
  already fits, `docs/ENGINEERING_JOURNAL.md:501-509`). Speculation: given
  the dev finding that same-line variants crowd rank 1
  (`docs/ENGINEERING_JOURNAL.md:741-745`), an LLM disambiguating near-duplicates
  listwise is plausibly worth +0.05–0.10 MRR, but this is unmeasured.

### 3.4 Multi-turn dialog state tracking

- Modern LLM DST: in-context learning (IC-DST,
  [arXiv:2203.08568](https://arxiv.org/abs/2203.08568)), zero-shot ChatGPT
  DST and its limits (Heck et al.,
  [arXiv:2306.01386](https://arxiv.org/abs/2306.01386)), function-calling
  DST ([arXiv:2402.10466](https://arxiv.org/abs/2402.10466)). The consistent
  finding: LLMs extract slots well zero-shot but make unreliable *state
  mutations* — which is precisely the repo's architecture thesis (LLM
  proposes a typed Turn Plan; deterministic code validates and commits
  atomically, `docs/ENGINEERING_JOURNAL.md:481-509`, ADRs 0001/0006). This is
  a genuinely strong, citable innovation story for judges.
- Slot accumulation vs override: the challenge's Intent Override (slot
  erasure) maps to classic DST value replacement; the repo's
  superseded/dismissed statuses and atomic Turn Plans
  (`starter/constraint_state.py`, `starter/replacement_evidence.py`) exceed
  what the evaluator strictly requires — presentation gold, modest score
  impact.

### 3.5 Clarification questions

- When to ask vs answer: risk-aware decision models (Wang & Ai,
  [arXiv:2101.06327](https://arxiv.org/abs/2101.06327)); datasets/benchmarks
  Qulac ([arXiv:1907.06554](https://arxiv.org/abs/1907.06554)) and ClariQ
  ([arXiv:2009.11352](https://arxiv.org/abs/2009.11352)); survey: Zamani et
  al., "Conversational Information Seeking"
  ([arXiv:2201.08808](https://arxiv.org/abs/2201.08808)). In conversational
  recommendation, EAR's ask-vs-recommend policy is the canonical treatment
  ([arXiv:2002.09102](https://arxiv.org/abs/2002.09102)).
- **This challenge nullifies the ask-cost side of the tradeoff**: the agent
  recommends 10 items *and* asks in the same turn, and a no-ask turn yields
  no disclosure (`evaluator/local_evaluator.py:316-321`). MTTC is charged per
  turn regardless. So the only decision is *which attribute* to ask —
  expected-information-gain selection over the live candidate pool, which the
  repo implements (`starter/agent.py:285`; principle in
  `docs/honest_optimizations.md:39-59`). Quantitatively: a wasted ask costs
  up to one extra turn to conversion ≈ 0.02 TechnicalScore for that session.
- Turn-limit edge: never let clarification continue past the point of
  recommending — the repo already recommends every turn, so the 10-turn hard
  limit only bites through slow convergence, not through ask loops.

### 3.6 Context distillation and personalization

- Session-state distillation: recursive dialogue summarization
  ([arXiv:2308.15022](https://arxiv.org/abs/2308.15022)), OS-style memory
  management (MemGPT, [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)).
  The repo's constraint-state + accumulated-dialog-text design is a compact,
  deterministic equivalent — frame it that way in the writeup.
- Long-term user profile: the evaluator provides only an aggregate
  `user_profile` (purchase-frequency, rating summaries, preference tags —
  `docs/competition_specification.md:21`), and the simulator never consults
  it when generating messages (`evaluate`, `evaluator/local_evaluator.py:404-462`
  uses only the intent card). **Profile use cannot move TechnicalScore**; it
  is purely an Innovation/Impact play (e.g., profile-conditioned tie-breaking
  or explanations). LLM-personalization benchmark for citation: LaMP
  ([arXiv:2304.11406](https://arxiv.org/abs/2304.11406)).
- Runtime workflow re-orchestration (pillar III): satisfy with the existing
  planner-owned route selection and fail-open degradation ladder
  (dense→lexical, reranker→fused ordering, connected→deterministic), each
  with measured evidence — this reads as "dynamic context programming"
  without new risk.

## 4. Hackathon-strategy synthesis

**Where the points are.** TechnicalScore is a fraction of the 35% Technical
Execution criterion; 65% is human-judged
([final_evaluation_faq.md](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/final_evaluation_faq.md)).
A 0.74-scoring agent with a compelling architecture story, honest robustness
evidence, and a flawless demo plausibly beats a 0.96 template-matcher that a
judge recognizes as private-label reconstruction — which the spec rules out
of scope (`docs/competition_specification.md:13`). The withdrawn-route saga
itself is presentation material: "we found the evaluator was exactly
reverse-engineerable, hit 0.96, and chose to remove it, with gates that make
that class of gain unshippable" demonstrates rare evaluation-integrity
maturity (`docs/ENGINEERING_JOURNAL.md:751-786`).

**The uncomfortable fact, stated honestly:** because the organizer guarantees
template stability, the exact-evidence route would almost certainly score
~0.96 on the private 800 too (its card derivation is target-agnostic and
parity-tested across the full 50k catalog,
`docs/ENGINEERING_JOURNAL.md:826-831`). If other teams discover the same
seam, the TechnicalScore leaderboard may be topped by template-aligned
agents. The hedge is not to join them but to (a) document the seam and the
decision in the writeup — converting a score deficit into an insight
advantage on the 20% Innovation criterion — and (b) maximize the honest
score so the gap is small. This recommendation is a judgment call, not a
provable optimum; the team should make it consciously.

**Where remaining engineering effort should go (given ~0.74 honest score):**

1. MRR via final-ordering work (largest metric headroom; see 3.3).
2. MTTC: dev MTTC 4.98 with Intent Override at 6.67
   (`benchmarks/honest_abc_dev160.json`) — faster post-override recovery
   (immediate re-query with the new hard constraint, which the override
   message discloses verbatim) is worth up to ~0.03 TechnicalScore.
3. Commit the working tree, regenerate the full evidence chain, and produce
   the missing robustness report.
4. Then stop tuning and build the demo, video, and Devpost writeup — the
   deliverables list is explicit (`docs/submission_rules.md:6-14`;
   `docs/competition_specification.md:95-100`), and the repo already has a
   demo storyline and presentation outline drafted
   (`docs/ENGINEERING_JOURNAL.md:587-639`).

## 5. Known pitfalls

- **Overfitting to public dev sessions.** All 200 public sessions (including
  the exposed 40-session ex-holdout) have influenced decisions
  (`docs/ENGINEERING_JOURNAL.md:191, 522`). Mitigation: the `novel` condition
  (unseen targets) as the release gate, plus predeclared frozen config before
  the private run.
- **Evaluator-alignment brittleness.** Any behavior that depends on exact
  simulator wording collapses under change and is judged as gaming; the
  paraphrase condition exists to catch this (`tools/robustness_eval.py:49-81`).
  Keep the gate even though the FAQ promises no paraphrases — it is cheap
  insurance and a judging asset.
- **Turn-limit and protocol edge cases.** Exceptions, invalid output, and
  timeouts count as misses (`docs/competition_specification.md:66`);
  `ask_attribute` outside the allowed set degrades to `other`
  (`evaluator/local_evaluator.py:322-323`); recommendations beyond 10 valid
  unique ASINs are ignored. The current build measures 0 exceptions / 0
  invalid responses — keep regression tests on this floor.
- **Clarification loops.** Re-asking answered/dismissed attributes wastes
  turns; the no-repeat policy is tested
  (`tests/test_honest_improvements.py:118-143`). Symmetrically, a *null* ask
  triggers the simulator's no-disclosure retry — never emit null.
- **External-API dependency in the live demo and final run.** The measured
  connected-planning regression (`docs/ENGINEERING_JOURNAL.md:833-844`) plus
  the rules requirement to document network dependence
  (`docs/submission_rules.md:54-64`) argue for: offline-first scoring path,
  any API strictly additive with bounded fallback, and a demo rehearsed with
  network disabled (the journal's demo plan already includes this,
  `docs/ENGINEERING_JOURNAL.md:596-597`).
- **Un-reproducible submission.** The organizer may invalidate runs that
  don't reproduce from the bundle (`docs/submission_rules.md:85-86`); the
  ~198 MB Qdrant artifact + 87 MB model need scripted, checksummed rebuild
  (already built: `docs/ENGINEERING_JOURNAL.md:283-289`).

## 6. Prioritized action list

Ordered by expected value; tagged **[score]** = protects the private-set
TechnicalScore, **[judges]** = wins human judging points.

1. **[score][judges]** Run `tools/robustness_eval.py` on the full 200 +
   paraphrase + 100 novel sessions; check in
   `benchmarks/robustness_final.json` and quote it in README/journal. The
   claim is already published internally but the artifact is missing.
2. **[score]** Commit the honest A/B/C working tree with its journal entry;
   tag the release configuration; declare the frozen config for the private
   run (no further tuning after the tag).
3. **[score]** Attack MRR: (a) improve rerank text rendering (title +
   matched-constraint evidence in the cross-encoder input), (b) re-benchmark
   bge-small / a larger cross-encoder now that latency gates are self-imposed,
   (c) prototype a final-top-10 LLM listwise pass with strict fail-open.
   Measure each on dev + novel before adoption.
4. **[score]** Post-override fast recovery: on the override turn, rebuild the
   query from the new hard constraint alone and verify the new-intent pool
   immediately contains the target (Intent Override MTTC 6.67 is the worst
   scenario axis).
5. **[judges]** Produce the demo: the scripted 6-step storyline with the
   constraint-state timeline, route contributions, and the network-disabled
   rerun (`docs/ENGINEERING_JOURNAL.md:587-605`). Record the video early;
   demo reliability is judged under Technical Execution.
6. **[judges]** Write the Devpost/report around the two differentiators:
   (i) validated Turn-Plan boundary (LLM proposes, deterministic code
   commits) with the DST literature contrast (arXiv:2306.01386), and
   (ii) the reward-hacking discovery-and-removal narrative with the
   robustness gate. Include the baseline→0.74 score progression and the
   honest limitations (MRR, unseen-product evidence).
7. **[judges]** Add one small, visible Innovation/Impact feature that costs
   nothing in score risk: profile-aware explanations or transparent
   "why this item" annotations from constraint evidence (pillar III/safe
   personalization, `docs/competition_specification.md:81-89`).
8. **[score]** Regression floor: keep the zero-exception/valid-output tests
   and the offline-mode evaluation artifact in CI before submission
   (`docs/ENGINEERING_JOURNAL.md:899-903`).
9. **[judges]** Decide explicitly, as a team, the evidence-route question
   (Section 4). If kept disabled, present it as an ablation ("what the
   evaluator rewards vs what a customer needs") — do not leave it
   undocumented, and do not ship it as the scoring path given
   `docs/competition_specification.md:13`.
10. **[score]** (Speculative, cheap) Measure disclosure yield per asked
    attribute on dev sessions and consider weighting the info-value ask
    toward attributes that historically yield disclosures (`feature`,
    `use_case`, `material`, `budget`) — without importing or mirroring any
    evaluator code. Validate on the novel condition to confirm it is not
    wording-coupled.
