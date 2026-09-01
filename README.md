# Beeline

**Team TechBros — TikTok TechJam 2026, Track 4 (Shopping Copilot)**

[Beeline on Devpost](https://devpost.com/software/beeline-by-team-techbros) · [Three-minute demo on YouTube](https://youtu.be/m86-2L3Drpc)

**Turn buying intent into purchase before the next swipe.**

Beeline is a conversational shopping agent that turns vague, changing,
everyday requests into ranked, catalog-valid product recommendations. It
remembers what matters, lets shoppers change their minds, asks useful
questions, and keeps working even when the network or an external model does
not. The name is the thesis: the shortest trustworthy path between a
shopper's attention and the right product.

## Inspiration

Shopping rarely starts with a perfect query. A customer might begin with “I
need boots,” later mention a budget, say waterproofing is essential, and then
change the request to slippers. Keyword search treats those messages as
isolated queries; a free-form chatbot can understand them but may forget a
constraint, invent a product, or fail when its model is unavailable.

Beeline bridges that gap. It combines conversational understanding with
deterministic state protection, catalog-grounded retrieval, measurable
latency and cost, and a complete offline fallback. This matters in live and
social commerce, where every unnecessary question or slow response gives the
shopper another chance to scroll away.

## What it does

On every turn, Beeline returns a natural-language response, an optional
clarification attribute, and up to ten ordered products from the frozen
50,000-item catalog. It carries requirements across turns, distinguishes hard
constraints from soft preferences, handles intent changes and “no preference”
answers, understands common budget language, and asks the question expected
to remove the most uncertainty from the current candidate pool.

## Competition contract

The official Python evaluator is the primary product contract. Beeline exports
`starter.agent.Agent` with:

```python
agent.reset(session_id: str, user_profile: dict) -> None
agent.respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict
```

Every response contains a customer-facing `message`, an allowed or null
`ask_attribute`, and ordered, unique, catalog-valid parent-ASIN
`recommendations`. Connected runs also report non-negative token `usage`.
Sessions are isolated by `session_id`, recommendations are returned on every
turn, and model or network failure falls back to schema-valid local behavior.

## For judges — run it in 4 commands

Python 3.11.9. With an `OPENAI_API_KEY` in `.env` (or exported), the bare
evaluator command below runs the full shipped configuration — the nano
ranking tournament, **0.806** — automatically. Without a key the same
command transparently runs the fully offline agent (0.756): no network, no
cost, never an error. The evaluator itself is never modified; the agent
simply configures itself.

```bash
python3 -m venv .venv && source .venv/bin/activate \
  && pip install -r requirements-dense.txt -r requirements-openai.txt
# catalog: download catalog.jsonl.gz from the GitHub Release, verify SHA256SUMS,
# then: gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
# connected mode (activated automatically when configured):
# echo 'OPENAI_API_KEY=sk-...' > .env
python -m tools.fetch_model && python -m tools.fetch_model \
  --identity cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --destination models/cross-encoder__ms-marco-MiniLM-L-6-v2 \
  --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  && python -m retrieval.build_dense_index --catalog data/catalog.jsonl --verify-load
python -m evaluator.local_evaluator          # official score, all 200 public sessions
python -m tools.demo_session --sample public_0044   # watch one annotated session
# (both auto-use the shipped LLM tournament if OPENAI_API_KEY is set;
#  export BEELINE_OFFLINE=1 to force the zero-cost offline agent)
```

### Required assets

The 50,000-row catalog and generated model/index artifacts are intentionally
kept out of Git. Download `catalog.jsonl.gz` and `SHA256SUMS` from the
[latest GitHub Release](https://github.com/Dharshan2004/beeline/releases/latest),
verify the published checksum, and place the decompressed catalog at
`data/catalog.jsonl`:

```bash
shasum -a 256 -c SHA256SUMS
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

`python -m tools.fetch_model` downloads the two pinned local model revisions;
`python -m retrieval.build_dense_index --catalog data/catalog.jsonl
--verify-load` builds and verifies the embedded index. Neither step downloads
anything during official evaluator turns.

### Demo

`python -m tools.demo_session --sample public_0044` replays an annotated
multi-turn session through the same Agent interface used by the evaluator.
The [three-minute public demo video](https://youtu.be/m86-2L3Drpc)
demonstrates the complete flow; its narration is preserved in
[`artifacts/team-techbros-storyboard/DEMO_SCRIPT.md`](artifacts/team-techbros-storyboard/DEMO_SCRIPT.md).

## Results

**The only valid score is the official evaluator's own output** (the command
above). The evaluator is **byte-identical** to the published original —
verify with `git diff 2a6cc8e HEAD -- evaluator/` (empty). Every team-side
extension (connected-model settings, latency reporting, holdout-scope
guards) lives outside it in `tools/evaluation_harness.py`, which imports
the pristine evaluator unchanged.
TechnicalScore = 0.5·HitRate@10 + 0.3·MRR + 0.2·Efficiency over all 200
public sessions. Weak BM25 starter baseline: **0.107**.

| Configuration (official evaluator) | TechnicalScore | HitRate@10 | MRR | MTTC | p95/turn | Cost/session |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Offline (no key, or `BEELINE_OFFLINE=1`) | **0.756** | 0.910 | 0.588 | 4.79 | 0.8 s | $0 |
| gpt-5.4-mini listwise rerank (opt-in) | **0.771** | 0.910 | 0.638 | 4.75 | ~4.6 s | ~$0.02 |
| Nano ranking tournament (**default with key**) | **0.806** | 0.960 | 0.628 | 4.13 | 3.4 s | ~$0.01 |

Supplementary anti-overfitting evidence (not the official metric): the same
unmodified `evaluate()` scorer replayed with every customer message fully
reworded, and on 100 generated sessions whose targets appear in no public
label. Scores that survive both are competence, not benchmark fit:

| Configuration | Exact | Paraphrased | Novel targets |
| --- | ---: | ---: | ---: |
| Offline | 0.756 | 0.725 | 0.718 |
| Mini rerank (opt-in) | 0.771 | 0.740 | 0.756 |
| Nano tournament (default with key) | 0.806 | 0.763 | 0.756 |

Run-to-run noise is ±0.013, quantified from identical-code runs. Reproduce:

```bash
python -m tools.robustness_eval --output benchmarks/robustness.json
python -m unittest discover -s tests   # 225 tests, always offline
```

## How we built it

Beeline separates understanding a shopper from protecting their intent. The
control plane converts the latest message, prior dialog, and budget evidence
into one Turn Plan; deterministic validation applies the entire update or none
of it. The data plane retrieves and locally reranks a catalog-valid candidate
pool. In the shipped connected configuration, that pool then enters the
parallel LLM ranking tournament by default. A provider failure returns the
valid local order rather than an error.

```mermaid
flowchart TD
    M[Customer message] --> TI[Turn interpreter<br/>deterministic turn plan]
    TI --> CS[Constraint state<br/>validated, atomic]
    M --> DE[Dialog evidence<br/>accumulated turns · budget parse · profile hint]
    CS --> R
    DE --> R
    subgraph R[Retrieval routes]
        S[Structured<br/>attribute index]
        B[BM25 dual-query<br/>full dialog + latest turn]
        D[Dense<br/>MiniLM + local Qdrant]
    end
    R --> F[Fusion<br/>frozen weights]
    F --> P[Popularity-aware pool admission<br/>bestseller prior, decays with evidence]
    P --> CE[Cross-encoder rerank<br/>local, 1.5 s deadline, fail-open]
    CE --> TB[Popularity tie-break<br/>within score bands, top-10 only]
    TB --> L[Parallel LLM ranking tournament<br/>default connected stage · fail-open]
    L --> O[Top-10 recommendations +<br/>highest-information-value question]
```

Every model stage fails open to a deterministic path: the agent cannot
crash, hang, or emit an invalid response because a model misbehaved. Key
behaviors:

- **Dialog accumulation** — retrieval conditions on everything the customer
  has said, minus superseded constraints (intent overrides drop replaced
  wording, structured state keeps what survives).
- **Information-value clarification** — each question maximally splits the
  live candidate pool; answered or dismissed attributes are never re-asked.
- **Popularity prior** — the hidden target is a real purchase; among
  near-identical listings a Bayesian bestseller prior is the honest
  tie-breaker, its weight decaying as the customer states requirements.
- **Budget understanding** — "under $50", "around $60" reorder toward
  in-budget products without eliminating unpriced ones.
- **Parallel ranking tournament** — at a fixed wall-clock budget,
  concurrency buys coverage: the whole 48-candidate pool gets a concurrent
  LLM read in chunks, chunk leaders meet in a final listwise call.

## Tradeoff analysis — built for live commerce

The obvious deployment for this agent is TikTok Shop–style conversational
commerce, and that context dictates our tradeoffs. A shopper who arrives
from a video has seconds of intent, not minutes: every extra turn and every
second of per-turn latency is attrition on a purchase that was almost made.

- **Turns are the scarcest resource.** MTTC is weighted at only 0.2 in the
  score, but in live commerce it *is* the business metric — each avoided
  clarification turn keeps a buyer who would otherwise scroll on. Our
  question policy asks only questions that provably shrink the candidate
  pool, and drops attributes the customer has answered or dismissed.
- **Per-turn latency is a hard product constraint, not a benchmark metric.**
  A 5-minute turn can buy benchmark points (LLM-reading thousands of
  candidates), but no live shopper waits for it. We keep the connected
  pipeline under 4 seconds (p95 3.4 s) by buying LLM coverage with
  *parallel* chunk calls rather than serial depth, and the offline path
  answers in under a second.
- **Cost must round to zero at platform scale.** The offline fallback costs
  nothing; the default connected path costs ~$0.01/session with no-reasoning
  nano calls doing the volume work and budget caps enforced per call.
- **Reliability beats brilliance.** When connected mode is configured, the
  LLM tournament is the default ranking stage. It remains fail-open: a
  timeout returns the local ordering, never an error. The agent's worst case
  is the offline agent, which already scores 0.756.
- **Generalization is a release gate.** The private evaluation uses different
  users and targets, so we validate the shipped configuration on fully
  reworded conversations and 100 targets absent from public labels, not only
  the released sessions.

## Generalization and evaluation discipline

- Production code contains no evaluator imports, no simulator template
  knowledge, no per-session logic — verified by an independent adversarial
  review (`docs/ENGINEERING_JOURNAL.md`).
- Exact / paraphrased / novel-target conditions are always reported
  separately so benchmark-specific gains are visible.
- Every lever is documented with its measurement — including rejections —
  in `docs/honest_optimizations.md`; the full iteration history, failures
  included, is in `docs/ENGINEERING_JOURNAL.md`.

## Connected mode — automatic when configured

**Network policy (per submission rules):** the agent never *requires*
network access. When an `OPENAI_API_KEY` is present, the agent automatically
uses the LLM ranking tournament as its default reranking path. If official
scoring disables network access—or the provider fails—the agent runs its
complete offline fallback (0.756) with zero code or config changes.

OpenAI Responses API with `store=false`, strict JSON-schema outputs,
per-call budget reservation under a $10 development cap, no retries,
unconditional fail-open. Prices versioned in `config/semantic_ranker*.json`.
Token usage is reported per turn through the official response schema.

```bash
# single mini rerank                      # nano ranking tournament
python -m tools.llm_rank_experiment \
  --role fast_ranker --sessions 60 \
  --max-candidates 12 --output benchmarks/llm_rank.json
python -m tools.llm_rank_experiment \
  --role nano_ranker --config config/semantic_ranker_nano.json \
  --tournament --sessions 60 --output benchmarks/llm_tournament.json
```

## Challenges and lessons

- **Candidate recall and ranking are different problems.** A reranker cannot
  recover a target excluded from its candidate pool, but increasing rerank
  depth from 50 to 80 added enough distractors to reduce quality. We learned
  to tune pool admission, depth, precision, and latency together.
- **Reproducible embeddings were unexpectedly subtle.** Dynamic batch padding
  made vectors depend on batch composition. Fixed 256-token padding, pinned
  model revisions, checksums, and staged index publication made the dense
  artifact reproducible and safe to rebuild.
- **Model intelligence cannot become a dependency.** Connected stages can
  time out, fail schema validation, or disappear with the network. Typed Turn
  Plans, atomic state transitions, deadlines, and fail-open stages keep the
  official Agent contract valid in every mode.
- **Conversational search is also state management.** Remembering “blue,”
  applying “actually, slippers” as an intent replacement, and respecting “I
  do not care about material” change which products remain eligible—not only
  how a query should be rewritten.
- **Evaluation is an engineering feature.** Identical runs vary by roughly
  ±0.013 TechnicalScore, so effects below about 0.02 are treated as noise.
  Released, paraphrased, and novel-target scores are reported separately, and
  plausible changes are rejected when evidence does not support them.

## Limitations

- Among near-identical listings, the exact purchased product is
  underdetermined by dialog text; the popularity prior is the honest ceiling
  on MRR.
- MTTC has a structural floor: intent overrides arrive at turn 3–4 and
  misses score as turn 11.
- With more time: LLM constraint extraction behind the existing fail-open
  planning seam, fusion-weight retraining on post-accumulation trajectories,
  a bge-small-en-v1.5 embedding swap (`docs/lever_catalog.md`).

## Repository map

```
starter/          agent, constraint state, retrieval, LLM stages (fail-open)
retrieval/        dense index, fusion, cross-encoder worker
evaluator/        official simulator + scorer (never modified)
tools/            robustness harness, demo replay, experiments
docs/             engineering journal, optimization log, specs, ADRs
tests/            225 tests incl. anti-coupling and fail-open coverage
```

## Team TechBros — contributions

Work was planned and tracked as vertical slices on this repository's issue
tracker (issues #1–#19); per-slice assignees:

- **Dharshan2004** — offline agent contract (Slice 01), embedded dense
  retrieval in live recommendations (05), fixed hybrid fusion (06),
  replayable fusion-training dataset (09), fusion-weight training and the
  frozen pool-aware policy (10–11), connected OpenAI model benchmarking (14),
  conditional second calls and budget enforcement (15, with dylothx),
  packaging/reproduction (17), plus this submission's honest-optimization
  wave (dialog accumulation, information-value asks, popularity priors,
  ranking tournament, robustness gates).
- **dylothx** — constraint handling end-to-end (02), Intent Overrides and
  Boundary Responses (03), validated LLM planning for overrides (12),
  Session Mode and clarification experiments (13), conditional calls (15,
  with Dharshan2004).
- **likalight** — versioned dense index (04), reranker benchmarking and
  deep-pool depth selection (07), live deep-pool reranking (08).

## Tools, models, and references

- **Development tools:** VS Code; Claude Code (Claude Fable 5) and OpenAI
  Codex as coding agents for implementation, measurement, and review;
  GitHub CLI for the issue-tracker workflow; ElevenLabs for the three-minute
  demo-video voiceover (presentation only, never part of the Agent or
  evaluation path).
- **APIs and models:** OpenAI Responses API — `gpt-5.4-mini` (listwise
  rerank), `gpt-5.4-nano` (chunk-tournament ranking, query rewriting),
  `gpt-5.6-sol` / `gpt-5.6-luna` (planning benchmarks, measured and not
  retained). Local models: `sentence-transformers/all-MiniLM-L6-v2`
  embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (pinned
  revisions).
- **Libraries:** PyTorch, Hugging Face Transformers, sentence-transformers,
  embedded Qdrant (local mode), SQLite FTS5, NumPy.
- **Dataset:** Amazon Reviews 2023 (McAuley Lab, UCSD) — organizer-frozen
  catalog and sessions; see `DATA_ATTRIBUTION.md`.
- **Referenced research** (how each mapped to our levers:
  `docs/lever_catalog.md`, `docs/honest_optimizations.md`):
  - Conversational query rewriting — [LLM4CS](https://arxiv.org/abs/2303.06573),
    [ConvGQR](https://arxiv.org/abs/2305.15645),
    [informative conversational rewriting](https://arxiv.org/abs/2310.09716),
    [selective dialog history for topic shifts](https://www.sciencedirect.com/science/article/pii/S2543925122001231)
  - Hybrid retrieval and fusion —
    [reciprocal rank fusion (Cormack et al., SIGIR 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf),
    [rerank-depth study (Elastic Search Labs)](https://www.elastic.co/search-labs/blog/elastic-semantic-reranker-part-3)
  - Document expansion — [doc2query](https://arxiv.org/abs/1904.08375),
    [Doc2Query++](https://arxiv.org/html/2510.09557v2)
  - Question/facet selection by expected information gain —
    [Vandic et al., CIKM 2013](https://personal.eur.nl/frasincar/papers/CIKM2013/cikm2013.pdf),
    [Interactive Classification](https://arxiv.org/abs/1911.03598),
    [BED-LLM](https://arxiv.org/abs/2508.21184),
    [SAUR aspect elicitation](https://par.nsf.gov/servlets/purl/10090082),
    [AGENT-CQ](https://arxiv.org/abs/2410.19692)
  - Listwise LLM reranking — [RankGPT](https://arxiv.org/abs/2304.09542),
    [RankZephyr](https://arxiv.org/abs/2312.02724)
  - Popularity and cold-start priors —
    [Bayesian-average ranking (Algolia)](https://www.algolia.com/doc/guides/managing-results/must-do/custom-ranking/how-to/bayesian-average),
    [Empirical-Bayes cold start in product search (Amazon, CIKM 2022)](https://assets.amazon.science/b5/2f/a9d9581d4f8eab473a4ab4a8ad35/addressing-cold-start-in-product-search-via-empirical-bayes.pdf)
  - Personalization —
    [Personalize-Before-Retrieve](https://arxiv.org/html/2510.08935v1),
    [Zero Attention Model for personalized product search](https://arxiv.org/abs/1908.11322)
  - Diversification — [intent-aware search diversification (IA-Select/xQuAD lineage)](https://dl.acm.org/doi/10.1145/2009916.2009997)
  - E-commerce relevance — [Amazon ESCI dataset](https://arxiv.org/abs/2206.06588)
  - Local models — [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
    [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
    [bge-small-en-v1.5 (future work)](https://huggingface.co/BAAI/bge-small-en-v1.5)
