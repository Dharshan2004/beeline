# Beeline — The shortest path from “I’m looking” to “that’s the one”

Beeline is a conversational shopping agent that turns vague, changing, everyday requests into ranked, catalog-valid product recommendations. It remembers what matters, lets shoppers change their minds, asks useful questions, and keeps working even when the network or an external model does not.

Built by **Team TechBros** for **TikTok TechJam 2026 — Track 4: Shopping Copilot**.

## Inspiration

Shopping rarely begins with a perfect query. A customer might start with “I need boots,” later mention a budget, say that waterproofing is essential, and then change the request to slippers altogether. Traditional search treats those messages as isolated keyword queries. A free-form chatbot may understand them, but it can also forget a constraint, invent a product, or fail when its model is unavailable.

We wanted to bridge that gap.

The name **Beeline** captures our goal: find the shortest trustworthy path between a shopper’s attention and the product they actually want. That felt especially relevant to live and social commerce, where every unnecessary question or slow response gives the shopper another chance to scroll away.

The challenge made this goal measurable. The official evaluator rewards finding a hidden target product, ranking it highly, and finding it early:

**TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency**

But we did not want to build something that only understood the evaluator. We wanted the score to reflect real shopping competence: remembering a conversation, handling ambiguity, surviving paraphrases, and behaving reliably under failure.

## What it does

On every turn, Beeline returns a natural-language response, an optional clarification attribute, and up to ten ordered product recommendations from the frozen 50,000-item catalog.

It can:

- carry requirements across a multi-turn conversation;
- distinguish hard constraints from soft preferences;
- retire stale requirements when the shopper changes intent;
- respect “no preference” answers and avoid asking the same question again;
- understand common budget language such as “under $50” or “around $60”;
- retrieve exact, lexical, and semantic matches in parallel;
- ask the question expected to eliminate the most uncertainty in the current candidate pool;
- use aggregate profile hints only while the shopper’s current intent is vague;
- rank a broad candidate pool with local models and an optional low-cost LLM tournament; and
- fail open to a complete offline agent if a model, network call, index, or reranker is unavailable.

The result is not a hosted storefront or a UI wrapper. It is a runnable Python shopping agent built directly around the competition’s official `Agent.reset(...)` and `Agent.respond(...)` contract.

## How we built it

We split Beeline into a **control plane** and a **data plane**.

### 1. A validated conversational control plane

The control plane interprets the latest message together with prior dialog and the shopper’s active constraints. It creates one typed **Turn Plan** describing additions, replacements, dismissals, and the next clarification.

The important design decision is that probabilistic interpretation does not directly mutate customer state. Deterministic validation checks the entire Turn Plan against an unchanged snapshot and either commits every transition atomically or commits none of them. This prevents malformed, contradictory, or stale model output from partially corrupting the shopper’s intent.

Constraints retain their source turn and status, so an override preserves history while removing old requirements from live retrieval. Hard constraints affect eligibility; soft preferences influence ordering without hiding otherwise valid products.

### 2. Multi-route retrieval

The data plane gathers independent evidence from three retrieval routes:

1. **Structured retrieval** matches known catalog attributes and hard requirements.
2. **SQLite FTS5 BM25** searches both the accumulated conversation and the newest message, keeping the stronger score per product.
3. **Dense retrieval** uses `sentence-transformers/all-MiniLM-L6-v2` embeddings with embedded Qdrant Local Mode to recover semantic matches and paraphrases.

We normalize and fuse these routes with a frozen non-negative policy. Candidate admission includes a catalog-derived popularity prior: when several near-identical listings satisfy the same request, products with stronger purchase-proxy evidence deserve consideration. That prior decays as the shopper provides more specific evidence.

### 3. Local reranking and a connected ranking tournament

The fused pool is reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` in a persistent local worker with a strict deadline. If it crashes, times out, or fails to load, Beeline returns the valid fused ordering.

When an OpenAI API key is available, Beeline adds a low-cost parallel ranking tournament. Several `gpt-5.4-nano` calls inspect chunks of the 48-product pool concurrently, then the chunk leaders meet in one final listwise ranking. Parallelism gives the model broader coverage without serial latency. Every call uses strict structured output, `store=false`, token and cost caps, no retry loop, and unconditional fail-open behavior.

### 4. Clarification by information value

Beeline does not ask attributes in a fixed order. For every eligible clarification, it estimates how much a definitive answer would split the live candidate pool, then asks the highest-value unanswered question. Active, dismissed, and previously asked attributes are excluded.

### 5. Evaluation as part of the architecture

We kept the official evaluator byte-identical and placed all experiments in separate tools. Every serious change was measured across three conditions:

- the released sessions;
- the same sessions with every customer message paraphrased; and
- generated sessions whose target products never appear in the public labels.

The agent’s production path imports no evaluator code and contains no per-session answers, scenario labels, or simulator-template branches.

## Challenges we ran into

### Candidate recall and final ranking are different problems

We initially focused on reranking, then discovered that no reranker can recover a target excluded from its candidate pool. We reordered our optimization work: first maximize honest target reachability, then improve final ordering.

More candidates were not automatically better, either. Expanding the rerank depth from 50 to 80 introduced enough distractors to reduce the score. That taught us to treat candidate depth as a precision–recall and latency decision, not a “larger is better” setting.

### Reproducible embeddings were unexpectedly subtle

Dynamic batch padding caused vector outputs to depend on batch composition. We fixed the embedding length at 256 tokens, pinned model revisions, checksummed the catalog and artifacts, and staged index publication so a failed rebuild could never overwrite the last valid version.

### Model intelligence could not be allowed to become a dependency

Connected models sometimes returned invalid plans, asked no useful question, exceeded latency budgets, or simply became unavailable. We built a deterministic fallback first and forced every connected stage to degrade to it. The agent’s worst case is therefore still a complete, schema-valid offline shopping system.

### Measurement noise was large enough to mislead us

Repeated identical runs varied by roughly **±0.013 TechnicalScore** because of floating-point and threaded reranking effects. We treated effects below about 0.02 as noise and rejected several plausible ideas—including deeper reranking, broad soft-preference boosts, and rank blending—when measurement did not support them.

## What we are proud of

The weak BM25 starter baseline scored **0.107**. On all 200 public sessions through the official evaluator, Beeline now reaches:

| Configuration | TechnicalScore | HitRate@10 | MRR | Mean turns | p95 latency | Cost/session |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fully offline | **0.756** | 0.910 | 0.588 | 4.79 | 0.8 s | $0 |
| Nano ranking tournament | **0.806** | 0.960 | 0.628 | 4.13 | 3.4 s | about $0.01 |

We are even prouder that the gains survive distribution shift:

| Configuration | Released sessions | Paraphrased sessions | Unseen target products |
| --- | ---: | ---: | ---: |
| Fully offline | 0.756 | 0.725 | 0.718 |
| Nano ranking tournament | 0.806 | 0.763 | 0.756 |

The connected path is optional. If credentials or network access disappear, the same evaluator command automatically falls back to the zero-cost offline configuration instead of failing.

## What we learned

We learned that reliable AI systems need a clear boundary between what a model may **suggest** and what the product must **guarantee**. LLMs are valuable for semantic judgment, but state integrity, catalog validity, deadlines, cost limits, and fallback behavior belong to deterministic code.

We learned that conversational search is a state-management problem as much as it is a retrieval problem. Remembering “blue,” understanding that “actually, slippers” retires an earlier request, and respecting “I don’t care about material” all change which products should remain eligible.

We learned that honest evaluation is itself an engineering feature. A leaderboard gain that disappears under paraphrasing is not generalization. Keeping failed experiments, measuring scenario regressions, and quantifying runtime noise gave us more confidence in the final system than the headline score alone could.

Finally, we learned that production constraints can inspire better architecture. A strict latency budget led us to parallel model calls. Offline requirements led us to embedded retrieval and local reranking. The need for valid output under every failure led us to typed plans, atomic commits, pinned artifacts, and fail-open stages.

## What is next

Beeline’s main remaining weakness is distinguishing the exact purchased item among near-identical variants. Our next steps would be:

- improve final ranking text and benchmark a stronger compact cross-encoder within the same latency gate;
- expand catalog-derived attribute understanding without turning preferences into filters;
- retrain fusion weights on the final accumulated-dialog trajectories; and
- add transparent recommendation explanations grounded only in active constraints and catalog fields.

Beyond the competition, the same architecture could support conversational discovery in TikTok Shop–style live commerce: low-latency intent capture, safe mid-conversation changes, proportional model cost, and a useful offline floor rather than a blank error screen.

## Built with

Python, OpenAI Responses API, PyTorch, Hugging Face Transformers, Sentence Transformers, embedded Qdrant, SQLite FTS5, NumPy, MiniLM embeddings, and a MiniLM cross-encoder.

Development and review used VS Code, GitHub, Claude Code, and OpenAI Codex. We used **ElevenLabs** to generate the voiceover for our three-minute demo video; it is a presentation tool and is not part of the shopping agent or evaluation path. Product data comes from the organizer-frozen **Amazon Reviews 2023** Clothing, Shoes, and Jewelry catalog.

## Team TechBros

- **Dharshan2004:** offline agent contract, dense retrieval integration, hybrid fusion and training, connected-model benchmarking, packaging, ranking tournament, and robustness gates.
- **dylothx:** constraint handling, intent overrides, boundary responses, validated planning, session-mode experiments, and conditional model calls.
- **likalight:** versioned dense index, reranker benchmarking, depth selection, and live local reranking.
