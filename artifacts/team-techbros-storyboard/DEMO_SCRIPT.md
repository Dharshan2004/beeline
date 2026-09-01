# Team TechBros — Beeline demo script

Runtime: **3:00 exactly**
Video: silent; deliver this script live or record it separately.

## 0:00–0:14 — Problem: attention is the scarce resource

> On TikTok Shop, you have seconds—not minutes. Every unnecessary question is another chance for the buyer to scroll on. Beeline is built around one number: how few turns it takes to reach the exact product. The name is the thesis.

## 0:14–0:31 — Why this matters for TikTok Shop

> TikTok creates enormous browsing traffic, but only a smaller group reveals real buying intent. That moment is valuable and short-lived: the next swipe can erase it. A shopping agent must recognize when curiosity becomes intent, remove uncertainty, and convert before attention moves on. Every extra turn adds attrition; every extra second risks losing the moment.

## 0:31–0:46 — Beeline session one: immediate conversion

> Now let's see this in real agent replays, produced by the repository's demo helper. In public zero zero four four, the shopper asks for men's jammers with a fabric requirement. The exact product is already rank one. One message in, the beeline is done—converted on turn one with zero connected-model tokens.

## 0:46–1:06 — Beeline session two: questions must earn their turn

> The next replay shows why Beeline sometimes spends a turn. Public zero zero one nine starts vague: rain footwear, still exploring. Instead of a fixed questionnaire, Beeline asks what best splits fifty candidates. Material can eliminate forty-six and, once dismissed, is never repeated. Color is dismissed next; feature then reveals a rubber sole and a five-and-a-half-inch shaft. The target reaches rank one on turn four. Every question reduces uncertainty.

## 1:06–1:21 — Beeline session three: intent override

> The third replay tests a change of mind. Public zero zero one three says, “Actually, ignore my earlier preference. What I need is a rubber sole.” Beeline applies one complete Turn Plan against one state revision. Superseded evidence is retired, useful context remains, and the correct product stays at rank one.

## 1:21–1:45 — Architecture: one correctness boundary, two planes

> That behavior comes from separating language understanding from state safety. The control plane turns the latest message, dialog, and budget into one Turn Plan. A validator applies the whole update or none, so contradictions cannot corrupt intent. The data plane retrieves through structured attributes, BM25 keyword search, and Qdrant semantic search. We fuse that evidence, then MiniLM reranks within a hard deadline. Beeline returns ten catalog-valid products or asks the most useful question. Models improve understanding; deterministic validation owns correctness.

## 1:45–2:03 — Architecture differentiator: parallel coverage

> The locally ranked pool then enters Beeline's default connected stage: an LLM tournament. Four GPT five point four nano calls each compare twelve products in parallel, covering forty-eight at roughly one-call latency. The top three per chunk become twelve finalists for one final LLM ranking. Coverage increases, not waiting time. If the service fails, Beeline returns the local order. Shipped p ninety-five is three point four four seconds.

## 2:03–2:23 — Development process: benchmarks made the decisions

> We chose that architecture through benchmarks, not intuition. Depth one hundred scored slightly higher, but projected to thirteen hundred sixty-seven seconds and failed the nine-hundred-second full-run gate, so depth fifty shipped. The first nano tournament measured four point two seconds p ninety-five, above our live-commerce limit. We tightened both deadlines, reran the robustness gates, and reached three point four four seconds without losing the zero point eight zero six score.

## 2:23–2:43 — Proof: the unmodified official evaluator

> Those decisions show up in the unmodified official evaluator. The weak BM25 starter scored zero point one zero seven. Beeline reaches zero point seven five six offline at zero cost, and zero point eight zero six with the nano tournament. Hit Rate at Ten is zero point nine six zero, M R R is zero point six two eight, mean turns to conversion is four point one three, and invalid outputs remain zero.

## 2:43–2:53 — Honesty: survive distribution shift

> Exact-set score is not enough. With every message reworded, Beeline scores zero point seven six three. Across one hundred targets absent from public labels, it scores zero point seven five six. Benchmark-specific gains do not ship. Connected failure returns the valid local order.

## 2:53–3:00 — Close

> Offline, Beeline is free. The default connected path costs about one cent per session. Beeline: the shortest honest path from attention to purchase.
