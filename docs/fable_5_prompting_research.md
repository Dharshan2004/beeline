# Prompting Claude Fable 5 for Shopping-Agent Optimization

Research date: 2026-08-31

## Executive answer

“Fable 5” refers to Anthropic's **Claude Fable 5**, model ID
`claude-fable-5`. Anthropic positions it as its most capable widely released
model for demanding reasoning, debugging, and long-horizon agentic work. It is
therefore a plausible choice for this repository-wide diagnosis, but prompt
wording cannot guarantee a TechnicalScore of 0.95. The score has to be earned
and reproduced through the repository's evaluator.

The claim that Fable 5 simply becomes “dumber” as a prompt gets longer is not
supported by the primary sources reviewed. The better-supported claim is:

- irrelevant, redundant, or stale context competes for a finite attention
  budget and can reduce recall or precision;
- requirements buried in a long context may receive less reliable attention;
- enough relevant context, motivation, and verification criteria still improve
  performance; and
- Fable 5 specifically needs **less prescriptive scaffolding** than older
  models, so inherited mega-prompts can degrade its output.

Use a concise job handoff, let Fable inspect the repository just in time, and
give it a deterministic measurement loop. For this task, select `xhigh` effort
if available (`high` is Anthropic's normal default) and start a fresh Claude
Code session so old conversational material does not dilute the task.

## What the primary sources actually say

Anthropic's [Fable 5 model documentation](https://platform.claude.com/docs/en/models/fable-5/introducing-claude-fable-5-and-claude-mythos-5)
identifies `claude-fable-5` as the broadly available model for demanding
reasoning and long-horizon agentic work. The dedicated
[Fable 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5)
reports stronger debugging and code-review recall, recommends `high` effort for
most tasks and `xhigh` for capability-sensitive work, and says brief
instructions can steer behavior that previously required enumeration. It also
warns that prompts and skills built for older models may be too prescriptive
and can degrade output quality.

That guide recommends giving Fable the reason behind a request, grounding
progress claims in tool results, using fresh-context verifier subagents, and
making verification explicit on long runs. It warns against asking the model to
reproduce hidden reasoning; this can trigger a `reasoning_extraction` refusal.
Ask for evidence and decisions, not private chain-of-thought.

Anthropic's [context-engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
says model focus and recall generally degrade as context grows, describing
context as a finite resource with diminishing returns. Crucially, it also says
that “minimal” does not necessarily mean short: an agent still needs enough
information to act correctly. It recommends the smallest **high-signal** set of
tokens, progressive discovery through file paths and tools, a few canonical
examples rather than an edge-case laundry list, and structured notes or
subagents for long-running work.

Anthropic's general [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
recommend clear constraints, relevant context, and structured sections. If a
large document must be placed directly in a prompt, Anthropic recommends putting
the documents first and the request at the end; its tests found that end-placed
queries could improve complex multi-document performance by up to 30%. This is
not a reason to paste the repository into chat: Claude Code can retrieve the
relevant files itself.

The original [*Lost in the Middle* paper](https://arxiv.org/abs/2307.03172)
found that long-context model performance varied with the location of relevant
information and was often worse when that information was in the middle. The
paper predates and did not test Fable 5, so it supports a general context-design
caution, not a Fable-specific degradation curve.

## How difficult coding tasks were made tractable

Anthropic's examples put most of the leverage in the environment and feedback
loop, not a magic preamble:

- In [its autonomous C-compiler experiment](https://www.anthropic.com/engineering/building-c-compiler),
  Anthropic used high-quality regression tests, concise machine-searchable
  diagnostics, deterministic fast samples, progress files, known-good oracles,
  delta debugging, and specialized parallel agents. The article explicitly
  says most effort went into tests, environment, and feedback.
- Its [long-running application harness](https://www.anthropic.com/engineering/harness-design-long-running-apps)
  used a short high-level request, let a planner define outcomes rather than
  premature implementation details, and separated the builder from a skeptical
  evaluator. It retained room to refine an approach or pivot when evidence
  plateaued.
- Its [agent-evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
  recommends stable environments, deterministic graders where possible, and
  both capability and regression evaluations. It also warns that leaked history
  can give an agent an unfair advantage.

For this repository, that implies: baseline first, diagnose from evaluator
traces, change one falsifiable bottleneck at a time when practical, retain only
measured gains, and use an independent verifier before accepting the final
result. Fable should be free to replace an architecture when the measurements
show that local tuning has reached a ceiling.

## What Boris Cherny, Anthropic, and Stripe actually put behind short prompts

The user's objection is correct: the first recommended prompt below was longer
than the visible prompts used in several strong first-party examples. The
important correction is not merely to delete words. It is to move stable detail
out of the turn prompt and into the **harness**: repository instructions,
retrievable documentation, tools, tests, progress artifacts, hooks, retries,
and independent verification.

### Boris Cherny: a short request sits on top of a maintained workflow

In [Boris Cherny's own workflow thread](https://x.com/bcherny/status/2007179832300581177),
he describes a deliberately “surprisingly vanilla” interactive setup. Most PR
sessions begin in Plan mode; the plan is reviewed before edits. The Claude Code
team checks a shared `CLAUDE.md` into git and adds lessons when Claude makes a
recurring mistake. Repeated workflows live in slash commands, formatting is a
hook, and code simplification and end-to-end verification are specialized
subagents. For long runs he uses a background verifier, a stop hook, or a Ralph
loop. His stated most important practice is giving Claude a way to verify the
result; he says that feedback loop improves final quality by 2–3x.

Cherny's later account goes one abstraction level higher. In the
[host's first-party recap of his Acquired Unplugged interview](https://workos.com/blog/boris-cherny-claude-code-acquired-interview-takeaways),
WorkOS reports that he now writes loops which prompt Claude and decide what to
work on, rather than manually writing every prompt. This is evidence for
automating the sense–act–check cycle, not evidence that an underspecified
one-liner can replace tests or project knowledge.

The literal user instruction is therefore only the tip of the system. The
larger context is supplied by the checked-in `CLAUDE.md`, plan, commands,
subagents, tools, hooks, repository, and executable verifier.

### Anthropic: the published one-line build used a large hidden harness

Anthropic's long-running application experiment gives the cleanest controlled
example. Its visible prompt was:

> Build a fully featured DAW in the browser using the Web Audio API.

That one sentence did **not** go directly to an unassisted coding model. A
planner expanded a 1–4 sentence request into a product specification; a
generator implemented it; a separate evaluator exercised the real application
with Playwright and returned defects for repair. The full run took about four
hours and cost $124.70. Anthropic explicitly says the planner should specify
outcomes and high-level design rather than granular implementation, because an
incorrect low-level decision in the upfront spec can cascade. See the
[official harness report](https://www.anthropic.com/engineering/harness-design-long-running-apps).

Anthropic's scientific-computing experiment follows the same pattern. The
human supplied a high-level parity and accuracy goal, while a maintained
`CLAUDE.md`, `CHANGELOG.md`, reference implementation, continuous tests, git
checkpoints, and an execution loop carried the operational detail. Its example
continuation prompt is essentially just “keep working until 0.1% accuracy,”
because the oracle and state already exist outside that prompt. See
[Anthropic's first-party write-up](https://www.anthropic.com/research/long-running-Claude).

### Stripe: tiny Slack triggers, industrial context and deterministic gates

Stripe's Minions are an even stronger warning against equating “short prompt”
with “small context.” A run begins from a Slack message or reaction and ends in
a human-reviewed, CI-passing pull request; Stripe reports more than 1,000 such
merged PRs per week with no human-written code. The public developer-keynote
example is a short outcome request: remove an obsolete public-preview badge,
plus a link to the relevant Slack discussion. See Stripe's
[keynote transcript](https://stripe.com/en-br/sessions/2026/developer-keynote)
and [Minions overview](https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents).

Behind that small trigger, Stripe supplies an isolated pre-warmed development
environment, the entire Slack thread and linked material, code search, internal
documentation and ticket context through MCP, repository-specific rule files,
local linting, selective tests from a multi-million-test corpus, CI feedback,
at most two repair rounds, and final human review. Stripe deliberately
interleaves probabilistic agent steps with deterministic git, lint, test, and
CI steps; the agent cannot decide to skip those gates. The implementation
details are in Stripe's
[Minions Part 2](https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents-part-2).

Stripe's separate integration benchmark reinforces the point: difficult
prompts worked in full repositories with databases and scripts, a terminal,
browser, Stripe search tools, test credentials, and deterministic API/UI
graders. The best runs averaged 63 turns. See Stripe's
[first-party benchmark report](https://stripe.com/blog/can-ai-agents-build-real-stripe-integrations).

Stripe's famous 3.7-million-line Flow-to-TypeScript migration is useful
engineering history but **not** evidence about AI prompting: it was published
in 2022, before this agent workflow. It succeeded through codemods, migration
tooling, type checking, and a coordinated cutover. It should not be cited as a
one-prompt autonomous refactor. See Stripe's
[migration report](https://stripe.dev/blog/migrating-to-typescript).

### Resulting rule

Use a short **goal prompt** when the repository already contains the durable
contract and a machine-checkable oracle. Do not paste the contract again. For
this project, `AGENTS.md`, the domain/evaluation documents, official upstream
kit, official evaluator, tests, benchmark results, git history, and development
data are the harness. The prompt should name the objective, authoritative
sources, non-negotiable boundary, and stop condition; Fable should discover the
rest just in time.

## Why 0.95 requires architectural headroom

The repository's current development result is documented as:

| Component | Current value |
|---|---:|
| TechnicalScore | 0.551831 |
| HitRate@10 | 0.656250 |
| MRR | 0.406937 |
| Efficiency | 0.508125 |

The evaluator computes:

```text
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

These facts are recorded in
[`benchmark_target_findings.md`](benchmark_target_findings.md) and
[`evaluation_config.json`](evaluation_config.json).

The current depth-50 Candidate Pool has observed development reachability of
0.825. Even under the deliberately generous assumptions that every reachable
target ranks first and Efficiency is 1.0, its score ceiling is:

```text
0.50 * 0.825 + 0.30 * 0.825 + 0.20 * 1.0 = 0.86
```

The older depth-300 union reached 0.90 pool recall, which gives the same loose,
optimistic ceiling of 0.92. Inverting that bound shows that a 0.95 score requires
candidate reachability of at least 0.9375 even while granting perfect
Efficiency and perfect rank for every reachable target:

```text
0.80 * reachability + 0.20 >= 0.95
reachability >= 0.9375
```

These are optimistic bounds. Misses receive turn 11 in the actual Efficiency
calculation, so misses also reduce Efficiency; the practical requirement can be
stricter. Therefore low-level reranker or weight optimization alone cannot
credibly reach 0.95. Fable must first test whether query construction, planning,
retrieval recall, candidate generation, or a replacement retrieval/ranking
architecture can lift reachability beyond the required range, and then improve
rank and conversion turn.

## Recommended copy-paste prompt

Use this in a **fresh Claude Code session with Claude Fable 5 at `xhigh`
effort**. This replaces the longer version: it intentionally leaves mechanics
to the repository and its executable evaluator.

```text
Raise this shopping agent's TechnicalScore from 0.551831 toward >=0.95. Read AGENTS.md, relevant repo docs, https://bit.ly/TikTokTechJam2026Info, and https://github.com/TechJam2026/techjam-conversational-search. Reproduce why Luna scored 0.2185 versus 0.6635 offline on the paired benchmark. Explore broadly: optimize or replace retrieval, ranking, planning, clarification, or the overall architecture. We have ample API credits for decisive experiments; track spend, and retain Luna calls only where they beat the local path. Use all 200 released public sessions for final evaluation. Never alter evaluation data or scoring, hard-code cases, or weaken tests. Continue until >=0.95 is reproduced or a quantified ceiling is proven; run the full suite and independently verify completion.
```

## Practical use

Do not add a second giant prompt after this one. If Fable asks for direction
that the repository can answer, tell it to inspect the relevant source and
continue. If it produces only a plan, reply with the short Fable-specific nudge
recommended by Anthropic:

```text
Continue and execute this end to end. Pause only for an irreversible action, a real scope change, or input only I can provide.
```

Review proposed evaluator or data-access changes manually. A model optimizing a
visible benchmark can overfit even without intending to, so a higher public
development score is evidence of progress, not proof of private-set
generalization.

## Conclusion

The best prompt is not the longest or shortest possible prompt. It is a compact
contract around a trustworthy optimization harness. Fable 5 should discover
details from the repository, measure the true bottleneck, and be allowed to
pivot architectures. The 0.95 target is far beyond the current measured
headroom and cannot be obtained by prompting alone.
