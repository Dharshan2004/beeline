# PRD: LLM-Centered Hybrid Shopping Agent

## Problem Statement

Customers often begin shopping with incomplete language, refine their needs over several turns, or replace an earlier preference with a new one. A keyword-only shopping search treats each turn in isolation, struggles with paraphrases, and can keep applying requirements that the customer has already withdrawn. For TikTok TechJam 2026 Track 4, this leads to late or missing placement of the hidden Target Product, especially in Browsing, Intent Override, and Boundary sessions.

The participant also needs a system that remains reproducible under the official evaluator. Connected LLMs may improve understanding, planning, and response quality, but network access may be disabled. External observability and retrieval services must not become scoring dependencies. The solution must therefore combine an LLM-centered experience with local retrieval, validated state, bounded cost, and a deterministic path that always returns a valid response.

## Solution

Build a multi-turn Shopping Agent whose semantic loop is led by an LLM when model access is available. On every turn, the LLM interprets the customer message, proposes changes to the Constraint State, selects narrow retrieval tools, and chooses whether a Clarification would be useful. Deterministic code validates those proposals, preserves superseded constraints, executes retrieval, applies the Fusion Policy, and guarantees a valid response.

The Shopping Agent retrieves candidates through structured catalog matching, BM25, and embedded dense vector search. It combines normalized route scores using weights selected by a controlled evaluation loop, then applies a bundled local cross-encoder before returning ten ranked products. A second LLM call may rerank or improve the customer-facing response only when ambiguity justifies its cost. If OpenAI, Langfuse, or the network is unavailable, local retrieval and deterministic response construction continue without error.

The official `Agent` interface remains the primary product contract. A single Docker image packages the agent, local indexes, selected models, configuration, and evaluation tooling for reproducible development and demonstration without making Docker or a separately hosted vector database mandatory for official scoring.

## User Stories

1. As a customer, I want the Shopping Agent to understand a direct product request, so that relevant products appear immediately.
2. As a customer, I want the Shopping Agent to understand paraphrases and everyday wording, so that I do not need to use catalog terminology.
3. As a customer, I want recommendations on every turn, so that I can make progress even while the agent asks a question.
4. As a customer, I want the Shopping Agent to remember active requirements across turns, so that I do not need to repeat myself.
5. As a customer, I want a later instruction to replace an incompatible earlier instruction, so that stale preferences do not distort my results.
6. As a customer, I want the Shopping Agent to retain the history of changed requirements, so that its behavior remains explainable.
7. As a customer, I want hard requirements treated differently from optional preferences, so that attractive alternatives do not violate essential needs.
8. As a customer, I want the Shopping Agent to recognize when I am browsing rather than ready to buy, so that it does not narrow too aggressively.
9. As a customer, I want the Shopping Agent to revise its view of my Session Mode as the conversation develops, so that early ambiguity does not lock me into the wrong strategy.
10. As a customer, I want useful Clarifications rather than repeated generic questions, so that each turn has a clear purpose.
11. As a customer, I want the Shopping Agent to stop asking about an attribute after I say I have no preference, so that Boundary Responses are respected.
12. As a customer, I want a new product type such as slippers to supersede an earlier request for shoes, so that Intent Overrides take effect immediately.
13. As a customer, I want recommendations ranked from strongest to weakest, so that the most promising options are easiest to inspect.
14. As a customer, I want recommendation text grounded in the catalog and active constraints, so that explanations remain faithful.
15. As a customer, I want aggregate profile preferences used only when they improve relevance, so that personalization does not overrule my current request.
16. As a customer, I want the Shopping Agent to keep working during a model or network failure, so that the conversation does not end with an error.
17. As an evaluator, I want the Shopping Agent to implement the required reset and response operations, so that it can run in the official harness.
18. As an evaluator, I want every response to contain a string message, an allowed `ask_attribute`, and ordered catalog-valid recommendations, so that scoring is reliable.
19. As an evaluator, I want the first ten unique valid recommendations to represent the agent's best ordering, so that HitRate@10 and MRR measure the intended ranking.
20. As an evaluator, I want Intent Override sessions to apply the new intent before conversion, so that the scenario rules are respected.
21. As an evaluator, I want sessions isolated by `session_id`, so that one customer's state cannot affect another session.
22. As an evaluator, I want reported model usage to contain non-negative token counts, so that feasibility can be assessed.
23. As an evaluator, I want the agent to operate without undeclared external services, so that final scoring is reproducible under restricted infrastructure.
24. As an evaluator, I want the agent to remain functional with network access disabled, so that connected intelligence is optional rather than required for validity.
25. As a participant, I want structured matching, BM25, and dense search to contribute independent evidence, so that the system handles both exact and semantic requests.
26. As a participant, I want Fusion Policy weights selected through an evaluation loop, so that route contributions are evidence-based rather than guessed.
27. As a participant, I want simple single-route and fixed-fusion baselines, so that the learned fusion improvement is measurable.
28. As a participant, I want per-scenario metrics and regression limits, so that an overall gain does not hide a major failure in a smaller scenario.
29. As a participant, I want a scenario-stratified development split and untouched holdout, so that tuning is less likely to overfit the public sessions.
30. As a participant, I want local embedding and reranking models selected using quality, latency, memory, and image-size evidence, so that accuracy remains practical.
31. As a participant, I want candidate depths and Fusion Policy weights frozen before the holdout is opened, so that the final result remains credible.
32. As a participant, I want the LLM to lead semantic decisions when available, so that the product behaves as a genuine agent rather than a fixed search script.
33. As a participant, I want deterministic validation around LLM decisions, so that invalid state changes or malformed outputs cannot break scoring.
34. As a participant, I want a provider-neutral model adapter, so that model selection can change without rewriting the Shopping Agent.
35. As a participant, I want the cheaper model compared with a stronger quality reference, so that routine evaluation does not spend more than necessary.
36. As a participant, I want a normal connected-development budget of $50, a warning at $40, and an absolute stop at $600, so that experiments cannot consume more than the authorized OpenAI credit.
37. As a participant, I want the second LLM call used only when it is likely to improve an ambiguous result, so that latency and cost remain controlled.
38. As a participant, I want local Constraint State to remain authoritative across model calls, so that provider conversation state is never the only copy.
39. As a participant, I want model, prompt, index, catalog, and Fusion Policy versions recorded together, so that a scored run can be reproduced.
40. As a developer, I want indexes prepared ahead of time and loaded once, so that individual turns do not pay indexing cost.
41. As a developer, I want dense search embedded in the agent process, so that no vector-service port or sidecar is required.
42. As a developer, I want route outputs cached during fusion experiments, so that weight searches do not repeatedly run expensive retrieval.
43. As a developer, I want failures classified by cause, so that model, validation, retrieval, and observability problems can be diagnosed separately.
44. As a developer, I want Langfuse traces grouped by evaluation session, so that a complete multi-turn strategy can be inspected.
45. As a developer, I want tracing to record structured decisions, counts, timings, usage, and fallback reasons without private chain-of-thought, so that observability is useful and safe.
46. As a developer, I want Langfuse export to fail open, so that missing credentials or telemetry outages never affect recommendations.
47. As a developer, I want deterministic fake model responses at the model boundary, so that agent behavior can be tested without spending API credit.
48. As a developer, I want network-disabled and invalid-model-output tests, so that the fallback path is continuously verified.
49. As a judge, I want a documented multi-turn demonstration including an Intent Override, so that the architecture's main differentiator is visible.
50. As a judge, I want architecture, model selection, latency, cost, limitations, and team contributions disclosed, so that the submission can be assessed beyond its raw score.
51. As a judge, I want the learned Fusion Policy compared against transparent baselines, so that the retrieval claim is supported by evidence.
52. As a judge, I want one reproducible container command and one official-harness command, so that the complete system can be evaluated easily.

## Implementation Decisions

- Preserve the participant starter repository and its required `Agent.reset(session_id, user_profile)` and `Agent.respond(session_id, user_message, turn, top_k)` contract.
- Keep Constraint State in memory and isolate it by `session_id`. Store raw wording, normalized attribute and value, hard-or-soft classification, source turn, confidence, and active, superseded, or dismissed status.
- Treat Intent Overrides as explicit state transitions. Never delete earlier evidence; deactivate superseded constraints and make the new constraint authoritative.
- Treat Boundary Responses as dismissed attributes that should not be requested again unless the customer later reintroduces them.
- Re-evaluate Session Mode as buying, browsing, or uncertain on every turn. Session Mode changes route weights and Clarification behavior but does not disable retrieval routes.
- Use an LLM-centered loop when a model is available. The first model call interprets the turn, proposes validated state changes, selects retrieval tools, and proposes a Clarification.
- Expose only narrow agent tools for structured lookup, BM25 search, dense search, and local reranking. Do not grant shell, web, arbitrary code execution, or catalog mutation capabilities.
- Require structured model output. Deterministic validation rejects unknown attributes, impossible mutations, invalid tool calls, malformed output, and recommendations outside the frozen catalog.
- Permit at most one bounded retry for model timeout or invalid output before deterministic takeover.
- Make the second model call conditional on ambiguity or expected ranking and response benefit. Otherwise finish with the local cross-encoder and deterministic response builder.
- Use a provider-neutral LLM adapter with OpenAI as the initial connected provider. Compare a stronger quality-reference model with a lower-cost high-volume model and keep the selected model configurable and pinned.
- Keep local Constraint State and recent turn history authoritative. Provider-side conversation identifiers and prompt caching may optimize calls but cannot become the only stored context.
- Implement three independent Retrieval Routes: deterministic structured catalog evidence, SQLite FTS5 BM25, and dense semantic retrieval.
- Use embedded Qdrant Local Mode for dense retrieval. The official path requires no listening port, network connection, or separately started vector service.
- Build versioned BM25 and vector indexes ahead of time, associate them with a catalog checksum, bundle them for scoring, and load them once during agent initialization.
- Benchmark compact local embedding and cross-encoder candidates using retrieval quality, latency, memory use, and packaged size. Bundle the selected models and prohibit runtime downloads in the scoring path.
- Retrieve up to 100 candidates per route, normalize scores over the union for the current turn, fuse them to 30 candidates, locally rerank, and return ten recommendations. Keep depths configurable during development and freeze them before holdout evaluation.
- Begin with one global non-negative Fusion Policy. Select weights through a simplex grid search against the official TechnicalScore using cached route outputs.
- Compare learned fusion with every single Retrieval Route and fixed Reciprocal Rank Fusion. Reject a candidate policy if a scenario's HitRate@10 drops by more than five percentage points against the agreed simple baseline.
- Split the 200 public sessions into 160 development sessions and a 40-session scenario-stratified locked holdout matching the official 40/40/15/5 scenario distribution.
- Tune on the development set, refine only stable weight plateaus, freeze configuration, and open the holdout once. Introduce observable state-specific weights only if repeated development splits and the untouched holdout support them.
- Return ranked recommendations on every turn, including turns that ask a Clarification. Use `ask_attribute` as the evaluator-facing question control.
- Choose a Clarification only when the expected information is useful, avoid dismissed attributes, and remain compatible with Buying, Browsing, Intent Override, and Boundary behavior.
- Guarantee that deterministic code can always produce a schema-valid message, allowed `ask_attribute`, ten-or-fewer unique catalog-valid recommendations, and non-negative usage counts.
- Activate deterministic fallback when model credentials are missing, the network is unavailable, a deadline is exceeded, structured output is invalid, a state proposal is rejected, or the connected cost budget is exhausted.
- Use $50 as the normal connected-development budget, warn at $40, require review before exceeding $50, and enforce $600 as the absolute programmatic ceiling. Exhausted budgets switch the system to local operation.
- Add Langfuse tracing for sessions and nested turn operations covering interpretation, retrieval routes, fusion, Clarification, reranking, response, and fallback.
- Record structured decisions, confidence, reason codes, candidate counts, timings, model usage, configuration identity, and failure causes. Do not request or store private chain-of-thought, credentials, full catalog records, or raw user profiles.
- Keep tracing outside the latency-critical result path. Missing Langfuse credentials, export failures, or lost network access silently disable telemetry without changing the result.
- Use one versioned runtime manifest for catalog and index checksums, model versions, prompt versions, Fusion Policy weights, route depths, timeouts, feature flags, observability state, and budget limits.
- Package the complete reproducible system in one Docker image while preserving direct execution through the official Python interface. An optional development-only hosted Qdrant profile may exist but cannot be required.
- Target an initial hybrid HitRate@10 of at least 0.60 and a submission target of at least 0.80, with MRR of at least 0.65, mean turns to conversion no greater than 4, and zero uncaught exceptions or invalid responses.
- Produce the official source submission, setup and reproduction instructions, short technical report, model and cost disclosure, limitations, team contributions, and one multi-turn demonstration.

## Testing Decisions

- Prefer the official evaluator as the highest test seam because it verifies the public `Agent` contract, the deterministic customer policy, exact catalog matching, ranking metrics, turn progression, and scenario behavior together.
- Test external behavior rather than private implementation. A good test provides catalog data, session inputs, model outcomes, or dependency failures and asserts returned state-independent facts: valid responses, ranking order, state-visible behavior, metric changes, and graceful fallback.
- Extend the existing evaluator tests, which already establish prior art for recommendation normalization, miss handling, hidden-field derivation, and end-to-end evaluation with a small fake Agent.
- Add public-contract tests for reset-before-response behavior, session isolation, turns 1 through 10, top-ten ordering, duplicate removal, catalog validity, allowed Clarification attributes, and non-negative model usage.
- Add table-driven conversation tests for hard constraints, Soft Preferences, Browsing progression, Intent Overrides, repeated overrides, Boundary Responses, contradictions, and empty or vague messages.
- Test the LLM adapter with deterministic fake responses for valid plans, invalid schemas, unknown tools, rejected state transitions, timeouts, retries, budget exhaustion, and provider unavailability.
- Test each Retrieval Route against a small fixed catalog, then test fusion using cached route-score fixtures. Assert ranking behavior and configuration handling rather than internal function calls.
- Benchmark each compact embedding and cross-encoder candidate under the same data and record quality, latency, memory, and artifact-size results before selecting the bundled model.
- Run single-route and fixed-fusion baselines before learned fusion. Evaluate candidate Fusion Policies on the 160-session development set with scenario-specific metrics and the five-point regression guardrail.
- Freeze the selected configuration before running the 40-session locked holdout. Do not use holdout results for repeated tuning.
- Add an end-to-end network-disabled run that starts from packaged assets, completes official sessions, and emits valid responses without OpenAI, Langfuse, or a hosted Qdrant process.
- Add Langfuse failure tests covering missing credentials, connection refusal, timeout, queue failure, and shutdown flushing. Assert that result content and scoring behavior are unchanged.
- Add cost tests using synthetic usage data to verify the $40 warning, $50 review boundary, $600 stop, and deterministic takeover.
- Add startup tests for catalog, index, model, prompt, and runtime-manifest checksum mismatches so incompatible artifacts fail clearly before a scored session.
- Add a Docker smoke test that imports the required Agent, loads bundled assets, runs a representative multi-turn Intent Override session, and exits successfully.
- Treat zero uncaught exceptions and zero invalid responses as release gates. Report HitRate@10, MRR, mean turns to conversion, efficiency, TechnicalScore, latency, token use, and per-scenario metrics for every release candidate.

## Out of Scope

- Modifying the frozen catalog or official evaluator.
- Reconstructing private labels, hidden intent cards, raw purchase histories, or organizer-only data.
- Real purchases, checkout, payments, inventory management, or retailer integrations.
- A mandatory user interface; the required deliverable is the Shopping Agent interface and demonstration.
- Multimodal image understanding or image-based retrieval.
- Training a foundation model or full cross-encoder from scratch.
- Requiring OpenAI, Langfuse, a hosted vector database, privileged host access, or any undeclared external service for official scoring.
- Making per-scenario routing decisions from hidden evaluator labels.
- Repeatedly tuning against the locked holdout or optimizing solely for public-session identifiers.
- Storing private chain-of-thought, secrets, raw user profiles, or the full catalog in observability payloads.
- Production-grade distributed session persistence, multi-region deployment, or infrastructure-heavy vector databases.

## Further Notes

- The weak BM25 starter baseline reports HitRate@10 of 0.125, MRR of 0.068034, mean turns to conversion of 9.81, and TechnicalScore of 0.10671 on 200 public sessions. Improvements must be measured against this reproducible starting point.
- The official private evaluation contains 800 undisclosed sessions with the same published scenario mix. Architecture and tuning choices must favor stable scenario-level gains over narrow public-set fitting.
- Research supporting score fusion and holdout discipline is captured by the agreed architecture: learned sparse-dense fusion can outperform fixed rank fusion, while repeated selection on limited data creates optimistic estimates. The PRD therefore combines transparent baselines, constrained global weights, scenario guardrails, and a one-time locked holdout.
- Langfuse is a development and runtime observability aid, not part of the scoring dependency chain.
- The absolute $600 connected-model limit represents authorization capacity, not a spending target. Normal development should remain within $50 unless reviewed.
- Architecture terminology and durable decisions are governed by the project glossary and architecture decision records.
