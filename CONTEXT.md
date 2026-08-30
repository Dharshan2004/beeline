# Conversational Shopping

This context describes the language used by the shopping agent and the official evaluation sessions it participates in.

## Language

**Shopping Agent**:
The participant-built system that converses with a customer and ranks catalog products within an evaluation session.
_Avoid_: Chatbot, search bot

**Target Product**:
The catalog product hidden by the evaluator that the Shopping Agent must place in its scored recommendations.
_Avoid_: Correct item, answer product

**Constraint**:
A customer requirement or preference that relates one attribute to one or more values and narrows or influences acceptable products. Multiple values use an explicit any-value or all-values relationship.
_Avoid_: Filter, slot value

**Hard Constraint**:
A requirement that a recommended product is expected to satisfy.
_Avoid_: Mandatory filter

**Soft Preference**:
A preference that improves a product's relevance without necessarily excluding alternatives.
_Avoid_: Optional filter

**Constraint State**:
The current, revisable account of a customer's active constraints, their origin, and whether they remain applicable.
_Avoid_: Memory, slot store

**Product Intent Constraint**:
A Constraint that applies only to the Target Product currently being described and does not carry across a broad Intent Override.
_Avoid_: Temporary constraint, product filter

**Session Constraint**:
A Constraint the customer explicitly establishes as applying across Target Product changes within the current evaluation session.
_Avoid_: Global constraint, permanent preference

**Product Intent**:
The cohesive description of one Target Product currently under consideration, including its Product Intent Constraints.
_Avoid_: Constraint group, search context

**Session Mode**:
The Shopping Agent's current view of whether the customer is ready to narrow toward a purchase, is exploring possibilities, or remains uncertain. A Session Mode may change as the conversation develops.
_Avoid_: Permanent intent class, traffic bucket

**Intent Override**:
A customer correction that explicitly replaces or withdraws an earlier Constraint or Product Intent. Mentioning a different product or value does not by itself establish an Intent Override.
_Avoid_: Preference update, changed slot

**Boundary Response**:
A customer response stating that a requested attribute has no useful preference and should not be asked again.
_Avoid_: Rejection, null answer

**Turn Plan**:
The complete proposed interpretation of one customer turn, including every intended Constraint State transition.
_Avoid_: Mutation list, model output

**Clarification**:
A question asked to obtain information expected to improve the next recommendation decision.
_Avoid_: Follow-up prompt

**Candidate Pool**:
The set of catalog products still under consideration for the current Constraint State.
_Avoid_: Search results, shortlist

**Route Candidate Set**:
The catalog products independently admitted by one Retrieval Route before evidence from different routes is combined.
_Avoid_: Route ranking, route output

**Deep Candidate Pool**:
The deduplicated union of every Route Candidate Set that remains eligible for local reranking.
_Avoid_: BM25-dense union, fused top 30

**Retrieval Route**:
A distinct method for producing or narrowing a Candidate Pool from the current customer evidence.
_Avoid_: Search engine, retriever

**Fusion Policy**:
The versioned rule that combines evidence from multiple Retrieval Routes into one product ordering.
_Avoid_: Magic weights, ensemble formula

**Conversion**:
The first evaluation turn on which the Target Product appears in the scored recommendations.
_Avoid_: Purchase, success

**Locked Holdout**:
The reserved evaluation sessions that cannot be executed, inspected, or used for quality, latency, or configuration decisions before the final human-reviewed release gate.
_Avoid_: Test split, timing set
