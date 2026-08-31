# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Use Python 3.11.9, the exact runtime used for the reported builds, benchmarks,
and test results. Install the local dense-route dependencies and prepare its
reproducible assets as described under **Dense Retrieval Route**. If they are
absent or incompatible, the agent remains valid by falling back to its
standard-library lexical route.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Validated Intent Override Planning

Planning contract `shopping-turn-planner-v3` accepts provider-neutral structured
Turn Plans while local Constraint State remains authoritative. One versioned
Explicit Replacement Evidence classifier is shared by connected validation and
the deterministic interpreter. Attribute-level correction replaces only the
affected Constraint; Product Intent replacement requires explicit product-type
replacement or whole-intent withdrawal plus a distinct supported successor in
the same atomic Turn Plan. Ambiguous mentions, stale state, invalid tools,
invalid schemas, timeouts, and provider failures receive one bounded retry and
then deterministic takeover without partial mutation.

The model may select only Candidate Pool-producing Retrieval Routes:
`structured`, `bm25`, and `dense`. Local reranking is the frozen post-fusion
policy and cannot be selected, disabled, or bypassed by a Turn Plan. Slice 12 is
provider-neutral and its connected behavior is tested with deterministic fake
providers; the measured concrete OpenAI adapter belongs to Slice 14.

## Session Mode and Clarifications

Session policy `shopping-session-policy-v1` revises the current mode on every
turn as `buying`, `browsing`, or `uncertain`. Explicit current-turn evidence wins
over earlier mode and aggregate profile hints. The connected Turn Plan proposes a
mode, while deterministic evidence handles explicit browsing, purchase, and
uncertainty language and provides the offline fallback.

Clarifications are selected only from evaluator-allowed, catalog-supported
attributes that are not already active, dismissed by a Boundary Response, or
previously asked for the current Product Intent. The mode determines attribute
priority, catalog answer diversity supplies the usefulness gate, and safe
aggregate profile tags may only break ties between otherwise eligible questions.
Profile data never becomes a Constraint. A Clarification is emitted only with a
non-empty ranked recommendation list, and a Product Intent replacement starts a
fresh asked-attribute history without reviving dismissed attributes.

Run the planning and Intent Override contract tests without API credentials or
spending:

```bash
.venv/bin/python -m unittest -v tests.test_planning tests.test_session_policy tests.test_turn_interpreter
```

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
retrieval/                        Retrieval Routes and their local index artifacts
tools/                            one-time model fetch, splits, and benchmarks
docs/dense_index.md               dense index build, verification, and measured scale
docs/fixed_hybrid_fusion.md       fixed policy, score normalization, and baselines
docs/reranker_benchmark.md        cross-encoder and deep-pool depth decision gate
docs/reranker_benchmark.json      machine-readable benchmark summary
```

## Dense Retrieval Route

The dense route searches a versioned artifact built ahead of time and loaded once
at startup, using embedded Qdrant Local Mode and a bundled embedding model. It
requires no listening port, no separate vector service, and no runtime download.
It is a local Python support module for the Shopping Agent, not a website, UI, or
hosted search service. `starter.Agent` queries the route at depth 100 and
combines its candidates with independent structured and BM25 evidence through
the validated `pool-aware-global-v2` policy. Hard Constraints govern eligibility
before fusion; missing,
incompatible, or failed dense assets disable only that route without
invalidating the response.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dense.txt
python -m tools.fetch_model
python -m retrieval.build_dense_index --catalog data/catalog.jsonl --verify-load
```

Before benchmarking, verify that the same interpreter used for scoring can load
and query the dense route. This command exits nonzero if the route silently
falls back because a dependency, model, catalog checksum, or index is missing or
incompatible:

```bash
.venv/bin/python - <<'PY'
from starter.agent import Agent

agent = Agent("data/catalog.jsonl")
before = agent.get_dense_route_metrics()
assert before["status"] == "available", before
agent.reset("dense-readiness", {})
agent.respond("dense-readiness", "comfortable house shoes for cold floors", 1, 10)
after = agent.get_dense_route_metrics()
assert after["status"] == "available", after
assert after["query_count"] == 1, after
assert after["last_candidate_count"] > 0, after
print(after)
PY
```

Do not benchmark with the system `python3` after installing into `.venv`; use
`.venv/bin/python` or activate the environment first. A valid readiness result
has `status: available`, `disabled_reason: None`, and a positive candidate count.

With those runtime dependencies and ignored local assets prepared, exercise the
real MiniLM-to-Qdrant paraphrase fixture through the public Agent path with:

```bash
python -m unittest \
  tests.test_agent.AgentContractTest.test_real_local_dense_route_recovers_paraphrase_through_agent
```

The test is skipped in the dependency-free fallback environment because the
large model and derived index are intentionally not committed.

See `docs/dense_index.md` for the manifest contents, determinism guarantees,
mismatch behavior, and live route metrics.

## Fixed Hybrid Fusion

The default policy min-max normalizes the three route scores per turn and combines
them with the Slice 11 frozen weights (`structured=0.02`, `bm25=0.64`,
`dense=0.34`). The live
Agent keeps the strongest 50 fused candidates for local reranking, while the
transparent no-reranker baseline intentionally preserves its historical depth
of 30. Constant-score and missing routes have deterministic behavior. Run those
baselines through the same official-scoring wrapper with
`python3 -m tools.evaluate_retrieval --policy <name>`. The official evaluator
remains unchanged and uses the fixed policy through the default `Agent`. See
`docs/fixed_hybrid_fusion.md` for the exact policy and commands.

## Reranking Benchmark

Slice 07 selects the bundled cross-encoder and the deepest Candidate Pool it may
rerank, on the 160-session development split only. Every model and depth is
compared on one cached replay of identical Deep Candidate Pools, so the comparison
never conflates a different pool with a different model. Runs are CPU-only and
network-disabled.

The frozen result is `cross-encoder/ms-marco-MiniLM-L-6-v2` at depth 50:
HitRate@10 0.600000, TechnicalScore 0.507240, p95 rerank latency 548.9 ms, and
an 800.6-second normalized 200-session wall-clock projection. The live Agent now
activates that choice in one persistent worker process. It verifies the exact
model revision before startup, applies a 1.5-second absolute deadline per turn,
and permanently returns the fused ordering after startup failure, worker crash,
malformed output, or timeout. The model is loaded with local-only flags and is
never downloaded in the scoring path.

```bash
python -m tools.fetch_model \
  --identity cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --destination models/cross-encoder__ms-marco-MiniLM-L-6-v2 \
  --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

Run the selected configuration end to end on the 160 development sessions only:

```bash
.venv/bin/python -m tools.evaluate_live_reranker \
  --output docs/live_reranker_evaluation.json
```

That report records the official aggregate and per-scenario metrics, depth-50
pool recall, recall-to-hit conversion, reranker p50/p95 latency, full-run wall
time, the normalized 200-session projection, route readiness, and the frozen
configuration beside the fused-30 baseline. `--sessions N` selects a deterministic
scenario-stratified development subset for smoke tests.

The checked-in isolated 160-session live run reached HitRate@10 **0.637500**,
MRR **0.405035**, and TechnicalScore **0.539511**, compared with 0.543750,
0.368237, and 0.467096 for a fresh fused-30 run through the same evaluator.
Depth-50 pool recall was 0.775000 and recall-to-hit conversion was 0.822581.
The persistent worker completed all 908 reranks without fallback at 368.5 ms
p50 and 392.9 ms p95 reranker latency. Complete Agent turn latency was 594.5 ms
p50 and 819.3 ms p95, compared with 202.5 ms and 455.5 ms for fused-30. The
548.8-second development run projects to 686.0 seconds for 200 sessions, below
the frozen 900-second gate. See `docs/live_reranker_evaluation.json` for
scenario metrics, baseline timing, and exact configuration.
The local reranker reports zero tokens and has **$0 incremental model/API cost**;
it uses only local CPU inference after the one-time model fetch. The configuration
in this Slice 08 report records the catalog and index checksums, embedding and
reranker revisions, planning prompt and connected-provider identity, Fusion
Policy, candidate depths, feature states, timeout, and declared cost thresholds.
Later slices may extend this versioned runtime manifest, but scored reports must
continue carrying all fields that apply to their build.

## Replayable Fusion-Training Dataset

Slice 09 freezes the 160-session development trajectory into a deterministic
training artifact containing session/scenario metadata, planning source,
Constraint State revision, selected Retrieval Routes, raw and normalized route
scores, the exact depth-50 Candidate Pool, local reranker scores for the complete
up-to-300 base-route union, target labels, and the final response ordering. Build
and replay it with:

```bash
.venv/bin/python -m tools.build_fusion_dataset build \
  --output benchmarks/fusion_training.jsonl \
  --report docs/fusion_training_dataset.json
.venv/bin/python -m tools.build_fusion_dataset replay \
  benchmarks/fusion_training.jsonl
```

The 41 MB JSONL and its adjacent manifest remain ignored under `benchmarks/`;
regenerate them locally rather than committing a large derived asset. The
checked-in `docs/fusion_training_dataset.json` records the artifact,
configuration, session-ID, catalog, dense-index, embedding-model, and reranker
checksums. The frozen artifact contains 160 sessions and 908 turns with scenario
counts 8 boundary / 64 browsing / 64 buying / 24 Intent Override. Model-free
replay reconstructs fixed fusion and reranking from cached scores and reproduces
HitRate@10 0.637500, MRR 0.405035, and TechnicalScore 0.539511 in under two
seconds on the recorded build. Missing fields, locked-holdout IDs,
wrong session proportions or identities, altered bytes, stale split/policy/model
identities, and inconsistent target labels are rejected before training.

## Fusion Weight Training

Slice 10 searches global non-negative, sum-to-one route weights entirely from
the replayable Slice 9 artifact:

```bash
.venv/bin/python -m tools.train_fusion_policy \
  benchmarks/fusion_training.jsonl \
  --output docs/fusion_policy_training.json
```

The deterministic 66-point coarse simplex plus 91-point local refinement chose
`structured=0.00`, `bm25=0.68`, and `dense=0.32` at Candidate Pool depth 50.
Across all 160 development sessions it raises pool recall from 0.7750 to 0.8250
and TechnicalScore from 0.539511 to 0.552275. The report includes official and
scenario metrics, the current fused-30 control, selected-depth-30 and
complete-union comparisons, and
single-route/RRF controls. This is a training result, not yet the live policy;
Slice 11 must validate the weight region across folds before activation. See
`docs/fusion_policy_training.md` for the selection rule and interpretation.

## Frozen Pool-Aware Fusion Policy

Slice 11 validates all 91 locally refined candidates across four deterministic,
scenario-balanced development folds and activates the center of the supported
plateau: `structured=0.02`, `bm25=0.64`, and `dense=0.34` at rerank depth 50.
The Locked Holdout remains unopened.

The frozen policy reaches 0.825 pool recall, 0.65625 HitRate@10, 0.406937 MRR,
and 0.551831 TechnicalScore. No scenario violates the five-point HitRate@10
guardrail; Intent Override improves from 0.375 to 0.500. A complete live run
matches those cached metrics exactly, with 348 ms rerank p95 latency and a
568.3-second projected 200-session wall time. See
`docs/fusion_policy_validation.md` and `docs/fusion_policy_freeze.json` for the
fold evidence, gap closure, checksums, and full frozen runtime identity.

`tools/dataset_split.py` computes the development/holdout partition used by every
development benchmark. The locked 40-session holdout is opened once, in the final
human-reviewed slice; Slice 07 measures all 160 development sessions and only
projects the resulting runtime to 200 sessions. See `docs/reranker_benchmark.md`
for the method, the selection rule, and the measured result.

### Reproducibility and external assets

The official harness entry point remains:

```bash
python3 -m evaluator.local_evaluator
```

Large reproducible assets are deliberately excluded by `.gitignore` and must not
be committed:

| Asset or dependency | Source and purpose | Reproduction |
| --- | --- | --- |
| Frozen Amazon Reviews 2023 catalog | Organizer release; evaluator and retrieval corpus | Download and verify as described in **Download the Catalog** above |
| `sentence-transformers/all-MiniLM-L6-v2` (`1110a243…`) | Local 384-dimensional embedding model; no scoring-time API | `python -m tools.fetch_model` |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` (`81d1926f…`), `MiniLM-L-2-v2` (`1b5cd67b…`), and `MiniLM-L-6-v2` (`233902d2…`) | Local Slice 07 reranking candidates; no scoring-time API | `python -m tools.fetch_model --identity <id> --destination models/<dir> --revision <sha>` |
| Dense Qdrant index | Derived locally from the frozen catalog and pinned MiniLM model | `python -m retrieval.build_dense_index --catalog data/catalog.jsonl --verify-load` |
| Qdrant, PyTorch, Transformers, NumPy | Local dense-route runtime dependencies | `pip install -r requirements-dense.txt` |

The dense route makes no runtime network call, uses no hosted API, and binds no
listening port. `models/`, `artifacts/`, and `data/catalog.jsonl` are ignored;
only source code, manifests, public evaluation data, and reproduction instructions
belong in the repository. Token usage and any connected APIs introduced by later
slices must continue to be reported through the official response schema.

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
