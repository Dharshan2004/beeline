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
`fixed-hybrid-v1`. Hard Constraints govern eligibility before fusion; missing,
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

The default policy min-max normalizes the three route scores per turn, combines
them with fixed weights (`structured=0.15`, `bm25=0.55`, `dense=0.30`), and
narrows the union to 30 candidates before returning at most ten. Constant-score
and missing routes have deterministic behavior. Run transparent baselines
through the same official-scoring wrapper with
`python3 -m tools.evaluate_retrieval --policy <name>`. The official evaluator
remains unchanged and uses the fixed policy through the default `Agent`. See
`docs/fixed_hybrid_fusion.md` for the exact policy and commands.

## Reranking Benchmark

Slice 07 selects the bundled cross-encoder and the deepest Candidate Pool it may
rerank, on the 160-session development split only. Every model and depth is
compared on one cached replay of identical base-route unions, so the comparison
never conflates a different pool with a different model. Runs are CPU-only and
network-disabled.

The frozen result is `cross-encoder/ms-marco-MiniLM-L-6-v2` at depth 50:
HitRate@10 0.600000, TechnicalScore 0.507240, p95 rerank latency 548.9 ms, and
an 800.4-second normalized 200-session wall-clock projection. Slice 08 activates
that choice through the live Agent; Slice 07 only records the decision.

```bash
python -m tools.fetch_model \
  --identity cross-encoder/ms-marco-MiniLM-L-2-v2 \
  --destination models/cross-encoder__ms-marco-MiniLM-L-2-v2 \
  --revision 1b5cd67b15209f24824c50370e0397743aa9b787
python -m tools.benchmark_reranker cache --output benchmarks/rerank_cache.jsonl
python -m tools.benchmark_reranker score \
  --identity cross-encoder/ms-marco-MiniLM-L-2-v2 \
  --cache benchmarks/rerank_cache.jsonl \
  --output benchmarks/rerank_MiniLM-L-2.json
python -m tools.benchmark_reranker summarize benchmarks/rerank_*.json \
  --output docs/reranker_benchmark.json
```

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
