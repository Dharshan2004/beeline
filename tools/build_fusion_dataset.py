"""Build and replay the deterministic Slice 09 fusion-training artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Iterable, Sequence

from evaluator.local_evaluator import catalog_index, evaluate, materialize_hidden_fields
from retrieval.fusion import LEGACY_FIXED_WEIGHTS, ROUTE_NAMES, FixedFusionPolicy
from retrieval.manifest import directory_sha256
from retrieval.reranker import (
    DEFAULT_RERANKER_DIR,
    DEFAULT_RERANKER_IDENTITY,
    FROZEN_RERANK_DEPTH,
    RERANKER_CANDIDATES,
    CrossEncoderReranker,
    order_by_scores,
)
from starter.agent import Agent
from tools.benchmark_reranker import _network_disabled, session_metrics
from tools.dataset_split import (
    FROZEN_HOLDOUT_SAMPLE_IDS,
    FROZEN_PUBLIC_SET_SHA256,
    SPLIT_VERSION,
    load_frozen_development_samples,
)


ARTIFACT_VERSION = "fusion-training-v1"
MAX_TRAINING_POOL_DEPTH = 300
DEVELOPMENT_SCENARIO_COUNTS = {
    "boundary": 8,
    "browsing": 64,
    "buying": 64,
    "intent_override": 24,
}
FROZEN_DEVELOPMENT_SAMPLE_IDS = {
    f"public_{index:04d}" for index in range(1, 201)
}.difference(FROZEN_HOLDOUT_SAMPLE_IDS)
FROZEN_TURN_COUNT = 908
FROZEN_TRAJECTORY_SHA256 = "a0cbf2bdb36bf866cfb523f594e6b1b5b1add3afca363fe9b5e26c7148849178"
FROZEN_CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
FROZEN_DENSE_STORE_SHA256 = "19cf21ba7063f8bfce474207fbe5126fded7afa00cfc4e9a7e7a010f4d891c9a"
FROZEN_EMBEDDING_FINGERPRINT = "ca8f4c28964cbcb918cda59f5e5aeb6b8f18f40e76c617babdcf82f65179131d"
FROZEN_RERANKER_DIRECTORY_SHA256 = "92df7334846cb53cf56721bf1c44f5f8dd9f7068b818940ef9b5646c6333e502"


class FusionDatasetError(ValueError):
    """The fusion-training artifact is incomplete, stale, or inconsistent."""


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def artifact_digest(records: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_path(path: Path) -> Path:
    return Path(str(path) + ".manifest.json")


def validate_training_records(
    records: Sequence[dict],
    *,
    expected_scenarios: dict[str, int] = DEVELOPMENT_SCENARIO_COUNTS,
    expected_session_count: int = 160,
    holdout_ids: set[str] | frozenset[str] = FROZEN_HOLDOUT_SAMPLE_IDS,
) -> None:
    if not records:
        raise FusionDatasetError("fusion-training artifact contains no turns")
    sessions: dict[str, str] = {}
    seen_turns: set[tuple[str, int]] = set()
    required = {
        "sample_id", "scenario_type", "turn", "target", "hit_eligible",
        "query", "planning", "route_candidates", "candidate_pool", "frozen_candidate_pool",
        "reranker_scores", "response_pool",
    }
    for index, record in enumerate(records, start=1):
        missing = sorted(required.difference(record))
        if missing:
            raise FusionDatasetError(
                f"turn record {index} is incomplete; missing {', '.join(missing)}"
            )
        sample_id = str(record["sample_id"])
        scenario = str(record["scenario_type"])
        if sample_id in holdout_ids:
            raise FusionDatasetError(f"locked holdout session {sample_id} is present")
        if sample_id in sessions and sessions[sample_id] != scenario:
            raise FusionDatasetError(f"session {sample_id} changes scenario type")
        sessions[sample_id] = scenario
        key = (sample_id, int(record["turn"]))
        if key[1] <= 0:
            raise FusionDatasetError(f"turn number must be positive for {sample_id}")
        if key in seen_turns:
            raise FusionDatasetError(f"duplicate turn {sample_id}/{record['turn']}")
        seen_turns.add(key)
        route_candidates = record["route_candidates"]
        if set(route_candidates) != set(ROUTE_NAMES):
            raise FusionDatasetError("route candidates must define every Retrieval Route")
        planning = record["planning"]
        if not isinstance(planning, dict) or not {
            "source", "state_revision", "retrieval_tools"
        }.issubset(planning):
            raise FusionDatasetError(f"planning metadata is incomplete for {sample_id}/{record['turn']}")
        for route_name, candidates in route_candidates.items():
            identifiers: set[str] = set()
            if len(candidates) > 100:
                raise FusionDatasetError(f"{route_name} exceeds its deterministic route cap")
            for candidate in candidates:
                if not {
                    "parent_asin", "raw_score", "normalized_score"
                }.issubset(candidate):
                    raise FusionDatasetError(
                        f"route candidate is incomplete for {sample_id}/{record['turn']}"
                    )
                identifier = str(candidate["parent_asin"])
                raw_score = float(candidate["raw_score"])
                normalized_score = float(candidate["normalized_score"])
                if (
                    not identifier
                    or identifier in identifiers
                    or not isfinite(raw_score)
                    or not isfinite(normalized_score)
                    or not 0.0 <= normalized_score <= 1.0
                ):
                    raise FusionDatasetError(
                        f"route candidate is invalid for {sample_id}/{record['turn']}"
                    )
                identifiers.add(identifier)
        pool = [str(identifier) for identifier in record["candidate_pool"]]
        if len(pool) > MAX_TRAINING_POOL_DEPTH or len(pool) != len(set(pool)):
            raise FusionDatasetError(f"candidate pool for {sample_id}/{record['turn']} is not unique")
        complete_union = reconstruct_fused_pool(
            record,
            {
                "fusion_and_retrieval": {
                    "weights": {"structured": 0.15, "bm25": 0.55, "dense": 0.3},
                    "fused_candidate_depth": MAX_TRAINING_POOL_DEPTH,
                }
            },
        )
        if pool != complete_union:
            raise FusionDatasetError(
                f"candidate pool is not the complete route union for {sample_id}/{record['turn']}"
            )
        frozen_pool = [str(identifier) for identifier in record["frozen_candidate_pool"]]
        if frozen_pool != pool[:FROZEN_RERANK_DEPTH]:
            raise FusionDatasetError(
                f"frozen Candidate Pool is not the depth-{FROZEN_RERANK_DEPTH} prefix for {sample_id}/{record['turn']}"
            )
        scored = record["reranker_scores"]
        if any(not {"parent_asin", "score", "is_target"}.issubset(item) for item in scored):
            raise FusionDatasetError(f"reranker score is incomplete for {sample_id}/{record['turn']}")
        if [str(item.get("parent_asin")) for item in scored] != pool:
            raise FusionDatasetError(
                f"reranker scores do not align with the Candidate Pool for {sample_id}/{record['turn']}"
            )
        if any(not isfinite(float(item.get("score"))) for item in scored):
            raise FusionDatasetError(f"reranker score is invalid for {sample_id}/{record['turn']}")
        target = str(record["target"])
        if any(bool(item.get("is_target")) != (str(item["parent_asin"]) == target) for item in scored):
            raise FusionDatasetError(f"target labels are inconsistent for {sample_id}/{record['turn']}")
        response_pool = [str(identifier) for identifier in record["response_pool"]]
        if len(response_pool) != len(frozen_pool) or set(response_pool) != set(frozen_pool):
            raise FusionDatasetError(f"response pool is not a Candidate Pool permutation for {sample_id}/{record['turn']}")

    if len(sessions) != expected_session_count:
        raise FusionDatasetError(
            f"expected {expected_session_count} development sessions, found {len(sessions)}"
        )
    actual_scenarios = dict(sorted(Counter(sessions.values()).items()))
    if actual_scenarios != dict(sorted(expected_scenarios.items())):
        raise FusionDatasetError(
            f"scenario counts do not match the frozen split: {actual_scenarios}"
        )
    if (
        expected_session_count == 160
        and expected_scenarios == DEVELOPMENT_SCENARIO_COUNTS
        and set(sessions) != FROZEN_DEVELOPMENT_SAMPLE_IDS
    ):
        raise FusionDatasetError("development session identities do not match the frozen split")
    if expected_session_count == 160 and expected_scenarios == DEVELOPMENT_SCENARIO_COUNTS:
        if len(records) != FROZEN_TURN_COUNT:
            raise FusionDatasetError(
                f"expected frozen {FROZEN_TURN_COUNT}-turn trajectory, found {len(records)}"
            )
        ordered_turns = [(str(record["sample_id"]), int(record["turn"])) for record in records]
        if _json_digest(ordered_turns) != FROZEN_TRAJECTORY_SHA256:
            raise FusionDatasetError("turn trajectory does not match the frozen development replay")
        by_session: dict[str, list[int]] = {}
        for sample_id, turn in ordered_turns:
            by_session.setdefault(sample_id, []).append(turn)
        if any(turns != list(range(1, len(turns) + 1)) for turns in by_session.values()):
            raise FusionDatasetError("per-session turns are incomplete or out of order")


def validate_current_identities(configuration: dict, identities: dict) -> None:
    """Match recorded provenance to the exact current frozen training build."""
    reranker_identity = identities.get("reranker", {})
    fusion_identity = configuration.get("fusion_and_retrieval", {})
    catalog_identity = configuration.get("catalog", {})
    dense_identity = identities.get("dense_index_and_model", {})
    dense_manifest = dense_identity.get("manifest") or {}
    embedding_model = dense_manifest.get("embedding_model") or {}
    vector_store = dense_manifest.get("vector_store") or {}
    if (
        reranker_identity.get("identity") != DEFAULT_RERANKER_IDENTITY
        or reranker_identity.get("revision")
        != RERANKER_CANDIDATES[DEFAULT_RERANKER_IDENTITY].revision
        or reranker_identity.get("directory_sha256")
        != FROZEN_RERANKER_DIRECTORY_SHA256
    ):
        raise FusionDatasetError("reranker identity is stale or incomplete")
    if (
        fusion_identity.get("policy_version") != "fixed-hybrid-v1"
        or fusion_identity.get("fused_candidate_depth") != FROZEN_RERANK_DEPTH
        or fusion_identity.get("reranker_revision")
        != RERANKER_CANDIDATES[DEFAULT_RERANKER_IDENTITY].revision
        or fusion_identity.get("weights")
        != {"structured": 0.15, "bm25": 0.55, "dense": 0.3}
    ):
        raise FusionDatasetError("Fusion Policy configuration is stale")
    if (
        catalog_identity.get("sha256") != FROZEN_CATALOG_SHA256
        or identities.get("catalog_sha256") != FROZEN_CATALOG_SHA256
        or (dense_manifest.get("catalog") or {}).get("file_sha256")
        != FROZEN_CATALOG_SHA256
    ):
        raise FusionDatasetError("catalog identity is stale or incomplete")
    if (
        dense_identity.get("status") != "available"
        or embedding_model.get("fingerprint_sha256")
        != FROZEN_EMBEDDING_FINGERPRINT
        or vector_store.get("directory_sha256") != FROZEN_DENSE_STORE_SHA256
    ):
        raise FusionDatasetError("dense index or embedding identity is stale")


def reconstruct_fused_pool(record: dict, configuration: dict | None = None) -> list[str]:
    fusion = (configuration or {}).get("fusion_and_retrieval", {})
    weights = fusion.get("weights") or {
        "structured": 0.15,
        "bm25": 0.55,
        "dense": 0.3,
    }
    depth = int(fusion.get("fused_candidate_depth", FROZEN_RERANK_DEPTH))
    fused: dict[str, float] = {}
    for route_name in ROUTE_NAMES:
        weight = float(weights[route_name])
        for candidate in record["route_candidates"][route_name]:
            identifier = str(candidate["parent_asin"])
            fused[identifier] = fused.get(identifier, 0.0) + (
                weight * float(candidate["normalized_score"])
            )
    return [
        identifier
        for identifier, _score in sorted(
            fused.items(),
            key=lambda item: (-item[1], item[0]),
        )[:depth]
    ]


def reconstruct_response_pool(record: dict, configuration: dict | None = None) -> list[str]:
    fused_pool = reconstruct_fused_pool(record, configuration)
    reranker_scores = {
        str(item["parent_asin"]): float(item["score"])
        for item in record["reranker_scores"]
    }
    try:
        scores = [reranker_scores[identifier] for identifier in fused_pool]
    except KeyError as error:
        raise FusionDatasetError(
            f"reranker score is missing for fused candidate {error.args[0]}"
        ) from error
    return order_by_scores(fused_pool, scores)


def validate_reconstruction(records: Sequence[dict], configuration: dict) -> None:
    for record in records:
        fused_pool = reconstruct_fused_pool(record, configuration)
        if fused_pool != [str(item) for item in record["frozen_candidate_pool"]]:
            raise FusionDatasetError(
                f"cached route scores do not reconstruct fusion for {record['sample_id']}/{record['turn']}"
            )
        response_pool = reconstruct_response_pool(record, configuration)
        if response_pool != [str(item) for item in record["response_pool"]]:
            raise FusionDatasetError(
                f"cached reranker scores do not reconstruct response for {record['sample_id']}/{record['turn']}"
            )


def replay_records(records: Sequence[dict], configuration: dict | None = None) -> dict:
    ranked = {
        (str(record["sample_id"]), int(record["turn"])):
            reconstruct_response_pool(record, configuration)[:10]
        for record in records
    }
    metrics = session_metrics(records, ranked)
    # Match the official evaluator exactly: its published aggregate components
    # are rounded before TechnicalScore is calculated.
    metrics["recommended_technical_score"] = round(
        0.50 * metrics["hit_rate_at_10"]
        + 0.30 * metrics["mrr"]
        + 0.20 * metrics["efficiency"],
        6,
    )
    return metrics


def write_artifact(
    records: Sequence[dict],
    path: str | Path,
    *,
    configuration: dict,
    expected_scenarios: dict[str, int] = DEVELOPMENT_SCENARIO_COUNTS,
    expected_session_count: int = 160,
    holdout_ids: set[str] | frozenset[str] = FROZEN_HOLDOUT_SAMPLE_IDS,
    identities: dict | None = None,
) -> dict:
    validate_training_records(
        records,
        expected_scenarios=expected_scenarios,
        expected_session_count=expected_session_count,
        holdout_ids=holdout_ids,
    )
    validate_reconstruction(records, configuration)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    session_ids = sorted({str(record["sample_id"]) for record in records})
    identity_payload = identities or {}
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "split_version": SPLIT_VERSION,
        "public_set_sha256": FROZEN_PUBLIC_SET_SHA256,
        "session_count": len(session_ids),
        "turn_count": len(records),
        "scenario_counts": dict(sorted(expected_scenarios.items())),
        "session_ids_sha256": _json_digest(session_ids),
        "artifact_sha256": artifact_digest(records),
        "configuration": configuration,
        "configuration_sha256": _json_digest(configuration),
        "identities": identity_payload,
        "identities_sha256": _json_digest(identity_payload),
        "replay_metrics": replay_records(records, configuration),
        "artifact_path": str(output),
    }
    _manifest_path(output).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_artifact(
    path: str | Path,
    *,
    expected_scenarios: dict[str, int] = DEVELOPMENT_SCENARIO_COUNTS,
    expected_session_count: int = 160,
    holdout_ids: set[str] | frozenset[str] = FROZEN_HOLDOUT_SAMPLE_IDS,
    enforce_current_identities: bool = True,
) -> tuple[list[dict], dict]:
    artifact = Path(path)
    manifest_path = _manifest_path(artifact)
    if not artifact.is_file() or not manifest_path.is_file():
        raise FusionDatasetError("artifact and manifest must both exist")
    try:
        records = [
            json.loads(line)
            for line in artifact.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FusionDatasetError(f"artifact is unreadable: {error}") from error
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise FusionDatasetError("artifact version is stale")
    if manifest.get("split_version") != SPLIT_VERSION:
        raise FusionDatasetError("split version is stale")
    if manifest.get("public_set_sha256") != FROZEN_PUBLIC_SET_SHA256:
        raise FusionDatasetError("public-set identity is stale")
    if manifest.get("artifact_sha256") != artifact_digest(records):
        raise FusionDatasetError("artifact checksum does not match its manifest")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or manifest.get("configuration_sha256") != _json_digest(configuration):
        raise FusionDatasetError("configuration checksum does not match its manifest")
    identities = manifest.get("identities")
    if not isinstance(identities, dict) or manifest.get("identities_sha256") != _json_digest(identities):
        raise FusionDatasetError("identity checksum does not match its manifest")
    if enforce_current_identities:
        validate_current_identities(configuration, identities)
    validate_training_records(
        records,
        expected_scenarios=expected_scenarios,
        expected_session_count=expected_session_count,
        holdout_ids=holdout_ids,
    )
    validate_reconstruction(records, configuration)
    session_ids = sorted({str(record["sample_id"]) for record in records})
    if manifest.get("session_count") != len(session_ids):
        raise FusionDatasetError("manifest session count is incomplete")
    if manifest.get("turn_count") != len(records):
        raise FusionDatasetError("manifest turn count is incomplete")
    if manifest.get("scenario_counts") != dict(sorted(expected_scenarios.items())):
        raise FusionDatasetError("manifest scenario counts are stale")
    if manifest.get("session_ids_sha256") != _json_digest(session_ids):
        raise FusionDatasetError("session identity checksum does not match")
    replay = replay_records(records, configuration)
    if manifest.get("replay_metrics") != replay:
        raise FusionDatasetError("stored replay metrics do not match the artifact")
    return records, manifest


class RecordingReranker:
    """In-process scorer that retains exact scores for one deterministic build."""

    configured = True
    identity = DEFAULT_RERANKER_IDENTITY

    def __init__(self, model_dir: Path = DEFAULT_RERANKER_DIR) -> None:
        self.model_dir = model_dir
        self.revision = RERANKER_CANDIDATES[self.identity].revision
        self.scorer = CrossEncoderReranker(model_dir, identity=self.identity)
        self.calls: list[dict] = []

    def rerank(self, query: str, candidates: Sequence[str], documents: Sequence[str]) -> list[str]:
        pool = list(candidates)[:FROZEN_RERANK_DEPTH]
        scores = self.scorer.score(query, list(documents)[:len(pool)])
        self.calls.append({"query": query, "candidates": pool, "scores": scores})
        return order_by_scores(pool, scores)

    def metrics(self) -> dict:
        return {
            "status": "available",
            "identity": self.identity,
            "revision": self.revision,
            "depth": FROZEN_RERANK_DEPTH,
            "deadline_seconds": 1.5,
        }

    def close(self) -> None:
        pass


def build_dataset(arguments: argparse.Namespace) -> dict:
    samples = load_frozen_development_samples(arguments.dataset)
    catalog_ids, categories, products = catalog_index(arguments.catalog)
    with _network_disabled():
        reranker = RecordingReranker(Path(arguments.model_dir))
        agent = Agent(
            arguments.catalog,
            reranker=reranker,
            fusion_policy=FixedFusionPolicy(
                weights=LEGACY_FIXED_WEIGHTS,
                version="fixed-hybrid-v1",
            ),
            candidate_pool_depth=FROZEN_RERANK_DEPTH,
            trace_pool_depths=(FROZEN_RERANK_DEPTH, MAX_TRAINING_POOL_DEPTH),
        )
        try:
            result = evaluate(agent, samples, catalog_ids, categories, products)
            runtime_configuration = agent.get_runtime_configuration()
            # Artifact v1 predates the explicit normalizer/prompt digests. Keep
            # its historical configuration bytes reproducible after Slice 11
            # exposes those identities in the live runtime manifest.
            runtime_configuration["fusion_and_retrieval"].pop("normalizer", None)
            runtime_configuration["fusion_and_retrieval"].pop(
                "reranker_directory_sha256", None
            )
            runtime_configuration["planning"].pop("prompt_sha256", None)
            runtime_configuration["reranker"].pop("directory_sha256", None)
            traces = agent.get_candidate_traces()
        finally:
            agent.close()

    flat_traces = [entry for session_entries in traces.values() for entry in session_entries]
    if len(flat_traces) != len(reranker.calls):
        raise FusionDatasetError("Agent traces and reranker score calls do not align")
    records: list[dict] = []
    call_index = 0
    with _network_disabled():
        for sample, session_entries in zip(samples, traces.values(), strict=True):
            target = str(sample["ground_truth"]["parent_asin"])
            _intent_card, behavior = materialize_hidden_fields(sample, products)
            override_turn = int((behavior.get("override") or {}).get("turn", 1))
            for trace in session_entries:
                call = reranker.calls[call_index]
                call_index += 1
                frozen_pool = list(trace["pools"][str(FROZEN_RERANK_DEPTH)])
                full_pool = list(trace["pools"][str(MAX_TRAINING_POOL_DEPTH)])
                if call["candidates"] != frozen_pool or full_pool[:len(frozen_pool)] != frozen_pool:
                    raise FusionDatasetError("reranker scores do not match the traced Candidate Pool")
                remaining = full_pool[len(frozen_pool):]
                scores = [
                    *call["scores"],
                    *reranker.scorer.score(
                        trace["query"],
                        agent.retrieval.rerank_documents(remaining),
                    ),
                ]
                records.append({
                    "sample_id": str(sample["sample_id"]),
                    "scenario_type": str(sample["scenario_type"]),
                    "turn": int(trace["turn"]),
                    "target": target,
                    "hit_eligible": (
                        sample["scenario_type"] != "intent_override"
                        or int(trace["turn"]) >= override_turn
                    ),
                    "query": trace["query"],
                    "planning": trace["planning"],
                    "route_candidates": trace["route_candidates"],
                    "candidate_pool": full_pool,
                    "frozen_candidate_pool": frozen_pool,
                    "reranker_scores": [
                        {
                            "parent_asin": parent_asin,
                            "score": score,
                            "is_target": parent_asin == target,
                        }
                        for parent_asin, score in zip(full_pool, scores, strict=True)
                    ],
                    "response_pool": list(trace["response_pool"]),
                })
                if call_index % 50 == 0:
                    print(
                        f"scored complete Candidate Pools for {call_index}/{len(flat_traces)} turns",
                        flush=True,
                    )

    identities = {
        "catalog_sha256": runtime_configuration["catalog"]["sha256"],
        "dense_index_and_model": runtime_configuration["dense_index_and_model"],
        "reranker": {
            "identity": reranker.identity,
            "revision": reranker.revision,
            "directory_sha256": directory_sha256(reranker.model_dir),
        },
    }
    manifest = write_artifact(
        records,
        arguments.output,
        configuration=runtime_configuration,
        identities=identities,
    )
    manifest["live_metrics"] = {
        key: value for key, value in result.items()
        if key not in {"sessions", "turn_latency"}
    }
    _manifest_path(Path(arguments.output)).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = Path(arguments.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def refresh_manifest(path: str | Path, report: str | Path) -> dict:
    """Recompute derived manifest metrics without retrieval or model loading."""
    artifact = Path(path)
    old_manifest = json.loads(_manifest_path(artifact).read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in artifact.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if old_manifest.get("artifact_sha256") != artifact_digest(records):
        raise FusionDatasetError("artifact checksum does not match its old manifest")
    manifest = write_artifact(
        records,
        artifact,
        configuration=old_manifest["configuration"],
        identities=old_manifest["identities"],
    )
    if "live_metrics" in old_manifest:
        manifest["live_metrics"] = old_manifest["live_metrics"]
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _manifest_path(artifact).write_text(rendered, encoding="utf-8")
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--catalog", default="data/catalog.jsonl")
    build.add_argument("--dataset", default="data/public_set.jsonl")
    build.add_argument("--model-dir", default=str(DEFAULT_RERANKER_DIR))
    build.add_argument("--output", default="benchmarks/fusion_training.jsonl")
    build.add_argument("--report", default="docs/fusion_training_dataset.json")
    replay = subparsers.add_parser("replay")
    replay.add_argument("artifact", nargs="?", default="benchmarks/fusion_training.jsonl")
    refresh = subparsers.add_parser("refresh-manifest")
    refresh.add_argument("artifact", nargs="?", default="benchmarks/fusion_training.jsonl")
    refresh.add_argument("--report", default="docs/fusion_training_dataset.json")
    arguments = parser.parse_args()
    if arguments.command == "build":
        output = build_dataset(arguments)
    elif arguments.command == "replay":
        records, manifest = load_artifact(arguments.artifact)
        output = {
            "artifact_sha256": manifest["artifact_sha256"],
            "configuration_sha256": manifest["configuration_sha256"],
            "metrics": replay_records(records, manifest["configuration"]),
        }
    else:
        output = refresh_manifest(arguments.artifact, arguments.report)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
