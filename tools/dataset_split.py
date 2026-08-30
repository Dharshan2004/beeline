"""The deterministic development / locked-holdout split of the public sessions.

The PRD reserves 40 of the 200 public sessions as a scenario-stratified holdout
that is opened once, in the final human-reviewed slice. Every development
benchmark, weight search, and regression check runs on the remaining 160.

The split algorithm remains available for tests and audits. Production
development tools use the frozen public-set checksum and holdout identifiers
below so protected rows can be discarded before JSON deserialization.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Sequence


SPLIT_VERSION = "public-split-v1"
HOLDOUT_SEED = 20260829
FROZEN_PUBLIC_SET_SHA256 = "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579"
FROZEN_HOLDOUT_SAMPLE_IDS = frozenset({
    "public_0008", "public_0009", "public_0014", "public_0018",
    "public_0026", "public_0028", "public_0030", "public_0033",
    "public_0045", "public_0048", "public_0050", "public_0069",
    "public_0070", "public_0072", "public_0074", "public_0076",
    "public_0080", "public_0081", "public_0082", "public_0084",
    "public_0086", "public_0099", "public_0105", "public_0109",
    "public_0112", "public_0117", "public_0122", "public_0126",
    "public_0130", "public_0135", "public_0136", "public_0141",
    "public_0142", "public_0148", "public_0166", "public_0174",
    "public_0191", "public_0194", "public_0195", "public_0199",
})

# The official scenario distribution, applied to a 40-session holdout.
HOLDOUT_SCENARIO_COUNTS: dict[str, int] = {
    "buying": 16,
    "browsing": 16,
    "intent_override": 6,
    "boundary": 2,
}


def load_frozen_development_samples(path: str | Path) -> list[dict]:
    """Deserialize development rows without opening locked holdout payloads."""
    dataset_path = Path(path)
    with dataset_path.open("rb") as binary_handle:
        digest = hashlib.file_digest(binary_handle, "sha256").hexdigest()
    if digest != FROZEN_PUBLIC_SET_SHA256:
        raise ValueError(
            "public set checksum does not match the frozen development split"
        )

    samples: list[dict] = []
    line_count = 0
    with dataset_path.open(encoding="utf-8") as handle:
        for line_count, line in enumerate(handle, start=1):
            expected_id = f"public_{line_count:04d}"
            if expected_id in FROZEN_HOLDOUT_SAMPLE_IDS:
                continue
            sample = json.loads(line)
            if str(sample.get("sample_id")) != expected_id:
                raise ValueError(
                    "public set ordering does not match the frozen split manifest"
                )
            samples.append(sample)
    if line_count != 200 or len(samples) != 160:
        raise ValueError("public set shape does not match the frozen split manifest")
    return samples


def _sample_ids_by_scenario(samples: Iterable[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["scenario_type"]), []).append(
            str(sample["sample_id"]),
        )
    return {scenario: sorted(ids) for scenario, ids in grouped.items()}


def holdout_sample_ids(samples: Sequence[dict]) -> set[str]:
    """Return the locked holdout identifiers. Never open these before Slice 18."""
    selected: set[str] = set()
    for scenario, identifiers in sorted(_sample_ids_by_scenario(samples).items()):
        count = HOLDOUT_SCENARIO_COUNTS.get(scenario, 0)
        if count > len(identifiers):
            raise ValueError(
                f"scenario {scenario!r} has {len(identifiers)} sessions but the "
                f"holdout requires {count}"
            )
        shuffled = list(identifiers)
        random.Random(f"{HOLDOUT_SEED}:{scenario}").shuffle(shuffled)
        selected.update(shuffled[:count])
    return selected


def development_samples(samples: Sequence[dict]) -> list[dict]:
    """Return the 160 sessions that development work is allowed to read."""
    locked = holdout_sample_ids(samples)
    return [
        sample for sample in samples
        if str(sample["sample_id"]) not in locked
    ]


def holdout_samples(samples: Sequence[dict]) -> list[dict]:
    locked = holdout_sample_ids(samples)
    return [
        sample for sample in samples
        if str(sample["sample_id"]) in locked
    ]


def stratified_subset(
    samples: Sequence[dict],
    size: int,
    *,
    seed: int,
) -> list[dict]:
    """Return a reproducible scenario-proportional subset of ``samples``.

    Used when a benchmark is too expensive to run over every development
    session. Largest-remainder allocation keeps the scenario mix of the input,
    and the seeded per-scenario shuffle makes the choice reproducible without
    depending on file order.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if size >= len(samples):
        return list(samples)
    grouped = _sample_ids_by_scenario(samples)
    total = sum(len(ids) for ids in grouped.values())
    exact = {
        scenario: len(ids) * size / total for scenario, ids in grouped.items()
    }
    allocation = {scenario: int(value) for scenario, value in exact.items()}
    remaining = size - sum(allocation.values())
    for scenario in sorted(
        exact,
        key=lambda name: (-(exact[name] - allocation[name]), name),
    )[:remaining]:
        allocation[scenario] += 1

    selected: set[str] = set()
    for scenario, identifiers in sorted(grouped.items()):
        shuffled = list(identifiers)
        random.Random(f"{seed}:{scenario}").shuffle(shuffled)
        selected.update(shuffled[: allocation[scenario]])
    return [
        sample for sample in samples
        if str(sample["sample_id"]) in selected
    ]
