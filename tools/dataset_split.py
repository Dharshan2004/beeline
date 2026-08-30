"""The deterministic development / locked-holdout split of the public sessions.

The PRD reserves 40 of the 200 public sessions as a scenario-stratified holdout
that is opened once, in the final human-reviewed slice. Every development
benchmark, weight search, and regression check runs on the remaining 160.

The split is computed, not stored, so that it cannot drift between tools: given
the same public set it is always the same 40 sessions. Selection is by a seeded
shuffle of each scenario's sorted sample identifiers, so it does not depend on
file order and does not simply take the numerically lowest identifiers.
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence


SPLIT_VERSION = "public-split-v1"
HOLDOUT_SEED = 20260829

# The official scenario distribution, applied to a 40-session holdout.
HOLDOUT_SCENARIO_COUNTS: dict[str, int] = {
    "buying": 16,
    "browsing": 16,
    "intent_override": 6,
    "boundary": 2,
}


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
