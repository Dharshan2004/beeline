from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence


ROUTE_NAMES = ("structured", "bm25", "dense")
POLICY_NAMES = ("fixed", "rrf", *ROUTE_NAMES)


class FusionPolicy(Protocol):
    version: str
    candidate_limit: int

    def rank(
        self,
        route_scores: Mapping[str, Sequence[tuple[str, float]]],
    ) -> list[str]: ...


def _normalized_scores(candidates: Sequence[tuple[str, float]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for parent_asin, raw_score in candidates:
        identifier = str(parent_asin)
        score = float(raw_score)
        if not identifier or not isfinite(score):
            continue
        if identifier not in best or score > best[identifier]:
            best[identifier] = score
    if not best:
        return {}
    low = min(best.values())
    high = max(best.values())
    if high == low:
        return {identifier: 1.0 for identifier in best}
    scale = high - low
    return {
        identifier: (score - low) / scale
        for identifier, score in best.items()
    }


@dataclass(frozen=True)
class FixedFusionPolicy:
    """Versioned weighted fusion over independently scored Retrieval Routes."""

    weights: Mapping[str, float] = field(default_factory=lambda: {
        "structured": 0.15,
        "bm25": 0.55,
        "dense": 0.3,
    })
    version: str = "fixed-hybrid-v1"
    candidate_limit: int = 30

    def __post_init__(self) -> None:
        if set(self.weights) != set(ROUTE_NAMES):
            raise ValueError(f"weights must define exactly: {', '.join(ROUTE_NAMES)}")
        numeric_weights = {
            name: float(value) for name, value in self.weights.items()
        }
        if any(
            not isfinite(value) or value < 0
            for value in numeric_weights.values()
        ):
            raise ValueError("route weights must be finite and non-negative")
        if abs(sum(numeric_weights.values()) - 1.0) > 1e-9:
            raise ValueError("route weights must sum to one")
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        object.__setattr__(self, "weights", MappingProxyType(numeric_weights))

    def rank(
        self,
        route_scores: Mapping[str, Sequence[tuple[str, float]]],
    ) -> list[str]:
        fused: dict[str, float] = {}
        for route_name in ROUTE_NAMES:
            weight = self.weights[route_name]
            for identifier, normalized in _normalized_scores(
                route_scores.get(route_name, ()),
            ).items():
                fused[identifier] = fused.get(identifier, 0.0) + weight * normalized
        return [
            identifier
            for identifier, _score in sorted(
                fused.items(),
                key=lambda item: (-item[1], item[0]),
            )[:self.candidate_limit]
        ]


@dataclass(frozen=True)
class SingleRoutePolicy:
    route_name: str
    candidate_limit: int = 30

    def __post_init__(self) -> None:
        if self.route_name not in ROUTE_NAMES:
            raise ValueError(f"unknown Retrieval Route: {self.route_name}")

    @property
    def version(self) -> str:
        return f"single-{self.route_name}-v1"

    def rank(
        self,
        route_scores: Mapping[str, Sequence[tuple[str, float]]],
    ) -> list[str]:
        best: dict[str, float] = {}
        for identifier, raw_score in route_scores.get(self.route_name, ()):
            score = float(raw_score)
            identifier = str(identifier)
            if identifier and isfinite(score) and (
                identifier not in best or score > best[identifier]
            ):
                best[identifier] = score
        return [
            identifier
            for identifier, _score in sorted(
                best.items(),
                key=lambda item: (-item[1], item[0]),
            )[:self.candidate_limit]
        ]


@dataclass(frozen=True)
class ReciprocalRankFusionPolicy:
    rank_constant: int = 60
    candidate_limit: int = 30
    version: str = "fixed-rrf-v1"

    def rank(
        self,
        route_scores: Mapping[str, Sequence[tuple[str, float]]],
    ) -> list[str]:
        fused: dict[str, float] = {}
        for route_name in ROUTE_NAMES:
            ordered = SingleRoutePolicy(
                route_name,
                candidate_limit=max(
                    self.candidate_limit,
                    len(route_scores.get(route_name, ())),
                ),
            ).rank(route_scores)
            for rank, identifier in enumerate(ordered, start=1):
                fused[identifier] = fused.get(identifier, 0.0) + (
                    1.0 / (self.rank_constant + rank)
                )
        return [
            identifier
            for identifier, _score in sorted(
                fused.items(),
                key=lambda item: (-item[1], item[0]),
            )[:self.candidate_limit]
        ]


def build_fusion_policy(name: str) -> FusionPolicy:
    if name == "fixed":
        return FixedFusionPolicy()
    if name == "rrf":
        return ReciprocalRankFusionPolicy()
    if name in ROUTE_NAMES:
        return SingleRoutePolicy(name)
    choices = ", ".join(POLICY_NAMES)
    raise ValueError(f"unknown Fusion Policy {name!r}; choose from {choices}")
