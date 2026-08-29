from __future__ import annotations

import unittest

from retrieval.fusion import FixedFusionPolicy, build_fusion_policy


class FixedFusionPolicyTest(unittest.TestCase):
    def test_weighted_fusion_ranks_from_route_score_fixtures(self) -> None:
        policy = FixedFusionPolicy(
            weights={"structured": 0.5, "bm25": 0.3, "dense": 0.2},
        )

        ranked = policy.rank({
            "structured": [("A", 3.0), ("B", 1.0)],
            "bm25": [("B", 9.0), ("C", 2.0)],
            "dense": [("C", 0.95), ("A", 0.80)],
        })

        self.assertEqual(ranked, ["A", "B", "C"])

    def test_missing_and_constant_score_routes_are_deterministic(self) -> None:
        policy = FixedFusionPolicy(
            weights={"structured": 0.4, "bm25": 0.3, "dense": 0.3},
        )
        fixtures = {
            "structured": [("B", 1.0), ("A", 1.0)],
            "dense": [("C", 0.7), ("B", 0.7)],
        }

        first = policy.rank(fixtures)
        second = policy.rank(fixtures)

        self.assertEqual(first, ["B", "A", "C"])
        self.assertEqual(second, first)

    def test_transparent_baselines_use_the_same_route_score_fixture(self) -> None:
        fixtures = {
            "structured": [("A", 4.0), ("B", 3.0)],
            "bm25": [("B", 5.0), ("C", 4.0)],
            "dense": [("C", 0.9), ("B", 0.8)],
        }

        self.assertEqual(build_fusion_policy("structured").rank(fixtures), ["A", "B"])
        self.assertEqual(build_fusion_policy("bm25").rank(fixtures), ["B", "C"])
        self.assertEqual(build_fusion_policy("dense").rank(fixtures), ["C", "B"])
        self.assertEqual(build_fusion_policy("rrf").rank(fixtures), ["B", "C", "A"])

    def test_default_policy_limits_the_fused_candidate_pool_to_thirty(self) -> None:
        ranked = FixedFusionPolicy().rank({
            "structured": [
                (f"A{index:02d}", float(40 - index))
                for index in range(40)
            ],
            "bm25": [],
            "dense": [],
        })

        self.assertEqual(len(ranked), 30)
        self.assertEqual(ranked[0], "A00")
        self.assertEqual(ranked[-1], "A29")


if __name__ == "__main__":
    unittest.main()
