from __future__ import annotations

import unittest

from starter.constraint_state import (
    AddConstraint,
    ConstraintState,
    DismissAttribute,
    PlanValidationError,
    ReintroduceConstraint,
    ReplaceProductIntent,
    TurnPlan,
)


SUPPORTED_VALUES = {
    "category": {"shoe", "slipper"},
    "material": {"cotton", "leather"},
    "color": {"blue", "red"},
}


class ConstraintStateTest(unittest.TestCase):
    def test_multiple_additions_commit_atomically_at_one_revision(self) -> None:
        state = ConstraintState()
        plan = TurnPlan(
            expected_state_revision=0,
            source_turn=1,
            mutations=(
                AddConstraint(
                    attribute="color",
                    values=("blue",),
                    match_rule="any",
                    classification="soft",
                    scope="product_intent",
                    raw_phrase="blue",
                    confidence=0.95,
                ),
                AddConstraint(
                    attribute="material",
                    values=("cotton",),
                    match_rule="all",
                    classification="hard",
                    scope="product_intent",
                    raw_phrase="cotton",
                    confidence=0.99,
                ),
            ),
        )

        state.apply(plan, SUPPORTED_VALUES)

        self.assertEqual(state.revision, 1)
        self.assertEqual(
            [
                (item.attribute, item.values, item.classification, item.status)
                for item in state.constraints
            ],
            [
                ("color", ("blue",), "soft", "active"),
                ("material", ("cotton",), "hard", "active"),
            ],
        )
        self.assertEqual(
            {item.product_intent_id for item in state.constraints},
            {state.active_product_intent_id},
        )

    def test_invalid_mutation_rolls_back_the_complete_plan(self) -> None:
        state = ConstraintState()
        original = state.as_dict()
        plan = TurnPlan(
            expected_state_revision=0,
            source_turn=1,
            mutations=(
                AddConstraint(
                    attribute="material",
                    values=("cotton",),
                    match_rule="all",
                    classification="hard",
                    scope="product_intent",
                    raw_phrase="cotton",
                    confidence=0.99,
                ),
                AddConstraint(
                    attribute="temperature_rating",
                    values=("warm",),
                    match_rule="all",
                    classification="hard",
                    scope="product_intent",
                    raw_phrase="warm",
                    confidence=0.8,
                ),
            ),
        )

        with self.assertRaises(PlanValidationError):
            state.apply(plan, SUPPORTED_VALUES)

        self.assertEqual(state.as_dict(), original)

    def test_stale_or_replayed_plan_changes_nothing(self) -> None:
        state = ConstraintState()
        plan = TurnPlan(
            expected_state_revision=0,
            source_turn=1,
            mutations=(
                AddConstraint(
                    attribute="material",
                    values=("cotton",),
                    match_rule="all",
                    classification="hard",
                    scope="product_intent",
                    raw_phrase="cotton",
                    confidence=0.99,
                ),
            ),
        )
        state.apply(plan, SUPPORTED_VALUES)
        committed = state.as_dict()

        with self.assertRaisesRegex(PlanValidationError, "stale Turn Plan"):
            state.apply(plan, SUPPORTED_VALUES)

        self.assertEqual(state.as_dict(), committed)

    def test_product_intent_replacement_retires_group_but_keeps_session_scope(self) -> None:
        state = ConstraintState()
        state.apply(
            TurnPlan(
                expected_state_revision=0,
                source_turn=1,
                mutations=(
                    AddConstraint(
                        attribute="category",
                        values=("shoe",),
                        match_rule="all",
                        classification="hard",
                        scope="product_intent",
                        raw_phrase="shoes",
                        confidence=0.99,
                    ),
                    AddConstraint(
                        attribute="material",
                        values=("leather",),
                        match_rule="all",
                        classification="hard",
                        scope="product_intent",
                        raw_phrase="leather",
                        confidence=0.95,
                    ),
                    AddConstraint(
                        attribute="color",
                        values=("blue",),
                        match_rule="any",
                        classification="soft",
                        scope="session",
                        raw_phrase="whatever I buy, blue",
                        confidence=0.95,
                    ),
                ),
            ),
            SUPPORTED_VALUES,
        )
        old_intent = state.active_product_intent_id

        state.apply(
            TurnPlan(
                expected_state_revision=1,
                source_turn=2,
                mutations=(
                    AddConstraint(
                        attribute="category",
                        values=("slipper",),
                        match_rule="all",
                        classification="hard",
                        scope="product_intent",
                        raw_phrase="slippers",
                        confidence=0.99,
                    ),
                    ReplaceProductIntent(
                        product_intent_id=old_intent,
                        raw_phrase="Actually, slippers instead of shoes.",
                    ),
                ),
            ),
            SUPPORTED_VALUES,
        )

        self.assertEqual(state.revision, 2)
        self.assertNotEqual(state.active_product_intent_id, old_intent)
        self.assertEqual(
            [
                (item.attribute, item.scope, item.status)
                for item in state.constraints
            ],
            [
                ("category", "product_intent", "superseded"),
                ("material", "product_intent", "superseded"),
                ("color", "session", "active"),
                ("category", "product_intent", "active"),
            ],
        )
        self.assertTrue(all(
            item.superseded_by == state.active_product_intent_id
            for item in state.constraints[:2]
        ))

    def test_boundary_dismissal_and_different_addition_commit_together(self) -> None:
        state = ConstraintState()
        state.apply(
            TurnPlan(
                expected_state_revision=0,
                source_turn=1,
                mutations=(
                    AddConstraint(
                        attribute="color",
                        values=("blue",),
                        match_rule="any",
                        classification="soft",
                        scope="product_intent",
                        raw_phrase="blue",
                        confidence=0.95,
                    ),
                ),
            ),
            SUPPORTED_VALUES,
        )

        state.apply(
            TurnPlan(
                expected_state_revision=1,
                source_turn=2,
                mutations=(
                    DismissAttribute(
                        attribute="color",
                        raw_phrase="I don't care about color",
                    ),
                    AddConstraint(
                        attribute="material",
                        values=("cotton",),
                        match_rule="all",
                        classification="hard",
                        scope="product_intent",
                        raw_phrase="need cotton",
                        confidence=0.99,
                    ),
                ),
            ),
            SUPPORTED_VALUES,
        )

        self.assertEqual(state.revision, 2)
        self.assertEqual(
            [(item.attribute, item.status) for item in state.constraints],
            [("color", "dismissed"), ("material", "active")],
        )
        self.assertEqual(state.dismissed_attributes["color"]["source_turn"], 2)

    def test_dismissed_attribute_requires_explicit_reintroduction(self) -> None:
        state = ConstraintState(
            dismissed_attributes={
                "color": {
                    "attribute": "color",
                    "raw_phrase": "no preference",
                    "source_turn": 1,
                    "status": "dismissed",
                }
            }
        )
        implicit_add = TurnPlan(
            expected_state_revision=0,
            source_turn=2,
            mutations=(
                AddConstraint(
                    attribute="color",
                    values=("red",),
                    match_rule="all",
                    classification="hard",
                    scope="product_intent",
                    raw_phrase="red",
                    confidence=0.95,
                ),
            ),
        )

        with self.assertRaisesRegex(PlanValidationError, "reintroduction"):
            state.apply(implicit_add, SUPPORTED_VALUES)

        state.apply(
            TurnPlan(
                expected_state_revision=0,
                source_turn=2,
                mutations=(
                    ReintroduceConstraint(
                        attribute="color",
                        values=("red",),
                        match_rule="all",
                        classification="hard",
                        scope="product_intent",
                        raw_phrase="Actually, red matters",
                        confidence=0.95,
                    ),
                ),
            ),
            SUPPORTED_VALUES,
        )

        self.assertNotIn("color", state.dismissed_attributes)
        self.assertEqual(state.constraints[-1].values, ("red",))

    def test_conflicting_additions_are_rejected_regardless_of_order(self) -> None:
        first = AddConstraint(
            attribute="color",
            values=("red",),
            match_rule="all",
            classification="hard",
            scope="product_intent",
            raw_phrase="red",
            confidence=0.95,
        )
        second = AddConstraint(
            attribute="color",
            values=("blue",),
            match_rule="all",
            classification="hard",
            scope="product_intent",
            raw_phrase="blue",
            confidence=0.95,
        )
        for mutations in ((first, second), (second, first)):
            with self.subTest(mutations=mutations):
                state = ConstraintState()
                original = state.as_dict()
                with self.assertRaises(PlanValidationError):
                    state.apply(
                        TurnPlan(
                            expected_state_revision=0,
                            source_turn=1,
                            mutations=mutations,
                        ),
                        SUPPORTED_VALUES,
                    )
                self.assertEqual(state.as_dict(), original)

    def test_blank_constraint_provenance_is_rejected(self) -> None:
        state = ConstraintState()
        with self.assertRaisesRegex(PlanValidationError, "provenance"):
            state.apply(
                TurnPlan(
                    expected_state_revision=0,
                    source_turn=1,
                    mutations=(
                        AddConstraint(
                            attribute="color",
                            values=("blue",),
                            match_rule="all",
                            classification="hard",
                            scope="product_intent",
                            raw_phrase="   ",
                            confidence=0.95,
                        ),
                    ),
                ),
                SUPPORTED_VALUES,
            )

    def test_product_intent_and_boundary_transitions_keep_provenance_history(self) -> None:
        state = ConstraintState()
        state.apply(
            TurnPlan(
                expected_state_revision=0,
                source_turn=1,
                mutations=(
                    AddConstraint(
                        attribute="category",
                        values=("shoe",),
                        match_rule="all",
                        classification="hard",
                        scope="product_intent",
                        raw_phrase="shoes",
                        confidence=0.95,
                    ),
                ),
            ),
            SUPPORTED_VALUES,
        )
        state.apply(
            TurnPlan(
                expected_state_revision=1,
                source_turn=2,
                mutations=(
                    ReplaceProductIntent(
                        product_intent_id="intent-1",
                        raw_phrase="Actually, slippers instead.",
                    ),
                    AddConstraint(
                        attribute="category",
                        values=("slipper",),
                        match_rule="all",
                        classification="hard",
                        scope="product_intent",
                        raw_phrase="slippers",
                        confidence=0.95,
                    ),
                ),
            ),
            SUPPORTED_VALUES,
        )
        state.apply(
            TurnPlan(
                expected_state_revision=2,
                source_turn=3,
                mutations=(DismissAttribute("color", "no color preference"),),
            ),
            SUPPORTED_VALUES,
        )

        self.assertEqual(
            [event["type"] for event in state.transition_history],
            ["product_intent_replaced", "attribute_dismissed"],
        )
        self.assertEqual(state.transition_history[0]["source_turn"], 2)
        self.assertEqual(
            state.transition_history[0]["raw_phrase"],
            "Actually, slippers instead.",
        )
        self.assertEqual(state.transition_history[1]["source_turn"], 3)


if __name__ == "__main__":
    unittest.main()
