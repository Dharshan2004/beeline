from __future__ import annotations

import unittest

from starter.constraint_state import (
    AddConstraint,
    ConstraintState,
    DismissAttribute,
    ReplaceConstraint,
    ReplaceProductIntent,
)
from starter.turn_interpreter import interpret_turn


SUPPORTED_VALUES = {
    "category": {"shoe", "slipper"},
    "material": {"cotton", "leather"},
    "color": {"blue", "red"},
    "feature": {"waterproof"},
}


class TurnInterpreterTest(unittest.TestCase):
    def test_mixed_soft_and_hard_requirements_produce_two_additions(self) -> None:
        state = ConstraintState()

        plan = interpret_turn(
            "I prefer blue, but must have cotton.",
            turn=1,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )

        additions = [item for item in plan.mutations if isinstance(item, AddConstraint)]
        self.assertEqual(
            [
                (item.attribute, item.values, item.classification)
                for item in additions
            ],
            [
                ("color", ("blue",), "soft"),
                ("material", ("cotton",), "hard"),
            ],
        )

    def test_boundary_and_new_requirement_share_one_turn_plan(self) -> None:
        state = ConstraintState()

        plan = interpret_turn(
            "I don't care about color, but I need cotton.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
            last_asked_attribute="color",
        )

        self.assertEqual(len(plan.mutations), 2)
        self.assertIsInstance(plan.mutations[0], DismissAttribute)
        self.assertEqual(plan.mutations[0].attribute, "color")
        self.assertIsInstance(plan.mutations[1], AddConstraint)
        self.assertEqual(plan.mutations[1].attribute, "material")

    def test_explicit_category_override_replaces_the_product_intent(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I need shoes and leather.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )

        plan = interpret_turn(
            "Actually, slippers instead of shoes.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )

        self.assertTrue(any(
            isinstance(item, ReplaceProductIntent)
            for item in plan.mutations
        ))
        additions = [item for item in plan.mutations if isinstance(item, AddConstraint)]
        self.assertEqual(
            [(item.attribute, item.values) for item in additions],
            [("category", ("slipper",))],
        )

    def test_attribute_correction_does_not_replace_product_intent(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I need leather shoes.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )
        original_intent = state.active_product_intent_id

        plan = interpret_turn(
            "Actually, I need waterproof.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )
        state.apply(plan, SUPPORTED_VALUES)

        self.assertFalse(any(
            isinstance(item, ReplaceProductIntent)
            for item in plan.mutations
        ))
        self.assertEqual(state.active_product_intent_id, original_intent)
        self.assertEqual(
            [item.normalized_value for item in state.constraints if item.status == "active"],
            ["shoe", "leather", "waterproof"],
        )

    def test_different_category_without_replacement_language_is_ambiguous(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I need shoes.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )

        plan = interpret_turn(
            "What about slippers?",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )

        self.assertEqual(plan.mutations, ())

    def test_explicit_preference_override_targets_the_obsolete_constraint(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I prefer blue.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )
        obsolete_id = state.constraints[0].constraint_id

        plan = interpret_turn(
            "Actually, ignore my earlier preference. What I need is cotton.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )

        replacements = [
            item for item in plan.mutations if isinstance(item, ReplaceConstraint)
        ]
        self.assertEqual(len(replacements), 1)
        self.assertEqual(replacements[0].constraint_id, obsolete_id)
        state.apply(plan, SUPPORTED_VALUES)
        self.assertEqual(
            [(item.normalized_value, item.status) for item in state.constraints],
            [("blue", "superseded"), ("cotton", "active")],
        )

    def test_preference_override_targets_unique_same_attribute_constraint(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I prefer leather, but I prefer blue.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )
        material_id = next(
            item.constraint_id
            for item in state.constraints
            if item.attribute == "material"
        )

        plan = interpret_turn(
            "Actually, ignore my previous preference. I need cotton.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )

        replacement = next(
            item for item in plan.mutations if isinstance(item, ReplaceConstraint)
        )
        self.assertEqual(replacement.constraint_id, material_id)
        state.apply(plan, SUPPORTED_VALUES)
        self.assertEqual(
            [
                (item.normalized_value, item.status)
                for item in state.constraints
            ],
            [
                ("leather", "superseded"),
                ("blue", "active"),
                ("cotton", "active"),
            ],
        )

    def test_attribute_withdrawal_does_not_license_product_intent_replacement(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I need blue shoes.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )

        plan = interpret_turn(
            "I no longer want blue; show me slippers.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )

        self.assertFalse(any(
            isinstance(item, ReplaceProductIntent)
            for item in plan.mutations
        ))

    def test_relation_to_attribute_does_not_license_product_intent_replacement(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I need cotton shoes.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )

        plan = interpret_turn(
            "I want slippers instead of cotton.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )

        self.assertFalse(any(
            isinstance(item, ReplaceProductIntent)
            for item in plan.mutations
        ))

    def test_multiple_values_preserve_explicit_any_or_all_meaning(self) -> None:
        cases = (
            ("Red or blue is fine.", ("red", "blue"), "any"),
            ("I need red and blue.", ("red", "blue"), "all"),
        )
        for message, values, match_rule in cases:
            with self.subTest(message=message):
                plan = interpret_turn(
                    message,
                    turn=1,
                    state=ConstraintState(),
                    supported_values=SUPPORTED_VALUES,
                )
                addition = next(
                    item
                    for item in plan.mutations
                    if isinstance(item, AddConstraint)
                )
                self.assertEqual(addition.values, values)
                self.assertEqual(addition.match_rule, match_rule)

    def test_session_scope_applies_only_to_the_explicit_clause(self) -> None:
        for message in (
            "Whatever I buy should be blue, but I need cotton shoes.",
            "Whatever I buy should be blue and I need cotton shoes.",
        ):
            with self.subTest(message=message):
                plan = interpret_turn(
                    message,
                    turn=1,
                    state=ConstraintState(),
                    supported_values=SUPPORTED_VALUES,
                )

                additions = [
                    item for item in plan.mutations
                    if isinstance(item, AddConstraint)
                ]
                self.assertEqual(
                    [(item.attribute, item.scope) for item in additions],
                    [
                        ("color", "session"),
                        ("category", "product_intent"),
                        ("material", "product_intent"),
                    ],
                )

    def test_explicit_new_category_replaces_intent_without_old_category(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I need leather.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )

        plan = interpret_turn(
            "Actually, I need slippers instead.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )

        self.assertTrue(any(
            isinstance(item, ReplaceProductIntent)
            for item in plan.mutations
        ))

    def test_compound_override_targets_obsolete_constraint_only_once(self) -> None:
        state = ConstraintState()
        state.apply(
            interpret_turn(
                "I prefer blue.",
                turn=1,
                state=state,
                supported_values=SUPPORTED_VALUES,
            ),
            SUPPORTED_VALUES,
        )

        plan = interpret_turn(
            "Actually, ignore my earlier preference. I need red and cotton.",
            turn=2,
            state=state,
            supported_values=SUPPORTED_VALUES,
        )
        state.apply(plan, SUPPORTED_VALUES)

        self.assertEqual(
            [(item.attribute, item.status) for item in state.constraints],
            [("color", "superseded"), ("color", "active"), ("material", "active")],
        )


if __name__ == "__main__":
    unittest.main()
