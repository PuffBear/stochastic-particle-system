from __future__ import annotations

import itertools
import json
from pathlib import Path
import unittest

import numpy as np

from particle_benchmark.communication import (
    aggregate_three_scalar_messages,
    validate_three_scalar_messages,
)


class TestThreeScalarAggregation(unittest.TestCase):
    def test_diagnostic_schema_freezes_four_by_four_topology_and_bandwidth(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "schemas" / "message_diagnostic.schema.json").read_text()
        )
        properties = schema["properties"]
        adjacency = properties["adjacency"]
        self.assertEqual((adjacency["minItems"], adjacency["maxItems"]), (4, 4))
        self.assertEqual(
            (adjacency["items"]["minItems"], adjacency["items"]["maxItems"]),
            (4, 4),
        )
        for name in (
            "sent_messages",
            "received_messages",
            "true_local_velocities",
            "estimator_squared_errors",
            "collector_actions",
            "pursuit_target_ids",
        ):
            self.assertEqual(
                (properties[name]["minItems"], properties[name]["maxItems"]),
                (4, 4),
            )
        message = schema["$defs"]["message3"]
        self.assertEqual((message["minItems"], message["maxItems"]), (3, 3))

    def test_exact_bandwidth_and_bounds_are_enforced(self) -> None:
        valid = np.array([[0.2, -0.4, 1.0], [-1.0, 1.0, 0.0]])
        self.assertEqual(validate_three_scalar_messages(valid).shape, (2, 3))
        for invalid in (
            np.zeros((2, 2)),
            np.zeros((2, 4)),
            np.array([[1.01, 0.0, 1.0]]),
            np.array([[0.0, 0.0, -0.01]]),
            np.array([[0.0, np.nan, 0.5]]),
        ):
            with self.assertRaises(ValueError):
                validate_three_scalar_messages(invalid)

    def test_both_arms_are_permutation_equivariant(self) -> None:
        messages = np.array(
            [[0.8, 0.1, 1.0], [-0.4, 0.2, 0.5], [0.0, -0.7, 0.25], [0.3, 0.4, 0.75]]
        )
        permutation = np.array([2, 0, 3, 1])
        for mode in ("independent", "all_to_all"):
            original = aggregate_three_scalar_messages(messages, mode=mode)
            permuted = aggregate_three_scalar_messages(
                messages[permutation], mode=mode
            )
            np.testing.assert_array_equal(permuted, original[permutation])

    def test_constant_field_messages_make_arms_equivalent(self) -> None:
        messages = np.tile(np.array([0.6, -0.2, 1.0]), (4, 1))
        independent = aggregate_three_scalar_messages(
            messages, mode="independent"
        )
        global_mean = aggregate_three_scalar_messages(
            messages, mode="all_to_all"
        )
        np.testing.assert_array_equal(independent, global_mean)

    def test_opposing_regions_cancel_only_under_global_pooling(self) -> None:
        messages = np.array(
            [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
        )
        independent = aggregate_three_scalar_messages(
            messages, mode="independent"
        )
        global_mean = aggregate_three_scalar_messages(
            messages, mode="all_to_all"
        )
        np.testing.assert_array_equal(independent, messages)
        np.testing.assert_array_equal(global_mean[:, :2], 0.0)
        np.testing.assert_array_equal(global_mean[:, 2], 1.0)

    def test_homogeneous_independent_noise_has_one_over_m_mean_variance(self) -> None:
        # Enumerating all 2^M Rademacher patterns is deterministic and yields
        # the exact expectation: Var(mean(error)) = Var(error) / M.
        local_squared_errors: list[float] = []
        pooled_squared_errors: list[float] = []
        for signs in itertools.product((-1.0, 1.0), repeat=4):
            messages = np.column_stack(
                (0.25 * np.asarray(signs), np.zeros(4), np.ones(4))
            )
            local = aggregate_three_scalar_messages(
                messages, mode="independent"
            )
            pooled = aggregate_three_scalar_messages(
                messages, mode="all_to_all"
            )
            local_squared_errors.extend(local[:, 0] ** 2)
            pooled_squared_errors.append(float(pooled[0, 0] ** 2))
        local_variance = float(np.mean(local_squared_errors))
        pooled_variance = float(np.mean(pooled_squared_errors))
        self.assertAlmostEqual(pooled_variance, local_variance / 4.0, places=12)

    def test_inputs_are_not_mutated_and_unknown_mode_is_rejected(self) -> None:
        messages = np.array([[0.1, 0.2, 0.3]])
        before = messages.copy()
        aggregate_three_scalar_messages(messages, mode="independent")
        np.testing.assert_array_equal(messages, before)
        with self.assertRaisesRegex(ValueError, "unknown aggregation mode"):
            aggregate_three_scalar_messages(messages, mode="other")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
