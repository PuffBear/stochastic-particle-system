"""Outcome-blind SPS-C04 design-contract tests; no experiment is executed."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import yaml

from particle_benchmark.aggregation_runner import (
    AggregationPairExperimentConfig,
    FROZEN_PHYSICAL_ENDPOINT_SECONDS,
    FROZEN_TIMESTEP_HORIZONS,
    _frozen_endpoint_status,
)
from particle_benchmark.environment import ParticleEnvConfig
from particle_benchmark.runner import RepositoryProvenance


ROOT = Path(__file__).resolve().parents[1]
DESIGN = yaml.safe_load(
    (ROOT / "configs/experiments/sps_c04_outcome_blind_design.yaml").read_text(
        encoding="utf-8"
    )
)


class TestSPSC04OutcomeBlindDesign(unittest.TestCase):
    def test_endpoint_step_semantics_are_exact(self) -> None:
        outcome = DESIGN["outcome"]
        self.assertEqual(outcome["physical_endpoint_seconds"], 1.34)
        self.assertEqual(outcome["first_included_decision_step"], 0)
        self.assertEqual(outcome["last_included_decision_step"], 66)
        self.assertEqual(outcome["last_included_outcome_step"], 67)
        self.assertEqual(outcome["first_excluded_outcome_step"], 68)
        self.assertIn("retain_every_transition", outcome["retention_rule"])

    def test_coupled_timestep_mappings_match_one_physical_endpoint(self) -> None:
        mappings = {
            float(row["dt"]): int(row["horizon_steps"])
            for row in DESIGN["timestep_matching"]["coupled_mappings"]
        }
        self.assertEqual(mappings, FROZEN_TIMESTEP_HORIZONS)
        self.assertEqual(FROZEN_PHYSICAL_ENDPOINT_SECONDS, 1.34)
        for dt, horizon in mappings.items():
            self.assertTrue(
                math.isclose(
                    dt * horizon,
                    FROZEN_PHYSICAL_ENDPOINT_SECONDS,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            self.assertEqual(
                _frozen_endpoint_status(dt=dt, horizon=horizon),
                "frozen_T_1.34_matched_physical_endpoint",
            )
        self.assertEqual(
            _frozen_endpoint_status(dt=0.02, horizon=68),
            "frozen_T_1.34_current_run_nonendpoint_diagnostic",
        )

    def test_mechanism_eligibility_and_mediator_exclusion_are_frozen(self) -> None:
        mechanism = DESIGN["analytic_mechanism"]
        self.assertEqual(len(mechanism["all_required"]), 5)
        self.assertIn("decision_step_is_at_least_1", mechanism["all_required"])
        self.assertIn(
            "all_four_collectors_have_nonempty_own_summaries",
            mechanism["all_required"],
        )
        self.assertIn(
            "no_selected_valid_particle_has_a_reflected_apparent_velocity",
            mechanism["all_required"],
        )
        self.assertIn(
            "row_is_ineligible",
            mechanism["treatment_events"]["cross_agent_rescue"],
        )
        self.assertIn(
            "does_not_make_an_otherwise_clean_row_ineligible",
            mechanism["treatment_events"]["zero_direction_cancellation"],
        )
        self.assertIn(
            "exclude_only_the_affected_current_collector_transition",
            mechanism["collector_reflection"]["action_to_displacement_mediator"],
        )
        self.assertIn("never_exclude", mechanism["collector_reflection"]["yield"])

    def test_diagnostic_thresholds_and_denominators_are_exact(self) -> None:
        thresholds = DESIGN["diagnostic_thresholds"]
        self.assertEqual(thresholds["minimum_arm_eta_analytic_eligibility_rate"], 0.80)
        self.assertEqual(
            thresholds["maximum_absolute_fraction_difference"],
            0.10,
        )
        self.assertIn(
            "10_percentage_points",
            thresholds["analytic_eligibility_arm_gap_interpretation"],
        )
        self.assertEqual(
            thresholds["minimum_individual_seed_arm_analytic_eligibility_rate"],
            0.50,
        )
        for key in (
            "maximum_own_empty_agent_row_rate",
            "maximum_fallback_agent_row_rate",
            "maximum_reflected_valid_slot_rate",
            "maximum_collector_reflected_action_transition_rate",
        ):
            self.assertEqual(thresholds[key], 0.05)
        self.assertEqual(thresholds["maximum_clipped_velocity_component_rate"], 0.01)
        self.assertEqual(thresholds["cross_agent_rescue_rate"], "report_without_a_cap")
        self.assertEqual(
            DESIGN["denominators"]["analytic_eligibility"],
            "post_history_transition_rows",
        )

    def test_only_frozen_long_horizons_are_runner_admissible(self) -> None:
        base = dict(
            particle_count=4,
            collector_count=4,
            diffusion_sigma=0.0,
            collector_max_speed=0.05,
            sensing_radius=2.0,
            collector_radius=0.0,
            particle_radius=0.0,
            nearest_particles_k=2,
            field_family="periodic_gaussian",
            signal_strength=0.01,
            field_kwargs={
                "correlation_length": 0.2,
                "component_variance": 0.01,
                "max_frequency": 3,
            },
            include_particle_velocity=True,
            include_teammates=False,
        )
        repository = RepositoryProvenance(
            full_name="PuffBear/stochastic-particle-system",
            branch="research-autonomy",
            commit_sha="a" * 40,
            dirty=False,
        )
        common = dict(
            experiment_id="SPS-C04-CONTRACT-TEST",
            scenario_seed=1,
            repository=repository,
            created_at_utc="2026-08-01T00:00:00Z",
            config_path="test-only",
            run_command="no execution",
        )
        for dt, horizon in FROZEN_TIMESTEP_HORIZONS.items():
            AggregationPairExperimentConfig(
                environment=ParticleEnvConfig(dt=dt, horizon=horizon, **base),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "frozen T=1.34"):
            AggregationPairExperimentConfig(
                environment=ParticleEnvConfig(dt=0.02, horizon=68, **base),
                **common,
            )

    def test_execution_firewall_remains_closed(self) -> None:
        firewall = DESIGN["execution_firewall"]
        self.assertFalse(firewall["scientific_seeds_authorized"])
        self.assertFalse(firewall["eta_grid_frozen"])
        self.assertFalse(firewall["scientific_seed_block_frozen"])
        self.assertFalse(firewall["effect_threshold_frozen"])
        self.assertFalse(firewall["canonical_environment_config_changed"])


if __name__ == "__main__":
    unittest.main()
