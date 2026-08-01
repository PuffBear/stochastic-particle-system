"""Tests for coordination metrics and communication ablation hooks.

CommNet ablation tests are skipped automatically if PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from particle_benchmark.metrics.coordination import (
    ce_report,
    per_seed_ce,
    spatial_coverage_metrics,
    split_field_informative,
    upstream_alignment_score,
    validate_execution_time_ablation,
)


# ---------------------------------------------------------------------------
# per_seed_ce
# ---------------------------------------------------------------------------

class TestPerSeedCE:

    def test_ce_zero_when_comm_equals_ablated(self):
        """CE = 0 when comm policy matches ablated (no communication benefit)."""
        ablated = np.array([0.3, 0.5, 0.4])
        oracle = np.array([0.8, 0.9, 0.7])
        ce = per_seed_ce(ablated, ablated, oracle)
        np.testing.assert_allclose(ce, 0.0, atol=1e-10)

    def test_ce_one_when_comm_equals_oracle(self):
        """CE = 1 when comm policy reaches oracle performance."""
        ablated = np.array([0.2, 0.3, 0.4])
        oracle = np.array([0.8, 0.9, 0.7])
        ce = per_seed_ce(oracle, ablated, oracle)
        np.testing.assert_allclose(ce, 1.0, atol=1e-10)

    def test_ce_intermediate_value(self):
        """CE is proportional to improvement between ablated and oracle."""
        ablated = np.array([0.0])
        oracle = np.array([1.0])
        comm = np.array([0.5])
        ce = per_seed_ce(comm, ablated, oracle)
        np.testing.assert_allclose(ce, 0.5, atol=1e-10)

    def test_ce_negative_when_comm_worse_than_ablated(self):
        """CE < 0 when communication hurts performance."""
        ablated = np.array([0.5])
        oracle = np.array([1.0])
        comm = np.array([0.2])
        ce = per_seed_ce(comm, ablated, oracle)
        assert ce[0] < 0.0

    def test_ce_undefined_when_oracle_equals_ablated(self):
        """CE is undefined when the oracle has no positive headroom."""
        val = np.array([0.5, 0.5])
        ce = per_seed_ce(val, val, val)
        assert np.all(np.isnan(ce))

    def test_ce_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="identical shapes"):
            per_seed_ce(np.ones(2), np.ones(3), np.ones(2))


class TestExecutionTimeAblationValidation:

    def test_rejects_ctde_as_communication(self):
        with pytest.raises(ValueError, match="no execution-time communication"):
            validate_execution_time_ablation(
                np.zeros((4, 2)), np.ones((4, 2)),
                execution_time_communication=False,
            )

    def test_rejects_identity_ablation(self):
        actions = np.arange(8, dtype=float).reshape(4, 2)
        with pytest.raises(ValueError, match="identity operation"):
            validate_execution_time_ablation(
                actions, actions.copy(), execution_time_communication=True
            )

    def test_accepts_genuine_message_intervention(self):
        validate_execution_time_ablation(
            np.zeros((4, 2)), np.ones((4, 2)),
            execution_time_communication=True,
        )

    def test_ce_output_shape(self):
        """Output shape matches input shape."""
        n = 17
        ce = per_seed_ce(
            np.random.rand(n),
            np.random.rand(n),
            np.random.rand(n) + 1.0,  # oracle > comm,ablated
        )
        assert ce.shape == (n,)
        assert ce.dtype == np.float64

    def test_ce_vectorised_known_values(self):
        comm    = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        ablated = np.zeros(5)
        oracle  = np.ones(5)
        expected = comm.copy()
        np.testing.assert_allclose(per_seed_ce(comm, ablated, oracle), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# split_field_informative
# ---------------------------------------------------------------------------

class TestSplitFieldInformative:

    def test_all_informative(self):
        """All seeds informative when oracle - stationary >> threshold."""
        oracle     = np.array([10.0, 12.0, 15.0])
        stationary = np.array([1.0,  2.0,  3.0])
        info, uninf = split_field_informative(oracle, stationary, threshold=4.0)
        assert np.all(info)
        assert not np.any(uninf)

    def test_all_uninformative(self):
        """All seeds uninformative when oracle - stationary < threshold."""
        oracle     = np.array([5.0, 5.0, 5.0])
        stationary = np.array([4.0, 4.5, 4.9])
        info, uninf = split_field_informative(oracle, stationary, threshold=4.0)
        assert not np.any(info)
        assert np.all(uninf)

    def test_mixed_split(self):
        oracle     = np.array([10.0, 5.0, 8.0, 3.0])
        stationary = np.array([1.0,  4.0, 3.0, 2.0])
        # Differences: 9, 1, 5, 1 — informative at > 4: indices 0, 2
        info, uninf = split_field_informative(oracle, stationary, threshold=4.0)
        np.testing.assert_array_equal(info, [True, False, True, False])
        np.testing.assert_array_equal(uninf, [False, True, False, True])

    def test_threshold_boundary_exclusive(self):
        """Exactly equal to threshold is NOT informative (strict >)."""
        oracle     = np.array([5.0])
        stationary = np.array([1.0])
        # difference = 4.0, which is NOT > 4.0
        info, uninf = split_field_informative(oracle, stationary, threshold=4.0)
        assert not info[0]
        assert uninf[0]

    def test_masks_are_complementary(self):
        oracle     = np.random.rand(20)
        stationary = np.zeros(20)
        info, uninf = split_field_informative(oracle, stationary, threshold=0.5)
        np.testing.assert_array_equal(info | uninf, np.ones(20, dtype=bool))
        np.testing.assert_array_equal(info & uninf, np.zeros(20, dtype=bool))

    def test_custom_threshold(self):
        oracle     = np.array([2.0])
        stationary = np.array([0.0])
        info, _ = split_field_informative(oracle, stationary, threshold=1.0)
        assert info[0]  # 2.0 - 0.0 = 2.0 > 1.0


# ---------------------------------------------------------------------------
# ce_report
# ---------------------------------------------------------------------------

class TestCEReport:

    def _make_report(self, ce_vals, info_mask, name="TestAlg"):
        return ce_report(np.array(ce_vals), np.array(info_mask, dtype=bool), name)

    def test_report_structure(self):
        ce_vals   = np.linspace(-0.2, 1.0, 20)
        info_mask = np.array([True] * 10 + [False] * 10)
        rep = self._make_report(ce_vals, info_mask)
        assert "algorithm_name" in rep
        assert "all_seeds" in rep
        assert "field_informative_seeds" in rep
        assert "field_uninformative_seeds" in rep
        assert "interpretation" in rep

    def test_all_seeds_stats(self):
        ce_vals = np.array([0.0, 0.5, 1.0])
        info_mask = np.array([True, True, True])
        rep = self._make_report(ce_vals, info_mask)
        all_s = rep["all_seeds"]
        assert all_s["n"] == 3
        assert pytest.approx(all_s["mean"], abs=1e-6) == float(np.mean(ce_vals))
        assert pytest.approx(all_s["median"], abs=1e-6) == float(np.median(ce_vals))
        assert "q10" in all_s
        assert "q90" in all_s
        assert pytest.approx(all_s["fraction_positive"], abs=1e-6) == 2 / 3

    def test_interpretation_exploits_field(self):
        """exploits_field: informative median > 0.1 AND uninformative mean < 0.05."""
        ce_vals = np.array(
            [0.5, 0.6, 0.7, 0.8,  # informative seeds (high CE)
             0.01, 0.02, 0.01]     # uninformative seeds (near-zero CE)
        )
        info_mask = np.array([True, True, True, True, False, False, False])
        rep = self._make_report(ce_vals, info_mask)
        assert rep["interpretation"] == "exploits_field"

    def test_interpretation_noise_injection(self):
        """noise_injection: uninformative mean > 0.05."""
        ce_vals = np.array(
            [0.0, 0.1,      # informative seeds (low CE)
             0.3, 0.4, 0.5] # uninformative seeds (high CE — noise injection)
        )
        info_mask = np.array([True, True, False, False, False])
        rep = self._make_report(ce_vals, info_mask)
        assert rep["interpretation"] == "noise_injection"

    def test_interpretation_harmful(self):
        """harmful: overall mean CE < 0."""
        ce_vals = np.array([-0.5, -0.3, -0.2, -0.1])
        info_mask = np.array([True, True, False, False])
        rep = self._make_report(ce_vals, info_mask)
        assert rep["interpretation"] == "harmful"

    def test_interpretation_neutral(self):
        """neutral: none of the other conditions met."""
        ce_vals = np.array([0.0, 0.02, 0.03, 0.01])
        info_mask = np.array([True, True, False, False])
        rep = self._make_report(ce_vals, info_mask)
        assert rep["interpretation"] == "neutral"

    def test_algorithm_name_preserved(self):
        rep = self._make_report([0.1, 0.2], [True, False], name="MyAlgorithm")
        assert rep["algorithm_name"] == "MyAlgorithm"

    def test_empty_informative_subset(self):
        """No seeds are field-informative — should not crash."""
        ce_vals = np.array([0.1, 0.2, 0.3])
        info_mask = np.zeros(3, dtype=bool)
        rep = self._make_report(ce_vals, info_mask)
        fi = rep["field_informative_seeds"]
        assert fi["n"] == 0
        assert np.isnan(fi["mean"])

    def test_empty_uninformative_subset(self):
        ce_vals = np.array([0.1, 0.2, 0.3])
        info_mask = np.ones(3, dtype=bool)
        rep = self._make_report(ce_vals, info_mask)
        fu = rep["field_uninformative_seeds"]
        assert fu["n"] == 0
        assert np.isnan(fu["mean"])


# ---------------------------------------------------------------------------
# upstream_alignment_score
# ---------------------------------------------------------------------------

class TestUpstreamAlignmentScore:

    def test_perfect_upstream_alignment(self):
        """Agents moving exactly against field direction should give alignment = 1."""
        field = np.array([1.0, 0.0])  # drift to the right
        # Agents move left (upstream = -field = [-1, 0])
        upstream_dir = np.array([-1.0, 0.0])
        movement = np.tile(upstream_dir, (10, 4, 1))  # (10, 4, 2)
        result = upstream_alignment_score(movement, field)
        assert pytest.approx(result["mean_alignment"], abs=1e-6) == 1.0

    def test_perfect_downstream_alignment(self):
        """Agents moving with field direction should give alignment = -1."""
        field = np.array([1.0, 0.0])
        downstream_dir = np.array([1.0, 0.0])
        movement = np.tile(downstream_dir, (10, 4, 1))
        result = upstream_alignment_score(movement, field)
        assert pytest.approx(result["mean_alignment"], abs=1e-6) == -1.0

    def test_perpendicular_movement(self):
        """Perpendicular movement should give alignment ≈ 0."""
        field = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        movement = np.tile(perp, (10, 4, 1))
        result = upstream_alignment_score(movement, field)
        assert pytest.approx(result["mean_alignment"], abs=1e-6) == 0.0

    def test_stationary_steps_excluded(self):
        """Steps with zero movement should not contribute to alignment."""
        field = np.array([1.0, 0.0])
        upstream = np.array([-1.0, 0.0])
        movement = np.zeros((10, 4, 2))
        movement[0] = upstream  # only first step moves, in upstream direction
        result = upstream_alignment_score(movement, field)
        # Only step 0 contributes; mean should be 1.0
        assert pytest.approx(result["mean_alignment"], abs=1e-6) == 1.0

    def test_per_agent_alignment_shape(self):
        field = np.array([0.0, 1.0])
        movement = np.random.randn(5, 3, 2)
        # Normalise to unit vectors.
        norms = np.linalg.norm(movement, axis=-1, keepdims=True)
        movement = movement / (norms + 1e-9)
        result = upstream_alignment_score(movement, field)
        assert result["per_agent_alignment"].shape == (3,)
        assert result["per_step_alignment"].shape == (5,)

    def test_non_unit_field_handled(self):
        """Field direction is normalised internally; magnitude does not matter."""
        field_unnorm = np.array([3.0, 0.0])
        field_unit   = np.array([1.0, 0.0])
        movement = np.tile(np.array([-1.0, 0.0]), (5, 2, 1))
        r1 = upstream_alignment_score(movement, field_unnorm)
        r2 = upstream_alignment_score(movement, field_unit)
        assert pytest.approx(r1["mean_alignment"], abs=1e-6) == r2["mean_alignment"]

    def test_alignment_std(self):
        """Std should be 0 when all agents move identically."""
        field = np.array([1.0, 0.0])
        movement = np.tile(np.array([-1.0, 0.0]), (5, 4, 1))
        result = upstream_alignment_score(movement, field)
        assert pytest.approx(result["alignment_std"], abs=1e-6) == 0.0


# ---------------------------------------------------------------------------
# spatial_coverage_metrics
# ---------------------------------------------------------------------------

class TestSpatialCoverageMetrics:

    def test_single_agent_no_pairs(self):
        """With one agent, pairwise metrics should be 0 and entropy well-defined."""
        pos = np.random.rand(10, 1, 2)
        result = spatial_coverage_metrics(pos)
        assert result["mean_pairwise_distance"] == 0.0
        assert result["redundancy_rate"] == 0.0
        assert result["coverage_entropy"] >= 0.0

    def test_redundancy_rate_fully_overlapping(self):
        """Agents at the same position should have redundancy rate = 1."""
        pos = np.zeros((20, 4, 2))  # all agents at (0, 0)
        result = spatial_coverage_metrics(pos, sensing_radius=0.16)
        assert pytest.approx(result["redundancy_rate"], abs=1e-6) == 1.0

    def test_redundancy_rate_far_apart(self):
        """Agents very far apart should have redundancy rate near 0."""
        pos = np.zeros((10, 4, 2))
        pos[:, 0, :] = [0.0, 0.0]
        pos[:, 1, :] = [1.0, 0.0]
        pos[:, 2, :] = [0.0, 1.0]
        pos[:, 3, :] = [1.0, 1.0]
        # All pairwise distances > sensing_radius = 0.16
        result = spatial_coverage_metrics(pos, sensing_radius=0.16)
        assert result["redundancy_rate"] == 0.0

    def test_mean_pairwise_distance_known(self):
        """Two agents at (0,0) and (1,0): mean pairwise distance = 1.0."""
        pos = np.zeros((5, 2, 2))
        pos[:, 1, 0] = 1.0  # agent 1 at x=1
        result = spatial_coverage_metrics(pos, sensing_radius=0.1)
        assert pytest.approx(result["mean_pairwise_distance"], abs=1e-6) == 1.0

    def test_coverage_entropy_concentrated(self):
        """All agents in one cell should give lower entropy than spread agents."""
        pos_concentrated = np.full((50, 4, 2), 0.05)  # all in top-left cell
        pos_spread = np.zeros((50, 4, 2))
        pos_spread[:, 0, :] = [0.1, 0.1]
        pos_spread[:, 1, :] = [0.9, 0.1]
        pos_spread[:, 2, :] = [0.1, 0.9]
        pos_spread[:, 3, :] = [0.9, 0.9]

        ent_conc = spatial_coverage_metrics(pos_concentrated)["coverage_entropy"]
        ent_spr  = spatial_coverage_metrics(pos_spread)["coverage_entropy"]
        assert ent_spr > ent_conc

    def test_coverage_entropy_non_negative(self):
        pos = np.random.rand(20, 4, 2)
        result = spatial_coverage_metrics(pos)
        assert result["coverage_entropy"] >= 0.0

    def test_output_keys_present(self):
        pos = np.random.rand(10, 3, 2)
        result = spatial_coverage_metrics(pos)
        assert "mean_pairwise_distance" in result
        assert "redundancy_rate" in result
        assert "coverage_entropy" in result

    def test_redundancy_rate_in_unit_interval(self):
        pos = np.random.rand(10, 4, 2)
        result = spatial_coverage_metrics(pos, sensing_radius=0.5)
        assert 0.0 <= result["redundancy_rate"] <= 1.0


# ---------------------------------------------------------------------------
# CommNet ablated_get_actions (requires torch)
# ---------------------------------------------------------------------------

class TestCommNetAblation:

    torch = pytest.importorskip("torch")

    def test_ablated_actions_shape(self):
        from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
        from particle_benchmark.marl.commnet import CommNet
        from particle_benchmark.marl.networks import compute_obs_dim

        config = ParticleEnvConfig(horizon=67, signal_strength=0.06)
        env = ParticleCollectorEnv(config)
        obs, _ = env.reset(seed=42)
        obs_dim = compute_obs_dim(obs)

        commnet = CommNet(obs_dim=obs_dim, n_agents=config.collector_count)
        ablated_actions = commnet.ablated_get_actions(obs)
        assert ablated_actions.shape == (config.collector_count, 2)

    def test_ablated_actions_in_unit_ball(self):
        from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
        from particle_benchmark.marl.commnet import CommNet
        from particle_benchmark.marl.networks import compute_obs_dim
        import numpy as np

        config = ParticleEnvConfig(horizon=67, signal_strength=0.06)
        env = ParticleCollectorEnv(config)
        obs, _ = env.reset(seed=42)
        obs_dim = compute_obs_dim(obs)

        commnet = CommNet(obs_dim=obs_dim, n_agents=config.collector_count)
        ablated_actions = commnet.ablated_get_actions(obs)
        norms = np.linalg.norm(ablated_actions, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_ablated_differs_from_comm_on_diverse_obs(self):
        """Ablating communication produces different outputs from full forward pass.

        This test uses heterogeneous observations so that the communication
        vectors carry meaningful information. With diverse observations the
        comm round changes the hidden states, leading to different action means.
        The test uses deterministic action means (not stochastic samples) to
        avoid test flakiness, by comparing CommNetModule forward_step outputs.
        """
        import torch
        from particle_benchmark.marl.commnet import CommNetModule
        from particle_benchmark.marl.networks import compute_obs_dim
        from particle_benchmark.environment import ParticleEnvConfig, ParticleCollectorEnv

        config = ParticleEnvConfig(horizon=67, signal_strength=0.06)
        env = ParticleCollectorEnv(config)
        obs, _ = env.reset(seed=123)
        obs_dim = compute_obs_dim(obs)

        torch.manual_seed(0)
        net = CommNetModule(obs_dim, h_dim=32, n_comm_rounds=2)

        from particle_benchmark.marl.networks import flatten_all_observations
        flat_obs = flatten_all_observations(obs)
        obs_t = torch.from_numpy(flat_obs)

        # Inject very different observations for each agent to maximise comm signal.
        obs_t = obs_t.clone()
        obs_t[0] += 50.0   # agent 0 has very different obs

        with torch.no_grad():
            mean_comm, _    = net.forward_step(obs_t, comm_ablated=False)
            mean_ablated, _ = net.forward_step(obs_t, comm_ablated=True)

        # With diverse observations and comm, outputs should differ.
        assert not torch.allclose(mean_comm, mean_ablated, atol=1e-4), (
            "Expected ablated and full-comm action means to differ, "
            "but they were identical. Check that _communicate respects comm_ablated."
        )

    def test_training_evaluation_preserves_agent_groups(self):
        from particle_benchmark.marl.commnet import CommNetModule

        net = CommNetModule(obs_dim=7, h_dim=8, n_comm_rounds=1)
        obs = torch.randn(3, 4, 7)
        actions = torch.randn(3, 4, 2)
        log_prob, entropy, value = net.evaluate_actions(obs, actions)
        assert log_prob.shape == (3, 4)
        assert entropy.shape == (3, 4)
        assert value.shape == (3, 4)

        with pytest.raises(ValueError, match="complete"):
            net.evaluate_actions(obs.reshape(12, 7), actions.reshape(12, 2))
