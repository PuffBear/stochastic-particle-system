"""Low-compute tests for FR-B3 observation and checkpoint transfer contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from analysis.evaluate_fr_b3_transfer import (
    canonical_config,
    design_summary,
    evaluate_bundle_episode,
    load_protocol,
    run_registered_evaluation,
)
from particle_benchmark.catchability import rescale_equivalent_config
from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.marl.representations import (
    AdaptedObservationEnv,
    FRB3ObservationAdapter,
    FrozenObservationStandardizer,
)
from particle_benchmark.observations import build_local_observations


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiments" / "fr_b3_learned_transfer.yaml"


def _physical_observations(
    config: ParticleEnvConfig,
    *,
    length_scale: float = 1.0,
    time_scale: float = 1.0,
) -> tuple:
    particles = length_scale * np.asarray(
        [[0.28, 0.26], [0.32, 0.21], [0.72, 0.76]], dtype=np.float64
    )
    collectors = length_scale * np.asarray(
        [[0.20, 0.20], [0.80, 0.80]], dtype=np.float64
    )
    velocities = (length_scale / time_scale) * np.asarray(
        [[0.08, -0.02], [0.03, 0.04], [-0.06, 0.01]], dtype=np.float64
    )
    return build_local_observations(
        particles,
        collectors,
        np.ones(3, dtype=np.bool_),
        arena_size=config.arena_size,
        sensing_radius=config.sensing_radius,
        nearest_particles_k=config.nearest_particles_k,
        particle_velocities=velocities,
        velocity_valid_mask=np.ones((2, 3), dtype=np.bool_),
        dt=config.dt,
        include_particle_velocity=True,
        include_teammates=True,
    )


def _base_config() -> ParticleEnvConfig:
    return ParticleEnvConfig(
        arena_size=(1.0, 1.0),
        dt=0.02,
        horizon=3,
        particle_count=8,
        collector_count=2,
        diffusion_sigma=0.06,
        collector_max_speed=0.12,
        sensing_radius=0.16,
        collector_radius=0.012,
        signal_strength=0.06,
        nearest_particles_k=4,
    )


@pytest.mark.parametrize(
    ("length_scale", "time_scale"),
    ((2.0, 1.0), (1.0, 4.0), (0.5, 0.25)),
)
def test_dimensionless_adapter_is_invariant_under_registered_rescalings(
    length_scale: float, time_scale: float
) -> None:
    base = ParticleEnvConfig(
        collector_count=2, nearest_particles_k=3, signal_strength=0.06
    )
    scaled = rescale_equivalent_config(
        base, length_scale=length_scale, time_scale=time_scale
    )
    canonical = FRB3ObservationAdapter(base, "dimensionless").adapt_all(
        _physical_observations(base)
    )
    transformed = FRB3ObservationAdapter(scaled, "dimensionless").adapt_all(
        _physical_observations(
            scaled, length_scale=length_scale, time_scale=time_scale
        )
    )
    for left, right in zip(canonical, transformed, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            if left[key].dtype == np.bool_:
                np.testing.assert_array_equal(left[key], right[key])
            else:
                np.testing.assert_allclose(left[key], right[key], rtol=1e-12, atol=1e-12)


def test_raw_adapter_recovers_physical_units_and_preserves_layout() -> None:
    base = ParticleEnvConfig(
        collector_count=2, nearest_particles_k=3, signal_strength=0.06
    )
    raw = FRB3ObservationAdapter(base, "raw_physical").adapt_all(
        _physical_observations(base)
    )
    dimensionless = FRB3ObservationAdapter(base, "dimensionless").adapt_all(
        _physical_observations(base)
    )
    for raw_observation, dim_observation in zip(raw, dimensionless, strict=True):
        assert raw_observation.keys() == dim_observation.keys()
        assert sum(np.asarray(value).size for value in raw_observation.values()) == 25
        np.testing.assert_array_equal(
            raw_observation["particle_mask"], dim_observation["particle_mask"]
        )
        np.testing.assert_array_equal(
            raw_observation["velocity_valid_mask"],
            dim_observation["velocity_valid_mask"],
        )
        np.testing.assert_allclose(
            raw_observation["particles"][:, 2:4] * base.dt / base.sensing_radius,
            dim_observation["particles"][:, 2:4],
        )


def test_frozen_standardizer_round_trip_and_masks_do_not_change(tmp_path: Path) -> None:
    config = _base_config()
    env = ParticleCollectorEnv(config)
    native, _ = env.reset(seed=101)
    adapted = FRB3ObservationAdapter(config, "dimensionless").adapt_all(native)
    standardizer = FrozenObservationStandardizer.fit(
        [adapted], representation="dimensionless"
    )
    before_hash = standardizer.sha256
    transformed = standardizer.transform_all(adapted)
    for source, target in zip(adapted, transformed, strict=True):
        np.testing.assert_array_equal(source["particle_mask"], target["particle_mask"])
        np.testing.assert_array_equal(
            source["velocity_valid_mask"], target["velocity_valid_mask"]
        )
    path = tmp_path / "standardizer.json"
    standardizer.save(path)
    restored = FrozenObservationStandardizer.load(path)
    assert restored.sha256 == before_hash
    assert standardizer.sha256 == before_hash
    assert not hasattr(standardizer, "update")


def test_adapted_environment_never_mutates_frozen_statistics() -> None:
    config = _base_config()
    adapter = FRB3ObservationAdapter(config, "raw_physical")
    fitting_env = ParticleCollectorEnv(config)
    native, _ = fitting_env.reset(seed=102)
    standardizer = FrozenObservationStandardizer.fit(
        [adapter.adapt_all(native)], representation="raw_physical"
    )
    before = standardizer.sha256
    wrapped = AdaptedObservationEnv(ParticleCollectorEnv(config), adapter, standardizer)
    observations, _ = wrapped.reset(seed=103)
    for _ in range(2):
        observations, _, terminated, truncated, _ = wrapped.step(
            np.zeros((config.collector_count, 2))
        )
        if terminated or truncated:
            break
    assert standardizer.sha256 == before


def test_candidate_design_is_complete_but_execution_is_blocked(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL)
    summary = design_summary(protocol)
    assert summary["training_run_count"] == 20
    assert summary["primary_evaluation_episode_count"] == 5120
    assert summary["commnet_ablation_episode_count"] == 2560
    assert summary["total_evaluation_episode_count"] == 7680
    with pytest.raises(RuntimeError, match="not registered"):
        run_registered_evaluation(
            protocol,
            bundle_dirs=[],
            repository_root=tmp_path,
        )


def test_candidate_cli_dry_run_and_fail_closed_execution(tmp_path: Path) -> None:
    command = [
        "python",
        str(ROOT / "analysis" / "evaluate_fr_b3_transfer.py"),
        "--config",
        str(PROTOCOL),
    ]
    dry_run = subprocess.run(
        [*command, "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert json.loads(dry_run.stdout)["total_evaluation_episode_count"] == 7680
    blocked = subprocess.run(
        [*command, "--output", str(tmp_path / "forbidden.json")],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert blocked.returncode != 0
    assert "not registered" in blocked.stderr
    assert not (tmp_path / "forbidden.json").exists()


def test_policy_bundle_writer_hashes_artifacts_without_training(tmp_path: Path) -> None:
    from particle_benchmark.marl.transfer import write_policy_bundle

    config = _base_config()
    env = ParticleCollectorEnv(config)
    native, _ = env.reset(seed=107)
    adapted = FRB3ObservationAdapter(config, "raw_physical").adapt_all(native)
    standardizer = FrozenObservationStandardizer.fit(
        [adapted], representation="raw_physical"
    )

    class FakePolicy:
        n_agents = config.collector_count
        obs_dim = 32

        @staticmethod
        def save(path: Path) -> None:
            path.write_bytes(b"development-only fake checkpoint")

    bundle_dir = tmp_path / "bundle"
    metadata = write_policy_bundle(
        bundle_dir,
        policy=FakePolicy(),
        architecture="ippo",
        representation="raw_physical",
        training_seed=108,
        checkpoint_episode=10,
        canonical_config=config,
        standardizer=standardizer,
    )
    assert metadata["adapter_contract_version"].startswith("fr_b3_")
    assert len(metadata["model"]["sha256"]) == 64
    assert len(metadata["standardizer"]["sha256"]) == 64
    with pytest.raises(FileExistsError, match="immutable policy bundle"):
        write_policy_bundle(
            bundle_dir,
            policy=FakePolicy(),
            architecture="ippo",
            representation="raw_physical",
            training_seed=108,
            checkpoint_episode=10,
            canonical_config=config,
            standardizer=standardizer,
        )


@pytest.mark.parametrize("architecture", ("ippo", "commnet"))
def test_checkpoint_bundle_is_deterministic_and_hash_checked(
    architecture: str, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from particle_benchmark.marl.networks import compute_obs_dim
    from particle_benchmark.marl.transfer import (
        deterministic_actions,
        load_policy_bundle,
        write_policy_bundle,
    )

    config = _base_config()
    env = ParticleCollectorEnv(config)
    native, _ = env.reset(seed=104)
    adapter = FRB3ObservationAdapter(config, "dimensionless")
    adapted = adapter.adapt_all(native)
    standardizer = FrozenObservationStandardizer.fit(
        [adapted], representation="dimensionless"
    )
    transformed = standardizer.transform_all(adapted)
    obs_dim = compute_obs_dim(transformed)
    if architecture == "ippo":
        from particle_benchmark.marl.ippo import IPPO

        policy = IPPO(obs_dim=obs_dim, n_agents=config.collector_count)
    else:
        from particle_benchmark.marl.commnet import CommNet

        policy = CommNet(
            obs_dim=obs_dim,
            n_agents=config.collector_count,
            h_dim=16,
            n_comm_rounds=1,
        )
    bundle_dir = tmp_path / architecture
    write_policy_bundle(
        bundle_dir,
        policy=policy,
        architecture=architecture,
        representation="dimensionless",
        training_seed=105,
        checkpoint_episode=10,
        canonical_config=config,
        standardizer=standardizer,
    )
    bundle = load_policy_bundle(bundle_dir, canonical_config=config)
    first = deterministic_actions(bundle, transformed)
    second = deterministic_actions(bundle, transformed)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.linalg.norm(first, axis=1) <= 1.0 + 1e-12)
    if architecture == "commnet":
        ablated = deterministic_actions(
            bundle, transformed, communication_ablated=True
        )
        assert ablated.shape == first.shape

    first_episode = evaluate_bundle_episode(
        bundle,
        environment=config,
        seed=106,
        communication_ablated=False,
    )
    second_episode = evaluate_bundle_episode(
        bundle,
        environment=config,
        seed=106,
        communication_ablated=False,
    )
    assert first_episode == second_episode

    with (bundle_dir / "model.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_policy_bundle(bundle_dir, canonical_config=config)
