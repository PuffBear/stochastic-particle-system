"""Minimal end-to-end multi-collector particle environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dynamics.capture import (
    CaptureState,
    resolve_captures,
    resolve_swept_fixed_captures,
)
from .dynamics.collectors import advance_collectors, bounded_velocity
from .dynamics.fields import field_velocity
from .dynamics.particles import advance_free_particles
from .initialization import sample_capture_free_initial_state
from .observations import LocalObservation, build_local_observations
from .seeding import ScenarioStreams, make_streams


@dataclass(frozen=True)
class ParticleEnvConfig:
    """Validated physical and observation parameters for one environment."""

    arena_size: tuple[float, float] = (1.0, 1.0)
    dt: float = 0.02
    horizon: int = 400
    particle_count: int = 256
    diffusion_sigma: float = 0.06
    collector_count: int = 4
    collector_max_speed: float = 0.12
    sensing_radius: float = 0.16
    collector_radius: float = 0.012
    # Point particles are the primary fixed-geometry contract. A nonzero
    # attached-disc/growing-aggregate radius requires a separately frozen model.
    particle_radius: float = 0.0
    field_family: str = "uniform"
    signal_strength: float = 0.20
    field_kwargs: dict[str, object] = field(default_factory=dict)
    capture_geometry: str = "fixed"
    nearest_particles_k: int = 32
    include_particle_velocity: bool = True
    include_teammates: bool = True

    def __post_init__(self) -> None:
        arena = np.asarray(self.arena_size, dtype=np.float64)
        if arena.shape != (2,) or not np.all(np.isfinite(arena)) or np.any(arena <= 0):
            raise ValueError("arena_size must contain two finite positive values")
        if not np.isfinite(self.dt) or self.dt <= 0 or self.horizon <= 0:
            raise ValueError("dt and horizon must be positive")
        if self.particle_count <= 0 or self.collector_count <= 0:
            raise ValueError("particle_count and collector_count must be positive")
        nonnegative = (
            self.diffusion_sigma,
            self.collector_max_speed,
            self.sensing_radius,
            self.collector_radius,
            self.particle_radius,
            self.signal_strength,
        )
        if not all(np.isfinite(value) and value >= 0 for value in nonnegative):
            raise ValueError("physical magnitudes must be finite and non-negative")
        if self.capture_geometry not in {"fixed", "growing"}:
            raise ValueError("capture_geometry must be 'fixed' or 'growing'")
        if self.nearest_particles_k <= 0:
            raise ValueError("nearest_particles_k must be positive")


class ParticleCollectorEnv:
    """A small parallel-agent environment with explicit reset/step semantics.

    Actions have shape ``(M, 2)`` and are normalized holonomic commands.
    Rewards are per-collector counts of newly captured particles. ``terminated``
    means every particle has been captured; ``truncated`` means the fixed
    horizon has been reached. Capture is evaluated after both populations move.
    """

    def __init__(self, config: ParticleEnvConfig | None = None) -> None:
        self.config = config or ParticleEnvConfig()
        self.particle_positions: NDArray[np.float64] | None = None
        self.collector_positions: NDArray[np.float64] | None = None
        self.capture_state: CaptureState | None = None
        self.step_count = 0
        self._particle_velocities: NDArray[np.float64] | None = None
        self._noise: NDArray[np.float32] | None = None
        self._streams: ScenarioStreams | None = None
        self._episode_field_kwargs: dict[str, object] | None = None
        self._last_visibility: NDArray[np.bool_] | None = None
        self._velocity_valid: NDArray[np.bool_] | None = None
        self.first_contact_step: int | None = None
        self.scenario_seed: int | None = None
        self.tie_scheme = "event_keyed_seed_step_particle_v1"
        self._done = False

    def reset(
        self, *, seed: int
    ) -> tuple[tuple[LocalObservation, ...], dict[str, Any]]:
        """Reset deterministically from ``seed`` with no initial capture."""
        self._streams = make_streams(seed)
        self.scenario_seed = seed
        initial = sample_capture_free_initial_state(
            self._streams.initialization,
            particle_count=self.config.particle_count,
            collector_count=self.config.collector_count,
            arena_size=self.config.arena_size,
            exclusion_radius=(
                self.config.collector_radius + self.config.particle_radius
            ),
            clearance=8.0 * np.finfo(np.float64).eps,
        )
        self.particle_positions = initial.particle_positions
        self.collector_positions = initial.collector_positions
        self.capture_state = CaptureState.empty(
            self.config.particle_count, self.config.collector_count
        )
        self._particle_velocities = np.zeros_like(self.particle_positions)
        self._noise = self._streams.noise.standard_normal(
            size=(self.config.horizon, self.config.particle_count, 2),
            dtype=np.float32,
        )
        self._episode_field_kwargs = dict(self.config.field_kwargs)
        if (
            self.config.field_family == "uniform"
            and "orientation" not in self._episode_field_kwargs
        ):
            self._episode_field_kwargs["orientation"] = float(
                self._streams.field.uniform(0.0, 2.0 * np.pi)
            )
        self.step_count = 0
        self.first_contact_step = None
        self._done = False
        self._velocity_valid = np.zeros(
            (self.config.collector_count, self.config.particle_count),
            dtype=np.bool_,
        )
        self._last_visibility = self._visibility()
        return self._observations(), {
            "step": 0,
            "captures": (),
            "captured_total": 0,
            "first_contact_step": None,
            "scenario_seed": self.scenario_seed,
            "tie_scheme": self.tie_scheme,
        }

    def _require_reset(self) -> None:
        if (
            self.particle_positions is None
            or self.collector_positions is None
            or self.capture_state is None
            or self._particle_velocities is None
            or self._noise is None
            or self._streams is None
            or self._episode_field_kwargs is None
            or self._last_visibility is None
            or self._velocity_valid is None
        ):
            raise RuntimeError("reset(seed=...) must be called before step")

    def _observations(self) -> tuple[LocalObservation, ...]:
        self._require_reset()
        assert self.capture_state is not None
        return build_local_observations(
            self.particle_positions,
            self.collector_positions,
            self.capture_state.owner < 0,
            arena_size=self.config.arena_size,
            sensing_radius=self.config.sensing_radius,
            nearest_particles_k=self.config.nearest_particles_k,
            particle_velocities=self._particle_velocities,
            velocity_valid_mask=self._velocity_valid,
            dt=self.config.dt,
            include_particle_velocity=self.config.include_particle_velocity,
            include_teammates=self.config.include_teammates,
        )

    def _visibility(self) -> NDArray[np.bool_]:
        """Return current per-collector visibility of free particles."""
        self._require_reset_except_visibility()
        assert self.particle_positions is not None
        assert self.collector_positions is not None
        assert self.capture_state is not None
        distances = np.linalg.norm(
            self.collector_positions[:, None, :]
            - self.particle_positions[None, :, :],
            axis=2,
        )
        return (distances <= self.config.sensing_radius) & (
            self.capture_state.owner[None, :] < 0
        )

    def _require_reset_except_visibility(self) -> None:
        if (
            self.particle_positions is None
            or self.collector_positions is None
            or self.capture_state is None
        ):
            raise RuntimeError("reset(seed=...) must be called first")

    def step(
        self, actions: ArrayLike
    ) -> tuple[
        tuple[LocalObservation, ...],
        NDArray[np.float64],
        bool,
        bool,
        dict[str, Any],
    ]:
        """Advance one step and return observations, rewards, done flags, info."""
        self._require_reset()
        if self._done:
            raise RuntimeError("episode is done; call reset before another step")
        assert self.particle_positions is not None
        assert self.collector_positions is not None
        assert self.capture_state is not None
        assert self._noise is not None
        assert self._streams is not None
        assert self._episode_field_kwargs is not None
        assert self._last_visibility is not None
        assert self.scenario_seed is not None

        prior_collectors = self.collector_positions.copy()
        collector_velocity = bounded_velocity(
            actions, max_speed=self.config.collector_max_speed
        )
        raw_collector_end = prior_collectors + self.config.dt * collector_velocity
        arena = np.asarray(self.config.arena_size, dtype=np.float64)
        collector_reflected = np.any(
            (raw_collector_end < 0.0) | (raw_collector_end > arena), axis=1
        )
        self.collector_positions = advance_collectors(
            prior_collectors,
            actions,
            dt=self.config.dt,
            max_speed=self.config.collector_max_speed,
            arena_size=self.config.arena_size,
        )
        prior_particles = self.particle_positions.copy()
        free = self.capture_state.owner < 0
        particle_reflected = np.zeros(self.config.particle_count, dtype=np.bool_)
        particle_unfolded_end = prior_particles.copy()
        if np.any(free):
            free_velocity = field_velocity(
                prior_particles[free],
                self.config.field_family,
                self.config.signal_strength,
                **self._episode_field_kwargs,
            )
            raw_particle_end = (
                prior_particles[free]
                + self.config.dt * free_velocity
                + self.config.diffusion_sigma
                * np.sqrt(self.config.dt)
                * self._noise[self.step_count, free]
            )
            particle_reflected[free] = np.any(
                (raw_particle_end < 0.0) | (raw_particle_end > arena), axis=1
            )
            particle_unfolded_end[free] = raw_particle_end
            self.particle_positions[free] = advance_free_particles(
                self.particle_positions[free],
                self._noise[self.step_count, free],
                dt=self.config.dt,
                diffusion_sigma=self.config.diffusion_sigma,
                arena_size=self.config.arena_size,
                field_family=self.config.field_family,
                signal_strength=self.config.signal_strength,
                field_kwargs=self._episode_field_kwargs,
            )
        self._particle_velocities = (
            self.particle_positions - prior_particles
        ) / self.config.dt

        if self.config.capture_geometry == "fixed":
            swept = resolve_swept_fixed_captures(
                prior_particles,
                self.particle_positions,
                prior_collectors,
                self.collector_positions,
                self.capture_state,
                collector_radius=self.config.collector_radius,
                particle_radius=self.config.particle_radius,
                event_seed=self.scenario_seed,
                step_index=self.step_count + 1,
                particle_reflected=particle_reflected,
                collector_reflected=collector_reflected,
                particle_unfolded_end_positions=particle_unfolded_end,
                collector_unfolded_end_positions=raw_collector_end,
                arena_size=self.config.arena_size,
            )
            events = [
                (event.particle_id, event.collector_id) for event in swept.events
            ]
            contact_times = tuple(event.time_fraction for event in swept.events)
            tie_provenance = tuple(
                {
                    "step_index": decision.step_index,
                    "particle_id": decision.particle_id,
                    "candidate_collector_ids": decision.candidate_collector_ids,
                    "chosen_collector_id": decision.chosen_collector_id,
                }
                for decision in swept.tie_decisions
            )
            guarded_reflection_pairs = swept.guarded_reflection_pairs
            contact_model = "fixed_piecewise_specular_reflection_exact_v1"
        else:
            # Growing aggregate paths are intentionally still endpoint-only.
            events = resolve_captures(
                self.particle_positions,
                self.collector_positions,
                self.capture_state,
                geometry=self.config.capture_geometry,
                collector_radius=self.config.collector_radius,
                particle_radius=self.config.particle_radius,
                tie_rng=self._streams.tie_breaking,
            )
            contact_times = tuple(1.0 for _ in events)
            tie_provenance = ()
            guarded_reflection_pairs = 0
            contact_model = "growing_endpoint_only_deferred"
        reward = np.zeros(self.config.collector_count, dtype=np.float64)
        for _, collector_id in events:
            reward[collector_id] += 1.0

        self.step_count += 1
        if events and self.first_contact_step is None:
            self.first_contact_step = self.step_count
        captured_total = int(np.count_nonzero(self.capture_state.owner >= 0))
        current_visibility = self._visibility()
        self._velocity_valid = self._last_visibility & current_visibility
        self._last_visibility = current_visibility
        terminated = captured_total == self.config.particle_count
        truncated = self.step_count >= self.config.horizon and not terminated
        self._done = terminated or truncated
        info = {
            "step": self.step_count,
            "captures": tuple(events),
            "captured_total": captured_total,
            "first_contact_step": self.first_contact_step,
            "scenario_seed": self.scenario_seed,
            "tie_scheme": (
                self.tie_scheme
                if self.config.capture_geometry == "fixed"
                else "stateful_endpoint_growing_deferred"
            ),
            "tie_provenance": tie_provenance,
            "contact_time_fractions": contact_times,
            "contact_model": contact_model,
            "guarded_reflection_pairs": guarded_reflection_pairs,
            "max_particle_chord": float(
                np.max(np.linalg.norm(self.particle_positions - prior_particles, axis=1))
            ),
            "max_collector_chord": float(
                np.max(np.linalg.norm(self.collector_positions - prior_collectors, axis=1))
            ),
            "continuous_contact_limitation": (
                "exact for piecewise-linear rectangular specular paths within one "
                "Euler step; nonlinear continuous-time dynamics still require a dt audit"
            ),
        }
        return self._observations(), reward, terminated, truncated, info
