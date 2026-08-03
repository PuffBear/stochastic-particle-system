"""Observation representations for the FR-B3 learned scale-transfer study.

The environment's native observation is mixed-unit: positions and distances
are normalized, while apparent particle velocities remain in physical
length/time units.  This module makes the representation intervention explicit
without changing entity selection, ordering, masks, or slot count.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from particle_benchmark.catchability import characteristic_length
from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.io import canonical_json_bytes
from particle_benchmark.observations import LocalObservation


Representation = Literal["raw_physical", "dimensionless"]
REPRESENTATIONS: tuple[Representation, ...] = (
    "raw_physical",
    "dimensionless",
)
ADAPTER_CONTRACT_VERSION = "fr_b3_observation_representation_v1"
STANDARDIZER_SCHEMA_VERSION = "fr_b3_frozen_standardizer_v1"
CONTINUOUS_KEYS = (
    "self_position",
    "particles",
    "teammate_relative_positions",
)


def _readonly_array(value: object, *, dtype: np.dtype = np.float64) -> NDArray:
    array = np.asarray(value, dtype=dtype).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("standardizer arrays must be finite")
    array.setflags(write=False)
    return array


def _validate_observation(observation: Mapping[str, object]) -> None:
    required = {
        "self_position",
        "particles",
        "particle_mask",
        "velocity_valid_mask",
    }
    missing = required - set(observation)
    if missing:
        raise ValueError(f"observation is missing required keys: {sorted(missing)}")
    self_position = np.asarray(observation["self_position"], dtype=np.float64)
    particles = np.asarray(observation["particles"], dtype=np.float64)
    particle_mask = np.asarray(observation["particle_mask"], dtype=np.bool_)
    velocity_mask = np.asarray(observation["velocity_valid_mask"], dtype=np.bool_)
    if self_position.shape != (2,):
        raise ValueError("self_position must have shape (2,)")
    if particles.ndim != 2 or particles.shape[1] != 5:
        raise ValueError("particles must have shape (K, 5)")
    if particle_mask.shape != (particles.shape[0],):
        raise ValueError("particle_mask must have shape (K,)")
    if velocity_mask.shape != (particles.shape[0],):
        raise ValueError("velocity_valid_mask must have shape (K,)")
    if not np.all(np.isfinite(self_position)) or not np.all(np.isfinite(particles)):
        raise ValueError("continuous observation slots must be finite")
    if "teammate_relative_positions" in observation:
        teammates = np.asarray(
            observation["teammate_relative_positions"], dtype=np.float64
        )
        if teammates.ndim != 2 or teammates.shape[1] != 2:
            raise ValueError("teammate_relative_positions must have shape (M-1, 2)")
        if not np.all(np.isfinite(teammates)):
            raise ValueError("teammate slots must be finite")


@dataclass(frozen=True)
class FRB3ObservationAdapter:
    """Convert native observations to one frozen FR-B3 representation."""

    config: ParticleEnvConfig
    representation: Representation

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation: {self.representation!r}")

    @property
    def contract_version(self) -> str:
        return ADAPTER_CONTRACT_VERSION

    def adapt(self, observation: Mapping[str, object]) -> LocalObservation:
        """Return a copy with identical layout and explicitly defined units."""

        _validate_observation(observation)
        arena = np.asarray(self.config.arena_size, dtype=np.float64)
        sensing = float(self.config.sensing_radius)
        length = characteristic_length(self.config.arena_size)

        native_self = np.asarray(observation["self_position"], dtype=np.float64)
        native_particles = np.asarray(observation["particles"], dtype=np.float64)
        physical_self = native_self * arena
        physical_particles = native_particles.copy()
        physical_particles[:, :2] *= sensing
        physical_particles[:, 4] *= sensing

        if self.representation == "raw_physical":
            adapted_self = physical_self
            adapted_particles = physical_particles
        else:
            adapted_self = physical_self / length
            adapted_particles = physical_particles.copy()
            adapted_particles[:, :2] /= sensing
            adapted_particles[:, 2:4] *= self.config.dt / sensing
            adapted_particles[:, 4] /= sensing

        adapted: LocalObservation = {
            "self_position": adapted_self,
            "particles": adapted_particles,
            "particle_mask": np.asarray(
                observation["particle_mask"], dtype=np.bool_
            ).copy(),
            "velocity_valid_mask": np.asarray(
                observation["velocity_valid_mask"], dtype=np.bool_
            ).copy(),
        }
        if "teammate_relative_positions" in observation:
            native_teammates = np.asarray(
                observation["teammate_relative_positions"], dtype=np.float64
            )
            physical_teammates = native_teammates * arena
            adapted["teammate_relative_positions"] = (
                physical_teammates
                if self.representation == "raw_physical"
                else physical_teammates / length
            )
        return adapted

    def adapt_all(
        self, observations: tuple[Mapping[str, object], ...]
    ) -> tuple[LocalObservation, ...]:
        return tuple(self.adapt(observation) for observation in observations)


@dataclass(frozen=True)
class FrozenObservationStandardizer:
    """Canonical-training statistics with no update or target-adaptation API."""

    representation: Representation
    sample_count: int
    means: Mapping[str, NDArray[np.float64]]
    scales: Mapping[str, NDArray[np.float64]]
    schema_version: str = STANDARDIZER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation: {self.representation!r}")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.schema_version != STANDARDIZER_SCHEMA_VERSION:
            raise ValueError("unsupported standardizer schema")
        if set(self.means) != set(self.scales):
            raise ValueError("mean and scale keys differ")
        if not set(self.means).issubset(CONTINUOUS_KEYS):
            raise ValueError("standardizer contains an unsupported feature key")
        means: dict[str, NDArray[np.float64]] = {}
        scales: dict[str, NDArray[np.float64]] = {}
        for key in sorted(self.means):
            mean = _readonly_array(self.means[key])
            scale = _readonly_array(self.scales[key])
            if mean.shape != scale.shape:
                raise ValueError(f"mean/scale shape mismatch for {key}")
            if np.any(scale <= 0.0):
                raise ValueError(f"standardizer scales must be positive for {key}")
            means[key] = mean
            scales[key] = scale
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)

    @classmethod
    def fit(
        cls,
        batches: Iterable[tuple[Mapping[str, object], ...]],
        *,
        representation: Representation,
        minimum_scale: float = 1e-8,
    ) -> "FrozenObservationStandardizer":
        """Fit once from canonical training observations and return frozen stats."""

        if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be finite and positive")
        values: dict[str, list[NDArray[np.float64]]] = {}
        count = 0
        expected_shapes: dict[str, tuple[int, ...]] | None = None
        for batch in batches:
            for observation in batch:
                _validate_observation(observation)
                current = {
                    key: np.asarray(observation[key], dtype=np.float64)
                    for key in CONTINUOUS_KEYS
                    if key in observation
                }
                shapes = {key: array.shape for key, array in current.items()}
                if expected_shapes is None:
                    expected_shapes = shapes
                elif shapes != expected_shapes:
                    raise ValueError("all fitted observations must have identical layout")
                for key, array in current.items():
                    values.setdefault(key, []).append(array.copy())
                count += 1
        if count == 0:
            raise ValueError("cannot fit a standardizer without observations")
        means: dict[str, NDArray[np.float64]] = {}
        scales: dict[str, NDArray[np.float64]] = {}
        for key, arrays in values.items():
            stacked = np.stack(arrays, axis=0)
            means[key] = np.mean(stacked, axis=0)
            standard_deviation = np.std(stacked, axis=0)
            scales[key] = np.where(
                standard_deviation >= minimum_scale, standard_deviation, 1.0
            )
        return cls(
            representation=representation,
            sample_count=count,
            means=means,
            scales=scales,
        )

    def transform(self, observation: Mapping[str, object]) -> LocalObservation:
        """Standardize continuous slots while copying binary masks unchanged."""

        _validate_observation(observation)
        missing = set(self.means) - set(observation)
        if missing:
            raise ValueError(f"observation is missing standardized keys: {sorted(missing)}")
        transformed: LocalObservation = {
            "self_position": np.asarray(
                observation["self_position"], dtype=np.float64
            ).copy(),
            "particles": np.asarray(observation["particles"], dtype=np.float64).copy(),
            "particle_mask": np.asarray(
                observation["particle_mask"], dtype=np.bool_
            ).copy(),
            "velocity_valid_mask": np.asarray(
                observation["velocity_valid_mask"], dtype=np.bool_
            ).copy(),
        }
        if "teammate_relative_positions" in observation:
            transformed["teammate_relative_positions"] = np.asarray(
                observation["teammate_relative_positions"], dtype=np.float64
            ).copy()
        for key, mean in self.means.items():
            value = np.asarray(observation[key], dtype=np.float64)
            if value.shape != mean.shape:
                raise ValueError(f"observation shape differs from fitted {key} layout")
            transformed[key] = (value - mean) / self.scales[key]
        return transformed

    def transform_all(
        self, observations: tuple[Mapping[str, object], ...]
    ) -> tuple[LocalObservation, ...]:
        return tuple(self.transform(observation) for observation in observations)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "representation": self.representation,
            "sample_count": self.sample_count,
            "features": {
                key: {
                    "shape": list(self.means[key].shape),
                    "mean": self.means[key].tolist(),
                    "scale": self.scales[key].tolist(),
                }
                for key in sorted(self.means)
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenObservationStandardizer":
        features = dict(payload["features"])
        means: dict[str, NDArray[np.float64]] = {}
        scales: dict[str, NDArray[np.float64]] = {}
        for key, raw_feature in features.items():
            feature = dict(raw_feature)
            shape = tuple(int(item) for item in feature["shape"])
            mean = np.asarray(feature["mean"], dtype=np.float64)
            scale = np.asarray(feature["scale"], dtype=np.float64)
            if mean.shape != shape or scale.shape != shape:
                raise ValueError(f"serialized standardizer shape mismatch for {key}")
            means[str(key)] = mean
            scales[str(key)] = scale
        return cls(
            representation=str(payload["representation"]),
            sample_count=int(payload["sample_count"]),
            means=means,
            scales=scales,
            schema_version=str(payload["schema_version"]),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"immutable standardizer already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> "FrozenObservationStandardizer":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class AdaptedObservationEnv:
    """Environment view that applies a representation and frozen statistics."""

    def __init__(
        self,
        env: ParticleCollectorEnv,
        adapter: FRB3ObservationAdapter,
        standardizer: FrozenObservationStandardizer | None = None,
    ) -> None:
        if env.config != adapter.config:
            raise ValueError("adapter config must match the wrapped environment")
        if standardizer is not None and (
            standardizer.representation != adapter.representation
        ):
            raise ValueError("standardizer and adapter representations differ")
        self.env = env
        self.adapter = adapter
        self.standardizer = standardizer

    @property
    def config(self) -> ParticleEnvConfig:
        return self.env.config

    def _transform(
        self, observations: tuple[Mapping[str, object], ...]
    ) -> tuple[LocalObservation, ...]:
        adapted = self.adapter.adapt_all(observations)
        return (
            adapted
            if self.standardizer is None
            else self.standardizer.transform_all(adapted)
        )

    def reset(self, *, seed: int) -> tuple[tuple[LocalObservation, ...], dict]:
        observations, info = self.env.reset(seed=seed)
        return self._transform(observations), info

    def step(self, actions: object) -> tuple:
        observations, reward, terminated, truncated, info = self.env.step(actions)
        return self._transform(observations), reward, terminated, truncated, info

    def __getattr__(self, name: str) -> object:
        return getattr(self.env, name)
