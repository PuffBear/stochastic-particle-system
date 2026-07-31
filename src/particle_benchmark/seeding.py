"""Independent random streams and exact matched-counterfactual noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScenarioStreams:
    """Random generators whose responsibilities may not be mixed."""

    initialization: np.random.Generator
    noise: np.random.Generator
    field: np.random.Generator
    tie_breaking: np.random.Generator


def make_streams(seed: int) -> ScenarioStreams:
    """Derive four independent streams deterministically from one root seed."""
    root = np.random.SeedSequence(seed)
    init_ss, noise_ss, field_ss, tie_ss = root.spawn(4)
    return ScenarioStreams(
        initialization=np.random.default_rng(init_ss),
        noise=np.random.default_rng(noise_ss),
        field=np.random.default_rng(field_ss),
        tie_breaking=np.random.default_rng(tie_ss),
    )


def brownian_tensor(
    seed: int, horizon: int, particle_count: int
) -> NDArray[np.float32]:
    """Pre-generate standard-normal forcing for an exact matched pair."""
    if horizon <= 0 or particle_count <= 0:
        raise ValueError("horizon and particle_count must be positive")
    streams = make_streams(seed)
    return streams.noise.standard_normal(
        size=(horizon, particle_count, 2), dtype=np.float32
    )

