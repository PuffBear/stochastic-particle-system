"""Latent transport fields for the stochastic-particle benchmark.

The ``EpisodeFrozenGaussianField`` below is the SPS-C04 field primitive.  It
uses a finite Fourier representation of a stationary Gaussian vector field on
a rectangular torus.  Its ensemble covariance is explicit (and evaluable by
``component_covariance``), each velocity component has the same pointwise
marginal variance for every declared correlation length, and one sampled
realization is immutable for the lifetime of an episode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class EpisodeFrozenGaussianField:
    r"""A stationary, episode-frozen Gaussian vector field on a 2-D torus.

    For velocity component ``r`` the finite spectral realization is

    .. math::

       v_r(x)=s_0 a_{r0}+\sum_k s_k
       [a_{rk}\cos(\omega_k^T x)+b_{rk}\sin(\omega_k^T x)],

    with independent standard-normal coefficients.  The exact ensemble
    covariance of either component is therefore

    .. math::

       C_K(h)=\sigma_v^2[w_0+\sum_k w_k\cos(\omega_k^T h)].

    The normalized spectral weights are a truncated periodic
    squared-exponential spectrum with declared kernel correlation length
    ``correlation_length``.  Because ``w_0 + sum(w_k) == 1``, the marginal
    component variance is exactly ``component_variance`` at every location and
    for every correlation length.  The two vector components are independent
    under the ensemble contract.

    Notes
    -----
    ``correlation_length`` is a kernel parameter in physical arena units.  The
    covariance used by this finite field is always the value returned by
    :meth:`component_covariance`; it is not the legacy vortex envelope scale.
    """

    correlation_length: float
    component_variance: float
    arena_size: tuple[float, float]
    max_frequency: int
    wavevectors: NDArray[np.float64]
    normalized_mode_weights: NDArray[np.float64]
    normalized_constant_weight: float
    constant_coefficients: NDArray[np.float64]
    cosine_coefficients: NDArray[np.float64]
    sine_coefficients: NDArray[np.float64]

    @classmethod
    def sample(
        cls,
        *,
        seed: int,
        correlation_length: float,
        component_variance: float = 1.0,
        arena_size: tuple[float, float] = (1.0, 1.0),
        max_frequency: int = 8,
    ) -> "EpisodeFrozenGaussianField":
        """Sample one immutable realization from the declared covariance."""
        if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if not np.isfinite(correlation_length) or correlation_length <= 0:
            raise ValueError("correlation_length must be finite and positive")
        if not np.isfinite(component_variance) or component_variance <= 0:
            raise ValueError("component_variance must be finite and positive")
        size = np.asarray(arena_size, dtype=np.float64)
        if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError("arena_size must contain two finite positive lengths")
        if not isinstance(max_frequency, (int, np.integer)) or max_frequency <= 0:
            raise ValueError("max_frequency must be a positive integer")

        # One representative from each +/- lattice-frequency pair.  Pairing a
        # cosine and sine coefficient produces a real stationary field.
        lattice = np.array(
            [
                (nx, ny)
                for nx in range(-int(max_frequency), int(max_frequency) + 1)
                for ny in range(-int(max_frequency), int(max_frequency) + 1)
                if nx > 0 or (nx == 0 and ny > 0)
            ],
            dtype=np.float64,
        )
        wavevectors = 2.0 * np.pi * lattice / size[None, :]
        base_weights = np.exp(
            -0.5 * float(correlation_length) ** 2
            * np.sum(wavevectors**2, axis=1)
        )
        # Each stored mode represents the +/- pair in the full spectrum.
        pair_weights = 2.0 * base_weights
        normalizer = 1.0 + float(np.sum(pair_weights))
        constant_weight = 1.0 / normalizer
        mode_weights = pair_weights / normalizer

        rng = np.random.default_rng(int(seed))
        constant = rng.standard_normal(2)
        cosine = rng.standard_normal((2, wavevectors.shape[0]))
        sine = rng.standard_normal((2, wavevectors.shape[0]))

        # Frozen dataclasses do not by themselves make ndarray contents
        # immutable.  Read-only arrays prevent a field from changing midway
        # through an episode through an accidental in-place write.
        for array in (wavevectors, mode_weights, constant, cosine, sine):
            array.setflags(write=False)

        return cls(
            correlation_length=float(correlation_length),
            component_variance=float(component_variance),
            arena_size=(float(size[0]), float(size[1])),
            max_frequency=int(max_frequency),
            wavevectors=wavevectors,
            normalized_mode_weights=mode_weights,
            normalized_constant_weight=float(constant_weight),
            constant_coefficients=constant,
            cosine_coefficients=cosine,
            sine_coefficients=sine,
        )

    def velocity(self, positions: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the frozen realization at positions with shape ``(N, 2)``."""
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2 or not np.all(np.isfinite(pos)):
            raise ValueError("positions must be finite with shape (N, 2)")
        phase = pos @ self.wavevectors.T
        constant_scale = np.sqrt(
            self.component_variance * self.normalized_constant_weight
        )
        mode_scale = np.sqrt(
            self.component_variance * self.normalized_mode_weights
        )
        values = constant_scale * self.constant_coefficients[:, None]
        values = values + np.sum(
            mode_scale[None, :, None]
            * (
                self.cosine_coefficients[:, :, None]
                * np.cos(phase).T[None, :, :]
                + self.sine_coefficients[:, :, None]
                * np.sin(phase).T[None, :, :]
            ),
            axis=1,
        )
        return values.T.copy()

    def component_covariance(self, displacements: ArrayLike) -> NDArray[np.float64]:
        """Return exact finite-basis covariance for either velocity component."""
        delta = np.asarray(displacements, dtype=np.float64)
        scalar_input = delta.shape == (2,)
        if scalar_input:
            delta = delta[None, :]
        if delta.ndim != 2 or delta.shape[1] != 2 or not np.all(np.isfinite(delta)):
            raise ValueError("displacements must be finite with shape (N, 2)")
        phase = delta @ self.wavevectors.T
        covariance = self.component_variance * (
            self.normalized_constant_weight
            + np.cos(phase) @ self.normalized_mode_weights
        )
        return covariance[0] if scalar_input else covariance

    def realization_sha256(self) -> str:
        """Return a canonical checksum for matched-arm provenance.

        Numeric values use network-order IEEE-754 bytes, making the digest
        independent of native byte order and Python's float representation.
        """
        digest = hashlib.sha256()
        digest.update(b"episode_frozen_gaussian_field_v1\0")
        for value in (
            self.correlation_length,
            self.component_variance,
            self.arena_size[0],
            self.arena_size[1],
            self.normalized_constant_weight,
        ):
            digest.update(struct.pack("!d", float(value)))
        digest.update(struct.pack("!I", self.max_frequency))
        for array in (
            self.wavevectors,
            self.normalized_mode_weights,
            self.constant_coefficients,
            self.cosine_coefficients,
            self.sine_coefficients,
        ):
            digest.update(struct.pack("!I", array.ndim))
            digest.update(np.asarray(array.shape, dtype=">u8").tobytes(order="C"))
            digest.update(np.asarray(array, dtype=">f8").tobytes(order="C"))
        return digest.hexdigest()


def field_velocity(
    positions: ArrayLike,
    family: str,
    signal_strength: float,
    *,
    orientation: float = 0.0,
    centre: tuple[float, float] = (0.5, 0.5),
    vortex_scale: float = 0.25,
    clockwise: bool = False,
    frozen_field: EpisodeFrozenGaussianField | None = None,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Return deterministic field velocity with shape ``(N, 2)``."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if not np.isfinite(signal_strength) or signal_strength < 0:
        raise ValueError("signal_strength must be finite and non-negative")

    if family == "null":
        return np.zeros_like(pos)
    if family == "uniform":
        direction = np.array([np.cos(orientation), np.sin(orientation)])
        return np.broadcast_to(signal_strength * direction, pos.shape).copy()
    if family == "periodic_gaussian":
        if frozen_field is None:
            raise ValueError(
                "periodic_gaussian requires an episode-frozen field realization"
            )
        return signal_strength * frozen_field.velocity(pos)
    if family != "vortex":
        raise ValueError(f"unknown field family: {family}")
    if vortex_scale <= 0 or not np.isfinite(vortex_scale):
        raise ValueError("vortex_scale must be finite and positive")

    centre_array = np.asarray(centre, dtype=np.float64)
    if centre_array.shape != (2,):
        raise ValueError("centre must be a pair")
    radius = pos - centre_array
    norm = np.linalg.norm(radius, axis=1, keepdims=True)
    tangent = np.concatenate((-radius[:, 1:2], radius[:, 0:1]), axis=1)
    tangent = tangent / (norm + epsilon)
    envelope = np.exp(-(norm**2) / (2.0 * vortex_scale**2))
    sign = -1.0 if clockwise else 1.0
    velocity = sign * signal_strength * envelope * tangent
    velocity[norm[:, 0] <= epsilon] = 0.0
    return velocity
