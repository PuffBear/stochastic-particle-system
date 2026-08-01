"""Theoretical analysis modules for the stochastic particle system paper.

Three results ground the experimental design and message-design choices:

1. matching   — variance reduction from matched counterfactual pairing
2. estimation — Fisher information and team-mean sufficiency for field direction
3. dilution   — SNR characterisation of the multi-agent observation-overlap trap
"""

from .matching import matched_variance_reduction, power_from_matching
from .estimation import (
    apparent_velocity_noise_std,
    fisher_information_field_direction,
    team_sensing_summary,
)
from .dilution import (
    overlap_probability,
    expected_unique_coverage,
    snr_per_seed,
    dilution_snr_ratio,
)

__all__ = [
    "matched_variance_reduction",
    "power_from_matching",
    "apparent_velocity_noise_std",
    "fisher_information_field_direction",
    "team_sensing_summary",
    "overlap_probability",
    "expected_unique_coverage",
    "snr_per_seed",
    "dilution_snr_ratio",
]
