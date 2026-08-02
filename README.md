# Stochastic Particle System

A controlled multi-agent benchmark and trajectory dataset for measuring when locally informed collectors can detect and exploit weak structure in stochastic particle motion.

## Primary research question

> At α=0.06, with four collectors and a 67-step fixed-horizon window, does one bounded three-number team velocity summary (shared_summary_v2) increase unique team capture yield relative to an identical-shape capacity-matched independent controller?

**Confirmed (SPS-C03, 2026-08-01):** Pre-registered one-sided studentized-bootstrap 95% lower bound = +0.459 > 0. Mean +1.19 unique particles, SD 2.44, 20/32 seeds positive. MARL baselines (6 archs × 8 seeds) currently running on ShARC HPC.

The first release studies four mobile collectors, 256 non-learning stochastic particles, local observations, and a uniform latent field. Aggregation, learned communication, and scale are downstream analyses.

Development occurs through tested, task-scoped changes on the `research-autonomy` branch.
