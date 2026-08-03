#!/usr/bin/env python3
"""FR-B4 supplemental: run intermediate L values to sharpen L_max estimates.

The main grid uses L ∈ {1, 3, 10, 30, 67}. The crossover is visible but the
L grid is too coarse to pin down L_max precisely, especially for slow (T_corr=33,
L_max≈10) and very_slow (T_corr=67, L_max=67). This script adds:

  L5:  between L3 and L10  — needed for mid (is L5 sig? → brackets [3,10])
  L20: between L10 and L30 — needed for slow (L10 sig, L30 not → brackets [10,30])
  L45: between L30 and Lall — needed for very_slow (Lall sig; is L45 sig?)

Runs all four non-stationary omega levels (very_slow, slow, mid, fast).
Uses the same 32 seeds (9001-9032 across 4 batches) as the main corrected grid.

Usage:
  python analysis/run_fr_b4_extra_L.py --main-seeds 9001:9009 --output results/FR-B4/fr_b4_extra_L_run1.json
  python analysis/run_fr_b4_extra_L.py --main-seeds 9009:9017 --output results/FR-B4/fr_b4_extra_L_run2.json
  ...
  python analysis/combine_fr_b4_runs.py results/FR-B4/fr_b4_corrected_combined_32seeds.json \
      results/FR-B4/fr_b4_extra_L_combined_32seeds.json \
      --output results/FR-B4/fr_b4_full_combined.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.policies import (
    capacity_matched_velocity_controller_v2_window,
    capacity_matched_velocity_controller_v2_decay,
)


STEPS  = 67
ALPHA  = 0.06
DT     = 0.02
DIFFUSION_SIGMA      = 0.06
PARTICLE_COUNT       = 256
COLLECTOR_COUNT      = 4
COLLECTOR_MAX_SPEED  = 0.12
SENSING_RADIUS       = 0.16

MAIN_SEEDS = list(range(9001, 9009))

OMEGA_LEVELS = {
    "very_slow": 0.75,
    "slow":      1.5,
    "mid":       5.0,
    "fast":      17.0,
}

# Intermediate L values filling the gaps in the main grid
EXTRA_L_LEVELS = {
    "L5":  5,
    "L20": 20,
    "L45": 45,
}

METHODS = ["window", "decay"]


def _run_episode(seed: int, omega: float, L: int, method: str, *, shared: bool) -> int:
    config = ParticleEnvConfig(
        horizon=STEPS,
        signal_strength=ALPHA,
        dt=DT,
        diffusion_sigma=DIFFUSION_SIGMA,
        particle_count=PARTICLE_COUNT,
        collector_count=COLLECTOR_COUNT,
        collector_max_speed=COLLECTOR_MAX_SPEED,
        sensing_radius=SENSING_RADIUS,
        omega=omega,
    )
    env = ParticleCollectorEnv(config)
    observations, _ = env.reset(seed=seed)
    history: list = []
    controller = (
        capacity_matched_velocity_controller_v2_window
        if method == "window"
        else capacity_matched_velocity_controller_v2_decay
    )
    for _ in range(STEPS):
        actions = controller(observations, history, L, shared=shared)
        history.append(observations)
        observations, _reward, terminated, truncated, _info = env.step(actions)
        if terminated or truncated:
            break
    assert env.capture_state is not None
    return int(np.count_nonzero(env.capture_state.owner >= 0))


def _run_seed_pair(args: tuple[int, float, int, str]) -> tuple[int, int, int]:
    seed, omega, L, method = args
    return (seed,
            _run_episode(seed, omega, L, method, shared=True),
            _run_episode(seed, omega, L, method, shared=False))


def _run_cell(seeds: list[int], omega: float, L: int, method: str, *, jobs: int = 1) -> dict[str, Any]:
    task_args = [(s, omega, L, method) for s in seeds]
    if jobs > 1:
        results_map: dict[int, tuple[int, int]] = {}
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_run_seed_pair, a): a[0] for a in task_args}
            for future in as_completed(futures):
                seed, y_shared, y_indep = future.result()
                results_map[seed] = (y_shared, y_indep)
        seed_results = [(s, *results_map[s]) for s in seeds]
    else:
        seed_results = [_run_seed_pair(a) for a in task_args]

    deltas = [ys - yi for _, ys, yi in seed_results]
    n = len(deltas)
    db = sum(deltas) / n
    sd = (sum((d - db) ** 2 for d in deltas) / (n - 1)) ** 0.5 if n > 1 else 0.0
    sc = sum(1 for d in deltas if d > 0)
    return {
        "delta_bar":  db,
        "delta_sd":   sd,
        "sign_count": sc,
        "n_seeds":    n,
        "per_seed":   [{"seed": s, "y_shared": ys, "y_indep": yi, "delta": ys - yi}
                       for s, ys, yi in seed_results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FR-B4 supplemental: intermediate L values")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    parser.add_argument("--main-seeds", type=str, default=None,
                        help="Seed range as 'start:stop' (e.g. '9001:9009')")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    seeds: list[int]
    if args.main_seeds is not None:
        start_s, stop_s = args.main_seeds.split(":")
        seeds = list(range(int(start_s), int(stop_s)))
    else:
        seeds = MAIN_SEEDS

    n = len(seeds)
    started = time.perf_counter()
    cells_out: list[dict[str, Any]] = []

    for omega_label, omega in OMEGA_LEVELS.items():
        t_corr = 1.0 / (omega * DT)
        for L_label, L in EXTRA_L_LEVELS.items():
            for method in METHODS:
                cell = _run_cell(seeds, omega=omega, L=L, method=method, jobs=jobs)
                print(f"  {omega_label} L={L_label} method={method} "
                      f"Δ̄={cell['delta_bar']:+.3f} sign={cell['sign_count']}/{n}",
                      file=sys.stderr)
                cells_out.append({
                    "omega_label": omega_label,
                    "omega":       omega,
                    "L_label":     L_label,
                    "L":           L,
                    "method":      method,
                    "gate_cell":   False,
                    **cell,
                })

    elapsed = time.perf_counter() - started
    out = {
        "extra_L_levels": EXTRA_L_LEVELS,
        "seeds": seeds,
        "elapsed_s": elapsed,
        "cells": cells_out,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nDone in {elapsed:.0f}s. Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
