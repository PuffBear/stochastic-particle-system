#!/usr/bin/env python3
"""FR-B4 supplemental: run one or more new omega values across all 8 L levels.

Purpose: add a third interior anchor point to the L_max ∝ T_corr scaling fit.
The primary target is omega=2.5 (T_corr=20, predicted L_max≈6 → L=5 in grid),
which falls between mid (T_corr=10, L_max=3) and slow (T_corr=33, L_max=10)
and is non-boundary-censored in both directions.

Usage:
  # Primary third anchor (T_corr=20, predicted L_max≈6):
  python analysis/run_fr_b4_new_omega.py --omega 2.5 --seeds 9001:9009 \
      --output results/FR-B4/fr_b4_new_omega_2p5_run1.json

  # Repeat for seed batches 2-4 (9009:9017, 9017:9025, 9025:9033) then combine:
  python analysis/combine_fr_b4_runs.py \
      results/FR-B4/fr_b4_new_omega_2p5_run{1,2,3,4}.json \
      --output results/FR-B4/fr_b4_new_omega_2p5_32seeds.json
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


STEPS               = 67
ALPHA               = 0.06
DT                  = 0.02
DIFFUSION_SIGMA     = 0.06
PARTICLE_COUNT      = 256
COLLECTOR_COUNT     = 4
COLLECTOR_MAX_SPEED = 0.12
SENSING_RADIUS      = 0.16

# Full L grid (same as main corrected grid + intermediate-L extension)
L_LEVELS = [1, 3, 5, 10, 20, 30, 45, 67]

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
    return (
        seed,
        _run_episode(seed, omega, L, method, shared=True),
        _run_episode(seed, omega, L, method, shared=False),
    )


def _run_cell(
    seeds: list[int], omega: float, L: int, method: str, *, jobs: int = 1
) -> dict[str, Any]:
    task_args = [(s, omega, L, method) for s in seeds]
    if jobs > 1:
        results_map: dict[int, tuple[int, int]] = {}
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_run_seed_pair, a): a[0] for a in task_args}
            for fut in as_completed(futures):
                seed, y_shared, y_indep = fut.result()
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
        "per_seed":   [
            {"seed": s, "y_shared": ys, "y_indep": yi, "delta": ys - yi}
            for s, ys, yi in seed_results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FR-B4: run a new omega value across all 8 L levels"
    )
    parser.add_argument(
        "--omega", type=float, required=True,
        help="Rotation speed in rad/step (e.g. 2.5 for T_corr=20)"
    )
    parser.add_argument(
        "--seeds", type=str, default="9001:9009",
        help="Seed range 'start:stop' (default: 9001:9009)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    parser.add_argument(
        "--methods", type=str, default="window,decay",
        help="Comma-separated methods to run (default: window,decay)"
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    start_s, stop_s = args.seeds.split(":")
    seeds = list(range(int(start_s), int(stop_s)))
    methods = [m.strip() for m in args.methods.split(",")]
    omega = args.omega
    t_corr = 1.0 / (omega * DT)
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)

    print(
        f"omega={omega} rad/step  T_corr={t_corr:.1f} steps  "
        f"predicted L_max≈{0.30 * t_corr:.1f}",
        file=sys.stderr,
    )

    started = time.perf_counter()
    cells_out: list[dict[str, Any]] = []

    for L in L_LEVELS:
        for method in methods:
            cell = _run_cell(seeds, omega=omega, L=L, method=method, jobs=jobs)
            n = len(seeds)
            print(
                f"  L={L:2d}  method={method}  "
                f"Δ̄={cell['delta_bar']:+.3f}  sign={cell['sign_count']}/{n}",
                file=sys.stderr,
            )
            cells_out.append({
                "omega_label": f"new_omega_{omega}",
                "omega":       omega,
                "t_corr":      t_corr,
                "L":           L,
                "method":      method,
                **cell,
            })

    elapsed = time.perf_counter() - started
    out = {
        "omega":     omega,
        "t_corr":    t_corr,
        "L_levels":  L_LEVELS,
        "seeds":     seeds,
        "elapsed_s": elapsed,
        "cells":     cells_out,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nDone in {elapsed:.0f}s. Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
