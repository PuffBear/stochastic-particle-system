#!/usr/bin/env python3
"""Γ-conservation sweep: find L_max*(ω) across a fine ω grid.

For each ω, sweeps L and records shared-arm reward to find the optimal
window length L_max*(ω).  The dimensionless group Γ(ω) = ω·dt·L_max*
should be approximately constant in the linear-rotation regime.

Predicted: L_max*(ω) ∝ 1/(ω·dt) = T_corr, so Γ = ω·dt·L_max* ≈ const.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.policies import capacity_matched_velocity_controller_v2_window

# ── Constants (frozen from main experiment) ───────────────────────────────────
STEPS               = 67
ALPHA               = 0.06
DT                  = 0.02
DIFFUSION_SIGMA     = 0.06
PARTICLE_COUNT      = 256
COLLECTOR_COUNT     = 4
COLLECTOR_MAX_SPEED = 0.12
SENSING_RADIUS      = 0.16

# ω grid: 7 values spanning from well below to well above the interior speeds.
# Excludes omega=0 (no rotation → no L_max) and fast (anomalous regime).
OMEGA_GRID = [0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 8.0]

# L sweep: 10 values covering 1–30 (coarser at large L to save time).
L_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45]

DEFAULT_SEEDS = list(range(9001, 9017))   # 16 matched seeds


def _run_shared_episode(seed: int, omega: float, L: int) -> int:
    """Shared arm only: unique captures through step STEPS."""
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

    for _ in range(STEPS):
        actions = capacity_matched_velocity_controller_v2_window(
            observations, history, L, shared=True
        )
        history.append(observations)
        observations, _r, terminated, truncated, _info = env.step(actions)
        if terminated or truncated:
            break

    assert env.capture_state is not None
    return int(np.count_nonzero(env.capture_state.owner >= 0))


def _worker(args: tuple) -> tuple:
    seed, omega, L = args
    return (seed, omega, L, _run_shared_episode(seed, omega, L))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Γ-conservation sweep: L_max*(ω) across ω grid"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="9001:9017",
                        help="Seed range start:stop (exclusive stop)")
    parser.add_argument("--omega-grid", type=str, default=None,
                        help="Comma-separated ω values; default: %(default)s")
    parser.add_argument("--l-grid", type=str, default=None,
                        help="Comma-separated L values")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    start_s, stop_s = args.seeds.split(":")
    seeds = list(range(int(start_s), int(stop_s)))

    omega_grid = (
        [float(x) for x in args.omega_grid.split(",")]
        if args.omega_grid else OMEGA_GRID
    )
    l_grid = (
        [int(x) for x in args.l_grid.split(",")]
        if args.l_grid else L_GRID
    )

    print(f"ω grid:  {omega_grid}", file=sys.stderr)
    print(f"L grid:  {l_grid}", file=sys.stderr)
    print(f"seeds:   {seeds[0]}–{seeds[-1]}  n={len(seeds)}", file=sys.stderr)
    print(f"jobs:    {args.jobs}", file=sys.stderr)
    print(f"cells:   {len(omega_grid) * len(l_grid)}", file=sys.stderr)
    print(file=sys.stderr)

    # Build all (seed, omega, L) task tuples
    tasks = [
        (s, omega, L)
        for omega in omega_grid
        for L in l_grid
        for s in seeds
    ]

    # Collect results into nested dict: results[omega][L] = [captures, ...]
    results: dict[float, dict[int, list[int]]] = {
        omega: {L: [] for L in l_grid} for omega in omega_grid
    }

    started = time.perf_counter()
    done = 0
    total = len(tasks)

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for future in as_completed(futures):
                seed, omega, L, cap = future.result()
                results[omega][L].append(cap)
                done += 1
                if done % max(1, total // 20) == 0:
                    pct = 100 * done / total
                    elapsed = time.perf_counter() - started
                    eta = elapsed / done * (total - done)
                    print(f"  {done}/{total} ({pct:.0f}%)  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                          file=sys.stderr)
    else:
        for t in tasks:
            seed, omega, L, cap = _worker(t)
            results[omega][L].append(cap)
            done += 1
            if done % max(1, total // 20) == 0:
                elapsed = time.perf_counter() - started
                print(f"  {done}/{total}  elapsed={elapsed:.0f}s", file=sys.stderr)

    elapsed = time.perf_counter() - started

    # Summarise per (omega, L)
    cells = []
    print("\nSummary:", file=sys.stderr)
    print(f"{'ω':>6}  {'T_corr':>7}  {'L':>4}  {'mean':>7}  {'se':>6}", file=sys.stderr)

    for omega in omega_grid:
        t_corr = 1.0 / (omega * DT)
        for L in l_grid:
            caps = results[omega][L]
            n = len(caps)
            mean = float(np.mean(caps))
            se   = float(np.std(caps, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            cells.append({
                "omega": omega, "t_corr": t_corr, "L": L,
                "mean": mean, "se": se, "n": n,
                "captures": caps,
            })
            print(f"{omega:6.2f}  {t_corr:7.1f}  {L:4d}  {mean:7.3f}  {se:6.3f}",
                  file=sys.stderr)

    out = {
        "omega_grid": omega_grid,
        "l_grid": l_grid,
        "seeds": seeds,
        "alpha": ALPHA,
        "dt": DT,
        "diffusion_sigma": DIFFUSION_SIGMA,
        "collector_count": COLLECTOR_COUNT,
        "elapsed_s": elapsed,
        "cells": cells,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nDone in {elapsed:.0f}s. Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
