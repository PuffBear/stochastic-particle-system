#!/usr/bin/env python3
"""FR-B4: Adaptive Coordination — rotating field experiment grid.

Runs 40 conditions (4 omega x 5 L x 2 methods) plus the omega=0 reproduction
gate.  For each main-grid condition, 8 paired seeds (9001-9008) are run: each
seed is run twice (shared arm and independent arm) with identical omega, L, and
method.  Unique particle captures through step 67 is the metric.

Pre-registered grid (from adaptive-coordination/experiments/grid-design.md):
  omega: 0, pi/200, pi/100, pi/50 rad/step
  L:     1, 3, 10, 30, 67 (=all)
  method: window, decay
  seeds:  gate=6001-6032 (32 confirmed), main=9001-9008 (8 matched)

Reproduction gate design note:
  Gate uses L=1 (stateless, matches SPS-C03 per-step behavior for shared arm)
  and 32 SPS-C03 confirmed seeds (6001-6032) for sufficient statistical power.
  The original design used L=all and seeds 1001-1008; this was corrected after
  finding that (a) temporal pooling at L=all changes both arms' estimates in a
  way that differs from the stateless SPS-C03 controller, and (b) 8 seeds give
  SE≈0.86 which is too wide to detect the +1.19 effect reliably.

Run order (per grid-design.md, priority):
  1. Reproduction gate: (omega=0, L=1, both methods, 32 seeds)
  2. Fast rotation diagnostic: (omega=pi/50, L in {1,3,10})
  3. Full slow + mid: (omega in {pi/200, pi/100}, all L)
  4. Complete fast: remaining L for omega=pi/50
  5. omega=0 at all non-all L levels (32 seeds)

Usage:
  python analysis/run_fr_b4_adaptive_coordination.py --output results/fr_b4.json
  python analysis/run_fr_b4_adaptive_coordination.py --output results/fr_b4.json -j 8
  python analysis/run_fr_b4_adaptive_coordination.py --output results/gate.json --conditions gate
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
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


# ── Frozen experimental constants ────────────────────────────────────────────

STEPS = 67
ALPHA = 0.06
DT = 0.02
DIFFUSION_SIGMA = 0.06
PARTICLE_COUNT = 256
COLLECTOR_COUNT = 4
COLLECTOR_MAX_SPEED = 0.12
SENSING_RADIUS = 0.16

# Reproduction gate uses the 32 confirmed SPS-C03 seeds (6001-6032) at L=1.
# L=1 is the stateless case that exactly matches SPS-C03's per-step behaviour.
# Seeds 1001-1008 have too high variance (8 seeds, SD≈2.44 gives SE≈0.86) to
# reliably detect a +1.19 effect, and L=all temporal pooling is not equivalent
# to the stateless SPS-C03 controller.
GATE_SEEDS = list(range(6001, 6033))   # 32 confirmed SPS-C03 seeds
GATE_L     = 1                         # L=1 matches SPS-C03 stateless controller
MAIN_SEEDS = list(range(9001, 9009))   # non-zero omega cells

L_ALL = STEPS  # sentinel: full episode history

OMEGA_LEVELS = {
    "stationary": 0.0,
    "slow":       math.pi / 200,
    "mid":        math.pi / 100,
    "fast":       math.pi / 50,
}

L_LEVELS = {
    "L1":  1,
    "L3":  3,
    "L10": 10,
    "L30": 30,
    "Lall": L_ALL,
}

METHODS = ["window", "decay"]

# Gate criterion: Δ̄ in [+0.69, +1.69] and sign count >= 20/32.
# Matches SPS-C03 confirmed result on seeds 6001-6032 at L=1 (stateless).
GATE_DELTA_LO  = 0.69
GATE_DELTA_HI  = 1.69
GATE_SIGN_MIN  = 20

# L_critical criterion: sign count >= 60% (>= 5/8 seeds)
LCRIT_SIGN_MIN = 5


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(
    seed: int,
    omega: float,
    L: int,
    method: str,
    *,
    shared: bool,
) -> int:
    """Run one episode; return unique particle captures through step STEPS."""
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
        observations, _reward, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    assert env.capture_state is not None
    return int(np.count_nonzero(env.capture_state.owner >= 0))


def _run_seed_pair(args: tuple[int, float, int, str]) -> tuple[int, int, int]:
    """Run one seed pair (shared + independent); returns (seed, y_shared, y_indep)."""
    seed, omega, L, method = args
    return (seed, _run_episode(seed, omega, L, method, shared=True),
            _run_episode(seed, omega, L, method, shared=False))


def _run_cell(
    seeds: list[int],
    omega: float,
    L: int,
    method: str,
    *,
    jobs: int = 1,
) -> dict[str, Any]:
    """Run one (omega, L, method) cell over all seeds; return cell summary."""
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

    deltas: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for seed, y_shared, y_indep in seed_results:
        delta = y_shared - y_indep
        deltas.append(float(delta))
        per_seed.append({"seed": seed, "shared": y_shared, "independent": y_indep, "delta": delta})

    deltas_arr = np.asarray(deltas)
    n = len(seeds)
    return {
        "delta_bar":  float(np.mean(deltas_arr)),
        "delta_sd":   float(np.std(deltas_arr, ddof=1)) if n > 1 else float("nan"),
        "sign_count": int(np.sum(deltas_arr > 0)),
        "n_seeds":    n,
        "per_seed":   per_seed,
    }


# ── Gate check ───────────────────────────────────────────────────────────────

def check_reproduction_gate(
    gate_results: dict[str, dict[str, Any]],
) -> bool:
    """Return True iff BOTH methods pass the omega=0, L=1, 32-seed gate.

    Gate criterion: Δ̄ ∈ [+0.69, +1.69] AND sign_count >= 20/32.
    This matches the SPS-C03 confirmed result (+1.19, 20/32 positive) on the
    same seeds (6001-6032) using the stateless L=1 controller.
    """
    passed = True
    for method in METHODS:
        cell = gate_results[method]
        db   = cell["delta_bar"]
        sc   = cell["sign_count"]
        n    = cell["n_seeds"]
        ok   = GATE_DELTA_LO <= db <= GATE_DELTA_HI and sc >= GATE_SIGN_MIN
        print(
            f"  gate [{method}]: Δ̄={db:+.3f}  sign={sc}/{n}  {'PASS' if ok else 'FAIL'}",
            file=sys.stderr,
        )
        passed = passed and ok
    return passed


# ── L_critical derivation ─────────────────────────────────────────────────────

def find_l_critical(
    omega_label: str,
    method: str,
    results: dict[tuple, dict[str, Any]],
) -> int | None:
    """Return smallest L with sign_count >= LCRIT_SIGN_MIN, or None."""
    for L_label in ["L1", "L3", "L10", "L30", "Lall"]:
        L = L_LEVELS[L_label]
        cell = results.get((omega_label, L_label, method))
        if cell is None:
            continue
        if cell["sign_count"] >= LCRIT_SIGN_MIN:
            return L
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FR-B4 adaptive coordination experiment")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--conditions",
        choices=["gate", "fast_diagnostic", "slow_mid", "full", "all"],
        default="all",
        help="Which conditions to run (gate=reproduction gate only; all=full grid)",
    )
    parser.add_argument(
        "--no-gate-halt",
        action="store_true",
        help="Continue even if reproduction gate fails (for debugging)",
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). Use os.cpu_count() for all cores.",
    )
    args = parser.parse_args()
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    all_results: dict[tuple, dict[str, Any]] = {}
    started = time.perf_counter()

    # ── Phase 1: Reproduction gate ────────────────────────────────────────────
    print(f"Phase 1: Reproduction gate (omega=0, L={GATE_L}, both methods, {len(GATE_SEEDS)} seeds)", file=sys.stderr)
    print("  L=1 matches the stateless SPS-C03 per-step controller exactly.", file=sys.stderr)
    gate_results: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        cell = _run_cell(GATE_SEEDS, omega=0.0, L=GATE_L, method=method, jobs=jobs)
        gate_results[method] = cell
        all_results[("stationary", "L1_gate", method)] = cell
        print(f"  gate omega=0 L={GATE_L} method={method} Δ̄={cell['delta_bar']:+.3f} sign={cell['sign_count']}/{len(GATE_SEEDS)}", file=sys.stderr)

    gate_passed = check_reproduction_gate(gate_results)
    if not gate_passed:
        msg = "REPRODUCTION GATE FAILED. Aborting (use --no-gate-halt to override)."
        print(msg, file=sys.stderr)
        if not args.no_gate_halt:
            sys.exit(1)

    if args.conditions == "gate":
        _write_output(args.output, all_results, gate_passed, started)
        return

    # ── Phase 2: Fast rotation diagnostic ────────────────────────────────────
    if args.conditions in ("fast_diagnostic", "slow_mid", "full", "all"):
        print("Phase 2: Fast rotation diagnostic (omega=pi/50, L in {1,3,10})", file=sys.stderr)
        for L_label in ["L1", "L3", "L10"]:
            L = L_LEVELS[L_label]
            for method in METHODS:
                cell = _run_cell(MAIN_SEEDS, omega=OMEGA_LEVELS["fast"], L=L, method=method, jobs=jobs)
                all_results[("fast", L_label, method)] = cell
                print(f"  fast L={L_label} method={method} Δ̄={cell['delta_bar']:+.3f} sign={cell['sign_count']}/8", file=sys.stderr)

        if args.conditions == "fast_diagnostic":
            _write_output(args.output, all_results, gate_passed, started)
            return

    # ── Phase 3: Full slow and mid ────────────────────────────────────────────
    if args.conditions in ("slow_mid", "full", "all"):
        for omega_label in ["slow", "mid"]:
            print(f"Phase 3: omega={omega_label} — all L levels", file=sys.stderr)
            for L_label, L in L_LEVELS.items():
                for method in METHODS:
                    cell = _run_cell(MAIN_SEEDS, omega=OMEGA_LEVELS[omega_label], L=L, method=method, jobs=jobs)
                    all_results[(omega_label, L_label, method)] = cell
                    print(f"  {omega_label} L={L_label} method={method} Δ̄={cell['delta_bar']:+.3f} sign={cell['sign_count']}/8", file=sys.stderr)

        if args.conditions == "slow_mid":
            _write_output(args.output, all_results, gate_passed, started)
            return

    # ── Phase 4: Complete fast rotation ──────────────────────────────────────
    if args.conditions in ("full", "all"):
        print("Phase 4: Fast rotation remaining L levels (L30, Lall)", file=sys.stderr)
        for L_label in ["L30", "Lall"]:
            L = L_LEVELS[L_label]
            for method in METHODS:
                cell = _run_cell(MAIN_SEEDS, omega=OMEGA_LEVELS["fast"], L=L, method=method, jobs=jobs)
                all_results[("fast", L_label, method)] = cell
                print(f"  fast L={L_label} method={method} Δ̄={cell['delta_bar']:+.3f} sign={cell['sign_count']}/8", file=sys.stderr)

    # ── Phase 5: omega=0 at non-all L levels ─────────────────────────────────
    if args.conditions == "all":
        print("Phase 5: omega=0 at non-all L levels", file=sys.stderr)
        for L_label in ["L1", "L3", "L10", "L30"]:
            L = L_LEVELS[L_label]
            for method in METHODS:
                cell = _run_cell(GATE_SEEDS, omega=0.0, L=L, method=method, jobs=jobs)
                all_results[("stationary", L_label, method)] = cell
                print(f"  stationary L={L_label} method={method} Δ̄={cell['delta_bar']:+.3f} sign={cell['sign_count']}/8", file=sys.stderr)

    _write_output(args.output, all_results, gate_passed, started)


def _write_output(
    path: Path,
    all_results: dict[tuple, dict[str, Any]],
    gate_passed: bool,
    started: float,
) -> None:
    # Compute L_critical per (omega, method)
    l_critical: dict[str, dict[str, int | None]] = {}
    for omega_label in OMEGA_LEVELS:
        l_critical[omega_label] = {}
        for method in METHODS:
            l_critical[omega_label][method] = find_l_critical(omega_label, method, all_results)

    # Extended L lookup that includes the gate-only key
    L_lookup = {**L_LEVELS, "L1_gate": GATE_L}

    # Serialise results (convert tuple keys to strings)
    cells_out: list[dict[str, Any]] = []
    for (omega_label, L_label, method), cell in sorted(all_results.items()):
        cells_out.append({
            "omega_label": omega_label,
            "omega": OMEGA_LEVELS.get(omega_label, 0.0),
            "L_label": L_label,
            "L": L_lookup.get(L_label, -1),
            "method": method,
            "gate_cell": L_label == "L1_gate",
            **cell,
        })

    output: dict[str, Any] = {
        "experiment_id": "FR-B4-ADAPTIVE-COORDINATION",
        "git_branch": "fr-b4-adaptive-coordination",
        "constants": {
            "steps": STEPS,
            "alpha": ALPHA,
            "dt": DT,
            "diffusion_sigma": DIFFUSION_SIGMA,
            "particle_count": PARTICLE_COUNT,
            "collector_count": COLLECTOR_COUNT,
        },
        "seeds": {"gate": GATE_SEEDS, "gate_L": GATE_L, "main": MAIN_SEEDS},
        "reproduction_gate_passed": gate_passed,
        "gate_criterion": {
            "condition": f"omega=0, L={GATE_L} (stateless), {len(GATE_SEEDS)} SPS-C03 confirmed seeds",
            "delta_bar_range": [GATE_DELTA_LO, GATE_DELTA_HI],
            "sign_count_min": GATE_SIGN_MIN,
            "note": "L=1 matches stateless SPS-C03; independent arm uses density blend unlike SPS-C03 v2",
        },
        "l_critical": l_critical,
        "l_critical_criterion": {"sign_count_min": LCRIT_SIGN_MIN, "n_seeds": 8},
        "cells": cells_out,
        "runtime_seconds": time.perf_counter() - started,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to {path}", file=sys.stderr)
    print(f"Reproduction gate: {'PASSED' if gate_passed else 'FAILED'}", file=sys.stderr)
    print("\nL_critical summary:", file=sys.stderr)
    for omega_label, by_method in l_critical.items():
        for method, lc in by_method.items():
            lc_str = str(lc) if lc is not None else "undefined"
            print(f"  {omega_label:12s} {method:6s}  L_critical={lc_str}", file=sys.stderr)


if __name__ == "__main__":
    main()
