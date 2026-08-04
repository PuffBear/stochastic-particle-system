#!/usr/bin/env python3
"""FR-B4 mechanism figure: angular dispersion and bias vs. L for slow condition.

For each L level, runs N_SEEDS episodes and records at every step:
  - true field direction theta_true = env._theta_seq[t]
  - shared arm direction estimate theta_shared (team windowed mean; same for all agents)
  - per-agent independent arm direction estimate theta_indep[agent] (own windowed mean)

Outputs JSON with per-L summary stats used to generate fig_dispersion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.policies import (
    _windowed_team_summary,
    local_velocity_summary,
)

STEPS               = 67
ALPHA               = 0.06
DT                  = 0.02
DIFFUSION_SIGMA     = 0.06
PARTICLE_COUNT      = 256
COLLECTOR_COUNT     = 4
COLLECTOR_MAX_SPEED = 0.12
SENSING_RADIUS      = 0.16
OMEGA               = 1.5   # slow condition

L_LEVELS            = [1, 3, 5, 10, 20, 30, 45, 67]
DEFAULT_SEEDS       = list(range(9001, 9033))   # same 32 seeds as main experiment


def _angle_of(vec: np.ndarray) -> float:
    """Signed angle (rad) of a 2-vector; 0 if near-zero magnitude."""
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return float("nan")
    return float(np.arctan2(vec[1], vec[0]))


def _circular_mean(angles: np.ndarray) -> float:
    return float(np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles))))


def _circular_std(angles: np.ndarray) -> float:
    R = float(np.sqrt(np.nanmean(np.sin(angles))**2 + np.nanmean(np.cos(angles))**2))
    return float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))


def _angle_diff(a: float, b: float) -> float:
    """Signed difference a - b, wrapped to (-pi, pi]."""
    d = a - b
    return float((d + np.pi) % (2 * np.pi) - np.pi)


def run_episode_instrumented(seed: int, L: int) -> dict:
    """Run one episode and return per-step direction-estimate data."""
    config = ParticleEnvConfig(
        horizon=STEPS,
        signal_strength=ALPHA,
        dt=DT,
        diffusion_sigma=DIFFUSION_SIGMA,
        particle_count=PARTICLE_COUNT,
        collector_count=COLLECTOR_COUNT,
        collector_max_speed=COLLECTOR_MAX_SPEED,
        sensing_radius=SENSING_RADIUS,
        omega=OMEGA,
    )
    env = ParticleCollectorEnv(config)
    observations, _ = env.reset(seed=seed)
    history: list = []

    # Storage: per step
    theta_true_steps   = []
    theta_shared_steps = []
    theta_indep_steps  = []  # shape: (STEPS, M)

    for t in range(STEPS):
        # True field direction
        theta_true = float(env._theta_seq[t]) if env._theta_seq is not None else 0.0
        theta_true_steps.append(theta_true)

        # Shared arm direction estimate: windowed team mean
        team_summary = _windowed_team_summary(history, observations, L)
        # team_summary[:2] is the estimated field velocity; _angle_of gives theta_hat.
        # Compare with theta_true (both in "downstream" frame) to measure directional lag.
        est_vel = team_summary[:2]
        theta_shared = _angle_of(est_vel)   # estimated field direction
        theta_shared_steps.append(theta_shared)

        # Independent arm direction estimate: each agent's own windowed mean
        agent_thetas = []
        for agent_id in range(COLLECTOR_COUNT):
            solo_history = [(obs_tuple[agent_id],) for obs_tuple in history]
            solo_summary = _windowed_team_summary(solo_history, (observations[agent_id],), L)
            est_vel_solo = solo_summary[:2]
            agent_thetas.append(_angle_of(est_vel_solo))   # estimated field direction
        theta_indep_steps.append(agent_thetas)

        # Advance environment with shared-arm actions (to keep positions realistic)
        from particle_benchmark.policies import _apply_field_density_blend
        actions = np.zeros((COLLECTOR_COUNT, 2), dtype=np.float64)
        for agent_id, obs in enumerate(observations):
            _apply_field_density_blend(team_summary, obs, actions, agent_id)
        history.append(observations)
        observations, _reward, terminated, truncated, _info = env.step(actions)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "L": L,
        "theta_true": theta_true_steps,
        "theta_shared": theta_shared_steps,
        "theta_indep": theta_indep_steps,   # (T, M)
    }


def summarise_L(episodes: list[dict]) -> dict:
    """Compute per-L summary stats from a list of episode dicts."""
    # For each episode, compute per-step errors then average over steps,
    # then aggregate stats over seeds.

    shared_biases = []    # per-seed mean signed error (shared arm)
    shared_abs   = []     # per-seed mean |error| (shared arm)
    indep_biases  = []    # per-seed mean signed error (independent, mean across agents)
    indep_abs     = []    # per-seed mean |error|
    inter_agent_disp_shared = []  # per-seed × step std across agents (shared)
    inter_agent_disp_indep  = []  # per-seed mean over steps of std across agents

    for ep in episodes:
        T = len(ep["theta_true"])
        true = np.array(ep["theta_true"])
        shared = np.array(ep["theta_shared"])
        indep = np.array(ep["theta_indep"])  # (T, M)

        # Signed errors: wrap to (-pi, pi]
        shared_err = np.array([_angle_diff(shared[t], true[t]) for t in range(T)])
        indep_err  = np.array([[_angle_diff(indep[t][m], true[t])
                                for m in range(indep.shape[1])] for t in range(T)])  # (T,M)

        shared_biases.append(float(np.nanmean(shared_err)))
        shared_abs.append(float(np.nanmean(np.abs(shared_err))))
        indep_biases.append(float(np.nanmean(indep_err)))
        indep_abs.append(float(np.nanmean(np.abs(indep_err))))

        # Inter-agent dispersion: std of direction estimates across M agents, per step
        # Shared: all agents same estimate → std = 0 always by construction
        # Confirm and record
        shared_per_step_std = 0.0   # exactly zero: all agents same team estimate
        inter_agent_disp_shared.append(shared_per_step_std)

        # Independent: std of agent directions at each step → average over steps
        if indep.shape[1] > 1:
            # Use circular std across agents at each step
            circ_stds = []
            for t in range(T):
                angles = indep[t]
                valid = angles[~np.isnan(angles)]
                if len(valid) < 2:
                    continue
                circ_stds.append(_circular_std(valid))
            inter_agent_disp_indep.append(float(np.mean(circ_stds)) if circ_stds else float("nan"))
        else:
            inter_agent_disp_indep.append(float("nan"))

    return {
        "L": episodes[0]["L"],
        "n_seeds": len(episodes),
        # Shared arm
        "shared_bias_mean":  float(np.nanmean(shared_biases)),
        "shared_bias_se":    float(np.nanstd(shared_biases, ddof=1) / np.sqrt(len(shared_biases))),
        "shared_abs_mean":   float(np.nanmean(shared_abs)),
        "shared_abs_se":     float(np.nanstd(shared_abs, ddof=1) / np.sqrt(len(shared_abs))),
        "shared_inter_disp": 0.0,   # by construction (all agents get same estimate)
        # Independent arm
        "indep_bias_mean":   float(np.nanmean(indep_biases)),
        "indep_bias_se":     float(np.nanstd(indep_biases, ddof=1) / np.sqrt(len(indep_biases))),
        "indep_abs_mean":    float(np.nanmean(indep_abs)),
        "indep_abs_se":      float(np.nanstd(indep_abs, ddof=1) / np.sqrt(len(indep_abs))),
        "indep_inter_disp_mean": float(np.nanmean(inter_agent_disp_indep)),
        "indep_inter_disp_se":   float(np.nanstd(inter_agent_disp_indep, ddof=1) /
                                       np.sqrt(len(inter_agent_disp_indep))),
        # Noise: std of shared bias across seeds (how variable is the shared estimate)
        "shared_bias_sd":    float(np.nanstd(shared_biases, ddof=1)),
        "indep_bias_sd":     float(np.nanstd(indep_biases, ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FR-B4 dispersion analysis: angular bias & dispersion vs. L at slow"
    )
    parser.add_argument("--seeds", type=str, default="9001:9033")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    start_s, stop_s = args.seeds.split(":")
    seeds = list(range(int(start_s), int(stop_s)))

    print(f"omega={OMEGA}  T_corr={1/(OMEGA*DT):.1f}  n_seeds={len(seeds)}", file=sys.stderr)

    started = time.perf_counter()
    summaries = []

    for L in L_LEVELS:
        print(f"  L={L:2d} ...", end="", flush=True, file=sys.stderr)
        episodes = [run_episode_instrumented(seed, L) for seed in seeds]
        summ = summarise_L(episodes)
        summaries.append(summ)
        print(
            f"  shared_abs={summ['shared_abs_mean']:.3f} rad  "
            f"indep_abs={summ['indep_abs_mean']:.3f} rad  "
            f"indep_disp={summ['indep_inter_disp_mean']:.3f} rad",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - started
    out = {
        "omega": OMEGA,
        "t_corr": 1.0 / (OMEGA * DT),
        "seeds": seeds,
        "elapsed_s": elapsed,
        "summaries": summaries,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nDone in {elapsed:.0f}s. Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
