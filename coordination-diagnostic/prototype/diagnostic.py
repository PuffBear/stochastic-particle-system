"""
Coordination Diagnostic Tool — v0.1

Three-condition test: actual messages / permuted messages / no messages.
Produces a provenance certificate (JSON) showing whether a multi-agent
communication protocol adds value beyond its structural format.

Usage:
    python diagnostic.py --traces-dir ./traces --output certificate.json

Trace format: one JSONL file per seed, each line a step dict with keys:
    seed, condition, step, messages, metric (final step only)

See methodology/diagnostic-spec.md for full input specification.
"""

import json
import random
import argparse
import math
from pathlib import Path
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_traces(traces_dir: Path) -> dict[int, dict[str, list]]:
    """
    Load trace files from a directory. Returns:
        {seed: {"actual": [steps], "permuted": [steps], "none": [steps]}}

    Expects files named <seed>_<condition>.jsonl where condition ∈
    {actual, permuted, none}.
    """
    traces: dict[int, dict[str, list]] = {}
    for f in sorted(traces_dir.glob("*.jsonl")):
        parts = f.stem.split("_", 1)
        if len(parts) != 2:
            continue
        seed, condition = int(parts[0]), parts[1]
        if condition not in ("actual", "permuted", "none"):
            continue
        steps = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        traces.setdefault(seed, {})[condition] = steps
    return traces


def extract_metric(steps: list[dict]) -> Optional[float]:
    """Return the outcome metric from a trace (last step that has it)."""
    for step in reversed(steps):
        if "metric" in step:
            return float(step["metric"])
        if "outcome" in step and "metric" in step["outcome"]:
            return float(step["outcome"]["metric"])
    return None


# ---------------------------------------------------------------------------
# Permutation utility (for generating Condition B from Condition A traces)
# ---------------------------------------------------------------------------

def permute_messages(steps: list[dict], rng: random.Random) -> list[dict]:
    """
    Given an actual-messages trace, return a new trace where at each step
    the messages are randomly permuted across receivers. Message content and
    format are preserved; only the sender-receiver assignment is scrambled.
    """
    permuted_steps = []
    for step in steps:
        new_step = dict(step)
        msgs = step.get("messages", [])
        if msgs:
            values = [m["value"] for m in msgs]
            rng.shuffle(values)
            new_step["messages"] = [
                {**m, "value": v} for m, v in zip(msgs, values)
            ]
        permuted_steps.append(new_step)
    return permuted_steps


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sign_count(xs: list[float]) -> int:
    return sum(1 for x in xs if x > 0)


def studentized_bootstrap_lower_bound(
    deltas: list[float], n_boot: int = 9999, confidence: float = 0.95, seed: int = 42
) -> float:
    """
    One-sided (lower) studentized bootstrap confidence bound.
    Returns the lower bound on E[Delta].
    """
    rng = random.Random(seed)
    n = len(deltas)
    mu_hat = mean(deltas)
    se_hat = std(deltas) / math.sqrt(n)

    t_stats = []
    for _ in range(n_boot):
        boot = [rng.choice(deltas) for _ in range(n)]
        mu_b = mean(boot)
        se_b = std(boot) / math.sqrt(n)
        if se_b > 0:
            t_stats.append((mu_b - mu_hat) / se_b)

    t_stats.sort()
    q = t_stats[int(confidence * len(t_stats))]
    return mu_hat - q * se_hat


def gate_threshold(n: int) -> int:
    """5% sign-test gate: minimum positive signs needed."""
    # Smallest k such that P(Binomial(n, 0.5) >= k) <= 0.05
    from math import comb
    total = 2 ** n
    for k in range(n, -1, -1):
        p_geq_k = sum(comb(n, j) for j in range(k, n + 1)) / total
        if p_geq_k > 0.05:
            return k + 1
    return n


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

DIAGNOSIS_PATTERNS = {
    "content_drives_gain":  "Content drives gain. The message content is informative; most benefit comes from what is communicated, not just that something is communicated.",
    "structure_drives_gain": "Structure drives gain. The communication scaffold provides most of the benefit; message content adds inconsistently. The content encoding is a candidate for improvement without changing channel architecture.",
    "content_harmful":       "Content is harmful. Scrambled messages outperform actual messages, indicating the protocol is causing correlated errors. The aggregation rule likely violates the sufficient statistic for the underlying quantity.",
    "communication_inert":   "Communication is inert. Neither content nor structure produces consistent benefit. Verify that the policy actually conditions on incoming messages and that assumption 3 (no leakage) holds.",
    "mixed":                 "Mixed result. Content and structure both contribute, or neither clearly dominates. Inspect per-seed breakdown for heterogeneous effects.",
}


def diagnose(
    content_gate: str,
    structure_gate: str,
    mean_content: float,
    mean_structure: float,
) -> tuple[str, str]:
    if content_gate == "FAIL" and structure_gate == "FAIL":
        if mean_content < 0:
            pattern = "content_harmful"
        else:
            pattern = "communication_inert"
    elif content_gate == "PASS" and structure_gate == "FAIL":
        pattern = "content_drives_gain"
    elif content_gate == "FAIL" and structure_gate == "PASS":
        pattern = "structure_drives_gain"
    else:
        pattern = "mixed"
    return pattern, DIAGNOSIS_PATTERNS[pattern]


# ---------------------------------------------------------------------------
# Certificate builder
# ---------------------------------------------------------------------------

def build_certificate(
    system_name: str,
    protocol_name: str,
    seed_results: list[dict],
    condition_descriptions: dict[str, str],
    assumptions: dict[str, str],
    engagement_type: str = "diagnostic",
    run_confirmatory: bool = False,
) -> dict:

    n = len(seed_results)
    deltas_content   = [r["delta_content"]   for r in seed_results]
    deltas_structure = [r["delta_structure"] for r in seed_results]
    deltas_net       = [r["delta_net"]       for r in seed_results]

    thresh = gate_threshold(n)

    sc_content   = sign_count(deltas_content)
    sc_structure = sign_count(deltas_structure)

    content_gate   = "PASS" if sc_content   >= thresh else "FAIL"
    structure_gate = "PASS" if sc_structure >= thresh else "FAIL"

    confirmatory_lb = None
    confirmatory_gate = None
    if run_confirmatory:
        confirmatory_lb = studentized_bootstrap_lower_bound(deltas_net)
        confirmatory_gate = "PASS" if confirmatory_lb > 0 else "FAIL"

    pattern, plain = diagnose(
        content_gate, structure_gate,
        mean(deltas_content), mean(deltas_structure),
    )

    cert_text = _render_certificate_text(
        system_name, protocol_name, n, engagement_type,
        mean(deltas_net), sc_content, n, content_gate,
        mean(deltas_content), sc_structure, structure_gate,
        mean(deltas_structure), pattern, confirmatory_lb, confirmatory_gate,
    )

    return {
        "meta": {
            "system_name": system_name,
            "protocol_name": protocol_name,
            "diagnostic_version": "0.1",
            "run_date": date.today().isoformat(),
            "seed_set": [r["seed"] for r in seed_results],
            "metric_higher_is_better": True,
            "engagement_type": engagement_type,
        },
        "conditions": condition_descriptions,
        "assumptions": assumptions,
        "results": {
            "per_seed": seed_results,
            "content": {
                "mean": round(mean(deltas_content), 4),
                "sd": round(std(deltas_content), 4),
                "sign_count": sc_content,
                "n": n,
                "gate_threshold": thresh,
                "gate_result": content_gate,
            },
            "structure": {
                "mean": round(mean(deltas_structure), 4),
                "sd": round(std(deltas_structure), 4),
                "sign_count": sc_structure,
                "n": n,
                "gate_threshold": thresh,
                "gate_result": structure_gate,
            },
            "net": {
                "mean": round(mean(deltas_net), 4),
                "sd": round(std(deltas_net), 4),
                "confirmatory_lower_bound": round(confirmatory_lb, 4) if confirmatory_lb is not None else None,
                "confirmatory_gate_result": confirmatory_gate,
            },
        },
        "diagnosis": {"pattern": pattern, "plain_language": plain},
        "certificate_text": cert_text,
    }


def _render_certificate_text(
    system_name, protocol_name, n, engagement_type,
    mean_net, sc_content, n_total, content_gate,
    mean_content, sc_structure, structure_gate,
    mean_structure, pattern, lb, conf_gate,
) -> str:
    lines = [
        "COORDINATION DIAGNOSTIC CERTIFICATE",
        "=" * 36,
        f"System:   {system_name}",
        f"Protocol: {protocol_name}",
        f"Date:     {date.today().isoformat()}",
        f"Seeds:    {n} ({engagement_type})",
        "",
        f"Net coordination value:  {mean_net:+.3f}  ({sc_content}/{n_total} seeds positive vs no-messages)",
        f"Gain from structure:     {mean_structure:+.3f}  ({sc_structure}/{n_total} positive)  — {structure_gate}",
        f"Gain from content:       {mean_content:+.3f}  ({sc_content}/{n_total} positive)  — {content_gate}",
    ]
    if lb is not None:
        lines.append(f"95% lower bound (net):   {lb:+.3f}  — {conf_gate}")
    lines += [
        "",
        f"Diagnosis: {pattern.upper().replace('_', ' ')}",
        "",
        "This certificate covers the specific conditions and seed set listed above.",
        "It does not certify behaviour under untested conditions.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostic(
    traces_dir: Path,
    output_path: Path,
    system_name: str,
    protocol_name: str,
    confirmatory: bool = False,
    permute_seed: int = 0,
):
    traces = load_traces(traces_dir)
    if not traces:
        raise ValueError(f"No valid traces found in {traces_dir}")

    seed_results = []
    for seed in sorted(traces.keys()):
        seed_traces = traces[seed]

        # Require at least Condition A; derive B and C if not present
        if "actual" not in seed_traces:
            print(f"  Seed {seed}: missing 'actual' condition — skipping")
            continue

        y_actual = extract_metric(seed_traces["actual"])

        if "permuted" in seed_traces:
            y_permuted = extract_metric(seed_traces["permuted"])
        else:
            rng = random.Random(permute_seed + seed)
            permuted_steps = permute_messages(seed_traces["actual"], rng)
            y_permuted = extract_metric(permuted_steps)
            if y_permuted is None:
                y_permuted = extract_metric(seed_traces["actual"])

        if "none" in seed_traces:
            y_none = extract_metric(seed_traces["none"])
        else:
            y_none = None

        if any(v is None for v in [y_actual, y_permuted]):
            print(f"  Seed {seed}: could not extract metric — skipping")
            continue

        delta_content   = (y_actual   - y_permuted)               if y_permuted is not None else None
        delta_structure = (y_permuted - y_none)                    if y_none     is not None else None
        delta_net       = (y_actual   - y_none)                    if y_none     is not None else None

        seed_results.append({
            "seed": seed,
            "Y_actual":        round(y_actual,   4) if y_actual   is not None else None,
            "Y_permuted":      round(y_permuted, 4) if y_permuted is not None else None,
            "Y_none":          round(y_none,     4) if y_none     is not None else None,
            "delta_content":   round(delta_content,   4) if delta_content   is not None else None,
            "delta_structure": round(delta_structure, 4) if delta_structure is not None else None,
            "delta_net":       round(delta_net,       4) if delta_net       is not None else None,
        })

    if not seed_results:
        raise ValueError("No seed results could be computed.")

    certificate = build_certificate(
        system_name=system_name,
        protocol_name=protocol_name,
        seed_results=seed_results,
        condition_descriptions={
            "A": "Actual protocol messages",
            "B": "Permuted messages (same format, scrambled content)",
            "C": "No messages (null baseline)",
        },
        assumptions={
            "matched_initialization": "declared",
            "matched_stochasticity":  "declared",
            "no_leakage":             "declared",
            "stationary_protocol":    "declared",
            "notes": "Assumptions declared by caller; not verified by this tool.",
        },
        engagement_type="confirmatory" if confirmatory else "diagnostic",
        run_confirmatory=confirmatory,
    )

    output_path.write_text(json.dumps(certificate, indent=2))
    print(certificate["certificate_text"])
    print(f"\nCertificate written to {output_path}")
    return certificate


def main():
    parser = argparse.ArgumentParser(description="Coordination Diagnostic Tool v0.1")
    parser.add_argument("--traces-dir",    type=Path, required=True, help="Directory containing trace JSONL files")
    parser.add_argument("--output",        type=Path, default=Path("certificate.json"))
    parser.add_argument("--system-name",   type=str,  default="Unknown System")
    parser.add_argument("--protocol-name", type=str,  default="Unknown Protocol")
    parser.add_argument("--confirmatory",  action="store_true", help="Run studentized bootstrap lower bound (requires N>=16)")
    parser.add_argument("--permute-seed",  type=int,  default=0, help="RNG seed for message permutation")
    args = parser.parse_args()

    run_diagnostic(
        traces_dir=args.traces_dir,
        output_path=args.output,
        system_name=args.system_name,
        protocol_name=args.protocol_name,
        confirmatory=args.confirmatory,
        permute_seed=args.permute_seed,
    )


if __name__ == "__main__":
    main()
