"""Compare immutable pilot summaries by scientific outcome keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict[tuple[object, ...], dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {
        (
            row["treatment"],
            row["policy_id"],
            row["collector_count"],
            row["rho"],
            row["scenario_seed"],
        ): row
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before = _read(args.before)
    after = _read(args.after)
    if set(before) != set(after):
        raise RuntimeError("pilot keys differ")
    changed: list[dict[str, object]] = []
    guard_before = 0
    guard_after = 0
    for key in sorted(before, key=str):
        left, right = before[key], after[key]
        guard_before += sum(int(left[arm]["guarded_reflection_pairs"]) for arm in ("null", "signal"))
        guard_after += sum(int(right[arm]["guarded_reflection_pairs"]) for arm in ("null", "signal"))
        left_outcome = (
            left["null"]["first_interception_step"],
            left["signal"]["first_interception_step"],
            left["matched_first_interception_gain"],
        )
        right_outcome = (
            right["null"]["first_interception_step"],
            right["signal"]["first_interception_step"],
            right["matched_first_interception_gain"],
        )
        if left_outcome != right_outcome:
            changed.append({"key": list(key), "before": list(left_outcome), "after": list(right_outcome)})
    result = {
        "comparison": "SPS-P01 guarded-reflection diagnostic vs SPS-P02 exact specular-reflection repair",
        "pair_count": len(before),
        "changed_first_interception_outcomes": len(changed),
        "guarded_reflection_pairs_before": guard_before,
        "guarded_reflection_pairs_after": guard_after,
        "changed_examples": changed[:20],
        "interpretation": "Same calibration seeds; this is a correctness regression comparison, not independent confirmation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
