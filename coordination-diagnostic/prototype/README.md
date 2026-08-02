# Coordination Diagnostic Prototype

Zero-dependency Python implementation of the three-condition coordination diagnostic. Requires Python 3.10+ and no external packages.

## Quick start

```bash
python diagnostic.py \
  --traces-dir ./my-traces \
  --system-name "My Robot Swarm" \
  --protocol-name "velocity-sharing-v2" \
  --output certificate.json
```

## Trace format

Create one JSONL file per seed per condition, named `<seed>_<condition>.jsonl`:
- `1001_actual.jsonl`   — real protocol messages
- `1001_permuted.jsonl` — scrambled messages (optional; derived from actual if absent)
- `1001_none.jsonl`     — no messages baseline

Each file contains one JSON object per timestep. The final step must include a `metric` field (higher = better):

```json
{"seed": 1001, "condition": "actual", "step": 66, "metric": 11.0}
```

Steps before the final one can include message logs for diagnostics but are not required:

```json
{"seed": 1001, "condition": "actual", "step": 0, "messages": [{"sender": 0, "receiver": 1, "value": [0.12, -0.34, 0.88]}]}
```

If you only have `actual` traces, the tool derives the permuted condition by scrambling messages in-memory. The `none` condition (null baseline) must be provided separately if you want the structure gain estimate.

## Output

The tool prints a plain-text certificate and writes `certificate.json`. See `methodology/certificate-format.md` for the full schema.

Example output:
```
COORDINATION DIAGNOSTIC CERTIFICATE
====================================
System:   My Robot Swarm
Protocol: velocity-sharing-v2
Date:     2026-08-01
Seeds:    8 (diagnostic)

Net coordination value:  +2.500  (8/8 seeds positive vs no-messages)
Gain from structure:     +1.625  (5/8 positive)  — PASS
Gain from content:       +0.875  (4/8 positive)  — FAIL

Diagnosis: STRUCTURE DRIVES GAIN

This certificate covers the specific conditions and seed set listed above.
It does not certify behaviour under untested conditions.
```

## Confirmatory mode

For a formal claim (N=32, pre-registered):

```bash
python diagnostic.py --traces-dir ./traces --confirmatory --output certificate.json
```

This adds a one-sided 95% studentized bootstrap lower bound on net coordination value. A positive lower bound supports a formal positive claim.

## Assumptions

The tool does not verify matched initialization or stochasticity — it trusts the caller. Before running:
1. Confirm episodes with the same seed start from identical initial states across all three conditions
2. Confirm stochastic disturbances are drawn from the same pre-generated sequence
3. Confirm the condition label does not influence environment dynamics

See `methodology/diagnostic-spec.md` for full assumption documentation.
