# FR-B3 Dimensionless Rescaling Audit

**Final status:** passed  
**Audit v2 date:** 3 August 2026  
**Audit v2 code snapshot:** `d58cf1205d04829267ab6076925d95d4bee2c52e`

## Result

The registered audit compared four physically different but dimensionlessly
equivalent environments under four fresh common seeds and two scripted
policies. All eight seed-policy comparisons had identical capture yield and
identical quantized normalized final-state checksums across every rescaling.

| Quantity | Result |
|---|---:|
| Physical rescalings | 4 |
| Fresh seeds | 4 (`7221-7224`) |
| Scripted policies | 2 |
| Episodes | 32 |
| Seed-policy comparisons | 8 |
| Failed comparisons | 0 |
| Audit decision | Pass |

The immutable machine-readable result is
[`results/analysis/FR-B3-CATCHABILITY-RESCALING-V2.json`](../results/analysis/FR-B3-CATCHABILITY-RESCALING-V2.json).
The raw episode summaries, design, and provenance manifest are in
[`results/raw/FR-B3-CATCHABILITY-RESCALING-V2`](../results/raw/FR-B3-CATCHABILITY-RESCALING-V2).

## Failed-first audit and correction

Audit v1 used seeds `7211-7214` against registration commit
`6390608ad935488e706fbd59c56ec7f32fa8d437`. Seven of eight comparisons passed.
For seed 7211 under `capacity_matched_independent`, capture yield was identical
but the normalized final-state checksum split into two values.

The first action divergence occurred at step 50. Apparent velocities were
expressed in physical units and clipped to `[-1, 1]`; the 2x length rescaling
caused one component to cross that absolute threshold and changed the action
direction. The v2 runner maps apparent-velocity slots to canonical units using
`time_scale / length_scale` before evaluating a scripted policy. This factor is
exactly one in every factorial cell, so the correction does not change the
registered factorial policies or estimand.

The failed output remains immutable at
[`results/raw/FR-B3-CATCHABILITY-RESCALING`](../results/raw/FR-B3-CATCHABILITY-RESCALING),
with its machine-readable failure report at
[`results/analysis/FR-B3-CATCHABILITY-RESCALING.json`](../results/analysis/FR-B3-CATCHABILITY-RESCALING.json).
Seeds `7211-7214` were excluded from v2 rather than reused.

## Interpretation boundary

This pass verifies implementation-level scale equivalence for the two scripted
controllers and frozen dynamics under the tested rescalings. It is not an
independent scientific replication and says nothing yet about learned-policy
scale generalization. No factorial seed (`7101-7164`) was consumed by either
audit.
