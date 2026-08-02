# Experiment Grid Design

**Status:** Pre-registered design — do not modify after first diagnostic run begins

---

## Fixed parameters (inherited from SPS-C03)

```
N = 256         # particles
M = 4           # collectors
dt = 0.02       # timestep
sigma = 0.06    # particle noise
steps = 67      # evaluation window
arena = [0,1]²  # unit square, reflecting boundaries
```

These are frozen. Any change to these parameters changes the environment contract and invalidates the comparison to SPS-C03.

---

## The 3×3 grid

### Step 1: Choose ρ levels

ρ = α · √dt / σ = α · √0.02 / 0.06

Target ρ values and implied α:

| ρ level | ρ value | α (derived) | Label |
|---|---|---|---|
| Low | 0.10 | α = ρ · σ / √dt = 0.10 · 0.06 / √0.02 ≈ 0.042 | easy sensing |
| Mid (confirmed) | 0.141 | α = 0.06 | confirmed C03 |
| High | 0.25 | α = 0.25 · 0.06 / √0.02 ≈ 0.106 | hard sensing |

Note: use α values rounded to 3 decimal places in configs; recompute ρ from exact α for reporting.

### Step 2: Choose κ levels

κ = α / v_max → v_max = α / κ

Target κ values and implied v_max (computed separately per row since α varies):

| κ level | κ value | v_max at ρ=low (α≈0.042) | v_max at ρ=mid (α=0.060) | v_max at ρ=high (α≈0.106) |
|---|---|---|---|---|
| Low | 0.10 | 0.42 | 0.60 | 1.06 |
| Mid (confirmed) | 0.20 | 0.21 | 0.30 | 0.53 |
| High | 0.40 | 0.105 | 0.15 | 0.265 |

### Step 3: Full parameter table

Each cell is identified by (ρ_level, κ_level). The confirmed SPS-C03 cell is (mid, mid).

| Cell | α | v_max | ρ (exact) | κ (exact) | Status |
|---|---|---|---|---|---|
| (low, low) | 0.042 | 0.420 | 0.099 | 0.100 | To run |
| (low, mid) | 0.042 | 0.210 | 0.099 | 0.200 | To run |
| (low, high) | 0.042 | 0.105 | 0.099 | 0.400 | To run |
| (mid, low) | 0.060 | 0.600 | 0.141 | 0.100 | To run |
| (mid, mid) | 0.060 | 0.300 | 0.141 | 0.200 | ✅ C03 confirmed |
| (mid, high) | 0.060 | 0.150 | 0.141 | 0.400 | To run |
| (high, low) | 0.106 | 1.060 | 0.250 | 0.100 | To run |
| (high, mid) | 0.106 | 0.530 | 0.250 | 0.200 | To run |
| (high, high) | 0.106 | 0.265 | 0.250 | 0.400 | To run |

---

## Seed plan

**Diagnostic phase (8 seeds per cell):**
- Seeds 8001–8008 for all 8 new cells
- Cell (mid, mid): reuse C03 seeds 1001–1008 (first 8 of the C03 confirmation set)
- Total new runs: 8 cells × 8 seeds × 2 arms = 128 episodes

**Confirmatory phase (additional 8 seeds, if diagnostic passes):**
- Seeds 8009–8016 for cells that pass the diagnostic gate
- Gate: ≥5/8 seeds positive for shared_v2 vs independent
- Total confirmatory runs (if all 8 pass): 8 cells × 8 additional seeds × 2 arms = 128 episodes

**Cell (mid, mid) confirmatory:** uses C03 seeds 1001–1032 (all 32), already complete.

---

## Pre-registered analysis

The following analysis is pre-registered before any diagnostic seeds are run. Do not modify.

### Primary (per-cell gate, diagnostic)
For each cell: compute Δ_s = Y_s(shared_v2) − Y_s(independent) for s ∈ 8001–8008. Gate passes if sign count ≥ 5/8.

### Secondary (separability test, after all 9 cells)
Compute mean Δ̄(i,j) for each cell. Fit:
```
log(Δ̄(i,j) + offset) = log(a) + b·log(ρ_i) + c·log(κ_j)
```
where offset is chosen to make all Δ̄ + offset > 0 (use offset = |min(Δ̄)| + 0.5 if any Δ̄ ≤ 0).

Compute R² of the fit. Primary separability criterion: R² ≥ 0.8.

### Exploratory (oracle gap per cell)
For each cell, compute oracle_gap = Y(oracle) − Y(shared_v2). Report alongside Δ̄ to show remaining headroom.

### Kill criterion
Stop confirmatory runs and redirect paper claim if:
- Fewer than 5/9 cells pass the diagnostic gate (coordination not reliably beneficial across regime)
- R² < 0.5 on multiplicative fit (separability rejected)
- Cell (mid, mid) diagnostic Δ̄ deviates from C03 by more than ±0.5 particles (infrastructure failure)

---

## Implementation notes

**Config changes needed:**
The existing SPS runner accepts `alpha` and `v_max` as config parameters. Grid runs use the same `run_sps_c03_confirmation.py` script with cell-specific configs.

**Matched noise:** Each cell uses independently generated Brownian noise tensors for its own seed set. The (mid, mid) cell reuses C03 noise. Cross-cell comparison is not planned and does not require matched noise.

**Runtime estimate:**
~10 seconds per episode on a single CPU core.
128 diagnostic episodes ≈ 21 minutes single-threaded.
With 8 cores: ≈ 3 minutes. No HPC required.

**Output format:** Same JSONL manifest and episode summary format as SPS-C03. Store per-cell results in `results/raw/FR-B3-CATCHABILITY/cell_{rho_level}_{kappa_level}/`.
