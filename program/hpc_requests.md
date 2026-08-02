# HPC Requests

## Active: SPS-WO-08 MARL Baselines — ShARC HPC (Ashoka University)

**Status:** ACTIVE — pilot jobs running (2026-08-02)

All upstream gates passed before HPC was requested:
1. ✅ Complete environment and observation contract (199 tests passing)
2. ✅ Leakage, limiting-case, and matched-counterfactual audits
3. ✅ Validated scripted baselines (SPS-WO-05, WO-07B, C03)
4. ✅ Pilot variance estimated from WO-07B (SD≈2.44)
5. ✅ Power analysis (80% power at SD=4.0, effect=2.0; 32 seeds recommended)

**Resource spec:** ShARC HPC, PBS `gpu` queue, `select=1:ncpus=8`, 24h walltime.
CPU-only PyTorch (environment physics is the bottleneck; GPU provides no meaningful speedup for small MLPs; eliminates CUDA driver compatibility issues).

**Job structure:**
- 6 architectures: IPPO, MAPPO, CommNet, COMA, VDN, MADDPG
- 8 training seeds per arch: 8001–8008
- 8 eval seeds per arch: 9001–9008 (never used for any other experiment)
- 20,000 training episodes per job
- Total: 48 jobs

**Pilot status (seed 8001, one job per arch):** All 5 on-policy jobs (IPPO, MAPPO, CommNet, COMA, VDN) running. MADDPG job queued separately (higher walltime concern; ~500k steps may approach 24h limit).

**Expected scientific decision:** Whether any MARL architecture exceeds `shared_summary_v2` (confirmed scripted coordination result, mean=+1.19 unique captures over independent). Negative result: scripted mechanism is sufficient. Positive: learning adds further gain.

**Aggregate analysis:** `analysis/aggregate_marl_results.py` post-processes all 48 per-seed JSONs once jobs complete.

---

## Historical

No HPC was requested before 2026-08-02. All prior work was correctness-bound and Codex cloud was sufficient.
