# FR-B3 Design Review: Ready for Human Sign-off

**Review date:** 3 August 2026  
**Audience:** FR-B3 authors and statistical reviewer  
**Scope:** axis ranges, factorial seed budget, and two-axis rejection rule only

## Technical summary

The three FR-B3 axes are internally coherent and cover a useful controlled
regime around the executed SPS-C03 anchor. The original 32-seed budget and
decision rule were not adequate: under a conservative simulation, they detected
a broad one-capture high-versus-low `eta` effect in only 18.8% of studies. The
revised 64-seed panel and two-part decision rule reach 83.8% simulated power
under zero cross-cell correlation and 95.0% at correlation 0.25, while producing
no false rejection in 500 simulated null studies for either setting.

**Recommendation:** approve the axis grid; approve 64 fresh common seeds; replace
the old rule with the revised practical-plus-statistical gate. The protocol
remains `proposed_not_preregistered` until a human advisor signs off and the
document is externally timestamped.

## The axis grid spans the intended dynamical regimes

All three axes are log-centered at the executed SPS-C03 point and vary by a
factor of two in each direction.

| Quantity | Low | Anchor | High |
|---|---:|---:|---:|
| `rho` | 0.07071 | 0.14142 | 0.28284 |
| `kappa` | 0.25 | 0.50 | 1.00 |
| `eta` | 0.004243 | 0.008485 | 0.016971 |

At fixed `dt=0.02`, `L=1`, and 67 steps, the full factorial implies:

| Derived quantity | Minimum | Maximum | Interpretation |
|---|---:|---:|---|
| `sigma` | 0.03 | 0.12 | diffusion coefficient |
| `alpha` | 0.015 | 0.24 | field drift speed |
| `v_max` | 0.015 | 0.96 | collector maximum speed |
| normalized drift per step | 0.0003 | 0.0048 | `rho * eta` |
| normalized control per step | 0.0003 | 0.0192 | `rho * eta / kappa` |
| 67-step drift distance | 0.0201 | 0.3216 | directed transport scale |
| 67-step control reach | 0.0201 | 1.2864 | actuation range |
| RMS diffusive excursion | 0.0347 | 0.1389 | `eta * sqrt(67)` |

The low end intentionally approaches a near-static reach floor. The high end
allows a collector to traverse an arena-scale distance, while `kappa=1` tests
the drift-speed/collector-speed boundary. The environment's swept-contact
calculation prevents the largest step size from tunneling through targets.

The grid is approved as a controlled SPS slice. It does not establish that the
same ranges represent real AUV, UAV, or search-and-rescue systems.

## Sixty-four seeds are required for the stated minimum effect

Calibration uses seed-level Gaussian paired contrasts with SD 2.442, taken from
the immutable SPS-C03 primary contrast. The simulated mean surface contains the
same quadratic two-axis terms used by the analysis and a broad linear `eta`
effect. A one-capture difference between high and low `eta` is the minimum
target because it is comparable to the historical +1.1875 coordination gain.

Each entry below is based on 500 simulated studies and 1,000 common-seed
bootstrap resamples per study. The complete deterministic output is preserved
in [design-calibration.json](design-calibration.json).

| Seeds per cell | Cross-cell correlation | Null rejection | Power at 1 capture |
|---:|---:|---:|---:|
| 32 | 0.00 | 0.0% | 18.8% |
| 32 | 0.25 | 0.0% | 49.6% |
| 48 | 0.00 | 0.0% | 55.8% |
| 48 | 0.25 | 0.0% | 83.4% |
| 64 | 0.00 | 0.0% | 83.8% |
| 64 | 0.25 | 0.0% | 95.0% |

Because the actual cross-cell correlation is unknown before running fresh
seeds, the zero-correlation row controls the recommendation. Sixty-four seeds
are the smallest evaluated budget that clears the 80% target in that row.

The calculation is reproducible with:

```bash
PYTHONPATH=src python analysis/calibrate_fr_b3_design.py \
  --trials 500 --bootstrap-draws 1000
```

## The revised decision rule separates effect size from uncertainty

The old rule required the upper endpoint of a bootstrap interval for the RMSE
ratio to fall below 0.80. That attempts to prove a full 20% improvement with
sampling uncertainty included and was unnecessarily underpowered.

The revised rule rejects two-axis sufficiency only when both conditions hold:

1. **Practical gate:** the observed leave-one-cell-out RMSE ratio is at most
   0.80.
2. **Uncertainty gate:** the one-sided 95% common-seed bootstrap lower bound for
   `RMSE(two-axis) - RMSE(three-axis)` is greater than zero.

The practical gate preserves the pre-specified requirement that `eta` improve
held-out prediction by at least 20%. The uncertainty gate asks the narrower
statistical question of whether the improvement is positive. Non-rejection
means only that both gates were not met; it is not evidence of equivalence or
proof that `(rho, kappa)` are sufficient.

## Limitations and robustness boundary

- The SD estimate comes from one operating point. Heavy tails or larger
  variance elsewhere would reduce power.
- The power claim assumes an `eta` effect spread across all nine `(rho, kappa)`
  slices. A single-cell or highly localized interaction is exploratory.
- The Gaussian simulation does not reproduce the discreteness of capture
  counts. The actual analysis remains nonparametric at the seed-resampling
  stage.
- The simulation checks operational rejection rates for the chosen model and
  rule; it is not a proof of exact frequentist size.
- Running four policies over 27 cells and 64 seeds requires 6,912 scripted
  episodes. This is computationally feasible but should be verified on the
  intended machine before registration.

## Human sign-off checklist

- [ ] Confirm that a broad one-capture high-versus-low `eta` effect is the
  smallest scientifically meaningful target.
- [ ] Accept 64 seeds per cell and the resulting 6,912-episode budget.
- [ ] Accept the two-part predictive decision rule.
- [ ] Accept that sparse cell-specific `eta` effects are exploratory.
- [ ] Externally timestamp the approved protocol.
- [ ] Change `protocol_status` to `registered` only after the preceding items.

No frozen factorial seed should be executed before all six items are resolved.
