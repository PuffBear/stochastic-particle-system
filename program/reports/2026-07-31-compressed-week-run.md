# Stochastic Particle Lab — Compressed-Week Research Report

**Date:** 31 July 2026  
**Active paper:** one  
**Branch:** `research-autonomy`  
**Scientific status:** exploratory calibration complete; confirmatory run blocked  
**HPC:** not needed and not requested

## Five-bullet TL;DR

1. The full seven-day correctness-to-evidence plan was compressed into one gated run: exact contact, deterministic ties, matched experiments, six scripted policies, calibrated inference, 288 matched pilot/repair pairs, manuscript update, fresh review, and program ledgers are complete.
2. The simulator now passes **79 tests**. Fixed-geometry contact is solved continuously along exact piecewise specular reflected Euler paths; signal/null pairs share initialization, Brownian noise, nuisance variables, policy randomness, and event-keyed tie provenance.
3. The 12-seed exact-contact calibration found **no supported boundary** through `rho=2.0`: every simultaneous one-sided 95% lower bound for `local_flow_v1` was negative. These seeds were calibration-only, so SPS-C01 remains proposed rather than supported or rejected.
4. The current research mechanism fails: at `rho=2`, stationary and pregenerated-random controls descriptively matched or exceeded `local_flow_v1`, and the four collectors are simply four independent copies with no teammate-dependent action. This supports no coordination, MARL, or AAMAS claim.
5. The fresh AAMAS review scored **3/10 (Reject), confidence 5/5**. Its strongest new threat is that the provisional `0.05` half-width target is much larger than observed mean effects of about `0.004–0.011`; therefore the mechanically suggested 24-seed run would be scientifically uninformative and will not be executed.

## Simple explanation

The project asks whether collectors can use a weak common flow hidden inside noisy particle motion to intercept a particle earlier. To answer that fairly, every signal episode is compared with a no-signal twin that receives the same starting positions and the same random noise. The only intended difference is the planted flow strength.

The engineering instrument now works well enough to run this comparison reproducibly. The first small calibration, however, says the current controller is not yet demonstrating the intended idea. Stronger flow made first interception slightly earlier on average, but passive stationary collectors and uninformed random movement showed changes of similar size. In addition, the four-controller policy does not share or use team information. In plain language: the current results mostly look like particles being carried by the flow, not a team intelligently exploiting the flow.

That negative diagnosis is valuable because it prevents an expensive but uninformative training or seed sweep. The next research task is to isolate policy-specific information value beyond passive transport and then introduce one genuinely multi-agent mechanism under matched budgets.

## Exact research question

> Under the frozen canonical uniform-drift task and scripted `local_flow_v1`, which preregistered value of `rho = alpha * sqrt(dt) / sigma` is the first grid point at which mean matched horizon-censored first-interception improvement has a strictly positive one-sided simultaneous 95% lower confidence bound?

This is a policy-relative and grid-relative question. It is not a claim about a universal physical detection threshold.

## What was completed

### Engineering

- Implemented continuous within-step contact for fixed geometry.
- Replaced the provisional wall-reflection guard with exact piecewise specular path segmentation, including arbitrary multi-wall overshoot, moving particle/collector pairs, and simultaneous corner reflections.
- Implemented earliest-contact ownership and deterministic event-keyed ties indexed by scenario seed, step, and stable particle ID.
- Implemented a strict paired runner that rejects any signal/null configuration difference other than signal strength.
- Verified matching of initial state, Brownian tensor, field orientation/nuisance, policy randomness, scenario seed, and tie-key provenance.
- Added immutable trajectory, summary, and manifest writing; closed-schema validation; and SHA-256 checksums.
- Added bounded pre-contact execution so each arm stops after its own first interception when only the primary endpoint is needed.

### Frozen scripted policies

- `stationary`: zero action.
- `pregenerated_random`: complete action tensor fixed before either paired episode.
- `coverage`: deterministic lane sweep without particle evidence.
- `density_greedy`: move toward the centroid of locally visible particles; ignores velocity.
- `local_flow_v1`: move opposite the mean causally valid local apparent velocity; stop when no valid velocity exists.
- `privileged_upstream_oracle`: move opposite the true field direction; this is a true-field diagnostic, not a full-state interception oracle.

### Statistical estimator

The estimator resamples complete scenario-seed rows across all five nonzero signal points and computes a one-sided simultaneous studentized maximum-bootstrap lower bound. Input row order is canonicalized, so a row permutation with the same bootstrap seed is bitwise invariant. Tests cover reproducibility, all-zero effects, constant positive effects, and the distinction between first and persistent crossings.

Synthetic calibration used 2,000 outer trials and 2,000 inner bootstrap draws per trial:

| Null family | False-crossing rate | Wilson 95% upper bound | Gate |
| --- | ---: | ---: | --- |
| Correlated continuous | 0.040 | 0.04951 | Pass (`<=0.07`) |
| Zero-inflated symmetric discrete | 0.006 | 0.01046 | Pass (`<=0.07`) |

This validates only the tested null families. It does not establish power or coverage under the real censored simulator distribution.

## Exploratory calibration results

SPS-P02 used the fixed calibration seeds `1001–1012`, four collectors, 256 particles, horizon 400, `dt=0.02`, `sigma=0.06`, and exact specular contact.

| rho | Mean paired gain | Paired SE | Simultaneous one-sided 95% LCB |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.005625 | 0.003387 | -0.015330 |
| 0.25 | 0.003750 | 0.004570 | -0.024523 |
| 0.50 | 0.007083 | 0.004229 | -0.019081 |
| 1.00 | 0.008958 | 0.004590 | -0.019440 |
| 2.00 | 0.011042 | 0.005144 | -0.020788 |

All lower bounds are negative. The exploratory grid boundary is therefore right-censored beyond `rho=2.0`. Because these seeds were reserved for calibration, this result cannot support or reject SPS-C01.

### Strongest-signal diagnostic controls

| Treatment at rho=2 | Mean gain | Median gain | Interpretation |
| --- | ---: | ---: | --- |
| Four independent `local_flow_v1` | 0.01104 | 0.00375 | Primary calibration treatment |
| Stationary | 0.01313 | 0.00625 | Passive flow change is at least as large descriptively |
| Pregenerated random | 0.01292 | 0.00875 | Uninformed movement matches/exceeds local flow descriptively |
| Privileged upstream | 0.01292 | 0.00625 | True field direction does not create headroom |
| Coverage | 0.00771 | 0.00625 | Descriptive only |
| Density greedy | 0.00625 | 0.00250 | Descriptive only |
| Single-collector local flow | 0.06938 | 0.01625 | Not a causal team-size comparison |

No multiplicity-controlled policy contrast was preregistered, so this table is diagnostic rather than confirmatory. Its practical meaning is still clear: the intended information-use mechanism has not been isolated.

## Contact-repair replication

SPS-P01 used a conservative endpoint guard whenever a particle or collector reflected within a step and logged 28,929 guarded particle/collector checks. SPS-P02 reran the same 144 calibration pairs after exact specular reflection contact was implemented.

- Guarded checks changed from **28,929 to 0**.
- First-interception outcomes changed in **0 of 144 pairs**.
- The complete calibration curve and control summaries were identical.

This is a same-seed correctness regression. It shows that the repaired implementation did not alter these recorded pilot outcomes; it is not independent scientific confirmation. A coupled-noise timestep-convergence study is still required because an Euler/Brownian chord is not a continuous Brownian bridge.

## Paper and reviewer status

The LaTeX manuscript is now a compiled four-page evidence snapshot. It contains the exact question, simulator/contact formalism, paired provenance, frozen policies, inference procedure, calibration tables, contact repair, seed-budget logic, and explicit AAMAS relevance failure. It retains only the two already verified references and makes no first/novelty claim.

Fresh review:

- **Overall:** 3/10 — Reject
- **Confidence:** 5/5
- **Strongest threat:** a `0.05` target simultaneous half-width is coarser than observed effects of `0.004–0.011`; a 24-seed run designed around that target could not demonstrate a positive lower bound even if the observed means were real.
- **Other major threats:** no multi-agent mechanism; passive/random controls match the policy; the current “oracle” is not a full-state interception oracle; missing nondimensional speed ratio `alpha / v_max`; missing timestep convergence; narrow literature map; no policy-activation or field-estimation diagnostics.

The provisional 24-seed calculation is therefore recorded but **not authorized**.

## Literature update

No new scientific or novelty claim was introduced today, so no unsupported citation expansion was added. The manuscript still uses only the verified task-level mobile-collector and local-sensing foraging neighbors. The fresh review identifies the next literature audit scope: multi-robot stochastic search, distributed field estimation, common-random-number simulation, cooperative benchmark design, and stochastic hitting/contact methods.

## Expansionist update

- Retired duplicate communication idea FR-001 into the more specific FR-009.
- Corrected FR-002: event-keyed ties are implemented, but the pilot recorded zero tie decisions, so empirical tie stress remains future work rather than an active blocker.
- Added FR-010: policy-specific signal value beyond passive transport.
- Added FR-011: resource-matched team-size and signal-value dilution.
- Added FR-012: power and boundary-error calibration for censored paired event-time inference.
- Corrected the industry contact-audit idea with the exact P01/P02 result and retained zero revenue/TAM precision.
- Added no new commercial idea because there is no customer, pricing, demand, procurement, or market-size evidence.

## Verified provenance

Primary commands:

```text
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests analysis
python .../research_program.py check .
PYTHONPATH=src python3 analysis/calibrate_inference.py --trials 2000 --bootstrap-draws 2000 ...
PYTHONPATH=src python3 analysis/run_scripted_pilot.py --experiment-id SPS-P02 ...
pdflatex -interaction=nonstopmode -halt-on-error main.tex  # twice
```

Verified environment: Python 3.12.13 and NumPy 2.3.5. Final test result: **79 passed, 0 failed**. The manuscript compiled twice without warnings, undefined references, or overfull/underfull boxes; PDF length is four pages.

Key artifacts:

- `results/raw/SPS-P02/pair_summaries.jsonl`
- `results/raw/SPS-P02/primary_analysis.json`
- `results/raw/SPS-P02/baseline_summary.json`
- `results/derived/SPS-P01-vs-P02.json`
- `results/derived/inference_calibration_2000x2000.json`
- `paper/manuscript/main.pdf`
- `paper/reviews/2026-07-31-compressed-week-fresh-aamas.md`
- `program/handoffs/SPS-WO-03-compressed-week.md`

## Blockers and compute

No HPC access is needed. The next blockers are conceptual and diagnostic, not computational:

1. Freeze a genuinely multi-agent bounded evidence-fusion policy and identical-budget independent ablation.
2. Isolate policy-specific value beyond passive transport with a preregistered difference-in-differences contrast.
3. Replace the true-field-direction diagnostic with a meaningful full-state interception oracle while retaining both controls separately.
4. Measure valid-track count, policy activation, local field-estimation error, action alignment, wall proximity, and absolute first-contact distributions.
5. Add `alpha / v_max` and other task-relevant nondimensional ratios; redesign the grid so observability is not conflated with catchability and wall accumulation.
6. Run coupled-noise timestep convergence before any confirmatory first-hitting-time claim.
7. Recompute power against a scientifically meaningful effect size; do not use the current 0.05 half-width target.

## Autonomous next 24 hours

1. Write a no-data design memo for FR-010: define the policy-minus-stationary difference-in-differences estimand, multiplicity family, kill criteria, and required diagnostics.
2. Implement instrumentation only—valid velocity-slot counts, activation fraction, local estimate error against hidden truth for diagnostics, action/field alignment, wall distance, and event-time survival outputs—without changing `local_flow_v1`.
3. Design a full-state nearest-interception/model-predictive oracle and deterministic microcases; do not tune against SPS-P02 seeds.
4. Specify a coupled Brownian timestep-refinement protocol at fixed physical horizon and field strength.
5. Draft the bounded-sharing treatment and independent ablation as a future AAMAS question, but do not train IPPO/MAPPO or spend HPC compute until scripted feasibility passes.
6. Expand the literature ledger only with verified primary sources relevant to the revised mechanism and statistical/contact methodology.

## No-reply action

No response is needed. Under the safe default, the program will continue with diagnostic instrumentation, formal contrast design, timestep protocol work, literature repair, and manuscript maintenance. It will not run the 24-seed confirmation, train learned policies, request HPC, or change the active claim without a new validity gate.
