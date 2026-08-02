# Research Questions: Offline MARL Evaluation

## Primary research question

Can a dataset of matched (signal, null) trajectory pairs from SPS support correct offline ranking of novel policies without running new simulations — and how does ranking accuracy degrade as dataset size decreases?

## Sub-questions

### Q1 — Offline ranking accuracy

**Setup:** Collect matched trajectory datasets for 5 policies with known online rankings:
1. `full_state_oracle` (mean yield ~18.4)
2. `shared_summary_v2` (~10.75)
3. `capacity_matched_independent` (~8.25)
4. `random_motion` (~4–5, estimated)
5. `stationary` (~8.75)

Collect N matched seed pairs per policy. Train three offline estimators. Evaluate on 3 novel held-out policies not in the training set.

**Hypothesis:** Pairwise ranking accuracy exceeds 80% for pairs with online yield gap ≥1.5 particles using N≥200 pairs. Accuracy drops below 60% for pairs with gap <0.5 particles (where SPS itself has high seed-level variance).

**Estimand:** Pairwise ranking accuracy as a function of N and the oracle yield gap between the pair.

**Estimators to compare:**

| Estimator | Description |
|---|---|
| PDIS | Per-decision importance sampling |
| DR-OPE | Doubly-robust estimator with learned value function |
| DM | Direct method: learn transition model, roll out for novel policies |
| Paired difference | Exploit matched structure directly: estimate E[Y(signal) − Y(null)] per policy |

The paired difference estimator is the novel contribution — it uses the counterfactual structure that generic offline estimators ignore.

### Q2 — Coverage requirements

**Hypothesis:** Correct top-3 ranking in M=4 SPS requires more data than an equivalent M=1 single-agent task because the joint observation space grows as M × obs_dim.

**Estimand:** Minimum N for 80% top-3 accuracy in M=4 vs. M=1. Report the ratio.

### Q3 — Does counterfactual pairing reduce estimator variance?

**Hypothesis:** Matched pairs reduce estimator variance by a factor ≈(1 − r), where r is the within-pair yield correlation. Based on SPS-C03 pilot data, r ≈ 0.3–0.5, predicting 30–50% variance reduction.

**Estimand:** Variance of offline return estimator under matched vs. unmatched dataset at N ∈ {100, 200, 500}.

## Kill criteria

- **Primary kill:** Ranking accuracy below 60% for pairs with online yield gap ≥2 particles using N=500 pairs. Counterfactual structure provides no useful signal.
- **Coverage kill:** Minimum N for 80% top-3 accuracy exceeds 2,000 pairs — beyond the feasibly collectable dataset (~5,000 pairs estimated). Offline evaluation is impractical for this task class.

## Dataset construction plan

- **Minimum viable:** 5 policies × 200 seeds × 2 arms = 2,000 trajectory pairs
- **Logged per step:** (local_obs, action, reward, team_message, field_direction)
- **Pairing enforced:** every (seed, policy, signal) trajectory matched to same (seed, policy, null) from identical Brownian noise tensor
- **Held-out evaluation:** 3 novel policies (e.g., modified blend_w values, alternative action rules) not in training data
- **Format:** one file per trajectory pair, JSON or HDF5; reproducible from frozen seed + policy + environment commit
