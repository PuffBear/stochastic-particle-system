# Offline Policy Evaluation from Matched Counterfactual Datasets

**Research idea:** FR-C1
**Target venue:** NeurIPS 2027 (Datasets and Benchmarks track)
**Status:** Long-horizon; confirmed baseline done, dataset not yet built

## Core idea

The SPS matched-pair runner generates a natural counterfactual dataset: for each seed, a (signal, null) trajectory pair where the only causal difference is the planted field strength α. Brownian noise tensor, initial positions, field nuisance variables, and tie-breaking randomness are identical across the pair.

This structure is unusual for MARL offline evaluation. Most offline datasets have no counterfactual pairing — policies are evaluated on trajectories from a different policy or environment state, requiring importance sampling to correct for distribution shift. The SPS paired structure removes one major confound and offers a direct path to estimating causal treatment effects.

This paper asks: can that structure support correct offline ranking of novel policies without new simulations?

## Why MARL offline evaluation is hard

In single-agent RL, offline evaluation via importance sampling is tractable because the environment is fixed. In MARL, each agent's policy affects other agents' observations — the joint observation distribution changes with the policy in a way importance sampling cannot easily correct. The SPS paired structure bypasses this by directly estimating causal effect on team yield.

## Files

- `research-questions.md` — estimands, estimator architectures, coverage requirements, dataset plan
