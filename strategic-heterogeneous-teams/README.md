# Strategic and Heterogeneous Teams: Beyond the Passive Homogeneous Baseline

**Research ideas:** FR-C2 (heterogeneous agents) + FR-C3 (strategic evaders)
**Target venue:** AAMAS 2027 (game-theoretic / multi-agent learning track)
**Status:** Long-horizon; passive homogeneous results must be published first

## Why these two ideas belong together

FR-C2 relaxes the *homogeneity* assumption: agents differ in sensing quality or actuation capacity. FR-C3 relaxes the *passivity* assumption: particles weakly evade collectors. Both ask the same structural question: does the team communication structure that worked for a simple, idealized baseline survive when that baseline is made more realistic?

The answer follows the same shape in both cases — the sufficient statistic changes, the optimal weighting changes, but the coordination benefit either persists or is eliminated — making them natural companion papers with a shared theoretical framework.

Combined, they argue: the SPS-C03 result is *robust in one direction* (heterogeneity changes weights but not the structure of the sufficient statistic) and *potentially fragile in another* (strategic evasion can eliminate the coordination benefit at sufficient evasion capacity). The contrast between these two outcomes is itself a publishable finding.

## Theoretical framing

**Heterogeneous teams (FR-C2):** If agent i has sensing quality α_i, Proposition 2 predicts the optimal message weights agent i's contribution by α_i² (inverse variance). A heterogeneity-aware controller should outperform the homogeneous v2 controller when α_i differ — a direct, testable prediction.

**Strategic evaders (FR-C3):** Particles that respond to collector proximity couple team communication with evasion. A communicating team implicitly discloses where it is concentrated; strategic particles can exploit this by avoiding the cluster that communication creates. Prediction: sharing helps at low evasion budgets and may hurt at high ones.

## Files

- `research-questions.md` — sub-questions, evader model, hypotheses, estimands, kill criteria
