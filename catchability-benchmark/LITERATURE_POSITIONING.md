# FR-B3 Literature Positioning

**Search date:** 3 August 2026  
**Status:** source-backed positioning draft; claims must be rechecked during
final manuscript review.

## Defensible position

FR-B3 should not claim to invent dimensional analysis, units-equivariant
learning, domain transfer, PPO-based MARL, or learned communication. Its
candidate contribution is narrower:

> a controlled multi-agent stochastic collection benchmark that first tests
> which dimensionless groups are needed to predict coordination gain, then
> evaluates whether explicitly dimensionless observations improve zero-shot
> scale transfer for independent and communicating learned policies.

The unusual combination is the contribution: a causal, matched-seed
coordination surface; a predictive sufficiency test for a third transport-scale
group; and a representation intervention evaluated on physically different but
dynamically equivalent environments.

## Closest literature lanes

### Dimensionless and units-equivariant machine learning

Villar et al. construct dimensionless features before inference to impose exact
equivariance to changes of units, and report in- and out-of-distribution
generalization benefits in regression and emulation examples. This is the
closest conceptual precedent for FR-B3's representation intervention, but it
does not study reinforcement learning, multi-agent coordination, or policy
communication.

- Soledad Villar, Weichi Yao, David W. Hogg, Ben Blum-Smith, and Bianca
  Dumitrascu. “Dimensionless machine learning: Imposing exact units
  equivariance.” JMLR 24(109), 2023.
  <https://www.jmlr.org/papers/v24/22-0680.html>

Bakarji et al. use the Buckingham Pi theorem as a constraint for discovering
dimensionless groups that collapse observed physical data. FR-B3 instead
derives candidate groups from known simulator dimensions and tests predictive
sufficiency under held-out factorial cells.

- Joseph Bakarji, Jared Callaham, Steven L. Brunton, and J. Nathan Kutz.
  “Dimensionally Consistent Learning with Buckingham Pi.” 2022.
  <https://arxiv.org/abs/2202.04643>

### Reinforcement-learning generalization

Invariant Policy Optimization formulates cross-domain policy generalization as
learning a representation supporting an action predictor that remains optimal
across training domains. FR-B3 is more controlled and less algorithmic: it
holds the learning algorithm fixed and intervenes on the units of the input
representation, while training at only one physical scale.

- Anoopkumar Sonar, Vincent Pacelli, and Anirudha Majumdar. “Invariant Policy
  Optimization: Towards Stronger Generalization in Reinforcement Learning.”
  L4DC, PMLR 144, 2021.
  <https://proceedings.mlr.press/v144/sonar21a.html>

Domain-randomization work trains over perturbed simulators to improve transfer
to unseen dynamics. FR-B3 asks a different question: whether an exact known
symmetry can be supplied through representation so that a policy trained at a
single scale transfers between equivalent systems without augmentation.

- Fabio Muratore, Felix Treede, Michael Gienger, and Jan Peters. “Domain
  Randomization for Simulation-Based Policy Optimization with Transferability
  Assessment.” CoRL, PMLR 87, 2018.
  <https://proceedings.mlr.press/v87/muratore18a.html>

### PPO-based cooperative MARL

Yu et al. establish IPPO and MAPPO as strong cooperative MARL baselines across
multiple testbeds. FR-B3 uses this family as a controlled learner rather than
claiming a new optimization algorithm. IPPO provides the non-communicating arm;
MAPPO is a robustness extension because centralized training would otherwise
complicate the clean independent-versus-communicating comparison.

- Chao Yu et al. “The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games.” NeurIPS 2022.
  <https://openreview.net/forum?id=YVXaxB6L2Pl>

### Learned communication

CommNet learns continuous inter-agent communication jointly with policy
behavior through mean-pooled hidden states. It is an appropriate communicating
baseline because FR-B3 can compare its full message channel with a zero-message
execution ablation already implemented in the repository.

- Sainbayar Sukhbaatar, Arthur Szlam, and Rob Fergus. “Learning Multiagent
  Communication with Backpropagation.” NeurIPS 2016.
  <https://proceedings.neurips.cc/paper/2016/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html>

## Gap statement for the introduction

Existing units-equivariant ML work motivates learning on dimensionless
features, RL-generalization work motivates invariance across domains, and MARL
communication work studies how agents learn to exchange information. These
lanes do not by themselves answer whether the coordination advantage in a
stochastic multi-collector system is governed by two or three dimensionless
groups, nor whether the corresponding dimensionless representation transfers a
learned communication policy between dynamically equivalent physical scales.

This is a **candidate gap**, not a priority claim. Before submission, the
authors should run a forward/backward citation search from Villar et al., Sonar
et al., and CommNet for any newer work directly combining dimensionless
representations with multi-agent reinforcement learning.

## Claims to avoid

- “first dimensionless reinforcement-learning method”;
- “first scale-invariant policy”;
- “first multi-agent particle-collection benchmark”;
- “dimensionless observations guarantee policy invariance”;
- “domain randomization is unnecessary”;
- “two-axis non-rejection proves sufficiency.”

The paper can claim only what the registered factorial and learned transfer
experiment actually establish.

## ICLR-specific framing

ICLR 2027 explicitly welcomes reinforcement learning, transfer learning,
representation interpretation, datasets and benchmarks, physics-informed ML,
robotics/autonomy, and physical-science applications. The main text is limited
to nine pages at submission and must include a mandatory AI-use disclosure.

- Author guidelines: <https://iclr.cc/Conferences/2027/AuthorGuidelines>
- Call for papers: <https://iclr.cc/Conferences/2027/CallForPapers>
- AI policy for authors: <https://iclr.cc/Conferences/2027/AIPolicyForAuthors>
