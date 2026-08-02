# Stochastic Particle System — FR-B3 Branch

This branch isolates the FR-B3 catchability question:

> Are drift SNR (`rho`) and the drift-to-collector speed ratio (`kappa`)
> sufficient to predict coordination gain, or must the benchmark also include
> absolute transport scale (`eta`)?

The historical SPS-C03 result supplies one anchor: at `alpha=0.06`,
`sigma=0.06`, `dt=0.02`, and the **executed** `v_max=0.12`, the bounded shared
summary improved mean unique capture yield by `+1.1875` over the
capacity-matched independent controller (one-sided lower bound `+0.4587`, 32
matched seeds). Its dimensionless coordinates are `rho=0.1414`, `kappa=0.50`,
and `eta=0.00849`.

FR-B3 tests whether that effect can be predicted across a controlled
`3 x 3 x 3` dimensionless regime and whether physically equivalent rescalings
preserve normalized behavior. The inherited environment, policies, immutable
SPS-C03 evidence, and tests remain because they are direct dependencies of the
FR-B3 audit; the new research assets are confined to the FR-B3 files listed in
[catchability-benchmark/README.md](catchability-benchmark/README.md).
