# Implementation Brief: Stochastic Particle Collection Benchmark

> **Audience:** Codex, Claude Code, or another coding agent operating directly in the repository.
>
> **Purpose:** Set up a complete, reproducible project skeleton and create the first runnable simulator, scripted baselines, trajectory-generation pipeline, validation tests, analysis scripts, and HPC launch scripts for a benchmark-and-dataset paper.
>
> **Time constraint:** The main empirical runs must be completed within a two-week HPC access window. Prioritize correctness, reproducibility, and experiment throughput over architectural novelty or software generality.

---

## 0. Core instruction to the coding agent

Build a research codebase for a **cooperative stochastic-particle collection benchmark**. A small team of mobile collectors observes only local particle motion. Particle trajectories contain stochastic noise and may also contain a weak latent transport field. Collectors capture particles, and in one condition captured particles remain attached and expand future capture geometry.

The project is a **benchmark and dataset paper**, not a new-learning-architecture paper. The first implementation must make it easy to answer four empirical questions:

1. Does a measurable signal-strength boundary separate unreliable from reliable exploitation?
2. Does irreversible growth amplify both genuine signal and accidental early contact?
3. Is coordination most valuable near the detectability boundary?
4. How does the boundary shift with collector count and one environmental scaling axis?

Do not overengineer the environment. Do not add strategic particle policies, learned evaders, complex communication protocols, deformable aggregates, multiple particle species, or elaborate vehicle dynamics.

The first successful milestone is:

```text
python -m scripts.smoke_env
```

This command must run deterministic episodes for the null, uniform-drift, and vortex environments under both fixed and growing capture geometry, print summary metrics, save a small trajectory artifact, and exit successfully.

---

# 1. Scientific contract

## 1.1 Central benchmark question

The benchmark studies:

> At what signal strength can locally informed collectors reliably detect and exploit hidden structure in stochastic particle motion, and how do coordination and irreversible capture growth change that threshold?

## 1.2 Strategic agents

Only collectors are agents.

Particles are non-learning environment entities following fixed stochastic transition rules. This distinction must be reflected in the API and documentation. Do not model particles as PettingZoo agents.

## 1.3 Canonical experiment

Use the following initial canonical configuration. These values are pilot defaults, not scientific constants. They must be configurable from YAML.

```yaml
arena:
  width: 1.0
  height: 1.0
  boundary: reflecting

time:
  dt: 0.02
  horizon: 400

collectors:
  count: 4
  max_speed: 0.30
  sensing_radius: 0.16
  capture_radius: 0.018
  initial_layout: corners_inset

particles:
  count: 256
  diffusion_sigma: 0.10
  particle_radius: 0.004
  collective_strength: 0.0

field:
  family: uniform
  signal_strength: 0.20
  orientation_rad: 0.0

capture:
  geometry: fixed
  attached_disc_radius: 0.008

observation:
  nearest_particles_k: 32
  include_particle_velocity: true
  velocity_estimation_window: 2
  include_teammates: true
  include_team_summary: false

reward:
  capture_scale: 1.0
  movement_cost: 0.001
```

Use `particles.count=256` for the first running version because it is fast enough for debugging and large enough to produce many-particle dynamics. Add `512` later as a scale condition.

## 1.4 Environment families

The first release must contain exactly three field families:

1. `null`: no latent transport field.
2. `uniform`: constant transport direction with configurable orientation and strength.
3. `vortex`: a curved local transport field around a configurable centre.

Do not implement moving fields until the core experiment pipeline is frozen.

## 1.5 Capture conditions

Implement exactly two capture geometries:

- `fixed`: the collector has a constant circular capture region.
- `growing`: captured particles remain attached to that collector and become additional circular capture sites.

The growing condition must create path dependence. It must not be implemented as a single radius that increases with capture count.

## 1.6 Information conditions

Support these two benchmark conditions:

- `independent`: each collector receives only its own local observation and teammate positions.
- `shared_summary`: each collector also receives a low-dimensional team summary consisting of per-agent local particle count and estimated local mean particle velocity.

This is an environment-controlled information ablation, not a proposed learned communication architecture.

## 1.7 Initial baseline set

The first complete codebase must include:

- `random`
- `coverage`
- `density_greedy`
- `local_flow`
- `team_flow`
- `oracle_field`

Add learning baselines only after all scripted baselines pass validation.

The intended learning baselines are:

- shared-parameter recurrent IPPO;
- one standard MAPPO implementation.

Learning baselines are benchmark participants, not contributions.

---

# 2. Non-goals

Do not spend time on any of the following during the first two-week cycle:

- a novel coordination network;
- strategic or learned particle evasion;
- general-purpose swarm simulation;
- realistic hydrodynamics;
- rigid-body collision physics between aggregates;
- aggregate rotation, bending, breaking, or topology optimization;
- heterogeneous collector bodies;
- multi-market or HFT reinterpretations;
- distributed training infrastructure beyond simple SLURM arrays;
- web dashboards;
- experiment tracking systems that require an external service;
- excessive hyperparameter optimization;
- a custom replay buffer unless required by a selected algorithm;
- a full GUI.

A lightweight Matplotlib renderer for debugging is sufficient.

---

# 3. Required repository layout

Create the following structure.

```text
particle-collection-benchmark/
├── README.md
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── configs/
│   ├── env/
│   │   ├── canonical.yaml
│   │   ├── null.yaml
│   │   ├── uniform.yaml
│   │   └── vortex.yaml
│   ├── baseline/
│   │   ├── random.yaml
│   │   ├── coverage.yaml
│   │   ├── density_greedy.yaml
│   │   ├── local_flow.yaml
│   │   ├── team_flow.yaml
│   │   └── oracle_field.yaml
│   ├── train/
│   │   ├── ippo.yaml
│   │   └── mappo.yaml
│   ├── experiments/
│   │   ├── pilot.yaml
│   │   ├── detectability.yaml
│   │   ├── aggregation.yaml
│   │   ├── coordination.yaml
│   │   └── scaling.yaml
│   └── hpc/
│       └── cluster.yaml
├── src/
│   └── particle_benchmark/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── seeding.py
│       ├── env.py
│       ├── state.py
│       ├── spaces.py
│       ├── dynamics/
│       │   ├── __init__.py
│       │   ├── boundaries.py
│       │   ├── particles.py
│       │   ├── collectors.py
│       │   ├── fields.py
│       │   └── capture.py
│       ├── observations/
│       │   ├── __init__.py
│       │   ├── local.py
│       │   └── team_summary.py
│       ├── policies/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── random_policy.py
│       │   ├── coverage_policy.py
│       │   ├── density_policy.py
│       │   ├── flow_policy.py
│       │   └── oracle_policy.py
│       ├── rl/
│       │   ├── __init__.py
│       │   ├── networks.py
│       │   ├── rollout_buffer.py
│       │   ├── ippo.py
│       │   └── mappo.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── episode_writer.py
│       │   ├── manifest.py
│       │   └── validation.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── episode.py
│       │   ├── matched.py
│       │   └── boundary.py
│       ├── rendering/
│       │   ├── __init__.py
│       │   └── matplotlib_renderer.py
│       └── utils/
│           ├── __init__.py
│           ├── io.py
│           └── logging.py
├── scripts/
│   ├── __init__.py
│   ├── smoke_env.py
│   ├── run_scripted.py
│   ├── generate_dataset.py
│   ├── train_ippo.py
│   ├── train_mappo.py
│   ├── evaluate_policy.py
│   ├── analyze_detectability.py
│   ├── analyze_aggregation.py
│   ├── analyze_coordination.py
│   ├── build_manifest.py
│   ├── validate_dataset.py
│   └── render_episode.py
├── slurm/
│   ├── smoke_cpu.sbatch
│   ├── scripted_array.sbatch
│   ├── dataset_array.sbatch
│   ├── train_ippo_array.sbatch
│   ├── train_mappo_array.sbatch
│   └── evaluate_array.sbatch
├── tests/
│   ├── test_seeding.py
│   ├── test_boundaries.py
│   ├── test_fields.py
│   ├── test_capture_fixed.py
│   ├── test_capture_growing.py
│   ├── test_observations.py
│   ├── test_env_determinism.py
│   ├── test_matched_pairs.py
│   ├── test_episode_writer.py
│   └── test_scripted_baselines.py
├── notebooks/
│   └── README.md
├── outputs/
│   └── .gitkeep
└── data/
    └── .gitkeep
```

All importable code must live under `src/particle_benchmark`. Files under `scripts/` should be thin command-line entry points.

---

# 4. Technology choices

## 4.1 Python and packaging

Use Python 3.11.

Use a standard `pyproject.toml` with editable installation:

```bash
python -m pip install -e ".[dev,rl]"
```

Suggested dependencies:

```toml
[project]
dependencies = [
  "numpy>=1.26",
  "scipy>=1.12",
  "pandas>=2.2",
  "pyarrow>=15",
  "matplotlib>=3.8",
  "pyyaml>=6.0",
  "omegaconf>=2.3",
  "typer>=0.12",
  "rich>=13.7",
  "tqdm>=4.66",
  "gymnasium>=0.29",
  "pettingzoo>=1.24",
]

[project.optional-dependencies]
rl = [
  "torch>=2.2",
  "tensorboard>=2.16",
]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.4",
  "mypy>=1.10",
  "pre-commit>=3.7",
]
```

Do not require CUDA for installation. GPU support should activate automatically when PyTorch detects it.

## 4.2 Environment API

Implement the environment using the PettingZoo `ParallelEnv` API, with collectors as the only agents.

Also provide a thin single-process vector wrapper later if needed for PPO throughput. Do not block the simulator milestone on vectorization.

The environment must expose:

```python
obs, infos = env.reset(seed=seed, options=options)
obs, rewards, terminations, truncations, infos = env.step(actions)
```

Agent names must be stable:

```python
collector_0
collector_1
collector_2
collector_3
```

## 4.3 Configuration

Use YAML files loaded with OmegaConf into typed dataclasses.

Support CLI overrides of the form:

```bash
python -m scripts.run_scripted \
  --config configs/experiments/pilot.yaml \
  field.family=uniform \
  field.signal_strength=0.2 \
  capture.geometry=growing \
  run.seed=7
```

Do not introduce Hydra run-directory magic. Explicitly define the output directory in the configuration.

## 4.4 Logging

Use:

- Python `logging` for human-readable logs;
- JSON Lines for per-episode metrics;
- TensorBoard only for learning curves;
- no external tracking service.

Every run directory must contain:

```text
config_resolved.yaml
run_metadata.json
metrics.jsonl
stdout.log
```

Training runs additionally contain:

```text
checkpoints/
tensorboard/
```

---

# 5. Exact environment design

## 5.1 Coordinate system

Use a continuous rectangular domain:

```text
x in [0, width]
y in [0, height]
```

Canonical width and height are `1.0`.

Use `float32` for stored trajectories and state arrays. Use `float64` only inside numerical analyses when necessary.

## 5.2 Collector motion

Use holonomic point-mass control to isolate inference and coordination from vehicle steering.

Each action is a two-dimensional desired velocity vector:

```python
action_i = np.ndarray(shape=(2,), dtype=np.float32)
action_i in [-1, 1]^2
```

Update:

```python
velocity = max_speed * clip_norm(action, max_norm=1.0)
position_next = position + dt * velocity
position_next = reflect(position_next)
```

Do not implement heading or angular velocity in version 1.

Record collector velocity in the state and trajectory.

## 5.3 Reflecting boundaries

Implement exact reflection, including overshoot.

For a scalar coordinate `q` and interval `[0, L]`, repeatedly reflect until it lies in bounds. A vectorized modulo-based implementation is preferable.

Add tests for:

- one-wall crossing;
- corner crossing;
- large overshoot crossing more than one wall length;
- no-op for an in-bounds coordinate.

## 5.4 Particle dynamics

For each free particle `k`:

```text
x_{k,t+1} = x_{k,t}
            + dt * field_velocity(x_{k,t})
            + sigma * sqrt(dt) * epsilon_{k,t}
```

Initially set collective interaction to zero. Keep a clean extension point:

```python
collective_velocity = collective_model.compute(...)
```

but do not spend time implementing a complex model before the main pipeline works.

If a minimal collective term is added later, implement only local alignment using a spatial grid. Do not use an O(N^2) neighbour search for production runs.

### Noise convention

Sample:

```python
epsilon ~ Normal(0, I_2)
```

The diffusion term must be scaled by `sqrt(dt)`.

## 5.5 Latent fields

All field implementations must return a velocity array of shape `(num_particles, 2)` and must be deterministic given positions and field parameters.

### Null field

```python
velocity = np.zeros_like(positions)
```

### Uniform field

For orientation `theta`:

```python
direction = [cos(theta), sin(theta)]
velocity = signal_strength * direction
```

The same vector applies everywhere.

### Vortex field

Let centre be `c=(cx, cy)` and `r=x-c`.

Use a tangential direction:

```text
tangent = (-r_y, r_x) / (||r|| + eps)
```

Use a bounded radial envelope to avoid singular behaviour and unrealistically constant speed far from the centre:

```text
envelope = exp(-||r||^2 / (2 * vortex_scale^2))
velocity = signal_strength * envelope * tangent
```

Support clockwise/counterclockwise with a sign parameter.

Add tests ensuring:

- the null field is zero;
- uniform magnitude is constant;
- vortex velocity is approximately orthogonal to the radius;
- field output is finite at the vortex centre.

## 5.6 Random streams and matched counterfactuals

This is scientifically critical.

Use independent RNG streams derived from one root seed via `numpy.random.SeedSequence`:

```python
root = np.random.SeedSequence(seed)
init_ss, noise_ss, field_ss, tie_ss = root.spawn(4)
```

Use separate generators for:

- initial particle/collector state;
- Brownian noise;
- field parameter sampling;
- tie-breaking.

A matched null/signal pair must share:

- initial particle positions;
- initial collector positions;
- Brownian noise sequence;
- field orientation/centre parameters;
- tie-breaking randomness where applicable.

They must differ only in whether the field strength is set to zero or the selected signal value.

The safest initial implementation is to pre-generate the complete Brownian noise tensor for each episode:

```python
noise.shape == (horizon, num_particles, 2)
```

For `400 x 256 x 2` float32 values, this is small enough for canonical runs and guarantees paired trajectories use identical stochastic forcing.

Add a `scenario_seed` distinct from `policy_seed`.

## 5.7 Initial states

Particles:

- sample uniformly over the arena;
- reject positions inside collector capture discs;
- free at time zero.

Collectors:

Use `corners_inset` for canonical runs:

```text
(inset, inset)
(width-inset, inset)
(inset, height-inset)
(width-inset, height-inset)
```

For non-four collector counts, support:

- evenly spaced perimeter positions;
- random non-overlapping positions.

All initial-state samplers must be deterministic under the scenario seed.

## 5.8 Capture order

Use the following per-step order consistently:

1. receive collector actions;
2. move collectors;
3. move attached aggregate centres with their parent collectors;
4. move free particles;
5. apply boundary reflection;
6. resolve captures;
7. build observations;
8. compute rewards and metrics.

Document this order in code and README. Do not silently change it later.

## 5.9 Fixed capture geometry

Collector `i` captures free particle `k` when:

```text
||particle_position - collector_position||
<= collector_capture_radius + particle_radius
```

Captured particles are removed from the free set and assigned to that collector, but they do not expand geometry.

For trajectory consistency, their final capture position and owner should still be recorded.

## 5.10 Growing capture geometry

Each collector has capture centres:

```text
[collector_root, attached_particle_1, attached_particle_2, ...]
```

Each attached particle stores its relative offset from the collector at capture time:

```python
relative_offset = particle_position - collector_position
```

The aggregate translates with the collector. Do not rotate the aggregate in version 1.

At each step:

```python
world_centres = collector_position + relative_offsets
```

A free particle is captured when it touches any world centre, using the corresponding disc radius.

Required invariants:

- capture ownership is permanent;
- a particle can be captured only once;
- aggregate node count never decreases;
- every attached node was in contact with the existing aggregate at capture time;
- fixed-geometry runs never use attached nodes for future capture;
- growing geometry never has a smaller instantaneous capture set than fixed geometry under the same state.

### Tie-breaking

If a free particle touches multiple collectors in the same step:

1. assign it to the collector with the minimum contact distance;
2. if exactly tied within tolerance, use the dedicated tie RNG;
3. log a tie event.

## 5.11 Computational implementation of capture

Start with a clear vectorized implementation.

For each collector:

1. obtain free particle positions `(F,2)`;
2. obtain aggregate centres `(A_i,2)`;
3. compute squared distances `(F,A_i)`;
4. find each particle's minimum distance to that collector;
5. combine across collectors and resolve owners.

This is acceptable for `N<=512`, `M<=8`, and short horizons.

Profile before optimizing. If capture becomes the bottleneck, introduce a uniform spatial hash. Do not prematurely use a complex geometry library.

---

# 6. Observation design

## 6.1 Goal

Observations must preserve local temporal evidence while remaining fixed-size for PPO.

## 6.2 Per-agent observation dictionary

Use a Gymnasium `Dict` space internally:

```python
{
    "self": Box(...),
    "particles": Box(shape=(K, particle_feature_dim)),
    "particle_mask": MultiBinary(K),
    "teammates": Box(shape=(M_max - 1, teammate_feature_dim)),
    "teammate_mask": MultiBinary(M_max - 1),
    "aggregate": Box(...),
    "team_summary": Box(...),
}
```

Provide a deterministic flattening utility for algorithms requiring vectors.

## 6.3 Self features

Include:

```text
normalized x position
normalized y position
normalized x velocity
normalized y velocity
normalized time remaining
captured count / N
```

## 6.4 Local particle features

Select the nearest `K=32` free particles within the sensing radius.

For each particle, include collector-relative:

```text
dx / sensing_radius
dy / sensing_radius
dvx / velocity_scale
dvy / velocity_scale
distance / sensing_radius
```

If fewer than `K` particles are visible, zero-pad and provide a mask.

Particle velocity should be the actual simulator velocity for the initial benchmark observation. Add a later ablation in which it is estimated from two positions. Do not complicate version 1 by requiring memory solely to recover velocity.

Sort visible particles by distance. This ensures deterministic ordering.

## 6.5 Teammate features

For each teammate, include:

```text
relative dx
relative dy
relative dvx
relative dvy
captured fraction
```

Sort by stable collector index, not distance.

## 6.6 Aggregate summary

Include:

```text
attached count / N
maximum relative extent / arena diagonal
mean relative extent / arena diagonal
x component of aggregate centroid offset
y component of aggregate centroid offset
```

Do not provide the entire aggregate graph to learning agents in version 1.

## 6.7 Shared team summary

When `include_team_summary=true`, provide one row per collector:

```text
visible particle count / K
mean local particle velocity x
mean local particle velocity y
local velocity dispersion
collector captured fraction
```

The summary is computed from information actually observed by each collector. It must not leak the true field or particles outside all sensing regions.

In the independent condition, return a zero vector of the same shape so the observation dimension remains unchanged.

## 6.8 Global state for centralized critics

Provide `env.state()` containing:

- all collector positions and velocities;
- all free particle positions and velocities;
- captured flags and owners;
- aggregate summaries;
- normalized time.

Do not include the latent field label in the default centralized critic state. Add an explicit oracle flag for experiments that require it.

---

# 7. Reward and termination

## 7.1 Shared reward

Use a shared team reward:

```text
reward_t = new_captures / N
           - movement_cost * mean(||action_i||^2)
```

The same scalar is returned for every collector.

Do not add dense rewards for moving in the correct field direction. The environment should reward collection, while diagnostics measure signal use.

## 7.2 Termination

An episode terminates early only if all particles are captured.

Otherwise it truncates at `horizon`.

## 7.3 Reward normalization

Because total capture reward sums to at most one, PPO should not require aggressive reward scaling. Keep optional reward normalization in the trainer, not the environment.

---

# 8. Episode metrics

Compute and log these metrics for every episode.

## 8.1 Primary metrics

```text
capture_fraction
captured_count
time_to_first_capture
first_capture_step
all_captured
mean_action_norm
path_length_per_collector
```

If there is no capture, store `first_capture_step=-1` and `time_to_first_capture=NaN` in tabular outputs.

## 8.2 Aggregation metrics

```text
post_first_capture_count
largest_collector_aggregate
aggregate_concentration
cascade_reached_k5
cascade_reached_k10
cascade_reached_k25
steps_first_to_k10
```

Define `post_first_capture_count` as total captures after the first capture event.

## 8.3 Pre-contact behavioural diagnostics

Before the first capture, compute:

- mean cosine alignment between each collector velocity and the true local field direction;
- distance from each collector to the nearest high-flow streamline proxy;
- fraction of time each collector remains stationary;
- team spatial dispersion.

For `null` episodes, alignment must be recorded as `NaN`, not zero.

These diagnostics use privileged information only for evaluation.

## 8.4 Coordination diagnostics

```text
mean_pairwise_collector_distance
minimum_pairwise_collector_distance
sensing_overlap_proxy
collector_capture_entropy
```

A simple sensing-overlap proxy may be computed from pairwise disc intersection area or by Monte Carlo sampling a fixed grid.

## 8.5 Matched-pair metrics

For a null/signal pair sharing the same scenario seed, compute:

```text
delta_capture_fraction
delta_time_to_first_capture
delta_post_contact_captures
delta_path_length
```

These are analysis outputs, not per-step environment rewards.

---

# 9. Scripted baselines

All scripted policies must implement a common interface:

```python
class Policy(Protocol):
    def reset(self, *, seed: int, agent_ids: list[str]) -> None: ...
    def act(
        self,
        observations: dict[str, dict[str, np.ndarray]],
        infos: dict[str, dict[str, Any]],
    ) -> dict[str, np.ndarray]: ...
```

Scripted policies may maintain internal state.

## 9.1 Random

Sample a persistent random direction for each agent. Resample with probability `p_turn` each step. This is better than independent white-noise actions because it produces meaningful movement.

Config:

```yaml
p_turn: 0.05
speed_fraction: 1.0
```

## 9.2 Coverage

Use deterministic lawnmower or waypoint coverage.

Requirements:

- different collectors receive offset lanes;
- waypoints adapt to collector count;
- no use of particle observations;
- deterministic given seed and config.

## 9.3 Density greedy

Move toward a weighted centroid of visible particles:

```text
target = sum(w_k * relative_position_k) / sum(w_k)
```

Use closer particles with larger weight. If no particle is visible, revert to coverage motion.

## 9.4 Local flow

Estimate mean visible particle velocity from the current local observation.

Move toward a short-horizon intercept point:

```text
estimated_target = mean_position + lookahead * mean_velocity
```

If no particles are visible, use coverage.

Do not use the true field.

## 9.5 Team flow

Use all collectors' shared summaries to estimate a team-average flow vector and allocate different perpendicular offsets to reduce redundant pursuit.

Keep this baseline transparent. It should not contain a trainable model.

Suggested behaviour:

1. average valid local flow estimates weighted by visible particle count;
2. compute common downstream direction;
3. assign collector-specific lateral offsets based on collector index;
4. move toward distinct interception lanes.

## 9.6 Oracle field

Use the true local field velocity at the collector's position.

The oracle should move toward downstream high-density interception rather than blindly following flow forever. A simple implementation can:

- query all particle positions;
- project particle positions forward by an oracle lookahead using the true field;
- assign each collector a cluster or quadrant;
- move toward the assigned projected centroid.

The oracle is allowed global privileged state. Clearly label it as an upper-reference policy.

## 9.7 Baseline validation expectations

At high uniform signal strength:

```text
oracle_field >= local_flow >= random
```

in median capture fraction over a small seed set.

This need not hold for every single seed, but the aggregate ordering should be evident. If it is not, debug environment calibration before training PPO.

---

# 10. Trajectory dataset specification

## 10.1 Storage strategy

Do not allow concurrent workers to write to one shared HDF5 file.

Each job writes independent compressed NPZ episode files and one JSONL metrics file under a unique task directory:

```text
data/raw/<experiment_id>/task_<array_id>/
├── config_resolved.yaml
├── metrics.jsonl
└── episodes/
    ├── episode_<scenario_seed>_<condition>.npz
    └── ...
```

After jobs complete, run a consolidation script that builds one Parquet manifest.

This avoids file locks and is robust on HPC filesystems.

## 10.2 Episode NPZ schema

Each episode file must contain:

```text
schema_version                  scalar string
scenario_seed                  scalar int64
policy_seed                    scalar int64
field_family                   scalar string
field_strength                 scalar float32
field_parameters               serialized JSON string
capture_geometry               scalar string
information_condition          scalar string
collector_count                scalar int32
particle_count                 scalar int32
horizon                        scalar int32
actual_length                  scalar int32

particle_position              float32 [T+1, N, 2]
particle_velocity              float32 [T+1, N, 2]
particle_free_mask             bool    [T+1, N]
particle_owner                 int16   [T+1, N]

collector_position             float32 [T+1, M, 2]
collector_velocity             float32 [T+1, M, 2]
collector_action               float32 [T, M, 2]
collector_reward               float32 [T, M]

capture_event_mask             bool    [T, N]
capture_event_owner            int16   [T, N]
aggregate_relative_offsets     object or padded representation

local_observation_flat         optional float32 [T+1, M, D]
team_summary                   optional float32 [T+1, M, S]
```

Avoid object arrays if possible because they require pickle. Prefer a padded representation:

```text
aggregate_offsets float32 [T+1, M, N+1, 2]
aggregate_mask    bool    [T+1, M, N+1]
```

This is larger but simple. To reduce dataset size, the initial dataset may omit full per-step aggregate offsets because they can be reconstructed from capture events and collector positions. In that case, store:

```text
capture_relative_offset float32 [N, 2]
capture_step            int32   [N]
particle_owner_final    int16   [N]
```

Use the reconstructable representation by default.

## 10.3 Manifest schema

The consolidated Parquet manifest contains one row per episode:

```text
episode_path
experiment_id
split
scenario_seed
policy_seed
policy_name
field_family
field_strength
field_orientation
vortex_center_x
vortex_center_y
capture_geometry
information_condition
collector_count
particle_count
sensing_radius
horizon
capture_fraction
first_capture_step
post_first_capture_count
cascade_reached_k10
mean_action_norm
matched_pair_id
config_hash
git_commit
created_at_utc
```

## 10.4 Dataset splits

Create deterministic splits by scenario seed:

```text
train: 70%
validation: 15%
test: 15%
```

Do not split paired null/signal episodes across different sets.

Use a hash of `matched_pair_id` to assign the split.

Add one explicit OOD split later:

- hold out orientations in `[45°, 90°)` for uniform drift; or
- hold out one arena size.

Start with held-out orientations because it is cheap.

## 10.5 Dataset validation

The validator must check:

- required keys exist;
- array shapes match metadata;
- no NaN or Inf in physical state arrays;
- free masks are monotone non-increasing per particle;
- owners never change once assigned;
- capture steps agree with free-mask transitions;
- collector positions remain in bounds;
- matched pairs have identical initial states and Brownian noise provenance;
- manifest paths exist;
- config hash is present;
- schema version is supported.

Return nonzero exit code on failure.

---

# 11. Initial CLI scripts

All scripts must expose `--help` and return nonzero status on invalid configuration.

## 11.1 `scripts.smoke_env`

Required command:

```bash
python -m scripts.smoke_env --output outputs/smoke
```

Behaviour:

1. instantiate each field family;
2. run fixed and growing geometry;
3. use random and oracle policies;
4. use two scenario seeds;
5. render one short GIF or MP4 if rendering dependencies are available;
6. save one NPZ trajectory;
7. print a compact table of capture fraction and first-capture time;
8. run internal assertions;
9. finish in under two minutes on one CPU core.

## 11.2 `scripts.run_scripted`

Example:

```bash
python -m scripts.run_scripted \
  --config configs/experiments/pilot.yaml \
  --policy local_flow \
  --seeds 0:5 \
  --output outputs/pilot_local_flow
```

Support:

- seed ranges such as `0:20`;
- comma-separated seeds;
- matched null/signal generation;
- one or more field strengths;
- fixed and growing geometry;
- no rendering by default;
- optional trajectory saving.

## 11.3 `scripts.generate_dataset`

Example:

```bash
python -m scripts.generate_dataset \
  --config configs/experiments/detectability.yaml \
  --task-index 7 \
  --num-tasks 64 \
  --output-root data/raw/detectability_v1
```

Requirements:

- deterministically partition experiment cells across tasks;
- be restart-safe;
- skip already completed valid episodes;
- write task-local metrics and episodes;
- save resolved config and git commit;
- flush metrics after every episode;
- support `--dry-run` to print assigned cells without executing.

## 11.4 `scripts.build_manifest`

Example:

```bash
python -m scripts.build_manifest \
  --input-root data/raw/detectability_v1 \
  --output data/manifests/detectability_v1.parquet
```

## 11.5 `scripts.validate_dataset`

Example:

```bash
python -m scripts.validate_dataset \
  --manifest data/manifests/detectability_v1.parquet \
  --workers 8
```

Print:

- number of valid episodes;
- number of invalid episodes;
- error counts by validation rule;
- matched-pair completeness;
- missing experiment cells.

## 11.6 `scripts.render_episode`

Example:

```bash
python -m scripts.render_episode \
  --episode data/raw/.../episode_0007_signal.npz \
  --output outputs/episode_0007.mp4
```

Renderer requirements:

- free particles as points;
- collectors as larger markers;
- attached particles visually distinguishable;
- sensing discs optional;
- field vectors optional;
- title with time, captures, field, and geometry;
- no dependency on a display server.

Use Matplotlib's noninteractive backend.

---

# 12. Learning baseline requirements

Do not begin this section until scripted baselines and dataset validation pass.

## 12.1 Shared-parameter recurrent IPPO

Use one shared actor network for all collectors.

Recommended initial network:

```text
flattened observation
→ LayerNorm
→ MLP(256, 256, tanh)
→ GRU(128)
→ Gaussian action head
```

Use one decentralized value head per agent for IPPO.

Initial hyperparameters:

```yaml
total_env_steps: 2_000_000
num_envs: 16
rollout_steps: 256
gamma: 0.99
gae_lambda: 0.95
learning_rate: 0.0003
clip_coef: 0.2
entropy_coef: 0.01
value_coef: 0.5
max_grad_norm: 0.5
update_epochs: 5
minibatches: 8
gru_hidden_size: 128
```

Save checkpoints at regular environment-step intervals and retain the best validation checkpoint.

## 12.2 MAPPO

Reuse the same actor.

Use a centralized critic that consumes `env.state()` plus the joint action or recurrent joint representation as appropriate.

Keep MAPPO implementation standard and documented. Do not modify it into a new method.

## 12.3 Training scope

Do not train over the full signal grid initially.

After scripted pilot calibration, select:

- one low signal;
- two transition-region signals;
- one high signal.

Train IPPO and MAPPO only on these values for the first paper pass.

Use three training seeds during debugging. Increase to five only for final selected cells.

## 12.4 Evaluation protocol

Evaluate each checkpoint on at least 20 held-out scenario seeds for final cells, using deterministic mean actions unless stochastic policy evaluation is explicitly required.

Store evaluation trajectories separately from training rollouts.

---

# 13. Detectability analysis

## 13.1 Primary curve

For each policy, field family, geometry, and information condition, plot:

```text
x-axis: signal strength or effective SNR
y-axis: capture fraction or probability of benchmark success
```

Show seed-level bootstrap confidence intervals.

## 13.2 Success event

The initial success event should be simple and pre-specified after pilots:

```text
success = capture_fraction >= c_star
```

Choose `c_star` using pilot calibration, not by maximizing significance.

Also report continuous capture fraction so conclusions do not depend solely on the threshold.

## 13.3 Boundary estimate

Estimate the signal level at which success probability crosses `tau=0.5`.

Implement two estimators:

1. logistic regression;
2. isotonic regression with interpolation.

Bootstrap at the scenario-seed level.

Return:

```text
boundary_estimate
95% bootstrap interval
number of seeds
fit diagnostics
out-of-range flag
```

If the fitted crossing lies outside the tested signal range, report it as censored rather than extrapolating confidently.

## 13.4 Matched analysis

Because null/signal episodes share scenario seeds, report paired differences and paired bootstrap intervals wherever possible.

Do not treat matched episodes as independent samples.

---

# 14. Experiment configurations

## 14.1 Pilot experiment

Purpose: calibrate difficulty and verify baseline ordering.

```yaml
field_families: [null, uniform, vortex]
signal_strengths: [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]
capture_geometries: [fixed, growing]
policies: [random, coverage, density_greedy, local_flow, oracle_field]
scenario_seeds: [0, 1, 2]
collector_count: 4
particle_count: 256
save_trajectories: true
```

These signal values are pilot values only. Replace them after inspecting whether they span failure, transition, and success.

## 14.2 Main detectability experiment

After pilot calibration, choose 6–8 strengths with denser spacing around the transition.

Target:

```text
5 seeds per full grid cell initially
15–20 seeds around the transition
20 seeds for final headline cells
```

Policies:

```text
random
coverage
density_greedy
local_flow
team_flow
oracle_field
```

Add IPPO/MAPPO only at selected strengths.

## 14.3 Aggregation experiment

Compare fixed and growing geometry under matched scenarios.

Must include:

- null episodes;
- transition-region signal;
- high signal;
- local_flow;
- team_flow;
- oracle_field;
- one learning baseline if available.

Primary outputs:

```text
capture fraction
post-first-capture captures
false cascade rate under null
cascade probability under signal
```

## 14.4 Coordination experiment

Compare:

- `local_flow` vs `team_flow`;
- IPPO independent observation vs MAPPO/shared-summary condition, if training is complete.

Use one low, two transition, and one high signal value.

The key hypothesis is a hump-shaped coordination benefit: small at very low and very high signal, largest near the transition.

## 14.5 Scaling experiment

Use collector count as the first axis:

```yaml
collector_counts: [1, 2, 4, 8]
```

Use sensing radius as the second axis only if compute permits:

```yaml
sensing_radii: [0.10, 0.16, 0.24]
```

Do not vary collector count, particle count, arena size, and sensing radius all at once.

---

# 15. HPC execution design

## 15.1 General principles

- use SLURM arrays;
- one task writes to one unique output directory;
- never rely on shared mutable state;
- use environment variables for scratch paths;
- copy final outputs back to persistent storage;
- make jobs restart-safe;
- save resolved configs and git commits;
- use CPU for scripted baselines and dataset generation;
- use GPU only for PPO/MAPPO.

## 15.2 Example CPU SLURM script

Create `slurm/scripted_array.sbatch` similar to:

```bash
#!/bin/bash
#SBATCH --job-name=pc-scripted
#SBATCH --array=0-63
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module purge
# Adapt modules to the actual cluster.
module load python/3.11

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m scripts.generate_dataset \
  --config configs/experiments/detectability.yaml \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks 64 \
  --output-root "${PROJECT_DATA}/detectability_v1"
```

Do not hard-code cluster-specific module names beyond a clearly marked placeholder.

## 15.3 Example GPU training script

Use one GPU per task initially.

```bash
#!/bin/bash
#SBATCH --job-name=pc-mappo
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m scripts.train_mappo \
  --config configs/train/mappo.yaml \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${PROJECT_OUTPUTS}/mappo_v1"
```

## 15.4 Dry-run manifests

Before submitting any array, generate a task manifest:

```text
task_index
policy
field_family
field_strength
capture_geometry
information_condition
seed
output_path
```

The script should print the number of cells and estimated episodes per task.

## 15.5 Job completion checks

Create a helper command or Make target that reports:

- expected task count;
- completed task count;
- failed task count;
- missing episode count;
- invalid episode count;
- aggregate runtime.

---

# 16. Tests and acceptance criteria

## 16.1 Unit tests

### Seeding

- same seed produces bitwise-identical initial state;
- same seed produces identical Brownian tensor;
- different seeds produce different scenarios;
- field toggle does not change Brownian noise.

### Boundaries

- all positions remain in bounds;
- reflection preserves overshoot distance.

### Capture

- no particle is captured twice;
- ownership is permanent;
- growing aggregate count is monotone;
- fixed geometry capture region remains constant;
- tie resolution is deterministic under tie seed.

### Observations

- shapes match declared spaces;
- masks match visible counts;
- no hidden-field leakage in non-oracle observations;
- particle ordering is deterministic;
- zero visible particles produces finite observations.

### Writer

- episode save/load round-trip preserves arrays;
- invalid schema is rejected;
- interrupted temporary files are not mistaken for completed episodes.

## 16.2 Integration tests

Run short episodes and verify:

- null, uniform, and vortex all execute;
- fixed and growing geometry both execute;
- all scripted policies execute;
- trajectories render;
- manifest builds;
- dataset validator passes.

## 16.3 Scientific sanity tests

These are not proofs, but the pipeline must flag failures.

1. Under `signal_strength=0`, uniform-field orientation must not affect dynamics.
2. At sufficiently high uniform signal, oracle performance should exceed random in median.
3. Growing geometry should never capture fewer particles than fixed geometry when replaying identical collector actions and particle paths, unless movement depends on aggregate size. Aggregate-dependent mobility is disabled in version 1.
4. Matched null/signal episodes must have identical initial states.
5. A zero-action policy must keep collectors stationary.
6. With `diffusion_sigma=0` and `signal_strength=0`, stationary particles must remain stationary.

## 16.4 Code quality gates

Before launching HPC jobs:

```bash
ruff check .
ruff format --check .
mypy src/particle_benchmark
pytest -q
python -m scripts.smoke_env
```

No HPC sweep should begin while these fail.

---

# 17. Implementation order

## Phase 1: Repository and configuration

Deliver:

- packaging;
- typed config loader;
- logging;
- seed utilities;
- canonical YAML;
- basic CI-friendly tests.

Acceptance command:

```bash
python -c "import particle_benchmark; print(particle_benchmark.__version__)"
```

## Phase 2: Simulator core

Deliver:

- state arrays;
- reflecting boundaries;
- collector motion;
- particle motion;
- three field families;
- fixed capture;
- growing capture;
- local observations;
- reward and metrics.

Acceptance:

```bash
pytest -q tests/test_env_determinism.py tests/test_capture_growing.py
```

## Phase 3: Scripted baselines and rendering

Deliver all six scripted baselines and `run_scripted`.

Acceptance:

```bash
python -m scripts.run_scripted \
  --config configs/experiments/pilot.yaml \
  --policy oracle_field \
  --seeds 0:3 \
  --output outputs/oracle_pilot
```

## Phase 4: Dataset generation

Deliver writer, manifest, validation, matched pairs, restart-safe task partitioning.

Acceptance:

```bash
python -m scripts.generate_dataset \
  --config configs/experiments/pilot.yaml \
  --task-index 0 \
  --num-tasks 1 \
  --output-root data/raw/pilot_v0

python -m scripts.build_manifest \
  --input-root data/raw/pilot_v0 \
  --output data/manifests/pilot_v0.parquet

python -m scripts.validate_dataset \
  --manifest data/manifests/pilot_v0.parquet
```

## Phase 5: Analysis

Deliver:

- summary tables;
- detectability curves;
- paired null/signal analysis;
- aggregation comparison;
- bootstrap intervals;
- boundary estimation.

## Phase 6: Learning baselines

Implement IPPO first, then MAPPO.

Do not delay scripted-data generation while waiting for RL.

## Phase 7: HPC arrays

Only after local end-to-end pipeline passes.

---

# 18. Two-week operational schedule

## Days 1–2

- complete Phases 1–3;
- run smoke tests;
- freeze step order and capture semantics;
- generate a tiny pilot dataset;
- inspect rendered episodes manually.

Hard freeze after Day 2: no mechanic changes except correctness fixes.

## Days 3–4

- run scripted pilot grid;
- recalibrate signal strengths;
- confirm oracle/random separation;
- confirm fixed/growing difference;
- choose final canonical signal grid.

## Days 5–8

- launch main scripted detectability and aggregation arrays;
- build dataset shards continuously;
- begin IPPO implementation/training in parallel;
- add seeds around the transition as results arrive.

## Days 9–10

- run coordination information ablation;
- train/evaluate MAPPO only on selected signal values;
- avoid a broad hyperparameter sweep.

## Days 11–12

- run collector-count scaling;
- optionally run sensing-radius scaling;
- fill missing transition cells.

## Days 13–14

- validate all datasets;
- rerun failed cells;
- freeze configs and git commit;
- generate final figures and tables;
- copy all raw and consolidated outputs to persistent storage;
- stop exploratory jobs.

---

# 19. Required outputs after the initial coding pass

The coding agent should leave the repository with all of the following working:

1. Editable installation from `pyproject.toml`.
2. Deterministic PettingZoo parallel environment.
3. Null, uniform, and vortex fields.
4. Fixed and growing capture geometry.
5. Independent and shared-summary observation conditions.
6. Six scripted baselines.
7. Smoke-test command.
8. Scripted-rollout command.
9. Restart-safe dataset-generation command.
10. NPZ trajectory schema.
11. Parquet manifest builder.
12. Dataset validator.
13. Matplotlib episode renderer.
14. Detectability analysis script.
15. Aggregation analysis script.
16. SLURM CPU-array templates.
17. Stubbed but documented IPPO/MAPPO entry points if full training is not yet complete.
18. Tests covering determinism, matched pairs, capture invariants, and storage.
19. README with exact local and HPC commands.
20. One committed example output under `outputs/example/` small enough for Git.

---

# 20. README content requirements

The repository README must contain:

1. one-paragraph scientific description;
2. explicit statement that particles are non-learning;
3. environment diagram or rendered frame;
4. installation commands;
5. smoke-test command;
6. scripted-baseline command;
7. dataset-generation command;
8. validation command;
9. training command placeholders;
10. output-directory explanation;
11. reproducibility and seeding explanation;
12. citation placeholder;
13. license placeholder;
14. known limitations.

Keep the README practical. Do not reproduce the full paper vision.

---

# 21. Makefile targets

Create at least:

```makefile
install:
	python -m pip install -e ".[dev,rl]"

format:
	ruff format .

lint:
	ruff check .
	mypy src/particle_benchmark

test:
	pytest -q

smoke:
	python -m scripts.smoke_env --output outputs/smoke

pilot:
	python -m scripts.run_scripted --config configs/experiments/pilot.yaml --output outputs/pilot

validate-pilot:
	python -m scripts.validate_dataset --manifest data/manifests/pilot.parquet
```

---

# 22. Reproducibility metadata

Every run must record:

```text
git commit hash
git dirty flag
Python version
NumPy version
PyTorch version if used
CUDA version if used
hostname
SLURM job ID and array ID if present
resolved configuration
scenario seeds
policy seeds
wall-clock start/end
schema version
```

Use UTC timestamps.

Configuration hashing must ignore output path and timestamp fields but include all scientific parameters.

---

# 23. Failure-handling rules

- Write episode files to a temporary path and atomically rename after successful validation.
- Flush `metrics.jsonl` after every episode.
- If an episode fails, log the full configuration and traceback to `failures.jsonl` and continue unless the error indicates global configuration invalidity.
- A resumed job must skip valid completed episodes.
- Never silently replace a file produced under a different config hash.
- Never mix schema versions in one manifest without explicit migration.

---

# 24. Performance guidance

Profile before optimizing.

Expected initial bottlenecks:

- capture distance calculations;
- repeated observation sorting;
- trajectory serialization;
- Python loops over collectors.

Safe early optimizations:

- vectorized NumPy state updates;
- `argpartition` for nearest-K particles;
- preallocated trajectory arrays;
- float32 storage;
- one compressed NPZ write per episode;
- batching independent environments for PPO.

Avoid:

- premature JAX rewrite;
- custom CUDA kernels;
- complex spatial trees that are rebuilt every step;
- multiprocessing inside each SLURM array task unless profiling shows a benefit.

---

# 25. Analysis outputs required for the paper

The analysis scripts must be able to produce these artifacts from manifests alone:

## Figure 1: Detectability curves

Facets:

```text
field family
capture geometry
```

Lines:

```text
scripted baselines and selected learned baselines
```

Show seed-bootstrap intervals.

## Figure 2: Aggregation signal versus accident

Plot:

```text
true cascade rate under signal
false cascade rate under null
```

