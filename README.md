# NOMAD

NOMAD trains a recurrent RL policy while adapting the distribution of environment parameters with a normalizing flow. The core idea is simple: the policy learns on a family of environments, and the flow gradually concentrates on contexts that are both informative and hard.

The method is generic. The current repository, however, contains one concrete instantiation: a one-zone RC5 thermal model with a heat pump and PID low-level control (this part is not generic yet).

## What is in this repository

The repository is split into a generic core and one concrete RC5 backend.

- `NOMAD/core/training.py`: generic training loop, callbacks, checkpointing, and PPO/ADR orchestration.
- `NOMAD/core/adr.py`: bounded normalizing-flow distribution and ADR update rule.
- `NOMAD/core/backend.py`: backend interface used to plug a simulator into the generic core.
- `NOMAD_RC5/sim.py`: RC5 context definition, bounds, data loading, and differentiable simulation helpers.
- `NOMAD_RC5/env.py`: optimized Torch RC5 batch simulator and its SB3 `VecEnv` adapter.
- `NOMAD_RC5/backend.py`: RC5 backend, default environment/policy/ADR overrides, and policy wiring.
- `NOMAD_RC5/training.py`: RC5 training entry point.
- `NOMAD_RC5/test/`: legacy Gym/JAX env and plot helpers for simax, MPC, and RL comparisons.
- `NOMAD/tests/test_core_smoke.py` and `NOMAD_test1/tests/test_smoke.py`: minimal smoke tests.

## Training loop

At a high level, training alternates between two coupled updates:

1. Sample a context `c` from the current flow.
2. Reset the environment with this context.
3. Train a recurrent PPO policy on the resulting trajectories.
4. Every few episodes, evaluate many candidate contexts in a differentiable batch simulator.
5. Refit the flow so that it assigns more mass to contexts with high ADR objective.

In the current code:

- the RL policy is `sb3_contrib.RecurrentPPO` with a custom `ValueCtxLstmPolicy` built on `MultiInputLstmPolicy`;
- the observation is a dict with a `now` vector and a `forecast` tensor;
- the action is a normalized residual around a base temperature setpoint (RC5-specific);
- ADR updates are done by `ADRFlows`, then pushed back to all vectorized environments through `set_sampling_dist`.

## RC5 Torch/GPU training

Use the optimized RC5 path:

```bash
python -m NOMAD_RC5.training
```

This path keeps RC5 simulation in Torch through two classes:

- `RC5TorchBatch`: simulates many RC5 environments as one tensor batch on `cuda`, `mps`, or `cpu`;
- `RC5TorchVecEnv`: thin Stable-Baselines3 adapter around `RC5TorchBatch`.

For PPO, `n_envs` is the batch size of parallel environments. Each row has its own context, start hour, exogenous window, state, reward, and done flag. PPO collects:

`rollout_size = n_envs * n_steps`

Then it computes advantages and minibatch updates from that rollout.

For ADR, `ADRFlows` also uses `RC5TorchBatch`, but directly. `n_sample` candidate contexts are evaluated in parallel on `adr_device`. After each ADR update, the learned context distribution is copied back to the PPO training env with `set_sampling_dist`.

One practical limitation remains: SB3 expects NumPy at the `VecEnv` boundary. The physics runs in Torch, but observations and rewards are converted back to NumPy before PPO consumes them. Larger `n_envs` helps amortize that boundary cost.

`torch.compile` is not enabled by default. Local tests showed speedups on CPU for the physics step after a large first compile cost, but MPS failed in Inductor. Treat it as an experiment, not as the default path.

## Environment interface

### Context

The environment is parameterized by a context vector `c`.

In the current RC5 implementation, `c` has 25 dimensions:

- 12 thermal parameters `TH`: capacitances, resistances, solar gain factor;
- 10 heat-pump parameters `PAC`: polynomial coefficients, gains, nominal temperatures;
- 3 PID gains: `kp`, `ki`, `kd`.

These parameters are sampled inside fixed box constraints. At the moment those bounds are RC5-specific:

- most thermal and heat-pump parameters use nominal value `+-20%`;
- `Tan` and `Tcn` use `+-5 K`;
- PID gains use `+-20%` for `kp` and `ki`, and `[0, 0.1]` for `kd`.

### Observation

Observations are split into present features and future exogenous information.

- `now`: current aggregated features for the control interval, plus the last zone temperature and last heat-pump power.
- `forecast`: `future_steps` future slices of exogenous features.

In the current RC5 setup:

- `now` has 13 features;
- `forecast` has shape `(future_steps, 13)`;
- the features include weather, solar gains, occupancy, electricity price, calendar embeddings, and comfort bounds (feature semantics are RC5/data-specific for now).

### Action

The policy outputs a normalized action in `[-1, 1]`. In the Torch env, it is mapped directly to a setpoint:

`setpoint = base_setpoint + max_dev * action`

The setpoint is then clipped to `[tz_min, tz_max]`.

This action design is generic in spirit, but the current implementation assumes a single scalar thermal setpoint (RC5-specific).

### Reward

Each control step integrates three costs over the interval:

- comfort violation;
- energy cost;
- actuator saturation.

The raw reward is the negative weighted sum of these terms:

`reward_raw = -(w_comfort * comfort + w_energy * energy + w_sat * saturation)`

The policy is not trained directly on `reward_raw`, but on a normalized improvement over a baseline controller:

- `reward_ref`: baseline reward on the same episode;
- `reward_ref_N`: baseline reward under unit occupancy and unit price, used only as a scale factor.

The actual training reward is:

`reward = (reward_raw - reward_ref) / max(-reward_ref_N, 1e-3)`

This normalization idea is generic. The present baseline is a fixed PID trajectory at `base_setpoint`, and the exact denominator construction is specific to the current RC5 implementation.

## The flow

### What the flow models

The flow models a distribution over contexts, not over states or trajectories.

If `c` denotes the context vector, NOMAD learns a bounded density `q(c)` over the admissible parameter box. Sampling from the flow means sampling a new environment instance.

### Current parameterization

`NormFlowDist` uses a masked autoregressive flow (`zuko.flows.MAF`) with rational-quadratic spline transforms. The support is bounded by construction:

1. the flow produces an unconstrained latent sample `y`;
2. `sigmoid(y)` maps it to `(0, 1)`;
3. an affine transform maps it to `[low, high]`.

So the learned distribution always stays inside the context bounds.

### ADR update in this code

The update implemented in `ADRFlows.update()` is:

1. Draw `n_sample` candidate contexts uniformly in the admissible box.
2. Roll out the current policy in the differentiable batch simulator `RC5TorchBatch`.
3. Compute an ADR objective for each candidate.
4. Optionally refine candidates by gradient ascent in context space for `refine_steps`.
5. Convert objectives to soft weights with a temperature-softmax.
6. Fit the flow to the weighted candidates.
7. Add a KL penalty so the new flow does not move too far from the previous one.

Two details matter:

- Candidate generation is currently uniform over the box, not sampled from the current flow.
- Refinement is projected back into the same box after every gradient step.

### ADR objective

For one candidate context, the objective used to rank samples is:

`objective = ret_coef * return + bonus_coef * bcs + surprise`

with:

- `return`: cumulative normalized RL reward of the current policy;
- `bcs`: baseline shaping term computed from the baseline rollout;
- `ret_coef`: weight applied to the RL return inside the ADR score;
- `bonus_coef`: weight applied to the baseline shaping term inside the ADR score;
- `surprise`: novelty bonus `surprise_coef * (-log q(c))`.

In the current code, `bcs` is built from baseline comfort and saturation costs, with an optional COP penalty:

- baseline comfort term;
- baseline saturation term;
- optional penalty if the effective COP leaves `cop_bounds`.

This makes the flow favor contexts that are both policy-relevant and physically informative. The COP component is currently tied to the RC5 heat-pump model.

### Flow fitting step

Once weighted candidates are obtained, the flow parameters are optimized with:

`loss = -sum_i w_i log q(c_i) + kl_beta * KL(q_new || q_old)`

where the KL term is estimated by Monte Carlo with `kl_M` samples. In practice:

- low `ess_bounds` makes the update concentrate on fewer candidates;
- high `kl_beta` keeps the flow conservative;
- high `surprise_coef` pushes exploration toward low-density regions;
- more `refine_steps` makes candidate search more adversarial.

The fit loss and surprise use unclipped finite log probabilities; clipping is kept only for score weighting and KL stabilization.

## Hyperparameters

Default values are split across:

- `NOMAD/core/training.py` for generic run, PPO, VecNormalize, and core ADR settings;
- `NOMAD_RC5/backend.py` for RC5-specific environment, policy, and ADR overrides;
- `NOMAD_RC5/training.py` for the assembled optimized RC5 `DEFAULT_CFG`.

The tables below report the effective default values of the current RC5 entry point.

For a complete Rorqual-ready configuration, use:

```bash
python -m NOMAD_RC5.training configs/rc5_reasonable.json
```

`configs/rc5_reasonable.json` is strict JSON because the trainer reads it with `json.loads`. Lightweight notes live in the top-level `__notes` field; avoid comment keys inside `ppo`, `env`, `policy`, `vecnorm`, or `adr`, because some of those dictionaries are passed directly into runtime constructors.

This config keeps the RC5 defaults except for cluster-oriented sizing: `n_envs=128`, `save_every_steps=200_000`, `ppo.batch_size=1024`, `adr.n_sample=1024`, and run-specific output paths.

For a short local smoke test before using the cluster:

```bash
python -m NOMAD_RC5.training configs/rc5_local_smoke_mps.json
```

Use `configs/rc5_local_smoke_cpu.json` instead if MPS is unavailable or unstable. These local configs intentionally shorten episodes, reduce model/ADR sizes, trigger a quick ADR update, and save checkpoints every `256` timesteps.

### Generic Hyperparameters

### Run-level

| Key | Default | Meaning | How to choose |
| --- | --- | --- | --- |
| `seed` | `0` | Global random seed. | Keep fixed for reproducibility; change it only to measure sensitivity to randomness. |
| `device` | auto: `cuda`, else `mps`, else `cpu` | Torch device used by PPO and the RC5 Torch batch env. | Prefer `cuda` when available. Use `mps` only after checking stability. |
| `adr_device` | same as `device` | Torch device used for ADR context rollouts. | Keep it equal to `device` unless PPO and ADR need different hardware. |
| `n_envs` | `64` | Number of parallel RC5 environments inside `RC5TorchBatch`. | Increase until steps/sec stops improving or memory becomes tight. |
| `total_timesteps` | `20_000_000` | Total PPO environment steps. | Set this from your time budget and convergence target; use shorter runs for quick sweeps, longer runs for final training. |
| `save_every_steps` | `100_000` | Period for checkpointing `model.zip`, `vecnormalize.pkl`, and `adr_flow.pt`. | Save often enough to recover useful milestones without spending too much on I/O. |
| `resume_dir` | `None` | Optional run directory to resume from. The latest numbered checkpoint is used when present. | Set it to an existing `save_dir` when continuing a stopped run. |
| `save_dir` | `NOMAD_RC5/runs/default` | Output directory. | Choose a run-specific directory when launching experiments you want to compare. |
| `init_flow_path` | `flows/collapsed_flow.pt` | Optional initial flow checkpoint. If the file does not exist, NOMAD falls back to a fresh bounded flow. The current default points to a bundled warm-start flow and is not generic. | Use an existing flow to warm-start ADR; use `None` when you want a clean run from scratch. |

### `ppo`

| Key | Default | Meaning | How to choose |
| --- | --- | --- | --- |
| `learning_rate_start` | `5e-4` | Initial PPO learning rate. | Lower it first if PPO is unstable. |
| `learning_rate_end` | `1e-4` | Final PPO learning rate for the linear schedule. | Lower it if you want gentler late training updates. |
| `n_steps` | `512` | Rollout length per environment before each PPO update. | Keep enough temporal horizon for the LSTM; reduce it when increasing `n_envs`. |
| `batch_size` | `256` | Minibatch size for PPO optimization. | Use the largest stable batch that fits your hardware. |
| `n_epochs` | `5` | Number of PPO passes over each rollout batch. | Increase if PPO underfits each batch; decrease if it becomes slow or overfits stale data. |
| `verbose` | `1` | Stable-Baselines verbosity. | Raise for debugging, lower for quieter runs. |
| `tensorboard_log` | `NOMAD_RC5/tensorboard_logs/tb` | TensorBoard output directory, used only if TensorBoard is installed. | Set a dedicated log directory when you want to compare runs visually in TensorBoard. |

### `vecnorm`

| Key | Default | Meaning | How to choose |
| --- | --- | --- | --- |
| `norm_obs` | `True` | Normalize observations online. | Keep enabled unless raw observation scale is already well controlled and you want maximal simplicity. |
| `norm_reward` | `True` | Normalize rewards online. | Keep enabled for PPO unless you have a good reason to preserve the raw reward scale. |
| `clip_obs` | `10.0` | Observation clipping after normalization. | Increase if clipping is too aggressive; decrease if outliers are destabilizing learning. |

### `adr` (generic)

| Key | Default | Meaning | How to choose |
| --- | --- | --- | --- |
| `transforms` | `3` | Number of autoregressive transforms in the flow. | Increase only if the learned context distribution seems too simple for the task. |
| `bins` | `8` | Number of spline bins per transform. | Increase for a more flexible flow; keep modest if ADR fitting becomes heavy. |
| `hidden` | `(64, 64)` | Hidden-layer sizes of the flow conditioner network. | Increase if the flow underfits candidate weights; decrease for a lighter ADR model. |
| `iters` | `50` | Number of optimizer steps used to fit the flow after each ADR update. | Increase if the flow adapts too slowly; decrease if ADR updates are too expensive. |
| `lr` | `1e-3` | Adam learning rate for the flow. | Lower it first if ADR updates are unstable; raise it only if adaptation is too slow. |
| `n_sample` | `512` | Number of candidate contexts evaluated per ADR update. | Increase when you need broader context-space coverage and can afford the cost. |
| `refine_steps` | `5` | Number of gradient-ascent steps in context space before fitting the flow. | Increase to search for harder contexts; reduce if ADR becomes too adversarial or costly. |
| `refine_lr` | `5e-3` | Step size for context refinement. | Lower it if refinement is erratic; raise it if refinement barely moves candidates. |
| `ret_coef` | `2.0` | Weight of the RL return in the ADR objective. | Increase it if ADR should rank contexts more by policy return than by shaping terms. |
| `bonus_coef` | `1.0` | Weight of the baseline shaping term in the ADR objective. | Increase it if ADR should care more about comfort, saturation, and COP shaping. |
| <span style="color: red;">`ess_bounds`</span> | <span style="color: red;">`(0.05, 0.2)`</span> | <span style="color: red;">Target ESS interval. Values `<= 1` are fractions of the weighted candidate count.</span> | <span style="color: red;">Raise it for broader ADR updates; lower it for sharper context selection.</span> |
| <span style="color: red;">`temp_bounds`</span> | <span style="color: red;">`(1e-3, 1e3)`</span> | <span style="color: red;">Allowed temperature interval for ESS adaptation.</span> | <span style="color: red;">Keep wide unless diagnostics show numerical extremes.</span> |
| `surprise_coef` | `5.0` | Coefficient of the novelty bonus `-log q(c)`. | Increase it if ADR collapses too quickly onto familiar contexts. |
| <span style="color: red;">`kl_beta`</span> | <span style="color: red;">`500.0`</span> | <span style="color: red;">Weight of the trust-region KL penalty between successive flows.</span> | <span style="color: red;">Increase it if the flow moves too abruptly; decrease it if adaptation is too conservative.</span> |
| `kl_M` | `1000` | Monte Carlo sample count for the KL estimate and entropy diagnostics. | Increase for less noisy estimates if compute allows; otherwise keep it moderate. |
| `update_every_episodes` | `100` | ADR update period, measured in completed episodes across vectorized environments. | Reduce it for faster ADR adaptation; increase it if the training distribution changes too often. |

### RC5-Specific Hyperparameters

These parameters come from the current RC5 backend and are not meant to be generic across simulators.

### `env` (RC5)

| Key | Default | Meaning | How to choose |
| --- | --- | --- | --- |
| `step_period` | `3600.0` | RL decision interval in seconds. In RC5, the data timestep is 30 s and `3600 s` means one action per hour. | Use a shorter period if you want finer control and can afford more computation. |
| `future_steps` | `24` | Forecast horizon length in RL steps. | Match this to how far ahead forecasts are useful for control. |
| `base_setpoint` | `294.15` | Reference setpoint around which residual actions are applied. | Set it to the nominal operating point around which residual actions should stay centered. |
| `max_dev` | `5.0` | Maximum residual deviation around `base_setpoint`. | Use the smallest range that still allows meaningful corrective actions. |
| `max_episode_length` | `504` | Episode length in RL steps. | Long episodes help the recurrent policy learn slow thermal effects. |
| `tz_min` | `288.15` | Hard lower bound on the final setpoint. | Set from the coldest admissible command you consider physically and operationally acceptable. |
| `tz_max` | `303.15` | Hard upper bound on the final setpoint. | Set from the hottest admissible command you consider physically and operationally acceptable. |
| `w_energy` | `1.0` | Weight of electricity cost in the reward. | Increase it if you want the policy to prioritize energy savings more strongly. |
| `w_comfort` | `5.0` | Weight of comfort violation in the reward. | Increase it if comfort matters more than energy savings in your objective. |
| `comfort_huber_k` | `0.0` | If `> 0`, applies a Huber penalty to comfort violation; if `0`, uses a linear penalty. | Keep `0` for the simplest setup; increase it if you want a smoother penalty near small violations. |
| `w_sat` | `0.2` | Weight of actuator saturation in the reward. | Increase it if you want to discourage extreme or clipped control effort. |

### `policy` (RC5)

| Key | Default | Meaning | How to choose |
| --- | --- | --- | --- |
| `critic_use_ctx` | `True` | If `True`, the critic receives the context vector `ctx` while the actor only sees `now` and `forecast`. | Keep it enabled unless you specifically want to remove context information from value estimation. |
| `policy_hidden` | `(128, 128)` | Hidden-layer sizes of the policy head. | Increase for a harder control task or richer observations; decrease if training is slow or unstable. |
| `value_hidden` | `(128, 128)` | Hidden-layer sizes of the value head. | Increase if the critic seems to underfit; decrease if you want a lighter model. |

### `adr` (RC5 additions)

| Key | Default | Meaning | How to choose |
| --- | --- | --- | --- |
| `baseline_cs_coef` | `50.0` | Global scale of the baseline shaping term `bcs`. | Increase it if baseline-derived shaping should matter more in context ranking. |
| `baseline_cop_coef` | `5.0` | Extra scale of the baseline COP penalty. | Increase it if you want ADR to reject implausible heat-pump behavior more strongly. |
| `max_episode_length` | `120` | ADR rollout horizon in RL steps. This is shorter than the training episode length to keep ADR updates cheaper. | Increase it if ADR needs longer-horizon behavior; decrease it if ADR updates dominate runtime. |
| `cop_bounds` | `(1.0, 5.0)` | Admissible effective COP interval used by the baseline COP penalty. This is heat-pump specific in the current code. | Set bounds from the physically acceptable COP range of your equipment model. |
| `php_min_w` | `100.0` | Minimum heat-pump power used before a COP sample is considered meaningful. | Raise it if low-power COP estimates are too noisy; lower it if you want to use more samples. |
| `cop_beta` | `1.0` | Softness parameter used in the smooth COP bound aggregation. | Increase it to approximate a harder worst-case penalty; decrease it for smoother aggregation. |

## Hyperparameter Tuning

The tables below give a first-order tuning intuition only. They summarize the usual effect of making a parameter larger/smaller or enabled/disabled; interactions still matter.

### Generic Tuning

### Run-level

| Key | Higher / enabled | Lower / disabled |
| --- | --- | --- |
| `seed` | No monotonic effect; changes randomness and can reveal another training trajectory. | No monotonic effect; changes randomness and can reveal another training trajectory. |
| `device` | Faster hardware reduces wall-clock time and makes larger experiments easier. | Slower hardware increases wall-clock time and practical tuning cost. |
| `n_envs` | More parallel rollouts, more diverse data per update, more RAM/compute use. | Cheaper and lighter, but data is noisier and less diverse per update. |
| `total_timesteps` | More opportunity to improve the policy, but longer runs. | Faster experiments, but higher risk of stopping before convergence. |
| `save_every_steps` | Less frequent saving, lower I/O overhead, fewer restore points. | More frequent saving, more disk I/O, finer-grained checkpoints. |
| `save_dir` | No learning effect; only changes where artifacts are written. | No learning effect; only changes where artifacts are written. |
| `init_flow_path` | Using a strong prior flow warm-starts ADR and biases search toward previous contexts. | `None` starts from a fresh bounded flow and explores from scratch. |

### `ppo`

| Key | Higher / enabled | Lower / disabled |
| --- | --- | --- |
| `learning_rate_start` | Faster early learning, but more instability. | Slower early learning, but usually more stable. |
| `learning_rate_end` | More aggressive late updates, which can speed progress or prevent settling. | More conservative late updates, which can help convergence but slow final improvement. |
| `n_steps` | Longer rollouts per update, better long-horizon signal, fewer but heavier PPO updates. | More frequent PPO updates, but noisier rollout estimates. |
| `batch_size` | Smoother gradients and fewer optimizer steps, but heavier memory use. | Noisier gradients and more updates per epoch, sometimes better exploration but less stable. |
| `n_epochs` | Reuses data more, which can improve fit but also overfit old rollouts. | Uses data less, faster updates, but can underfit each rollout batch. |
| `verbose` | More logs and easier debugging. | Less console noise. |
| `tensorboard_log` | If set and TensorBoard is installed, more training visibility. | No TensorBoard traces. |

### `vecnorm`

| Key | Higher / enabled | Lower / disabled |
| --- | --- | --- |
| `norm_obs` | `True`: observation scale is stabilized, which usually helps PPO. | `False`: raw observations are used; simpler but often harder to optimize. |
| `norm_reward` | `True`: reward scale is stabilized, which often helps PPO updates. | `False`: raw reward scale is preserved; simpler but can be noisier. |
| `clip_obs` | Less clipping of extreme normalized observations, more information but more sensitivity to outliers. | More clipping, more robustness, but possible information loss. |

### `adr` (generic)

| Key | Higher / enabled | Lower / disabled |
| --- | --- | --- |
| `transforms` | More expressive flow, better fit to complex context distributions, but slower/harder to train. | Simpler flow, faster and more stable, but less expressive. |
| `bins` | More flexible spline transforms, but more parameters and heavier optimization. | Simpler transforms, faster fitting, but less local flexibility. |
| `hidden` | Larger conditioner network, more expressive flow, but slower and easier to overfit. | Smaller conditioner, lighter and simpler, but less expressive. |
| `iters` | Fits the flow more strongly after each ADR update, but increases compute and can overfit the candidate set. | Faster ADR updates, but the flow may underfit the weighted candidates. |
| `lr` | Faster flow adaptation, but higher risk of unstable ADR updates. | Slower, smoother flow updates. |
| `n_sample` | Evaluates more candidate contexts, improving coverage, but increasing ADR cost. | Cheaper ADR updates, but weaker coverage of the context space. |
| `refine_steps` | More adversarial candidate refinement, usually producing harder contexts, but at higher cost. | Less refinement; contexts stay closer to the initial uniform draw. |
| `refine_lr` | Larger refinement jumps, which can find harder contexts faster but be unstable. | Smaller refinement steps, which are safer but less aggressive. |
| `ret_coef` | ADR ranking depends more on policy return. | ADR ranking depends less on policy return. |
| `bonus_coef` | ADR ranking depends more on baseline shaping terms. | ADR ranking depends less on baseline shaping terms. |
| <span style="color: red;">`ess_bounds`</span> | <span style="color: red;">Broader target ESS, so ADR uses more candidates.</span> | <span style="color: red;">Sharper target ESS, so ADR concentrates more strongly.</span> |
| `surprise_coef` | Stronger novelty pressure toward low-density contexts and exploration. | Less novelty pressure; ADR stays closer to already visited high-value regions. |
| <span style="color: red;">`kl_beta`</span> | <span style="color: red;">More conservative flow updates; the distribution moves less each ADR step.</span> | <span style="color: red;">More aggressive flow drift toward current weighted candidates.</span> |
| `kl_M` | More accurate KL/entropy estimates, but more compute. | Cheaper estimates, but noisier regularization diagnostics. |
| `update_every_episodes` | ADR updates less often, which is cheaper and more stable but slower to adapt. | ADR updates more often, adapting faster but changing the training distribution more aggressively. |

### RC5-Specific Tuning

### `env` (RC5)

| Key | Higher / enabled | Lower / disabled |
| --- | --- | --- |
| `step_period` | Coarser control, fewer decisions, cheaper training, but less reactive control. | Finer control, more reactive policy, but longer sequences and more compute. |
| `future_steps` | Longer forecast horizon, more anticipative behavior, larger observation/model burden. | Shorter lookahead, simpler observation, but less anticipation. |
| `base_setpoint` | Warmer default operating point; usually favors comfort over savings. | Colder default operating point; usually favors savings over comfort. |
| `max_dev` | Larger action freedom and exploration, but more aggressive setpoints. | Tighter action range, safer behavior, but less corrective power. |
| `max_episode_length` | Longer horizons and richer credit assignment, but slower episodes. | Shorter episodes, faster iteration, but less long-horizon signal. |
| `tz_min` | Raises the minimum setpoint, protecting comfort but limiting energy-saving actions. | Allows colder actions, which can save more energy but increases comfort risk. |
| `tz_max` | Raises the maximum setpoint, giving more recovery authority but allowing hotter actions. | Tighter upper cap, safer range, but less ability to recover quickly. |
| `w_energy` | Makes the policy care more about energy cost, often at the expense of comfort. | Makes energy less important relative to the other terms. |
| `w_comfort` | Makes the policy protect comfort more strongly, often with higher energy use. | Makes comfort violations less costly, which can encourage cheaper operation. |
| `comfort_huber_k` | Softer comfort penalty around violations; large errors are smoothed more. | Sharper penalty; at `0` it becomes the linear penalty used in the code. |
| `w_sat` | Penalizes saturation more, encouraging smoother and less extreme control. | Allows more aggressive saturated control when it helps return. |

### `policy` (RC5)

| Key | Higher / enabled | Lower / disabled |
| --- | --- | --- |
| `critic_use_ctx` | `True`: critic sees context and can value trajectories more accurately. | `False`: simpler critic, but less information for value estimation. |
| `policy_hidden` | Larger actor capacity, potentially better policies, but slower and easier to overfit. | Smaller actor, faster and simpler, but less expressive. |
| `value_hidden` | Larger critic capacity, potentially better value estimates, but slower and easier to overfit. | Smaller critic, faster and simpler, but less expressive. |

### `adr` (RC5 additions)

| Key | Higher / enabled | Lower / disabled |
| --- | --- | --- |
| `baseline_cs_coef` | Gives more weight to baseline comfort/saturation/COP shaping inside ADR. | ADR ranking depends more on policy return and less on baseline shaping. |
| `baseline_cop_coef` | Penalizes implausible COP behavior more strongly inside ADR. | Weakens or removes the COP-based shaping term. |
| `cop_bounds` | Wider bounds make the COP penalty activate less often. | Tighter bounds make the COP penalty activate more often. |
| `php_min_w` | Ignores more low-power COP samples, reducing noise but discarding more information. | Includes more low-power samples, which uses more information but can make COP estimates noisier. |
| `cop_beta` | Makes the COP penalty closer to a hard worst-case violation over time. | Makes the COP penalty smoother and more averaged over time. |

## Outputs

Each training run saves in `save_dir`:

- `model.zip`: PPO policy;
- `vecnormalize.pkl`: observation and reward normalization statistics;
- `adr_flow.pt`: learned flow over contexts;
- `<timesteps>/`: periodic snapshots with the same three files.

## Minimal usage

For a complete single-GPU run with explicit hyperparameters:

```bash
python -m NOMAD_RC5.training configs/rc5_reasonable.json
```

For a MacBook smoke test:

```bash
python -m NOMAD_RC5.training configs/rc5_local_smoke_mps.json
tensorboard --logdir NOMAD_RC5/tensorboard_logs --port 16006
```

Each ADR update logs `adr/params_std_pct_mean`, the mean relative context standard deviation of the current flow, in percent.

The optimized RC5 training path logs TensorBoard and saves checkpoints, but it does not currently generate episode plots. Plotting helpers live in the legacy `NOMAD_RC5/test` environment.

If you want to start from a fresh flow instead of the bundled legacy checkpoint, set `init_flow_path` to `None`.

```python
from NOMAD_RC5.training import run_training

run_training(
    {
        "save_dir": "NOMAD_RC5/runs/my_run",
        "device": "cuda",      # use "mps" or "cpu" if needed
        "adr_device": "cuda",
        "n_envs": 128,
        "total_timesteps": 1_000_000,
        "init_flow_path": None,
        "ppo": {"n_steps": 256, "batch_size": 1024},
        "env": {"future_steps": 24, "max_episode_length": 24 * 21},
        "adr": {"update_every_episodes": 100, "n_sample": 1000, "ret_coef": 2.0, "bonus_coef": 1.0},
    }
)
```

or:

```bash
python -m NOMAD_RC5.training config.json
```

To resume from a saved training folder, set `resume_dir`. The trainer will load `model.zip`, `vecnormalize.pkl`, and `adr_flow.pt` from that folder, and if numbered checkpoint subfolders exist it will pick the latest one automatically.

```python
run_training(
    {
        "resume_dir": "NOMAD_RC5/runs/my_run",
        "save_dir": "NOMAD_RC5/runs/my_run_resume",
        "total_timesteps": 500_000,
    }
)
```

or:

```bash
python -m NOMAD_RC5.training config.json
```

where `config.json` can contain:

```json
{
  "resume_dir": "NOMAD_RC5/runs/my_run",
  "save_dir": "NOMAD_RC5/runs/my_run_resume",
  "total_timesteps": 500000
}
```

## What is generic, and what is not

Generic pieces:

- recurrent RL policy trained on a parameterized environment;
- bounded context distribution learned with a normalizing flow;
- ADR update based on candidate evaluation, optional adversarial refinement, and KL-regularized refitting.

Current repository-specific pieces:

- RC5 thermal dynamics and dataset format;
- one-dimensional temperature-setpoint action;
- context semantics (`TH`, `PAC`, `PID`) and their bounds;
- baseline definition and COP penalty.

To port NOMAD to another simulator, the main replacements are the context definition, the environment wrapper, and the differentiable batch evaluator used inside `ADRFlows`.
