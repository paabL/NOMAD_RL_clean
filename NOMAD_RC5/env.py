from __future__ import annotations

from pathlib import Path

import equinox as eqx
import gymnasium as gym
from gymnasium import spaces
import jax.numpy as jnp
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from NOMAD.simax import Controller_PID
from .sim import (
    BASE_SETPOINT,
    CTX_NAMES,
    DT,
    FUTURE_STEPS,
    LOWER_K,
    PAC,
    PAC_KEYS,
    PID,
    PID_KEYS,
    Q_INTERNAL_CON_W,
    Q_INTERNAL_RAD_W,
    TH,
    TH_KEYS,
    TZ_MAX_K,
    TZ_MIN_K,
    UPPER_K,
    build_rc5_simulation,
    load_rc5_data,
    nominal_pid,
    nominal_theta,
    pack_context,
    rc5_steady_state_tz_fixed,
    split_context_torch,
    unpack_context,
)

ROOT = Path(__file__).resolve().parent
BAD_SCORE = -1e6


def interval_reward_and_terms(
    *,
    t_step_s,
    tz_seq_k,
    lower_seq_k,
    upper_seq_k,
    occ_seq,
    php_w_seq,
    price_seq,
    delta_sat_seq,
    comfort_huber_k,
    w_energy=1.0,
    w_comfort=5.0,
    w_sat=0.2,
):
    t_s = np.asarray(t_step_s, dtype=np.float64)
    tz_k = np.asarray(tz_seq_k, dtype=np.float64)
    lower_k = np.asarray(lower_seq_k, dtype=np.float64)
    upper_k = np.asarray(upper_seq_k, dtype=np.float64)
    occ = np.asarray(occ_seq, dtype=np.float64)
    php_w = np.asarray(php_w_seq, dtype=np.float64)
    price = np.asarray(price_seq, dtype=np.float64)
    delta_sat = np.asarray(delta_sat_seq, dtype=np.float64)

    v = np.maximum(lower_k - tz_k, 0.0) + np.maximum(tz_k - upper_k, 0.0)
    hub = float(comfort_huber_k)
    if hub > 0:
        v = np.where(v <= hub, 0.5 * v * v / hub, v - 0.5 * hub)

    comfort_kh = np.trapezoid(v * occ, x=t_s) / 3600.0
    sat_uh = np.trapezoid(np.abs(delta_sat), x=t_s) / 3600.0
    energy_eur = np.trapezoid(price * np.maximum(php_w, 0.0) / 1000.0, x=t_s / 3600.0)

    comfort_term = -float(w_comfort) * comfort_kh
    energy_term = -float(w_energy) * energy_eur
    sat_term = -float(w_sat) * sat_uh
    return float(comfort_term + energy_term + sat_term), (float(comfort_term), float(energy_term), float(sat_term))


def cop_penalty_torch(cops, *, cop_bounds=(1.0, 5.0), beta=1.0):
    cop_lo, cop_hi = map(float, cop_bounds)
    x = float(beta) * cops
    x0 = x.max(dim=0).values
    eff = (x0 + torch.log(torch.mean(torch.exp(x - x0), dim=0) + 1e-12)) / float(beta)
    sp = torch.nn.functional.softplus
    return 100.0 * sp(cop_lo - eff) ** 2 + sp(eff - cop_hi) ** 2


class _PIDSeed(eqx.Module):
    pid: Controller_PID
    state0: object

    def init_state(self):
        return self.state0

    def compute_control(self, **kwargs):
        return self.pid.compute_control(**kwargs)


class NomadEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        data=None,
        simulation=None,
        sampling_dist=None,
        step_period=3600.0,
        future_steps=FUTURE_STEPS,
        warmup_steps=3 * 24,
        base_setpoint=BASE_SETPOINT,
        max_episode_length=24 * 7,
        tz_min=TZ_MIN_K,
        tz_max=TZ_MAX_K,
        w_energy=1.0,
        w_comfort=5.0,
        comfort_huber_k=0.0,
        w_sat=0.2,
        pid0=None,
        flat_obs=False,
        include_ctx=False,
        excluding_periods=None,
        auto_plot=False,
        plot_every_episodes=0,
        rollout_dir: str | Path | None = None,
        env_id: int = 0,
    ):
        super().__init__()
        self.data = load_rc5_data() if data is None else data
        self.simulation = build_rc5_simulation(self.data, base_setpoint=base_setpoint) if simulation is None else simulation
        self.sampling_dist = sampling_dist
        self.include_ctx = bool(include_ctx)
        self.step_period = float(step_period)
        self.future_steps = int(future_steps)
        self.warmup_steps = int(warmup_steps)
        self.base_setpoint = float(base_setpoint)
        self.max_episode_length = int(max_episode_length)
        self.tz_min, self.tz_max = float(tz_min), float(tz_max)
        self.w_energy, self.w_comfort = float(w_energy), float(w_comfort)
        self.comfort_huber_k, self.w_sat = float(comfort_huber_k), float(w_sat)
        self.pid_default = nominal_pid() if pid0 is None else tuple(map(float, pid0))
        self.flat_obs = bool(flat_obs)
        self.auto_plot = bool(auto_plot)
        self.plot_every_episodes = int(plot_every_episodes)
        self.rollout_dir = str(rollout_dir) if rollout_dir is not None else None
        self.env_id = int(env_id)

        self._time_np = self.data.time_np
        self._dist_matrix = self.data.dist_matrix
        self.n = int(self._time_np.shape[0])
        self.dataset_dt = float(self._time_np[1] - self._time_np[0])
        self.step_n = max(1, int(round(self.step_period / self.dataset_dt)))
        self.warmup_steps_dataset = self.warmup_steps * self.step_n
        self.idx_min = self.warmup_steps_dataset
        self.idx_max = self.n - 1 - (self.future_steps + 1) * self.step_n
        self.idx_max_start = self.idx_max - self.max_episode_length * self.step_n

        self.theta = nominal_theta()
        self.pid_kp, self.pid_ki, self.pid_kd = self.pid_default
        self.ctx = None
        self.idx = int(self.idx_min)
        self.ep_steps = 0
        self.total_timesteps = 0
        self._episode_count = 0
        self._pid_cache = {}
        self._ctx_cache = None
        self._ctx_cache_idx = 0
        self._ctx_cache_n = 64
        self._excluded_mask = np.zeros((self.n,), dtype=bool)
        if excluding_periods:
            for t0, t1 in excluding_periods:
                i0 = int(np.searchsorted(self._time_np, float(t0), side="left"))
                i1 = int(np.searchsorted(self._time_np, float(t1), side="left"))
                self._excluded_mask[max(i0, 0) : max(i1, 0)] = True

        self.x = np.zeros((5,), dtype=np.float32)
        self.pid_state = None
        self.tz_last = np.float32(self.base_setpoint)
        self.sp_last = np.float32(self.base_setpoint)
        self.php_last = np.float32(0.0)

        self.reward_ref = np.zeros((0,), dtype=np.float32)
        self.reward_ref_terms = np.zeros((0, 3), dtype=np.float32)
        self.reward_ref_N = np.zeros((0,), dtype=np.float32)

        self.warmup = {}
        self._last_episode = None
        self._last_saved = None
        self._rollout_stats = {"n": 0, "mean": {}, "m2": {}}

        self.now_phys_dim = 4
        self.n_phys_features_past = 5
        self.n_time_features = int(self._dist_matrix.shape[1] - self.n_phys_features_past)
        self.forecast_phys_dim = 6
        self.ctx_dim = len(CTX_NAMES)
        self.now_dim = self.now_phys_dim + self.n_time_features + 2
        self.forecast_feat_dim = self.forecast_phys_dim + self.n_time_features
        if self.flat_obs:
            n = int(self.now_dim + self.future_steps * self.forecast_feat_dim + (self.ctx_dim if self.include_ctx else 0))
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)
        else:
            obs_spaces = {
                "now": spaces.Box(low=-np.inf, high=np.inf, shape=(self.now_dim,), dtype=np.float32),
                "forecast": spaces.Box(low=-np.inf, high=np.inf, shape=(self.future_steps, self.forecast_feat_dim), dtype=np.float32),
            }
            if self.include_ctx:
                obs_spaces["ctx"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.ctx_dim,), dtype=np.float32)
            self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(low=self.tz_min, high=self.tz_max, shape=(1,), dtype=np.float32)
        self._reset_episode_logs()

    def _reset_episode_logs(self):
        self.ep_time = []
        self.ep_setpoint = []
        self.ep_reward = []
        self.ep_reward_norm = []
        self.ep_terms = []
        self.ep_idx_30s = []
        self.ep_tz_30s = []
        self.ep_u_30s = []
        self.ep_qc_30s = []
        self.ep_qe_30s = []
        self.ep_php_30s = []

    def _record_ctx_stats(self):
        if self.ctx is None:
            return
        stats = self._rollout_stats
        values = dict(zip(CTX_NAMES, np.asarray(self.ctx, dtype=np.float64)))
        stats["n"] += 1
        n = stats["n"]
        for key, value in values.items():
            mu = float(stats["mean"].get(key, 0.0))
            m2 = float(stats["m2"].get(key, 0.0))
            delta = value - mu
            mu += delta / n
            m2 += delta * (value - mu)
            stats["mean"][key] = mu
            stats["m2"][key] = m2

    def get_rollout_std_pct_mean(self):
        stats = self._rollout_stats
        if int(stats.get("n", 0)) < 2:
            return 0.0
        vals = []
        for key in CTX_NAMES:
            mu = float(stats["mean"][key])
            std = np.sqrt(float(stats["m2"][key]) / (stats["n"] - 1))
            vals.append(100.0 * std / max(abs(mu), 1e-12))
        return float(np.mean(vals))

    def set_sampling_dist(self, dist):
        self.sampling_dist = dist
        self._ctx_cache = None
        self._ctx_cache_idx = 0

    def set_rollout_dir(self, rollout_dir: str | Path | None):
        self.rollout_dir = str(rollout_dir) if rollout_dir is not None else None

    def _to_numpy_ctx(self, ctx):
        if hasattr(ctx, "detach"):
            ctx = ctx.detach().cpu().numpy()
        return np.asarray(ctx, dtype=np.float32).reshape(-1)

    def _apply_ctx(self, ctx):
        self.theta, (self.pid_kp, self.pid_ki, self.pid_kd) = unpack_context(ctx)
        self.ctx = self._to_numpy_ctx(ctx)
        self._record_ctx_stats()

    def _make_pid(self, setpoint, horizon_len):
        sp = jnp.full((horizon_len,), float(setpoint), dtype=jnp.float32)
        gains = tuple(jnp.asarray(v, dtype=jnp.float32) for v in (self.pid_kp, self.pid_ki, self.pid_kd))
        pid = self._pid_cache.get(horizon_len)
        if pid is None:
            pid = Controller_PID(k_p=gains[0], k_i=gains[1], k_d=gains[2], n=1, verbose=False, SetPoints=sp)
        pid = eqx.tree_at(lambda c: (c.SetPoints, c.k_p, c.k_i, c.k_d), pid, (sp, *gains))
        self._pid_cache[horizon_len] = pid
        return pid

    def _aggregate_step_features(self, start_idx):
        return self._dist_matrix[start_idx : start_idx + self.step_n].mean(axis=0)

    def _build_observation(self):
        full = self._aggregate_step_features(self.idx).astype(np.float32)
        now = np.concatenate([full[:2], full[2:3], full[4:5], full[5:], np.asarray([self.tz_last, self.php_last], dtype=np.float32)], axis=0)
        forecast = np.zeros((self.future_steps, self.forecast_feat_dim), dtype=np.float32)
        for k in range(1, self.future_steps + 1):
            fk = self._aggregate_step_features(self.idx + k * self.step_n)
            phys = np.asarray([fk[0], fk[1], fk[3], fk[4], LOWER_K, UPPER_K], dtype=np.float32)
            forecast[k - 1] = np.concatenate([phys, fk[self.n_phys_features_past :]], axis=0).astype(np.float32)
        if self.flat_obs:
            x = [now.reshape(-1), forecast.reshape(-1)]
            if self.include_ctx:
                x.append(self.ctx.reshape(-1))
            return np.concatenate(x, axis=0).astype(np.float32)
        obs = {"now": now, "forecast": forecast}
        if self.include_ctx:
            obs["ctx"] = self.ctx.astype(np.float32)
        return obs

    def _init_state(self, idx):
        row = self._dist_matrix[int(idx)]
        q_con = float(row[2]) * Q_INTERNAL_CON_W
        q_rad = float(row[2]) * Q_INTERNAL_RAD_W
        t0, _ = rc5_steady_state_tz_fixed(float(row[0]), float(row[1]), q_con, q_rad, self.base_setpoint, self.theta)
        return np.asarray(t0, dtype=np.float32)

    def _run_warmup(self, start_idx, end_idx):
        x_init = self._init_state(start_idx)
        time_slice = self.data.time[start_idx : end_idx + 1]
        _, y_seq, states, controls, self.pid_state = self.simulation.run(
            self.theta,
            time_grid=time_slice,
            controller=self._make_pid(self.base_setpoint, len(time_slice)),
            x0=x_init,
            return_ctrl_state=True,
        )
        if not (np.isfinite(y_seq).all() and np.isfinite(states).all()):
            self.pid_state = self._make_pid(self.base_setpoint, len(time_slice)).init_state()
            self.x = x_init
            self.tz_last = np.float32(self.base_setpoint)
            self.php_last = np.float32(0.0)
            return

        y = np.asarray(y_seq, dtype=np.float32)
        tz, qc, qe = y[:, 0], y[:, 1], y[:, 2]
        php = qc - np.abs(qe)
        k0 = max(0, (self.warmup_steps - 1) * self.step_n)
        k1 = min(k0 + self.step_n + 1, int(php.shape[0]))
        s0 = min(k0 + 1, k1)
        self.tz_last = np.float32(tz[s0:k1].mean()) if s0 < k1 else np.float32(tz[k1 - 1])
        self.php_last = np.float32(php[s0:k1].mean()) if s0 < k1 else np.float32(php[k1 - 1])
        self.x = np.asarray(states[-1], dtype=np.float32)
        u = np.asarray(controls.get("oveHeaPumY_u", np.zeros((len(time_slice),), dtype=np.float32)), dtype=np.float32)
        self.warmup = {
            "idx_30s": np.arange(int(start_idx), int(end_idx) + 1, dtype=np.int64),
            "time": np.asarray(time_slice, dtype=np.float32) / 86400.0,
            "tz": (tz - 273.15).astype(np.float32),
            "u": u[: len(time_slice)],
            "qc": qc.astype(np.float32),
            "qe": qe.astype(np.float32),
            "php": php.astype(np.float32),
            "idx": (int(start_idx), int(end_idx)),
        }

    def _sample_ctx(self):
        if self.sampling_dist is None:
            return None
        if self._ctx_cache is None or self._ctx_cache_idx >= len(self._ctx_cache):
            sampler = getattr(self.sampling_dist, "sample", None) or getattr(self.sampling_dist, "rsample")
            self._ctx_cache = np.asarray(sampler((self._ctx_cache_n,)).detach().cpu().numpy(), dtype=np.float32)
            self._ctx_cache_idx = 0
        ctx = self._ctx_cache[self._ctx_cache_idx]
        self._ctx_cache_idx += 1
        return self._to_numpy_ctx(ctx)

    def reset(self, *, seed=None, options=None, ctx=None):
        super().reset(seed=seed)
        self._episode_count += 1
        self._reset_episode_logs()

        ctx = self._sample_ctx() if ctx is None else ctx
        if ctx is not None:
            self._apply_ctx(ctx)
        else:
            self.theta = nominal_theta()
            self.pid_kp, self.pid_ki, self.pid_kd = self.pid_default
            self.ctx = pack_context(self.theta, self.pid_default)
            self._record_ctx_stats()

        rng = np.random.default_rng(seed)
        opts = options or {}
        forced = "start_time_s" in opts or "start_idx" in opts
        for _ in range(10_000):
            raw = int(np.searchsorted(self._time_np, float(opts["start_time_s"]), side="left")) if "start_time_s" in opts else int(opts.get("start_idx", rng.integers(self.idx_min, self.idx_max_start + 1)))
            idx = raw - ((raw - self.idx_min) % self.step_n)
            if idx < self.idx_min:
                idx += self.step_n
            idx = min(int(idx), int(self.idx_max_start))
            if self._excluded_mask.any():
                w0 = idx - self.warmup_steps_dataset
                w1 = idx + (self.max_episode_length + self.future_steps + 1) * self.step_n
                if w0 < 0 or w1 > self.n or self._excluded_mask[w0:w1].any():
                    if forced:
                        raise ValueError("reset: start index in excluded window")
                    continue
            self.idx = idx
            break
        else:
            raise ValueError("reset: no valid start index")

        self.pid_state = None
        self._run_warmup(self.idx - self.warmup_steps_dataset, self.idx)
        self.sp_last = np.float32(self.base_setpoint)
        self.ep_steps = 0

        t_ep = self.data.time[self.idx : self.idx + self.max_episode_length * self.step_n + 1]
        _, y_ref, _, u_ref = self.simulation.run(
            self.theta,
            time_grid=t_ep,
            controller=_PIDSeed(self._make_pid(self.base_setpoint, len(t_ep)), self.pid_state),
            x0=self.x,
        )
        y_ref = np.asarray(y_ref, dtype=np.float32)
        php = y_ref[:, 1] - np.abs(y_ref[:, 2])
        delta = np.asarray(u_ref.get("delta_sat", np.zeros((len(t_ep),), dtype=np.float32)), dtype=np.float32)
        ref_rewards, ref_terms, ref_rewards_n = [], [], []
        for k in range(0, self.max_episode_length * self.step_n, self.step_n):
            occ = self._dist_matrix[self.idx + k : self.idx + k + self.step_n, 2]
            price = self._dist_matrix[self.idx + k : self.idx + k + self.step_n, 4]
            kw = dict(
                t_step_s=t_ep[k : k + self.step_n],
                tz_seq_k=y_ref[k + 1 : k + self.step_n + 1, 0],
                lower_seq_k=np.full((self.step_n,), LOWER_K, dtype=np.float32),
                upper_seq_k=np.full((self.step_n,), UPPER_K, dtype=np.float32),
                php_w_seq=php[k : k + self.step_n],
                delta_sat_seq=delta[k : k + self.step_n],
                w_energy=self.w_energy,
                w_comfort=self.w_comfort,
                comfort_huber_k=self.comfort_huber_k,
                w_sat=self.w_sat,
            )
            reward, terms = interval_reward_and_terms(**kw, occ_seq=occ, price_seq=price)
            reward_n, _ = interval_reward_and_terms(**kw, occ_seq=np.ones_like(occ), price_seq=np.ones_like(price))
            ref_rewards.append(reward)
            ref_terms.append(terms)
            ref_rewards_n.append(reward_n)
        self.reward_ref = np.asarray(ref_rewards, dtype=np.float32)
        self.reward_ref_terms = np.asarray(ref_terms, dtype=np.float32)
        self.reward_ref_N = np.asarray(ref_rewards_n, dtype=np.float32)
        return self._build_observation(), {"context": self.ctx.copy()}

    def step(self, action):
        idx_start = int(self.idx)
        a0 = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        if not np.isfinite(a0):
            a0 = self.base_setpoint
        self.sp_last = np.float32(np.clip(a0, self.tz_min, self.tz_max))

        time_slice = self.data.time[self.idx : self.idx + self.step_n + 1]
        _, y_seq, states, controls, self.pid_state = self.simulation.run(
            self.theta,
            time_grid=time_slice,
            controller=_PIDSeed(self._make_pid(float(self.sp_last), len(time_slice)), self.pid_state),
            x0=self.x,
            return_ctrl_state=True,
        )
        if not (np.isfinite(y_seq).all() and np.isfinite(states).all()):
            obs = self._build_observation()
            return obs, BAD_SCORE, False, True, {"idx": int(self.idx), "setpoint": float(self.sp_last), "context": self.ctx.copy(), "nan": 1}

        y = np.asarray(y_seq, dtype=np.float32)
        tz, qc, qe = y[:, 0], y[:, 1], y[:, 2]
        php = qc - np.abs(qe)
        u = np.asarray(controls.get("oveHeaPumY_u", np.zeros((len(time_slice),), dtype=np.float32)), dtype=np.float32)
        delta_sat = np.asarray(controls.get("delta_sat", np.zeros((len(time_slice),), dtype=np.float32)), dtype=np.float32)
        self.tz_last = np.float32(tz[1:].mean()) if tz.shape[0] > 1 else np.float32(tz[0])
        self.php_last = np.float32(php[1:].mean()) if php.shape[0] > 1 else np.float32(php[0])
        self.x = np.asarray(states[-1], dtype=np.float32)

        n_inner = min(len(time_slice) - 1, len(u))
        self.ep_idx_30s.append(idx_start)
        self.ep_tz_30s.append(float(tz[0]))
        self.ep_u_30s.append(float(u[0] if u.size else 0.0))
        self.ep_qc_30s.append(float(qc[0]))
        self.ep_qe_30s.append(float(qe[0]))
        self.ep_php_30s.append(float(php[0]))
        for k in range(1, n_inner + 1):
            self.ep_idx_30s.append(idx_start + k)
            self.ep_tz_30s.append(float(tz[k]))
            self.ep_u_30s.append(float(u[k - 1]))
            self.ep_qc_30s.append(float(qc[k]))
            self.ep_qe_30s.append(float(qe[k]))
            self.ep_php_30s.append(float(php[k]))

        self.idx += self.step_n
        self.ep_steps += 1
        self.total_timesteps += 1
        terminated = False
        truncated = self.idx > self.idx_max or self.ep_steps >= self.max_episode_length
        if self.idx > self.idx_max:
            self.idx = int(self.idx_max)

        rows_step = self._dist_matrix[idx_start : idx_start + self.step_n + 1]
        reward, terms = interval_reward_and_terms(
            t_step_s=np.asarray(time_slice, dtype=np.float32),
            tz_seq_k=tz,
            lower_seq_k=np.full((rows_step.shape[0],), LOWER_K, dtype=np.float32),
            upper_seq_k=np.full((rows_step.shape[0],), UPPER_K, dtype=np.float32),
            occ_seq=rows_step[:, 2],
            php_w_seq=php,
            price_seq=rows_step[:, 4],
            delta_sat_seq=delta_sat,
            w_energy=self.w_energy,
            w_comfort=self.w_comfort,
            comfort_huber_k=self.comfort_huber_k,
            w_sat=self.w_sat,
        )
        ref = float(self.reward_ref[self.ep_steps - 1])
        ref_n = float(self.reward_ref_N[self.ep_steps - 1])
        reward_norm = (reward - ref) / max(-ref_n, 1e-3)
        if not np.isfinite(reward_norm):
            reward_norm = BAD_SCORE

        self.ep_time.append(float(self._time_np[idx_start]))
        self.ep_setpoint.append(float(self.sp_last))
        self.ep_reward.append(float(reward))
        self.ep_reward_norm.append(float(reward_norm))
        self.ep_terms.append([float(terms[0]), float(terms[1]), float(terms[2])])

        obs = self._build_observation()
        info = {
            "idx": int(self.idx),
            "Tz": float(self.tz_last),
            "setpoint": float(self.sp_last),
            "context": self.ctx.copy(),
            "reward_raw": float(reward),
            "reward_ref": float(ref),
            "reward_ref_N": float(ref_n),
        }
        if terminated or truncated:
            self._last_episode = self._episode_arrays()
        if self.auto_plot and truncated and self.plot_every_episodes > 0 and self._episode_count % self.plot_every_episodes == 0:
            self.plot_last_episode()
        return obs, float(reward_norm), terminated, truncated, info

    def _episode_arrays(self):
        idx_30s = np.asarray(self.ep_idx_30s, dtype=np.int64)
        warm = self.warmup or {}
        warm_idx = np.asarray(warm.get("idx_30s", []), dtype=np.int64)
        return {
            "context": np.asarray(self.ctx, dtype=np.float32),
            "idx_30s": idx_30s,
            "time_30s": self._time_np[idx_30s] if idx_30s.size else np.zeros((0,), dtype=np.float32),
            "tz_30s": np.asarray(self.ep_tz_30s, dtype=np.float32),
            "u_30s": np.asarray(self.ep_u_30s, dtype=np.float32),
            "qc_30s": np.asarray(self.ep_qc_30s, dtype=np.float32),
            "qe_30s": np.asarray(self.ep_qe_30s, dtype=np.float32),
            "php_30s": np.asarray(self.ep_php_30s, dtype=np.float32),
            "time_rl": np.asarray(self.ep_time, dtype=np.float32),
            "setpoint_rl": np.asarray(self.ep_setpoint, dtype=np.float32),
            "reward_raw_rl": np.asarray(self.ep_reward, dtype=np.float32),
            "reward_norm_rl": np.asarray(self.ep_reward_norm, dtype=np.float32),
            "terms_rl": np.asarray(self.ep_terms, dtype=np.float32),
            "reward_ref_rl": self.reward_ref[: len(self.ep_reward)].astype(np.float32),
            "reward_ref_terms_rl": self.reward_ref_terms[: len(self.ep_reward)].astype(np.float32),
            "reward_ref_N_rl": self.reward_ref_N[: len(self.ep_reward)].astype(np.float32),
            "warm_idx_30s": warm_idx,
            "warm_time_30s": self._time_np[warm_idx] if warm_idx.size else np.zeros((0,), dtype=np.float32),
            "warm_tz_30s": np.asarray(warm.get("tz", []), dtype=np.float32),
            "warm_u_30s": np.asarray(warm.get("u", []), dtype=np.float32),
            "warm_qc_30s": np.asarray(warm.get("qc", []), dtype=np.float32),
            "warm_qe_30s": np.asarray(warm.get("qe", []), dtype=np.float32),
            "warm_php_30s": np.asarray(warm.get("php", []), dtype=np.float32),
        }

    def _episode_payload(self):
        return self._last_episode or self._episode_arrays()

    def save_last_episode(self, path: str | Path | None = None):
        arrays = self._episode_payload()
        if path is None:
            out_dir = Path(self.rollout_dir) if self.rollout_dir else ROOT / "runs"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"episode_{self._episode_count:06d}_env{self.env_id}.npz"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        self._last_saved = path
        return path

    def plot_last_episode(self, path: str | Path | None = None):
        arrays = self._episode_payload()
        if path is None:
            out_dir = Path(self.rollout_dir) if self.rollout_dir else ROOT / "runs"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"episode_{self._episode_count:06d}_env{self.env_id}.png"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        plt.ioff()

        idx_main = np.asarray(arrays["idx_30s"], dtype=np.int64)
        t_days_main = np.asarray(arrays["time_30s"], dtype=float) / 86400.0
        tz_c = np.asarray(arrays["tz_30s"], dtype=float) - 273.15
        u_arr = np.asarray(arrays["u_30s"], dtype=float)
        qc_arr = np.asarray(arrays["qc_30s"], dtype=float)
        qe_arr = np.asarray(arrays["qe_30s"], dtype=float)
        php_arr = np.asarray(arrays["php_30s"], dtype=float)
        t_days_rl = np.asarray(arrays["time_rl"], dtype=float) / 86400.0
        sp_rl = np.asarray(arrays["setpoint_rl"], dtype=float) - 273.15
        costs = -np.asarray(arrays["reward_raw_rl"], dtype=float)
        parts = -np.asarray(arrays["terms_rl"], dtype=float) if np.asarray(arrays["terms_rl"]).size else None
        pid_costs = -np.asarray(arrays["reward_ref_rl"], dtype=float)
        pid_parts = -np.asarray(arrays["reward_ref_terms_rl"], dtype=float) if np.asarray(arrays["reward_ref_terms_rl"]).size else None
        rew_norm = np.asarray(arrays["reward_norm_rl"], dtype=float)
        warm_idx = np.asarray(arrays["warm_idx_30s"], dtype=np.int64)
        warm_t = np.asarray(arrays["warm_time_30s"], dtype=float) / 86400.0
        warm_tz = np.asarray(arrays["warm_tz_30s"], dtype=float)
        warm_u = np.asarray(arrays["warm_u_30s"], dtype=float)
        warm_qc = np.asarray(arrays["warm_qc_30s"], dtype=float)
        warm_qe = np.asarray(arrays["warm_qe_30s"], dtype=float)
        warm_php = np.asarray(arrays["warm_php_30s"], dtype=float)
        has_warmup = warm_t.size > 0
        warm_span = (float(warm_t[0]), float(warm_t[-1])) if has_warmup else None

        fig = plt.figure(figsize=(12, 12), dpi=200, constrained_layout=True)
        axs = fig.subplots(9, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1, 1, 1, 1]})

        def plot_cost(ax, t, total, parts_arr, *, color, ls):
            label = f"cost (sum={float(total.sum()):.2f}EUR)"
            if parts_arr is None or parts_arr.ndim != 2 or parts_arr.shape[1] != 3:
                ax.plot(t, total, color=color, linewidth=1, linestyle=ls, label=label)
                ax.legend(loc="lower left", fontsize=7)
                return
            sums = parts_arr.sum(axis=0)
            safe = float(total.sum()) if abs(float(total.sum())) > 1e-6 else 1.0
            pct = 100.0 * sums / safe
            ax.plot(t, total, color=color, linewidth=1, linestyle=ls, label=label)
            ax.plot(t, parts_arr[:, 0], "r", linewidth=1, linestyle=ls, label=f"comfort ({pct[0]:.0f}%)")
            ax.plot(t, parts_arr[:, 1], "g", linewidth=1, linestyle=ls, label=f"energy ({pct[1]:.0f}%)")
            ax.plot(t, parts_arr[:, 2], "m", linewidth=1, linestyle=ls, label=f"sat ({pct[2]:.0f}%)")
            ax.legend(loc="lower left", fontsize=7)

        if warm_span:
            for ax in axs:
                ax.axvspan(warm_span[0], warm_span[1], color="khaki", alpha=0.15, zorder=0)
            axs[0].plot([warm_span[0]], [np.nan], color="khaki", alpha=0.3, linewidth=6, label="warmup")

        if idx_main.size:
            rows = self._dist_matrix[idx_main]
            ta = rows[:, 0].astype(float) - 273.15
            qsol = rows[:, 1].astype(float)
            occ = rows[:, 2].astype(float)
            occ_exp = rows[:, 3].astype(float)
            price = rows[:, 4].astype(float)
            lower_c = np.full_like(t_days_main, LOWER_K - 273.15, dtype=float)
            upper_c = np.full_like(t_days_main, UPPER_K - 273.15, dtype=float)
            warm_rows = self._dist_matrix[warm_idx] if warm_idx.size else None

            axs[0].plot(t_days_main, lower_c, "--", color="seagreen", linewidth=1, label="Comfort band")
            axs[0].plot(t_days_main, upper_c, "--", color="seagreen", linewidth=1)
            if warm_rows is not None:
                axs[0].plot(warm_t, np.full_like(warm_t, self.base_setpoint - 273.15), "-", color="gray", linewidth=1, alpha=0.8)
                axs[0].plot(warm_t, warm_tz, "-", color="darkorange", linewidth=1, alpha=0.8)
            axs[0].step(t_days_rl, sp_rl, where="post", color="gray", linewidth=1, label="Setpoint")
            axs[0].plot(t_days_main, tz_c, "-", color="darkorange", linewidth=1, label="Tz")
            axs[0].set_ylabel("Tz / setpoint\n(degC)")
            axs[0].set_ylim(15.0, 30.0)

            axs[1].plot(t_days_main, u_arr, "-", color="slateblue", linewidth=1)
            if warm_t.size:
                n = min(warm_t.size, warm_u.size)
                axs[1].plot(warm_t[:n], warm_u[:n], "-", color="slateblue", linewidth=1, alpha=0.7)
            axs[1].set_ylabel("Control\n(-)")

            axs[2].plot(t_days_main, php_arr, "-", color="black", linewidth=1, label="P_hp")
            if warm_t.size:
                n = min(warm_t.size, warm_php.size)
                axs[2].plot(warm_t[:n], warm_php[:n], "-", color="black", linewidth=1, alpha=0.7)
            axcop = axs[2].twinx()
            cop = np.divide(qc_arr, php_arr, out=np.zeros_like(qc_arr), where=php_arr > 0.0)
            axcop.plot(t_days_main, cop, "-", color="darkorange", linewidth=1)
            if warm_t.size:
                n = min(warm_t.size, warm_qc.size, warm_php.size)
                axcop.plot(
                    warm_t[:n],
                    np.divide(warm_qc[:n], warm_php[:n], out=np.zeros_like(warm_qc[:n]), where=warm_php[:n] > 0.0),
                    "-",
                    color="darkorange",
                    linewidth=1,
                    alpha=0.7,
                )
            axs[2].set_ylabel("P_hp (W)")
            axcop.set_ylabel("COP (-)")
            axs[2].legend(loc="upper right", fontsize=7)

            plot_cost(axs[3], t_days_rl, costs, parts, color="b", ls="-")
            axs[3].set_ylabel("Costs (EUR)")

            n_pid = min(costs.size, pid_costs.size)
            plot_cost(axs[4], t_days_rl[:n_pid], pid_costs[:n_pid], pid_parts[:n_pid] if pid_parts is not None else None, color="black", ls="--")
            axs[4].set_ylabel("PID cost\n(EUR)")

            rn_sum = float(rew_norm.sum()) if rew_norm.size else 0.0
            rn_mean = float(rew_norm.mean()) if rew_norm.size else 0.0
            axs[5].plot(t_days_rl, rew_norm, color="black", linewidth=1, label=f"sum={rn_sum:.2f}, mean={rn_mean:.2f}")
            axs[5].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
            axs[5].set_ylabel("Reward\n(norm)")
            axs[5].legend(loc="lower left", fontsize=7)

            axs[6].plot(t_days_main, ta, color="royalblue", linewidth=1, label="Ta")
            axq = axs[6].twinx()
            axq.plot(t_days_main, qsol, color="gold", linewidth=1, label="Qsol")
            if warm_rows is not None:
                n = min(warm_t.size, warm_rows.shape[0])
                axs[6].plot(warm_t[:n], warm_rows[:n, 0].astype(float) - 273.15, "-", color="royalblue", linewidth=1, alpha=0.7)
                axq.plot(warm_t[:n], warm_rows[:n, 1].astype(float), "-", color="gold", linewidth=1, alpha=0.7)
            axs[6].set_ylabel("Ta (degC)")
            axq.set_ylabel("Qsol (W)")
            axs[6].legend(loc="upper left", fontsize=7)
            axq.legend(loc="upper right", fontsize=7)

            axs[7].step(t_days_main, occ, where="post", color="black", linewidth=1)
            axs[7].plot(t_days_main, occ_exp, color="red", linewidth=0.8)
            if warm_rows is not None:
                n = min(warm_t.size, warm_rows.shape[0])
                axs[7].step(warm_t[:n], warm_rows[:n, 2].astype(float), where="post", color="black", linewidth=1, alpha=0.7)
                axs[7].plot(warm_t[:n], warm_rows[:n, 3].astype(float), color="red", linewidth=0.8, alpha=0.7)
            axs[7].set_ylabel("Occ\n(-)")

            if price.size:
                p_max = price.max()
                mask = np.isclose(price, p_max, rtol=1e-5, atol=1e-8)
                if mask.any():
                    idxs = np.where(mask)[0]
                    start = prev = int(idxs[0])
                    first = True
                    for k in idxs[1:]:
                        k = int(k)
                        if k != prev + 1:
                            axs[0].axvspan(t_days_main[start], t_days_main[prev], color="lightcoral", alpha=0.18, zorder=0, label="Max price" if first else None)
                            first = False
                            start = k
                        prev = k
                    axs[0].axvspan(t_days_main[start], t_days_main[prev], color="lightcoral", alpha=0.18, zorder=0, label="Max price" if first else None)

            if occ_exp.size:
                o_min = occ_exp.min()
                mask = np.isclose(occ_exp, o_min, rtol=1e-5, atol=1e-8)
                if mask.any():
                    idxs = np.where(mask)[0]
                    start = prev = int(idxs[0])
                    first = True
                    for k in idxs[1:]:
                        k = int(k)
                        if k != prev + 1:
                            axs[0].axvspan(
                                t_days_main[start],
                                t_days_main[prev],
                                facecolor="cornflowerblue",
                                edgecolor="cornflowerblue",
                                alpha=0.25,
                                hatch="//",
                                zorder=0,
                                label="Min occ." if first else None,
                            )
                            first = False
                            start = k
                        prev = k
                    axs[0].axvspan(
                        t_days_main[start],
                        t_days_main[prev],
                        facecolor="cornflowerblue",
                        edgecolor="cornflowerblue",
                        alpha=0.25,
                        hatch="//",
                        zorder=0,
                        label="Min occ." if first else None,
                    )

            axs[0].legend(fontsize=7)

            axs[8].plot(t_days_main, price, color="black", linewidth=1)
            if warm_rows is not None:
                n = min(warm_t.size, warm_rows.shape[0])
                axs[8].plot(warm_t[:n], warm_rows[:n, 4].astype(float), "-", color="black", linewidth=1, alpha=0.7)
            axs[8].set_ylabel("Price\n(EUR/kWh)")
            axs[8].set_xlabel("Time (days)")

            if t_days_main.size >= 2:
                energy_kwh = float(np.trapezoid(np.maximum(php_arr, 0.0) / 1000.0, x=t_days_main * 24.0))
            else:
                energy_kwh = 0.0
            fig.suptitle(f"timestep={self.total_timesteps} | consumption={energy_kwh:.1f} kWh")

        fig.savefig(path)
        plt.close(fig)
        return path

    def close(self):
        return None


NOMAD = NomadEnv


class ResidualActionWrapper(gym.ActionWrapper):
    def __init__(self, env, base_action: float, max_dev: float):
        super().__init__(env)
        self.base_action = float(base_action)
        self.max_dev = float(max_dev)
        self._low = env.action_space.low.astype(np.float32)
        self._high = env.action_space.high.astype(np.float32)
        self.action_space = spaces.Box(low=-self.max_dev, high=self.max_dev, shape=env.action_space.shape, dtype=np.float32)

    def action(self, delta):
        delta = np.asarray(delta, dtype=np.float32)
        return np.clip(self.base_action + np.clip(delta, -self.max_dev, self.max_dev), self._low, self._high)

    def reset(self, *, seed=None, options=None, **kwargs):
        return self.env.reset(seed=seed, options=options, **kwargs)


class NormalizeAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._low = env.action_space.low.astype(np.float32)
        self._high = env.action_space.high.astype(np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32)
        return self._low + (a + 1.0) * 0.5 * (self._high - self._low)

    def reset(self, *, seed=None, options=None, **kwargs):
        return self.env.reset(seed=seed, options=options, **kwargs)


class RC5TorchBatch:
    """Batched RC5 simulator used by GPU PPO training and ADR.

    One tensor row is one independent environment: its own context, start hour,
    state, action, exogenous window, reward, and done flag.
    """
    def __init__(
        self,
        *,
        data=None,
        device="cpu",
        n_envs=1,
        step_period=3600.0,
        future_steps=FUTURE_STEPS,
        max_episode_length=24 * 4,
        base_setpoint=BASE_SETPOINT,
        max_dev=5.0,
        tz_min=TZ_MIN_K,
        tz_max=TZ_MAX_K,
        w_energy=1.0,
        w_comfort=5.0,
        comfort_huber_k=0.0,
        w_sat=0.2,
        baseline_cs_coef=1.0,
        baseline_cop_coef=0.0,
        cop_bounds=(1.0, 5.0),
        php_min_w=100.0,
        cop_beta=1.0,
        sampling_dist=None,
        pid=None,
        th=None,
        pac=None,
    ):
        self.data = load_rc5_data() if data is None else data
        self.device = torch.device(device)
        self.n_envs = int(n_envs)
        self.step_period = float(step_period)
        self.step_n = max(1, int(round(self.step_period / DT)))
        self.future_steps = int(future_steps)
        self.max_episode_length = int(max_episode_length)
        self.base_setpoint = float(base_setpoint)
        self.max_dev = float(max_dev)
        self.tz_min, self.tz_max = float(tz_min), float(tz_max)
        self.w_energy, self.w_comfort, self.w_sat = float(w_energy), float(w_comfort), float(w_sat)
        self.comfort_huber_k = float(comfort_huber_k)
        self.baseline_cs_coef = float(baseline_cs_coef)
        self.baseline_cop_coef = float(baseline_cop_coef)
        self.cop_bounds = tuple(map(float, cop_bounds))
        self.php_min_w = float(php_min_w)
        self.cop_beta = float(cop_beta)
        self.sampling_dist = sampling_dist

        # Exogenous data stays on the target device; each env indexes its own hour.
        usable = (self.data.dist_matrix.shape[0] // self.step_n) * self.step_n
        dist = self.data.dist_matrix[:usable]
        dist_h = dist.reshape(-1, self.step_n, dist.shape[1]).mean(axis=1)
        self.dist = torch.as_tensor(dist, device=self.device)
        self.dist_h = torch.as_tensor(dist_h, device=self.device)
        self.n_hours = int(self.dist_h.shape[0])
        self.idx_max_h = self.n_hours - 1 - (self.future_steps + 1)
        self.max_start_h = max(0, int(self.idx_max_h - self.max_episode_length))
        self.dist2 = self.dist.reshape(self.n_hours, self.step_n, -1)
        self._k_future = torch.arange(1, self.future_steps + 1, device=self.device)
        self._lower_fc = torch.full((self.n_envs, self.future_steps, 1), LOWER_K, device=self.device)
        self._upper_fc = torch.full((self.n_envs, self.future_steps, 1), UPPER_K, device=self.device)
        self.n_phys_features_past = 5
        self.n_time_features = int(self.dist.shape[1] - self.n_phys_features_past)
        self.now_dim = 4 + self.n_time_features + 2
        self.forecast_feat_dim = 6 + self.n_time_features
        self.ctx_dim = len(CTX_NAMES)

        def _as(value):
            t = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            return t if t.ndim else t.expand(self.n_envs)

        th0 = TH if th is None else th
        pac0 = PAC if pac is None else pac
        pid0 = PID if pid is None else pid
        self.th = {k: _as(th0[k]) for k in TH_KEYS}
        self.pac = {k: _as(pac0[k]) for k in PAC_KEYS}
        self.kp, self.ki, self.kd = (_as(pid0[k]) for k in PID_KEYS)

        self.i_err = torch.zeros((self.n_envs,), device=self.device)
        self.e_prev = torch.zeros((self.n_envs,), device=self.device)
        self.d_err = torch.zeros((self.n_envs,), device=self.device)
        self.tz_last = torch.full((self.n_envs,), self.base_setpoint, device=self.device)
        self.php_last = torch.zeros((self.n_envs,), device=self.device)
        self.state = torch.zeros((self.n_envs, 5), device=self.device)
        self.i_err_ref = torch.zeros((self.n_envs,), device=self.device)
        self.e_prev_ref = torch.zeros((self.n_envs,), device=self.device)
        self.d_err_ref = torch.zeros((self.n_envs,), device=self.device)
        self.tz_last_ref = torch.full((self.n_envs,), self.base_setpoint, device=self.device)
        self.php_last_ref = torch.zeros((self.n_envs,), device=self.device)
        self.state_ref = torch.zeros((self.n_envs, 5), device=self.device)
        self.h_idx = torch.zeros((self.n_envs,), dtype=torch.long, device=self.device)
        self.steps = torch.zeros((self.n_envs,), dtype=torch.long, device=self.device)
        self._cop_history = []
        self._cop_bonus_prev = torch.zeros((self.n_envs,), device=self.device)
        self.ctx = torch.as_tensor(pack_context({"th": TH, "pac": PAC}, nominal_pid()), device=self.device, dtype=torch.float32).repeat(self.n_envs, 1)
        self.set_ctx(self.ctx)

    def set_sampling_dist(self, dist):
        self.sampling_dist = dist

    def set_ctx(self, ctx_batch):
        # One context per env, split into thermal dynamics, heat pump, and PID gains.
        ctx_batch = torch.as_tensor(ctx_batch, device=self.device, dtype=torch.float32).reshape(-1, self.ctx_dim)
        if ctx_batch.shape[0] == 1 and self.n_envs > 1:
            ctx_batch = ctx_batch.expand(self.n_envs, -1)
        ctx_batch = ctx_batch.reshape(self.n_envs, -1)
        self.ctx = ctx_batch
        th, pac, pid = split_context_torch(self.ctx)
        self.th = th
        self.pac = pac
        self.kp, self.ki, self.kd = (pid[k] for k in ("kp", "ki", "kd"))

    def _index_tensor(self, indices):
        if indices is None:
            return torch.arange(self.n_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(indices, device=self.device, dtype=torch.long).reshape(-1)

    def _sample_ctx(self, n):
        if self.sampling_dist is None:
            return self.ctx[:1].expand(int(n), -1).clone()
        sampler = getattr(self.sampling_dist, "sample", None) or getattr(self.sampling_dist, "rsample")
        return torch.as_tensor(sampler((int(n),)), device=self.device, dtype=torch.float32)

    def _sample_start_hour(self, n, start_hour=None):
        if start_hour is None:
            return torch.randint(0, self.max_start_h + 1, (int(n),), device=self.device)
        hours = torch.as_tensor(start_hour, device=self.device, dtype=torch.long).reshape(-1)
        if hours.numel() == 1:
            hours = hours.expand(int(n)).clone()
        return hours.clamp(0, self.max_start_h)

    def _set_ctx_subset(self, indices, ctx_batch):
        idx = self._index_tensor(indices)
        ctx_batch = torch.as_tensor(ctx_batch, device=self.device, dtype=torch.float32).reshape(-1, self.ctx_dim)
        if ctx_batch.shape[0] == 1 and idx.numel() > 1:
            ctx_batch = ctx_batch.expand(idx.numel(), -1)
        self.ctx[idx] = ctx_batch
        self.set_ctx(self.ctx)

    def reset_indices(self, indices=None, *, start_hour=None, ctx=None):
        # Reset all envs, or only the finished subset selected by SB3.
        idx = self._index_tensor(indices)
        if idx.numel() == 0:
            return self._build_obs()
        if ctx is None and self.sampling_dist is not None:
            ctx = self._sample_ctx(idx.numel())
        if ctx is not None:
            self._set_ctx_subset(idx, ctx)
        h0 = self._sample_start_hour(idx.numel(), start_hour=start_hour)
        self.h_idx[idx] = h0
        self.steps[idx] = 0
        ta0 = self.dist_h[h0, 0]
        tz0 = torch.full((idx.numel(),), self.base_setpoint, device=self.device)
        tw0 = (ta0 * self.th["R_w2"][idx] + tz0 * self.th["R_w1"][idx]) / (self.th["R_w1"][idx] + self.th["R_w2"][idx])
        state0 = torch.stack([tz0, tw0, tz0, tz0, tz0], dim=1)
        self.state[idx] = state0
        self.i_err[idx] = 0.0
        self.e_prev[idx] = 0.0
        self.d_err[idx] = 0.0
        self.tz_last[idx] = tz0
        self.php_last[idx] = 0.0
        self.i_err_ref[idx] = 0.0
        self.e_prev_ref[idx] = 0.0
        self.d_err_ref[idx] = 0.0
        self.tz_last_ref[idx] = tz0
        self.php_last_ref[idx] = 0.0
        self.state_ref[idx] = state0
        if idx.numel() == self.n_envs:
            self._cop_history = []
        else:
            for cop in self._cop_history:
                cop[idx] = 1.0
        self._cop_bonus_prev[idx] = 0.0
        return self._build_obs()

    def _build_obs(self):
        full = self.dist_h[self.h_idx]
        now = torch.cat([full[:, :2], full[:, 2:3], full[:, 4:5], full[:, 5:], self.tz_last[:, None], self.php_last[:, None]], dim=1)
        idxs = self.h_idx[:, None] + self._k_future[None, :]
        fk = self.dist_h[idxs]
        forecast = torch.cat([fk[:, :, :2], fk[:, :, 3:4], fk[:, :, 4:5], self._lower_fc, self._upper_fc, fk[:, :, 5:]], dim=2)
        return {"now": now, "forecast": forecast, "ctx": self.ctx}

    def reset(self, *, start_hour=None):
        return self.reset_indices(start_hour=start_hour)

    def _step_one(self, state, i_err, e_prev, d_err, sp, ta, qsol, occ, price):
        # Pure tensor physics over shape (2, n_envs, ...): agent and reference.
        kp, ki, kd = self.kp, self.ki, self.kd
        th, pac = self.th, self.pac
        rinf, rw1, rw2, rf, ri, rc = (th[k] for k in ("R_inf", "R_w1", "R_w2", "R_f", "R_i", "R_c"))
        cz, cw, ci, cf, cc, g_a = (th[k] for k in ("C_z", "C_w", "C_i", "C_f", "C_c", "gA"))
        kc, ke = pac["k_c"], pac["k_e"]
        ac, ae, bc, be, cc_p, ce = (pac[k] for k in ("a_c", "a_e", "b_c", "b_e", "c_c", "c_e"))
        tcn, tan = pac["Tcn"], pac["Tan"]

        tz_sum = torch.zeros_like(sp)
        php_sum = torch.zeros_like(sp)
        qc_sum = torch.zeros_like(sp)
        qe_sum = torch.zeros_like(sp)
        u_sum = torch.zeros_like(sp)
        comfort_kh = torch.zeros_like(sp)
        comfort_kh_n = torch.zeros_like(sp)
        energy_eur = torch.zeros_like(sp)
        energy_eur_n = torch.zeros_like(sp)
        sat_uh = torch.zeros_like(sp)

        hub = float(self.comfort_huber_k)
        for k in range(self.step_n):
            tz, tw, ti, tf, tc = (state[..., j] for j in range(5))
            err = sp - tz
            i_err = torch.clamp(i_err + err * DT, -100.0, 100.0)
            d_err = ((err - e_prev) / DT + d_err) * 0.5
            u_raw = kp * err + kd * d_err + ki * i_err
            u = torch.clamp(u_raw, 0.0, 1.0)
            delta_sat = u_raw - u
            e_prev = err

            ta_k = ta[:, k][None, :]
            qc = kc * (ac + bc * (tc - tcn) + cc_p * (ta_k - tan)) * u
            qe = -ke * (ae + be * (tc - tcn) + ce * (ta_k - tan)) * u
            php = qc - torch.abs(qe)
            qc_sum += qc
            qe_sum += qe
            u_sum += u

            occ_k = occ[:, k][None, :]
            q_occ = occ_k * (Q_INTERNAL_CON_W + Q_INTERNAL_RAD_W)
            d_tz = ((ta_k - tz) / rinf + (tw - tz) / rw2 + (tf - tz) / rf + (ti - tz) / ri + g_a * qsol[:, k][None, :] + q_occ) / cz
            d_tw = ((ta_k - tw) / rw1 + (tz - tw) / rw2) / cw
            d_ti = ((tz - ti) / ri) / ci
            d_tf = ((tz - tf) / rf + (tc - tf) / rc) / cf
            d_tc = ((tf - tc) / rc + qc) / cc
            state = state + DT * torch.stack([d_tz, d_tw, d_ti, d_tf, d_tc], dim=-1)

            tz_k = state[..., 0]
            tz_sum += tz_k
            php_sum += php
            v = torch.relu(LOWER_K - tz_k) + torch.relu(tz_k - UPPER_K)
            if hub > 0:
                v = torch.where(v <= hub, 0.5 * v * v / hub, v - 0.5 * hub)
            comfort_kh += v * occ_k * (DT / 3600.0)
            comfort_kh_n += v * (DT / 3600.0)
            p_k = price[:, k][None, :]
            energy_eur += p_k * torch.relu(php) * (DT / 3600.0) / 1000.0
            energy_eur_n += torch.relu(php) * (DT / 3600.0) / 1000.0
            sat_uh += torch.abs(delta_sat) * (DT / 3600.0)

        tz_last = tz_sum / self.step_n
        php_last = php_sum / self.step_n
        rew = -(self.w_comfort * comfort_kh + self.w_energy * energy_eur + self.w_sat * sat_uh)
        rew_n = -(self.w_comfort * comfort_kh_n + self.w_energy * energy_eur_n + self.w_sat * sat_uh)
        return state, i_err, e_prev, d_err, tz_last, php_last, rew, rew_n, qc_sum / self.step_n, qe_sum / self.step_n, u_sum / self.step_n, comfort_kh, energy_eur, sat_uh

    def step(self, action):
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(-1)
        sp = torch.clamp(self.base_setpoint + self.max_dev * a, self.tz_min, self.tz_max)

        row = self.dist2[self.h_idx]
        ta = row[:, :, 0]
        qsol = row[:, :, 1]
        occ = row[:, :, 2]
        price = row[:, :, 4]

        # Simulate the policy trajectory and the fixed-setpoint baseline together.
        sp2 = torch.stack([sp, torch.full_like(sp, self.base_setpoint)], dim=0)
        state2 = torch.stack([self.state, self.state_ref], dim=0)
        i_err2 = torch.stack([self.i_err, self.i_err_ref], dim=0)
        e_prev2 = torch.stack([self.e_prev, self.e_prev_ref], dim=0)
        d_err2 = torch.stack([self.d_err, self.d_err_ref], dim=0)
        state2, i_err2, e_prev2, d_err2, tz_last2, php_last2, rew2, rew_n2, qc2, qe2, u2, comfort2, energy2, sat2 = self._step_one(
            state2, i_err2, e_prev2, d_err2, sp2, ta, qsol, occ, price
        )
        self.state, self.state_ref = state2[0], state2[1]
        self.i_err, self.i_err_ref = i_err2[0], i_err2[1]
        self.e_prev, self.e_prev_ref = e_prev2[0], e_prev2[1]
        self.d_err, self.d_err_ref = d_err2[0], d_err2[1]
        self.tz_last, self.tz_last_ref = tz_last2[0], tz_last2[1]
        self.php_last, self.php_last_ref = php_last2[0], php_last2[1]

        # PPO sees improvement over the baseline; ADR only needs adr_bonus below.
        reward_raw, reward_ref, reward_ref_n = rew2[0], rew2[1], rew_n2[1]
        rew = (reward_raw - reward_ref) / torch.clamp(-reward_ref_n, min=1e-3)
        comfort_bonus = -(self.w_comfort * comfort2[1])
        sat_bonus = -(self.w_sat * sat2[1])
        cop_bonus = torch.zeros_like(rew)
        if self.baseline_cop_coef:
            php = self.php_last_ref
            mask = php > self.php_min_w
            cop = torch.where(mask, qc2[1] / torch.where(mask, php, torch.ones_like(php)), torch.ones_like(php))
            self._cop_history.append(cop)
            total = -self.baseline_cop_coef * cop_penalty_torch(torch.stack(self._cop_history, dim=0), cop_bounds=self.cop_bounds, beta=self.cop_beta)
            cop_bonus = total - self._cop_bonus_prev
            self._cop_bonus_prev = total
        adr_bonus = self.baseline_cs_coef * (comfort_bonus + sat_bonus + cop_bonus)
        bad = ~(torch.isfinite(state2).all(dim=(0, 2)) & torch.isfinite(rew2).all(dim=0) & torch.isfinite(rew_n2).all(dim=0) & torch.isfinite(adr_bonus))
        if bad.any():
            bad_idx = torch.nonzero(bad, as_tuple=False).reshape(-1)
            self.reset_indices(bad_idx, start_hour=self.h_idx[bad_idx], ctx=self.ctx[bad_idx])
            rew[bad_idx] = BAD_SCORE
            adr_bonus[bad_idx] = BAD_SCORE
            reward_raw[bad_idx] = BAD_SCORE
            reward_ref[bad_idx] = 0.0
            reward_ref_n[bad_idx] = 1.0
        self.h_idx += 1
        self.steps += 1
        done = (self.steps >= self.max_episode_length) | (self.h_idx >= self.idx_max_h)
        done = done | bad
        return self._build_obs(), rew, done, {"adr_bonus": adr_bonus}


class RC5TorchVecEnv(VecEnv):
    """Thin SB3 adapter around RC5TorchBatch.

    The simulator runs in Torch. This wrapper only implements the VecEnv API,
    converts tensors to NumPy for SB3, and resets finished rows of the batch.
    """
    metadata = {"render_modes": []}

    def __init__(self, *, data=None, sampling_dist=None, device="cpu", n_envs=1, **batch_kwargs):
        self.batch = RC5TorchBatch(data=data, sampling_dist=sampling_dist, device=device, n_envs=n_envs, **batch_kwargs)
        self.render_mode = None
        self.envs = [self]
        observation_space = spaces.Dict(
            {
                "now": spaces.Box(low=-np.inf, high=np.inf, shape=(self.batch.now_dim,), dtype=np.float32),
                "forecast": spaces.Box(low=-np.inf, high=np.inf, shape=(self.batch.future_steps, self.batch.forecast_feat_dim), dtype=np.float32),
                "ctx": spaces.Box(low=-np.inf, high=np.inf, shape=(self.batch.ctx_dim,), dtype=np.float32),
            }
        )
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.actions = None
        super().__init__(int(n_envs), observation_space, action_space)

    def get_wrapper_attr(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        if hasattr(self.batch, name):
            return getattr(self.batch, name)
        raise AttributeError(name)

    def set_sampling_dist(self, dist):
        self.batch.set_sampling_dist(dist)

    def _obs_to_numpy(self, obs):
        # SB3 VecEnv boundary: observations/rewards must cross back to NumPy here.
        return {k: v.detach().cpu().numpy().astype(np.float32, copy=False) for k, v in obs.items()}

    def _slice_obs(self, obs, idx):
        return {k: v[idx].copy() for k, v in obs.items()}

    def _reset_info(self, idx):
        return {"start_hour": int(self.batch.h_idx[idx].item())}

    def reset(self):
        maybe_options = self._options[0] if self._options else {}
        obs = self.batch.reset_indices(start_hour=maybe_options.get("start_hour"))
        self.reset_infos = [self._reset_info(i) for i in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return self._obs_to_numpy(obs)

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        obs, reward, done, _ = self.batch.step(self.actions)
        obs_np = self._obs_to_numpy(obs)
        reward_np = reward.detach().cpu().numpy().astype(np.float32, copy=False)
        done_np = done.detach().cpu().numpy().astype(bool, copy=False)
        info_list = [{"TimeLimit.truncated": bool(flag)} for flag in done_np]
        done_idx = np.flatnonzero(done_np)
        for i in done_idx.tolist():
            info_list[i]["terminal_observation"] = self._slice_obs(obs_np, i)
        if done_idx.size:
            # VecEnv auto-reset: SB3 gets terminal_observation, then the row is reused.
            obs_reset = self._obs_to_numpy(self.batch.reset_indices(done_idx))
            for i in done_idx.tolist():
                for key in obs_np:
                    obs_np[key][i] = obs_reset[key][i]
                self.reset_infos[i] = self._reset_info(i)
        return obs_np, reward_np, done_np, info_list

    def close(self):
        return None

    def get_images(self):
        return [None for _ in range(self.num_envs)]

    def get_attr(self, attr_name, indices=None):
        value = self.get_wrapper_attr(attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        target = self if hasattr(self, attr_name) else self.batch
        setattr(target, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        result = self.get_wrapper_attr(method_name)(*method_args, **method_kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
