from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

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
    load_rc5_data,
    nominal_pid,
    pack_context,
    split_context_torch,
)

BAD_SCORE = -1e6


def cop_penalty_torch(cops, *, cop_bounds=(1.0, 5.0), beta=1.0):
    cop_lo, cop_hi = map(float, cop_bounds)
    x = float(beta) * cops
    x0 = x.max(dim=0).values
    eff = (x0 + torch.log(torch.mean(torch.exp(x - x0), dim=0) + 1e-12)) / float(beta)
    sp = torch.nn.functional.softplus
    return 100.0 * sp(cop_lo - eff) ** 2 + sp(eff - cop_hi) ** 2


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
