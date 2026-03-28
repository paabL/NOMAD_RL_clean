from __future__ import annotations

import copy
import time

import numpy as np
import torch
import zuko

from .backend import NomadBackend


def normalize_obs(obs, stats):
    clip = float(stats["clip_obs"])
    eps = float(stats["eps"])
    out = {}
    for key, value in obs.items():
        mean = stats.get(f"{key}_mean")
        var = stats.get(f"{key}_var")
        out[key] = value if mean is None else torch.clamp((value - mean) / torch.sqrt(var + eps), -clip, clip)
    return out


class NormFlowDist:
    def __init__(self, low, high, *, flow=None, transforms=3, bins=8, hidden=(64, 64), device="cpu"):
        if low is None or high is None:
            raise ValueError("NormFlowDist requires explicit low/high bounds.")
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        self.scale = self.high - self.low
        self.log_scale = torch.log(self.scale)
        self.hidden = tuple(hidden)
        self.transforms = int(transforms)
        self.bins = int(bins)
        self.ndim = int(self.low.numel())
        self.flow = flow if flow is not None else zuko.flows.MAF(
            features=self.ndim,
            context=0,
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes=((self.bins,), (self.bins,), (self.bins - 1,)),
            hidden_features=self.hidden,
            transforms=self.transforms,
        ).to(self.low.device)

    def get_params(self):
        return self.flow.parameters()

    def clone(self):
        return NormFlowDist(self.low, self.high, flow=copy.deepcopy(self.flow), transforms=self.transforms, bins=self.bins, hidden=self.hidden, device=self.low.device)

    def sample(self, shape):
        n = int(shape[0]) if shape else 1
        with torch.no_grad():
            y = self.flow().sample((n,))
        return self.low + self.scale * torch.sigmoid(y)

    def rsample(self, shape):
        n = int(shape[0]) if shape else 1
        y = self.flow().rsample((n,))
        return self.low + self.scale * torch.sigmoid(y)

    def log_prob(self, x, eps=1e-6):
        u = ((x - self.low) / self.scale).clamp(eps, 1.0 - eps)
        y = torch.logit(u)
        logp_y = self.flow().log_prob(y)
        log_det = (self.log_scale + torch.log(u) + torch.log1p(-u)).sum(-1)
        return logp_y - log_det

    def state_dict(self):
        return {
            "flow": self.flow.state_dict(),
            "low": self.low.detach().cpu(),
            "high": self.high.detach().cpu(),
            "hidden": self.hidden,
            "transforms": self.transforms,
            "bins": self.bins,
        }

    def _compat_flow_state_dict(self, state):
        flow_state = dict(state["flow"])
        expected = set(self.flow.state_dict())
        for old, new in (("base._0", "base.loc"), ("base._1", "base.scale")):
            if new in expected and new not in flow_state and old in flow_state:
                flow_state[new] = flow_state[old]
            if old in expected and old not in flow_state and new in flow_state:
                flow_state[old] = flow_state[new]
            if old not in expected:
                flow_state.pop(old, None)
            if new not in expected:
                flow_state.pop(new, None)
        return flow_state

    def load_state_dict(self, state):
        self.flow.load_state_dict(self._compat_flow_state_dict(state))
        return self


class RecurrentPPOPolicyEvaluator:
    def __init__(self, model, device="cpu", obs_norm=None):
        self.device = torch.device(device)
        policy_kwargs = copy.deepcopy(getattr(model, "policy_kwargs", {}))
        self.policy = model.policy_class(model.observation_space, model.action_space, model.lr_schedule, **policy_kwargs).to(self.device)
        self.obs_norm = obs_norm
        self.lstm_states = None
        self.episode_starts = None
        self.sync(model, obs_norm=obs_norm)

    def _set_eval_mode(self):
        if hasattr(self.policy, "set_training_mode"):
            self.policy.set_training_mode(False)
        else:
            self.policy.eval()

    def sync(self, model, obs_norm=None):
        self.policy.load_state_dict(model.policy.state_dict())
        self.obs_norm = obs_norm
        self._set_eval_mode()

    def reset(self, n_envs=1):
        shape = list(getattr(self.policy, "lstm_hidden_state_shape", (1, 1, 1)))
        shape[1] = int(n_envs)
        h = torch.zeros(tuple(shape), device=self.device)
        c = torch.zeros(tuple(shape), device=self.device)
        self.lstm_states = (h, c)
        self.episode_starts = torch.ones((int(n_envs),), device=self.device)

    def act(self, obs):
        obs = {k: v.to(self.device) for k, v in obs.items()}
        if self.obs_norm is not None:
            obs = normalize_obs(obs, self.obs_norm)
        with torch.no_grad():
            dist, self.lstm_states = self.policy.get_distribution(obs, self.lstm_states, self.episode_starts)
            action = dist.get_actions(deterministic=True)
        low = torch.as_tensor(self.policy.action_space.low, dtype=action.dtype, device=action.device)
        high = torch.as_tensor(self.policy.action_space.high, dtype=action.dtype, device=action.device)
        self.episode_starts = torch.zeros_like(self.episode_starts)
        return torch.clamp(action, low, high)


class ADRFlows:
    def __init__(
        self,
        backend: NomadBackend,
        dist: NormFlowDist | None = None,
        *,
        device="cpu",
        iters=30,
        lr=1e-3,
        n_sample=64,
        refine_steps=0,
        refine_lr=1e-3,
        temp_init=1.0,
        kl_beta=0.1,
        kl_M=1024,
        ret_coef=1.0,
        bonus_coef=1.0,
        surprise_coef=1.0,
    ):
        self.backend = backend
        self.device = torch.device(device)
        self.current = dist if dist is not None else NormFlowDist(*backend.flow_bounds(self.device), device=self.device)
        self.opt = torch.optim.Adam(self.current.get_params(), lr=float(lr))
        self.ev = None

        self.iters = int(iters)
        self.n_sample = int(n_sample)
        self.refine_steps = int(refine_steps)
        self.refine_lr = float(refine_lr)
        self.temp = float(temp_init)
        self.kl_beta = float(kl_beta)
        self.kl_M = int(kl_M)
        self.ret_coef = float(ret_coef)
        self.bonus_coef = float(bonus_coef)
        self.surprise_coef = float(surprise_coef)

    def set_policy(self, model, obs_norm=None):
        stats = None
        if obs_norm is not None:
            stats = {
                k: (torch.as_tensor(v, device=self.device, dtype=torch.float32) if isinstance(v, np.ndarray) else v)
                for k, v in obs_norm.items()
            }
        if self.ev is None:
            self.ev = RecurrentPPOPolicyEvaluator(model, device=self.device, obs_norm=stats)
        else:
            self.ev.sync(model, obs_norm=stats)

    def get_train_dist(self):
        return self.get_train_dist_on(self.device)

    def get_train_dist_on(self, device):
        device = torch.device(device)
        if device == self.device:
            return self.current
        state = self.state_dict()
        dist = NormFlowDist(
            state["low"],
            state["high"],
            transforms=int(state.get("transforms", 3)),
            bins=int(state.get("bins", 8)),
            hidden=tuple(state.get("hidden", (64, 64))),
            device=device,
        )
        dist.load_state_dict(state)
        return dist

    def sample(self, n):
        return self.current.sample((int(n),))

    def state_dict(self):
        return self.current.state_dict()

    def load_state_dict(self, state):
        self.current = NormFlowDist(
            state["low"],
            state["high"],
            transforms=int(state.get("transforms", 3)),
            bins=int(state.get("bins", 8)),
            hidden=tuple(state.get("hidden", (64, 64))),
            device=self.device,
        )
        self.current.load_state_dict(state)
        self.opt = torch.optim.Adam(self.current.get_params(), lr=self.opt.param_groups[0]["lr"])
        return self

    def update(self):
        if self.ev is None:
            raise ValueError("ADRFlows.set_policy(model, obs_norm) must be called before update().")

        t0 = time.perf_counter()
        low, high = self.current.low, self.current.high
        x = (low + (high - low) * torch.rand((self.n_sample, low.numel()), device=self.device)).requires_grad_(True)
        env = self.backend.make_adr_env(device=self.device, n_envs=self.n_sample)
        xs, objs = [], []
        ret_pre = bonus_pre = None
        surprise_pre = None

        def rollout_objective():
            env.set_ctx(x)
            self.ev.reset(self.n_sample)
            obs = env.reset()
            ret = torch.zeros((self.n_sample,), device=self.device)
            bonus = torch.zeros((self.n_sample,), device=self.device)
            for _ in range(env.max_episode_length):
                obs, reward, done, info = env.step(self.ev.act(obs))
                ret += reward
                bonus += info["adr_bonus"]
            return self.ret_coef * ret + self.bonus_coef * bonus, ret, bonus

        for _ in range(self.refine_steps):
            obj, ret, bonus = rollout_objective()
            if ret_pre is None:
                ret_pre, bonus_pre = ret.detach(), bonus.detach()
            surprise = self.surprise_coef * (-self.current.log_prob(x)) if self.surprise_coef else torch.zeros_like(obj)
            surprise_pre = surprise.detach() if surprise_pre is None else surprise_pre
            total = obj + surprise
            xs.append(x.detach().clone())
            objs.append(total.detach())
            grad = torch.autograd.grad(total.sum(), x)[0]
            with torch.no_grad():
                x.add_(self.refine_lr * grad).clamp_(low, high)
            x = x.detach().requires_grad_(True)

        obj, ret, bonus = rollout_objective()
        if ret_pre is None:
            ret_pre, bonus_pre = ret.detach(), bonus.detach()
        surprise = self.surprise_coef * (-self.current.log_prob(x)) if self.surprise_coef else torch.zeros_like(obj)
        surprise_pre = surprise.detach() if surprise_pre is None else surprise_pre
        xs.append(x.detach())
        objs.append((obj + surprise).detach())

        x_data = torch.cat(xs, dim=0)
        obj_all = torch.cat(objs, dim=0)
        with torch.no_grad():
            weights = torch.softmax(obj_all / self.temp, dim=0)
        ess = 1.0 / torch.sum(weights * weights)

        ref = self.current.clone()
        for p in ref.get_params():
            p.requires_grad_(False)

        loss_fit_val = torch.zeros((), device=self.device)
        kl_val = torch.zeros((), device=self.device)
        for _ in range(self.iters):
            self.opt.zero_grad(set_to_none=True)
            loss_fit = -(weights * self.current.log_prob(x_data)).sum()
            x_kl = self.current.rsample((self.kl_M,))
            kl = (self.current.log_prob(x_kl) - ref.log_prob(x_kl)).mean()
            loss_fit_val = loss_fit.detach()
            kl_val = kl.detach()
            (loss_fit + self.kl_beta * kl).backward()
            self.opt.step()

        with torch.no_grad():
            x_mc = self.current.sample((self.kl_M,))
            entropy = (-self.current.log_prob(x_mc)).mean().item()
        ret_np = ret.detach().cpu().numpy()
        bonus_mean = float(bonus.mean().item())
        bonus_mean_pre = float(bonus_pre.mean().item())
        return {
            "dt": float(time.perf_counter() - t0),
            "B": int(self.n_sample),
            "obj_mean": float((obj + surprise).mean().item()),
            "ret_mean": float(ret_np.mean()),
            "ret_std": float(ret_np.std()),
            "bonus_mean": bonus_mean,
            "bonus_mean_pre": bonus_mean_pre,
            "bcs_mean": bonus_mean,
            "bcs_mean_pre": bonus_mean_pre,
            "ret_mean_pre": float(ret_pre.mean().item()),
            "surprise_mean_pre": float(surprise_pre.mean().item()),
            "surprise_mean": float(surprise.mean().item()),
            "surprise_std": float(surprise.std().item()),
            "entropy": float(entropy),
            "ess": float(ess.item()),
            "loss_fit": float(loss_fit_val.item()),
            "beta_kl": float((self.kl_beta * kl_val).item()),
        }
