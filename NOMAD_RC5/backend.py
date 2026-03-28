from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from NOMAD.core.backend import PolicySpec
from .env import NOMAD, NormalizeAction, RC5TorchBatch, ResidualActionWrapper
from .sim import BASE_SETPOINT, FUTURE_STEPS, TZ_MAX_K, TZ_MIN_K, context_low_high, load_rc5_data

DEFAULT_ENV_CFG = {
    "step_period": 3600.0,
    "future_steps": FUTURE_STEPS,
    "warmup_steps": 2 * 24,
    "base_setpoint": BASE_SETPOINT,
    "max_dev": 5.0,
    "max_episode_length": 24 * 21, #3 semaines, relativement long : pour le LSTM
    "tz_min": TZ_MIN_K,
    "tz_max": TZ_MAX_K,
    "w_energy": 1.0,
    "w_comfort": 5.0,
    "comfort_huber_k": 0.0,
    "w_sat": 0.2,
}

DEFAULT_POLICY_CFG = {
    "critic_use_ctx": True,
    "policy_hidden": (128, 128), #default (64, 64)
    "value_hidden": (128, 128), #default (64, 64)
}

DEFAULT_ADR_CFG = {
    "baseline_cs_coef": 50.0,
    "baseline_cop_coef": 5.0,
    "max_episode_length": 24*5, #5 jours, pour accélérer les itérations d'ADR, les episodes de training sont longs mais pas besoin d'autant pour avoir une idée des perfs de la policy
    "cop_bounds": (1.0, 5.0),
    "php_min_w": 100.0,
    "cop_beta": 1.0,
}


def merge_dict(base, extra):
    out = deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out


class ConvForecastTemporalFuseExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, d_conv=64, kernel_size=3, n_conv_layers=2, d_out=64, hidden_fuse=128, use_ctx=False):
        now_space = observation_space.spaces["now"]
        forecast_space = observation_space.spaces["forecast"]
        ctx_space = observation_space.spaces.get("ctx")
        self.now_dim = int(now_space.shape[0])
        self.f_dim = int(forecast_space.shape[1])
        self.d_conv = int(d_conv)
        self.use_ctx = bool(use_ctx and ctx_space is not None)
        self.ctx_dim = int(ctx_space.shape[0]) if self.use_ctx else 0
        super().__init__(observation_space, features_dim=int(d_out))
        padding = int(kernel_size) // 2
        layers = []
        in_ch = self.f_dim
        for _ in range(int(n_conv_layers)):
            layers.extend([nn.Conv1d(in_ch, self.d_conv, kernel_size=int(kernel_size), padding=padding), nn.ReLU()])
            in_ch = self.d_conv
        self.conv_net = nn.Sequential(*layers) if layers else nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.now_dim + self.d_conv + self.ctx_dim, int(hidden_fuse)),
            nn.ReLU(),
            nn.Linear(int(hidden_fuse), int(d_out)),
            nn.ReLU(),
        )

    def forward(self, obs):
        x_now = obs["now"]
        x_fc = obs["forecast"]
        if x_fc.shape[1] == 0:
            z_fc = torch.zeros((x_now.shape[0], self.d_conv), device=x_now.device, dtype=x_now.dtype)
        else:
            z_fc = self.pool(self.conv_net(x_fc.transpose(1, 2))).squeeze(-1)
        x = [x_now, z_fc]
        if self.use_ctx:
            x.append(obs["ctx"])
        return self.fuse_mlp(torch.cat(x, dim=1))


class ValueCtxLstmPolicy(MultiInputLstmPolicy):
    def __init__(self, *args, critic_use_ctx=True, **kwargs):
        self.critic_use_ctx = bool(critic_use_ctx)
        if self.critic_use_ctx:
            kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def make_features_extractor(self):
        kwargs = dict(self.features_extractor_kwargs or {})
        kwargs["use_ctx"] = self.critic_use_ctx and hasattr(self, "pi_features_extractor")
        return self.features_extractor_class(self.observation_space, **kwargs)


class RC5Backend:
    def __init__(self, *, env_cfg=None, policy_cfg=None, adr_cfg=None, data=None):
        self.env_cfg = merge_dict(DEFAULT_ENV_CFG, env_cfg)
        self.policy_cfg = merge_dict(DEFAULT_POLICY_CFG, policy_cfg)
        self.adr_cfg = merge_dict(DEFAULT_ADR_CFG, adr_cfg)
        self.data = load_rc5_data() if data is None else data

    def flow_bounds(self, device):
        return context_low_high(device=device)

    def make_train_env(self, *, sampling_dist, env_id, rollout_dir, plot_every_episodes):
        env = NOMAD(
            data=self.data,
            sampling_dist=sampling_dist,
            step_period=self.env_cfg["step_period"],
            future_steps=self.env_cfg["future_steps"],
            warmup_steps=self.env_cfg["warmup_steps"],
            base_setpoint=self.env_cfg["base_setpoint"],
            max_episode_length=self.env_cfg["max_episode_length"],
            tz_min=self.env_cfg["tz_min"],
            tz_max=self.env_cfg["tz_max"],
            w_energy=self.env_cfg["w_energy"],
            w_comfort=self.env_cfg["w_comfort"],
            comfort_huber_k=self.env_cfg["comfort_huber_k"],
            w_sat=self.env_cfg["w_sat"],
            include_ctx=bool(self.policy_cfg["critic_use_ctx"]),
            rollout_dir=rollout_dir,
            auto_plot=False,
            plot_every_episodes=plot_every_episodes,
            env_id=env_id,
        )
        env = ResidualActionWrapper(env, base_action=self.env_cfg["base_setpoint"], max_dev=self.env_cfg["max_dev"])
        env = NormalizeAction(env)
        return Monitor(env)

    def make_adr_env(self, *, device, n_envs):
        max_episode_length = self.env_cfg["max_episode_length"] if self.adr_cfg["max_episode_length"] is None else self.adr_cfg["max_episode_length"]
        return RC5TorchBatch(
            data=self.data,
            device=device,
            n_envs=n_envs,
            step_period=self.env_cfg["step_period"],
            future_steps=self.env_cfg["future_steps"],
            max_episode_length=max_episode_length,
            base_setpoint=self.env_cfg["base_setpoint"],
            max_dev=self.env_cfg["max_dev"],
            tz_min=self.env_cfg["tz_min"],
            tz_max=self.env_cfg["tz_max"],
            w_energy=self.env_cfg["w_energy"],
            w_comfort=self.env_cfg["w_comfort"],
            comfort_huber_k=self.env_cfg["comfort_huber_k"],
            w_sat=self.env_cfg["w_sat"],
            baseline_cs_coef=self.adr_cfg["baseline_cs_coef"],
            baseline_cop_coef=self.adr_cfg["baseline_cop_coef"],
            cop_bounds=self.adr_cfg["cop_bounds"],
            php_min_w=self.adr_cfg["php_min_w"],
            cop_beta=self.adr_cfg["cop_beta"],
        )

    def policy_spec(self):
        return PolicySpec(
            policy=ValueCtxLstmPolicy,
            policy_kwargs={
                "features_extractor_class": ConvForecastTemporalFuseExtractor,
                "critic_use_ctx": bool(self.policy_cfg["critic_use_ctx"]),
                "net_arch": {"pi": list(self.policy_cfg["policy_hidden"]), "vf": list(self.policy_cfg["value_hidden"])},
            },
        )
