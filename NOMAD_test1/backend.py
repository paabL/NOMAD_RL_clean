from __future__ import annotations

from copy import deepcopy

from stable_baselines3.common.monitor import Monitor

from NOMAD.core.backend import PolicySpec

from .env import SwingEnv, SwingTorchBatch, context_low_high

DEFAULT_ENV_CFG = {
    "dt": 0.2,
    "max_speed": 8.0,
    "action_coef": 0.05,
    "max_episode_length": 64,
    "init_angle_noise": 0.2,
    "init_speed_noise": 0.1,
    "difficulty_bonus_coef": 1.0,
}

DEFAULT_ADR_CFG = {
    "iters": 10,
    "n_sample": 64,
    "refine_steps": 1,
    "refine_lr": 1e-2,
    "kl_M": 64,
    "surprise_coef": 1.0,
    "kl_beta": 1.0,
    "update_every_episodes": 8,
}


def merge_dict(base, extra):
    out = deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out


class SwingBackend:
    def __init__(self, *, env_cfg=None):
        self.env_cfg = merge_dict(DEFAULT_ENV_CFG, env_cfg)

    def flow_bounds(self, device):
        return context_low_high(device=device)

    def make_train_env(self, *, sampling_dist, env_id, rollout_dir, plot_every_episodes):
        return Monitor(
            SwingEnv(
                sampling_dist=sampling_dist,
                dt=self.env_cfg["dt"],
                max_speed=self.env_cfg["max_speed"],
                action_coef=self.env_cfg["action_coef"],
                max_episode_length=self.env_cfg["max_episode_length"],
                init_angle_noise=self.env_cfg["init_angle_noise"],
                init_speed_noise=self.env_cfg["init_speed_noise"],
                env_id=env_id,
                rollout_dir=rollout_dir,
            )
        )

    def make_adr_env(self, *, device, n_envs):
        return SwingTorchBatch(
            device=device,
            n_envs=n_envs,
            dt=self.env_cfg["dt"],
            max_speed=self.env_cfg["max_speed"],
            action_coef=self.env_cfg["action_coef"],
            max_episode_length=self.env_cfg["max_episode_length"],
            difficulty_bonus_coef=self.env_cfg["difficulty_bonus_coef"],
            init_angle_noise=self.env_cfg["init_angle_noise"],
        )

    def policy_spec(self):
        return PolicySpec(policy="MultiInputLstmPolicy", policy_kwargs={})
