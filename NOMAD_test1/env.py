from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

CTX_CENTER = np.asarray([1.0, 0.1], dtype=np.float32)
CTX_LOW = np.asarray([0.7, 0.02], dtype=np.float32)
CTX_HIGH = np.asarray([1.3, 0.25], dtype=np.float32)
CTX_HALF_RANGE = 0.5 * (CTX_HIGH - CTX_LOW)

DEFAULT_DT = 0.2
DEFAULT_MAX_SPEED = 8.0
DEFAULT_ACTION_COEF = 0.05
DEFAULT_INIT_ANGLE_NOISE = 0.2
DEFAULT_INIT_SPEED_NOISE = 0.1
DEFAULT_DIFFICULTY_BONUS_COEF = 1.0


def context_low_high(device="cpu"):
    dev = torch.device(device)
    return (
        torch.tensor(CTX_LOW, dtype=torch.float32, device=dev),
        torch.tensor(CTX_HIGH, dtype=torch.float32, device=dev),
    )


def _wrap_np(theta):
    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_torch(theta):
    return torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi


def _difficulty_np(ctx):
    z = (np.asarray(ctx, dtype=np.float32) - CTX_CENTER) / CTX_HALF_RANGE
    return float(np.mean(z * z))


def _difficulty_torch(ctx):
    center = torch.tensor(CTX_CENTER, dtype=torch.float32, device=ctx.device)
    half = torch.tensor(CTX_HALF_RANGE, dtype=torch.float32, device=ctx.device)
    z = (ctx - center) / half
    return z.square().mean(dim=1)


def _step_np(theta, omega, action, ctx, dt, max_speed):
    g_scale, damping = np.asarray(ctx, dtype=np.float32)
    omega_dot = float(g_scale) * math.sin(float(theta)) + float(action) - float(damping) * float(omega)
    omega = float(np.clip(float(omega) + float(dt) * omega_dot, -float(max_speed), float(max_speed)))
    theta = float(_wrap_np(float(theta) + float(dt) * omega))
    return theta, omega


def _step_torch(theta, omega, action, ctx, dt, max_speed):
    g_scale = ctx[:, 0]
    damping = ctx[:, 1]
    omega_dot = g_scale * torch.sin(theta) + action - damping * omega
    omega = torch.clamp(omega + float(dt) * omega_dot, -float(max_speed), float(max_speed))
    theta = _wrap_torch(theta + float(dt) * omega)
    return theta, omega


def _obs_np(theta):
    return {"now": np.asarray([math.sin(float(theta)), math.cos(float(theta))], dtype=np.float32)}


def _obs_torch(theta):
    return {"now": torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)}


class SwingEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        *,
        sampling_dist=None,
        dt=DEFAULT_DT,
        max_speed=DEFAULT_MAX_SPEED,
        action_coef=DEFAULT_ACTION_COEF,
        max_episode_length=64,
        init_angle_noise=DEFAULT_INIT_ANGLE_NOISE,
        init_speed_noise=DEFAULT_INIT_SPEED_NOISE,
        env_id=0,
        rollout_dir=None,
    ):
        super().__init__()
        self.sampling_dist = sampling_dist
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.action_coef = float(action_coef)
        self.max_episode_length = int(max_episode_length)
        self.init_angle_noise = float(init_angle_noise)
        self.init_speed_noise = float(init_speed_noise)
        self.env_id = int(env_id)
        self.rollout_dir = None if rollout_dir is None else str(rollout_dir)

        self.observation_space = spaces.Dict({"now": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)})
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.ctx = CTX_CENTER.copy()
        self.theta = math.pi
        self.omega = 0.0
        self.steps = 0
        self._episode_count = 0
        self._last_episode = None
        self._reset_episode_log()

    def set_sampling_dist(self, dist):
        self.sampling_dist = dist

    def set_rollout_dir(self, rollout_dir):
        self.rollout_dir = None if rollout_dir is None else str(rollout_dir)

    def _reset_episode_log(self):
        self.ep_theta = []
        self.ep_omega = []
        self.ep_action = []
        self.ep_reward = []

    def _record_state(self):
        self.ep_theta.append(float(self.theta))
        self.ep_omega.append(float(self.omega))

    def _episode_arrays(self):
        theta = np.asarray(self.ep_theta, dtype=np.float32)
        omega = np.asarray(self.ep_omega, dtype=np.float32)
        action = np.asarray(self.ep_action, dtype=np.float32)
        reward = np.asarray(self.ep_reward, dtype=np.float32)
        return {
            "context": np.asarray(self.ctx, dtype=np.float32),
            "time_state": self.dt * np.arange(theta.size, dtype=np.float32),
            "time_step": self.dt * np.arange(1, action.size + 1, dtype=np.float32),
            "theta": theta,
            "omega": omega,
            "action": action,
            "reward": reward,
        }

    def _episode_payload(self):
        return self._last_episode or self._episode_arrays()

    def save_last_episode(self, path: str | Path | None = None):
        if path is None:
            out_dir = Path(self.rollout_dir or Path(__file__).resolve().parent / "runs")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"episode_{self._episode_count:06d}_env{self.env_id}.npz"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self._episode_payload())
        return path

    def plot_last_episode(self, path: str | Path | None = None):
        if path is None:
            out_dir = Path(self.rollout_dir or Path(__file__).resolve().parent / "runs")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"episode_{self._episode_count:06d}_env{self.env_id}.png"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        data = self._episode_payload()
        t_state, t_step = data["time_state"], data["time_step"]
        theta, omega = data["theta"], data["omega"]
        action, reward = data["action"], data["reward"]
        fig, axs = plt.subplots(3, 1, figsize=(9, 8), dpi=180, constrained_layout=True)
        fig.patch.set_facecolor("#f8f7f4")
        for ax in axs:
            ax.set_facecolor("#fcfbf8")
        axs[0].axhspan(-0.15, 0.15, color="#d9f0d8", alpha=0.9, lw=0)
        axs[0].plot(t_state, theta / math.pi, color="#1d3557", lw=2.4, label=r"$\theta/\pi$")
        axs[0].plot(t_state, np.cos(theta), color="#2a9d8f", lw=1.8, label=r"$\cos\theta$")
        axs[0].set_ylabel("state")
        axs[0].legend(loc="upper right")
        axs[1].plot(t_state, omega, color="#457b9d", lw=2.0, label=r"$\omega$")
        axs[1].step(t_step, action, where="post", color="#e76f51", lw=1.8, label="action")
        axs[1].set_ylabel("control")
        axs[1].legend(loc="upper right")
        phase = axs[2].scatter(theta[1:] / math.pi, omega[1:], c=t_step, cmap="viridis", s=28, edgecolors="none")
        axs[2].plot(theta / math.pi, omega, color="#1d3557", alpha=0.18, lw=2)
        axs[2].set_xlabel("theta / pi")
        axs[2].set_ylabel("omega")
        axs[2].set_title(
            f"return={reward.sum():.2f}  g={self.ctx[0]:.2f}  damp={self.ctx[1]:.3f}",
            fontsize=11,
        )
        fig.colorbar(phase, ax=axs[2], label="time [s]")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _sample_ctx(self):
        if self.sampling_dist is None:
            return CTX_CENTER.copy()
        return np.asarray(self.sampling_dist.sample((1,)).detach().cpu().numpy()[0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        self.steps = 0
        self.ctx = self._sample_ctx()
        self.theta = float(_wrap_np(math.pi + self.np_random.uniform(-self.init_angle_noise, self.init_angle_noise)))
        self.omega = float(self.np_random.uniform(-self.init_speed_noise, self.init_speed_noise))
        self._reset_episode_log()
        self._record_state()
        return _obs_np(self.theta), {"context": self.ctx.copy(), "difficulty": _difficulty_np(self.ctx)}

    def step(self, action):
        action = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        self.theta, self.omega = _step_np(self.theta, self.omega, action, self.ctx, self.dt, self.max_speed)
        reward = float(math.cos(self.theta) - self.action_coef * action * action)
        self.ep_action.append(action)
        self.ep_reward.append(reward)
        self._record_state()
        self.steps += 1
        if self.steps >= self.max_episode_length:
            self._last_episode = self._episode_arrays()
        return _obs_np(self.theta), reward, False, self.steps >= self.max_episode_length, {"context": self.ctx.copy(), "reward_raw": reward}


class SwingTorchBatch:
    def __init__(
        self,
        *,
        device="cpu",
        n_envs=1,
        dt=DEFAULT_DT,
        max_speed=DEFAULT_MAX_SPEED,
        action_coef=DEFAULT_ACTION_COEF,
        max_episode_length=64,
        difficulty_bonus_coef=DEFAULT_DIFFICULTY_BONUS_COEF,
        init_angle_noise=DEFAULT_INIT_ANGLE_NOISE,
    ):
        self.device = torch.device(device)
        self.n_envs = int(n_envs)
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.action_coef = float(action_coef)
        self.max_episode_length = int(max_episode_length)
        self.difficulty_bonus_coef = float(difficulty_bonus_coef)
        self.init_angle_noise = float(init_angle_noise)

        self.ctx = torch.tensor(CTX_CENTER, dtype=torch.float32, device=self.device).repeat(self.n_envs, 1)
        self.theta = torch.zeros((self.n_envs,), dtype=torch.float32, device=self.device)
        self.omega = torch.zeros((self.n_envs,), dtype=torch.float32, device=self.device)
        self.steps = torch.zeros((self.n_envs,), dtype=torch.long, device=self.device)

    def set_ctx(self, ctx_batch):
        self.ctx = torch.as_tensor(ctx_batch, dtype=torch.float32, device=self.device).reshape(self.n_envs, 2)

    def reset(self):
        self.theta = torch.full((self.n_envs,), math.pi - self.init_angle_noise, dtype=torch.float32, device=self.device)
        self.omega = torch.zeros((self.n_envs,), dtype=torch.float32, device=self.device)
        self.steps.zero_()
        return _obs_torch(self.theta)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(self.n_envs).clamp(-1.0, 1.0)
        self.theta, self.omega = _step_torch(self.theta, self.omega, action, self.ctx, self.dt, self.max_speed)
        reward = torch.cos(self.theta) - self.action_coef * action.square()
        bonus = self.difficulty_bonus_coef * _difficulty_torch(self.ctx) / float(self.max_episode_length)
        self.steps += 1
        return _obs_torch(self.theta), reward, self.steps >= self.max_episode_length, {"adr_bonus": bonus}
