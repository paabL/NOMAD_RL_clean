from __future__ import annotations

import ast
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from NOMAD.core.adr import ADRFlows, NormFlowDist
from NOMAD.core.backend import PolicySpec
from NOMAD.core.training import run_training as run_core_training, vecnorm_stats


class ToyTrainEnv(gym.Env):
    metadata = {}

    def __init__(self, sampling_dist=None, horizon=4):
        super().__init__()
        self.sampling_dist = sampling_dist
        self.horizon = int(horizon)
        self.observation_space = spaces.Dict(
            {
                "now": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                "forecast": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.state = 0.0
        self.steps = 0
        self.ctx = np.zeros((2,), dtype=np.float32)

    def set_sampling_dist(self, dist):
        self.sampling_dist = dist

    def _sample_ctx(self):
        if self.sampling_dist is None:
            return np.zeros((2,), dtype=np.float32)
        return np.asarray(self.sampling_dist.sample((1,)).detach().cpu().numpy()[0], dtype=np.float32)

    def _obs(self):
        return {
            "now": np.asarray([self.state, self.ctx[1]], dtype=np.float32),
            "forecast": np.asarray([[self.ctx[0], self.ctx[1]]], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = 0.0
        self.ctx = self._sample_ctx()
        return self._obs(), {"context": self.ctx.copy()}

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self.state = 0.5 * self.state + a + float(self.ctx[0])
        reward = -((a - float(self.ctx[1])) ** 2 + 0.1 * self.state * self.state)
        self.steps += 1
        return self._obs(), float(reward), False, self.steps >= self.horizon, {"context": self.ctx.copy()}


class ToyBatchEnv:
    def __init__(self, *, device="cpu", n_envs=1, horizon=4):
        self.device = torch.device(device)
        self.n_envs = int(n_envs)
        self.max_episode_length = int(horizon)
        self.ctx = torch.zeros((self.n_envs, 2), device=self.device)
        self.state = torch.zeros((self.n_envs,), device=self.device)
        self.steps = torch.zeros((self.n_envs,), dtype=torch.long, device=self.device)

    def set_ctx(self, ctx_batch):
        self.ctx = torch.as_tensor(ctx_batch, device=self.device, dtype=torch.float32).reshape(self.n_envs, 2)

    def _obs(self):
        return {"now": torch.stack([self.state, self.ctx[:, 1]], dim=1), "forecast": self.ctx[:, None, :]}

    def reset(self):
        self.state = self.ctx[:, 0].clone()
        self.steps.zero_()
        return self._obs()

    def step(self, action):
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.n_envs)
        reward = -((a - self.ctx[:, 1]) ** 2 + 0.1 * self.state.square())
        bonus = 0.05 * self.ctx[:, 0]
        self.state = 0.5 * self.state + a + self.ctx[:, 0]
        self.steps += 1
        done = self.steps >= self.max_episode_length
        return self._obs(), reward, done, {"adr_bonus": bonus}


class ToyBackend:
    def flow_bounds(self, device):
        return (
            torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device),
            torch.tensor([1.0, 1.0], dtype=torch.float32, device=device),
        )

    def make_train_env(self, *, sampling_dist, env_id, rollout_dir, plot_every_episodes):
        return Monitor(ToyTrainEnv(sampling_dist=sampling_dist))

    def make_adr_env(self, *, device, n_envs):
        return ToyBatchEnv(device=device, n_envs=n_envs)

    def policy_spec(self):
        return PolicySpec(policy="MultiInputLstmPolicy", policy_kwargs={})


def test_core_adr_smoke():
    backend = ToyBackend()
    venv = DummyVecEnv([lambda: backend.make_train_env(sampling_dist=None, env_id=0, rollout_dir=None, plot_every_episodes=0)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = RecurrentPPO("MultiInputLstmPolicy", venv, n_steps=8, batch_size=4, n_epochs=1, learning_rate=1e-4, verbose=0, device="cpu")
    model.learn(total_timesteps=16)

    adr = ADRFlows(backend, device="cpu", n_sample=2, iters=1, refine_steps=0, kl_M=4, surprise_coef=0.1)
    adr.set_policy(model, obs_norm=vecnorm_stats(venv))
    stats = adr.update()
    assert np.isfinite(stats["obj_mean"])
    assert np.isfinite(stats["ret_mean"])
    assert np.isfinite(stats["bonus_mean"])
    assert np.isfinite(stats["loss_fit"])
    venv.close()


def test_core_adr_set_policy_reuses_evaluator():
    backend = ToyBackend()
    venv = DummyVecEnv([lambda: backend.make_train_env(sampling_dist=None, env_id=0, rollout_dir=None, plot_every_episodes=0)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = RecurrentPPO("MultiInputLstmPolicy", venv, n_steps=8, batch_size=4, n_epochs=1, learning_rate=1e-4, verbose=0, device="cpu")
    adr = ADRFlows(backend, device="cpu", n_sample=2, iters=1, refine_steps=0, kl_M=4, surprise_coef=0.1)
    obs_norm = vecnorm_stats(venv)
    obs = backend.make_adr_env(device="cpu", n_envs=2).reset()
    adr.set_policy(model, obs_norm=obs_norm)
    ev0 = adr.ev
    ev0.reset(2)
    act0 = ev0.act(obs)
    adr.set_policy(model, obs_norm=obs_norm)
    assert adr.ev is ev0
    adr.ev.reset(2)
    act1 = adr.ev.act(obs)
    assert torch.allclose(act0, act1)
    venv.close()


def test_core_adr_get_train_dist_on_cpu():
    backend = ToyBackend()
    adr = ADRFlows(backend, device="cpu", n_sample=2, iters=1, refine_steps=0, kl_M=4, surprise_coef=0.1)
    train_dist = adr.get_train_dist_on("cpu")
    assert train_dist.low.device.type == "cpu"
    assert train_dist is adr.current


def test_core_training_smoke_and_flow_load(tmp_path):
    backend = ToyBackend()
    low, high = backend.flow_bounds("cpu")
    init_path = tmp_path / "init_flow.pt"
    torch.save(NormFlowDist(low, high, device="cpu").state_dict(), init_path)
    out = run_core_training(
        backend,
        {
            "save_dir": str(tmp_path / "core_run"),
            "init_flow_path": str(init_path),
            "n_envs": 1,
            "total_timesteps": 16,
            "save_every_steps": 8,
            "plot_every_episodes": 0,
            "ppo": {"n_steps": 8, "batch_size": 4, "n_epochs": 1, "learning_rate": 1e-4, "verbose": 0},
            "adr": {"n_sample": 2, "iters": 1, "refine_steps": 0, "kl_M": 4, "surprise_coef": 0.1, "update_every_episodes": 1},
        },
    )
    assert Path(out["model_path"]).exists()
    assert Path(out["vecnorm_path"]).exists()
    assert Path(out["flow_path"]).exists()


def test_core_flow_load_legacy_base_keys():
    backend = ToyBackend()
    low, high = backend.flow_bounds("cpu")
    dist = NormFlowDist(low, high, device="cpu")
    state = dist.state_dict()
    flow = dict(state["flow"])
    flow["base._0"] = flow.pop("base.loc")
    flow["base._1"] = flow.pop("base.scale")
    dist.load_state_dict({**state, "flow": flow})


def test_core_has_no_backend_imports():
    core_dir = Path(__file__).resolve().parents[1] / "core"
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("NOMAD_RC5", "NOMAD_test1"))
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(("NOMAD_RC5", "NOMAD_test1"))
