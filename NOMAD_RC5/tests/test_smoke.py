from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from NOMAD_RC5.backend import DEFAULT_POLICY_CFG, RC5Backend
from NOMAD_RC5.env import NomadEnv, NormalizeAction, RC5TorchBatch, RC5TorchVecEnv, ResidualActionWrapper
from NOMAD_RC5.sim import BASE_SETPOINT, build_rc5_simulation, context_low_high, load_rc5_data
from NOMAD_RC5.training import DEFAULT_CFG, run_training
from NOMAD_RC5.training_gpu import run_training as run_training_gpu


def make_rc5_env(*, future_steps=12, warmup_steps=4, max_episode_length=4, include_ctx=True):
    env = NomadEnv(data=load_rc5_data(), future_steps=future_steps, warmup_steps=warmup_steps, max_episode_length=max_episode_length, include_ctx=include_ctx)
    env = ResidualActionWrapper(env, base_action=BASE_SETPOINT, max_dev=5.0)
    env = NormalizeAction(env)
    return env


def test_simax_rc5_smoke(tmp_path):
    data = load_rc5_data()
    sim = build_rc5_simulation(data)
    sim_small = sim.copy(time_grid=data.time[:60], d={k: v[:60] for k, v in data.dataset.d.items()})
    t, y, states, controls = sim_small.run()
    assert t.shape == (60,)
    assert y.shape == (60, 3)
    assert states.shape == (60, 5)
    assert "oveHeaPumY_u" in controls

    plot_path = tmp_path / "sim.png"
    sim_small.plot(path=plot_path)
    assert plot_path.exists()

    save_path = tmp_path / "sim.pkl"
    sim_small.save_simulation(save_path)
    with save_path.open("rb") as f:
        loaded = pickle.load(f)
    t2, y2, *_ = loaded.run()
    assert t2.shape == t.shape
    assert y2.shape == y.shape


def test_env_smoke(tmp_path):
    env = NomadEnv(data=load_rc5_data(), future_steps=12, warmup_steps=4, max_episode_length=2)
    obs, info = env.reset(seed=0)
    assert obs["now"].shape == (13,)
    assert obs["forecast"].shape == (12, 13)
    assert "context" in info

    for _ in range(2):
        obs, reward, term, trunc, info = env.step([BASE_SETPOINT])
        assert np.isfinite(reward)
        assert np.isfinite(info["reward_raw"])
        if term or trunc:
            break

    plot_path = env.plot_last_episode(tmp_path / "episode.png")
    save_path = env.save_last_episode(tmp_path / "episode.npz")
    assert plot_path.exists()
    assert save_path.exists()


def test_env_last_episode_survives_dummy_vec_env_reset(tmp_path):
    venv = DummyVecEnv(
        [lambda: NomadEnv(data=load_rc5_data(), future_steps=12, warmup_steps=4, max_episode_length=2, rollout_dir=tmp_path)]
    )
    venv.reset()
    for _ in range(2):
        _, _, done, _ = venv.step(np.array([[BASE_SETPOINT]], dtype=np.float32))
        if done[0]:
            break
    save_path = venv.env_method("save_last_episode", path=tmp_path / "episode.npz", indices=0)[0]
    plot_path = venv.env_method("plot_last_episode", path=tmp_path / "episode.png", indices=0)[0]
    arr = np.load(save_path)
    assert arr["reward_raw_rl"].shape == (2,)
    assert arr["time_30s"].size > 0
    assert Path(plot_path).exists()
    venv.close()


def test_training_rc5_smoke(tmp_path):
    out = run_training(
        {
            "save_dir": str(tmp_path / "run"),
            "n_envs": 1,
            "total_timesteps": 16,
            "save_every_steps": 8,
            "plot_every_episodes": 0,
            "ppo": {"n_steps": 8, "batch_size": 4, "n_epochs": 1, "learning_rate": 1e-4, "verbose": 0},
            "env": {"future_steps": 12, "warmup_steps": 4, "max_episode_length": 4},
            "adr": {"n_sample": 2, "iters": 1, "refine_steps": 0, "kl_M": 4, "surprise_coef": 0.1, "update_every_episodes": 1, "baseline_cs_coef": 1.0},
        }
    )
    assert Path(out["model_path"]).exists()
    assert Path(out["vecnorm_path"]).exists()
    assert Path(out["flow_path"]).exists()


def test_training_rc5_rollout_plot_smoke(tmp_path):
    out = run_training(
        {
            "save_dir": str(tmp_path / "run"),
            "n_envs": 1,
            "total_timesteps": 8,
            "save_every_steps": 8,
            "plot_every_episodes": 1,
            "ppo": {"n_steps": 4, "batch_size": 4, "n_epochs": 1, "learning_rate": 1e-4, "verbose": 0},
            "env": {"future_steps": 12, "warmup_steps": 4, "max_episode_length": 2},
            "adr": {"n_sample": 2, "iters": 1, "refine_steps": 0, "kl_M": 4, "surprise_coef": 0.1, "update_every_episodes": 1, "baseline_cs_coef": 1.0},
        }
    )
    rollout_dir = Path(out["save_dir"]) / "rollouts"
    npz_files = sorted(rollout_dir.glob("*.npz"))
    png_files = sorted(rollout_dir.glob("*.png"))
    assert npz_files
    assert png_files
    arr = np.load(npz_files[0])
    assert arr["reward_raw_rl"].shape == (2,)
    assert arr["time_30s"].size > 0


def test_training_rc5_default_devices():
    assert DEFAULT_CFG["device"] == "cpu"
    assert DEFAULT_CFG["plot_every_episodes"] == 100
    assert DEFAULT_CFG["adr_device"] == "cpu"


def test_policy_net_arch_cfg():
    spec = RC5Backend().policy_spec()
    assert spec.policy_kwargs["net_arch"] == {
        "pi": list(DEFAULT_POLICY_CFG["policy_hidden"]),
        "vf": list(DEFAULT_POLICY_CFG["value_hidden"]),
    }
    custom = RC5Backend(policy_cfg={"policy_hidden": (32,), "value_hidden": (48, 24)}).policy_spec()
    assert custom.policy_kwargs["net_arch"] == {"pi": [32], "vf": [48, 24]}


def test_torch_batch_contract():
    env = RC5TorchBatch(data=load_rc5_data(), n_envs=2, future_steps=12, max_episode_length=2)
    low, high = context_low_high(device="cpu")
    ctx = low + (high - low) * 0.5
    env.set_ctx(ctx.repeat(2, 1))
    obs = env.reset(start_hour=10)
    assert obs["now"].shape == (2, 13)
    assert obs["forecast"].shape == (2, 12, 13)
    obs, reward, done, info = env.step(np.zeros((2, 1), dtype=np.float32))
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["adr_bonus"].shape == (2,)
    assert np.isfinite(reward.detach().cpu().numpy()).all()
    assert "reward_ref" in info


def test_torch_vec_env_smoke():
    env = RC5TorchVecEnv(data=load_rc5_data(), device="cpu", n_envs=2, future_steps=12, max_episode_length=2)
    obs = env.reset()
    assert obs["now"].shape == (2, 13)
    assert obs["forecast"].shape == (2, 12, 13)
    assert obs["ctx"].shape == (2, context_low_high(device="cpu")[0].numel())
    obs, reward, done, info = env.step(np.zeros((2, 1), dtype=np.float32))
    assert obs["ctx"].shape == (2, context_low_high(device="cpu")[0].numel())
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert np.isfinite(reward).all()
    assert len(info) == 2
    env.close()


def test_torch_vec_env_partial_autoreset():
    env = RC5TorchVecEnv(data=load_rc5_data(), device="cpu", n_envs=2, future_steps=12, max_episode_length=2)
    env.reset()
    h_idx_prev = env.batch.h_idx.clone()
    env.batch.steps[0] = env.batch.max_episode_length - 1
    obs, reward, done, info = env.step(np.zeros((2, 1), dtype=np.float32))
    assert done[0]
    assert not done[1]
    assert "terminal_observation" in info[0]
    assert "terminal_observation" not in info[1]
    assert int(env.batch.steps[0].item()) == 0
    assert int(env.batch.steps[1].item()) == 1
    assert int(env.batch.h_idx[1].item()) == int(h_idx_prev[1].item()) + 1
    assert env.reset_infos[0]["start_hour"] == int(env.batch.h_idx[0].item())
    assert obs["now"].shape == (2, 13)
    assert reward.shape == (2,)
    env.close()


def test_training_rc5_gpu_smoke(tmp_path):
    out = run_training_gpu(
        {
            "device": "cpu",
            "adr_device": "cpu",
            "save_dir": str(tmp_path / "run_gpu"),
            "n_envs": 2,
            "total_timesteps": 16,
            "save_every_steps": 8,
            "ppo": {"n_steps": 8, "batch_size": 4, "n_epochs": 1, "learning_rate": 1e-4, "verbose": 0},
            "env": {"future_steps": 12, "max_episode_length": 4},
            "adr": {"n_sample": 2, "iters": 1, "refine_steps": 0, "kl_M": 4, "surprise_coef": 0.1, "update_every_episodes": 1, "baseline_cs_coef": 1.0},
        }
    )
    assert Path(out["model_path"]).exists()
    assert Path(out["vecnorm_path"]).exists()
    assert Path(out["flow_path"]).exists()
