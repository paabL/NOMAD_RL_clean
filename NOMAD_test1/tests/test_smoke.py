from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from NOMAD_test1 import SwingEnv, SwingTorchBatch, context_low_high, run_training


def test_env_smoke(tmp_path):
    env = SwingEnv(max_episode_length=2)
    obs, info = env.reset(seed=0)
    assert obs["now"].shape == (2,)
    assert info["context"].shape == (2,)

    for _ in range(2):
        obs, reward, term, trunc, info = env.step([0.0])
        assert obs["now"].shape == (2,)
        assert np.isfinite(reward)
        assert np.isfinite(info["reward_raw"])
        if term or trunc:
            break

    plot_path = env.plot_last_episode(tmp_path / "episode.png")
    save_path = env.save_last_episode(tmp_path / "episode.npz")
    assert plot_path.exists()
    assert save_path.exists()


def test_torch_batch_contract():
    env = SwingTorchBatch(n_envs=2, max_episode_length=2)
    low, high = context_low_high(device="cpu")
    ctx = low + 0.5 * (high - low)
    env.set_ctx(ctx.repeat(2, 1))
    obs = env.reset()
    assert obs["now"].shape == (2, 2)
    obs, reward, done, info = env.step(np.zeros((2, 1), dtype=np.float32))
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["adr_bonus"].shape == (2,)
    assert np.isfinite(reward.detach().cpu().numpy()).all()


def test_randomization_changes_transition():
    env = SwingTorchBatch(n_envs=2, max_episode_length=1)
    env.set_ctx(torch.tensor([[0.7, 0.02], [1.3, 0.25]], dtype=torch.float32))
    env.reset()
    obs, reward, _, _ = env.step(torch.zeros((2, 1), dtype=torch.float32))
    assert not torch.allclose(obs["now"][0], obs["now"][1])
    assert not torch.isclose(reward[0], reward[1])


def test_training_smoke(tmp_path):
    save_dir = tmp_path / "run"
    out = run_training(
        {
            "save_dir": str(save_dir),
            "n_envs": 1,
            "total_timesteps": 16,
            "save_every_steps": 8,
            "plot_every_episodes": 1,
            "ppo": {"n_steps": 8, "batch_size": 4, "n_epochs": 1, "learning_rate": 1e-4, "verbose": 0},
            "env": {"max_episode_length": 8},
            "adr": {"n_sample": 2, "iters": 1, "refine_steps": 0, "kl_M": 4, "surprise_coef": 0.1, "update_every_episodes": 1},
        }
    )
    assert Path(out["model_path"]).exists()
    assert Path(out["vecnorm_path"]).exists()
    assert Path(out["flow_path"]).exists()
    assert list((save_dir / "rollouts").glob("*.png"))
    assert list((save_dir / "rollouts").glob("*.npz"))
