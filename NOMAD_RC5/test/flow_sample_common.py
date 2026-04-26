from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from NOMAD.core.adr import NormFlowDist


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "20000000"


def load_flow(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    dist = NormFlowDist(
        state["low"],
        state["high"],
        transforms=int(state.get("transforms", 3)),
        bins=int(state.get("bins", 8)),
        hidden=tuple(state.get("hidden", (64, 64))),
        device=device,
    )
    return dist.load_state_dict(state)


def plot_mean_cum_reward(rewards, step_period, path, *, label="mean"):
    rewards = np.asarray(rewards, dtype=np.float32)
    cum = np.cumsum(rewards, axis=0)
    days = np.arange(1, cum.shape[0] + 1) * float(step_period) / 86400.0
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180, constrained_layout=True)
    ax.plot(days, cum, color="0.7", linewidth=0.8, alpha=0.45)
    ax.plot(days, cum.mean(axis=1), color="black", linewidth=2.0, label=label)
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return days, cum


def save_summary(path, *, method, step_period, rewards, temp_days, tz_c):
    rewards = np.asarray(rewards, dtype=np.float32)
    days = np.arange(1, rewards.shape[0] + 1) * float(step_period) / 86400.0
    np.savez(
        path,
        method=str(method),
        reward_days=days,
        rewards=rewards,
        cum_rewards=np.cumsum(rewards, axis=0),
        temp_days=np.asarray(temp_days, dtype=np.float32),
        tz_c=np.asarray(tz_c, dtype=np.float32),
    )
