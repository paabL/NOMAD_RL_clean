from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from NOMAD.simax.Controller import Controller_MPC
from NOMAD_RC5.backend import DEFAULT_ENV_CFG
from NOMAD_RC5.sim import LOWER_K, UPPER_K, build_rc5_simulation, load_rc5_data
from NOMAD_RC5.test.env import NomadEnv, interval_reward_and_terms
from NOMAD_RC5.test.flow_sample_common import DEFAULT_RUN_DIR, load_flow, plot_mean_cum_reward, save_summary


def plot_sample(path, t_days, tz_c, rewards):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), dpi=180, constrained_layout=True)
    axs[0].plot(t_days, tz_c, color="darkorange", linewidth=1)
    axs[0].axhline(LOWER_K - 273.15, color="seagreen", linestyle="--", linewidth=1)
    axs[0].axhline(UPPER_K - 273.15, color="seagreen", linestyle="--", linewidth=1)
    axs[0].set_ylabel("Tz (degC)")
    axs[1].plot(np.cumsum(rewards), color="black", linewidth=1)
    axs[1].set_ylabel("Cum. reward")
    axs[1].set_xlabel("Step")
    fig.savefig(path)
    plt.close(fig)


def run_one(ctx, i, args, cfg, data, out_dir):
    env_cfg = {k: v for k, v in cfg.items() if k != "max_dev"}
    env = NomadEnv(include_ctx=True, env_id=i, rollout_dir=out_dir, **env_cfg)
    env.reset(seed=args.seed + i, options={"start_time_s": float(args.start_day) * 86400.0}, ctx=ctx)

    n = cfg["max_episode_length"] * env.step_n + 1
    idx0, idx1 = int(env.idx), int(env.idx) + n
    base = build_rc5_simulation(data, theta=env.theta, base_setpoint=cfg["base_setpoint"])
    sim = base.copy(
        time_grid=data.time[idx0:idx1],
        d={k: v[idx0:idx1] for k, v in base.d.items()},
        x0=env.x,
    )
    ctrl = Controller_MPC(
        sim=sim,
        window_size=min(n, int(args.horizon_hours * env.step_n) + 1),
        n=env.step_n,
        SetPoints=jnp.full((n,), cfg["base_setpoint"], dtype=jnp.float32),
        verbose=False,
    )
    t, y, _states, controls, *_ = sim.run_numpy(x0=env.x, controller=ctrl)
    php = y[:, 1] - np.abs(y[:, 2])
    rewards = []
    for k in range(cfg["max_episode_length"]):
        a, b = k * env.step_n, (k + 1) * env.step_n + 1
        rows = data.dist_matrix[idx0 + a : idx0 + b]
        reward, _ = interval_reward_and_terms(
            t_step_s=t[a:b],
            tz_seq_k=y[a:b, 0],
            lower_seq_k=np.full((b - a,), LOWER_K, dtype=np.float32),
            upper_seq_k=np.full((b - a,), UPPER_K, dtype=np.float32),
            occ_seq=rows[:, 2],
            php_w_seq=php[a:b],
            price_seq=rows[:, 4],
            delta_sat_seq=np.zeros((b - a,), dtype=np.float32),
            comfort_huber_k=cfg["comfort_huber_k"],
            w_energy=cfg["w_energy"],
            w_comfort=cfg["w_comfort"],
            w_sat=cfg["w_sat"],
        )
        rewards.append((reward - float(env.reward_ref[k])) / max(-float(env.reward_ref_N[k]), 1e-3))

    payload = {
        "context": ctx,
        "time_30s": np.asarray(t, dtype=np.float32),
        "tz_30s": y[:, 0].astype(np.float32),
        "u_30s": np.asarray(controls.get("oveHeaPumY_u", np.zeros_like(t)), dtype=np.float32),
        "php_30s": php.astype(np.float32),
        "reward_norm_rl": np.asarray(rewards, dtype=np.float32),
    }
    np.savez(out_dir / f"sample_{i:03d}.npz", **payload)
    plot_sample(out_dir / f"sample_{i:03d}.png", payload["time_30s"] / 86400.0, payload["tz_30s"] - 273.15, rewards)
    return np.asarray(rewards, dtype=np.float32), payload["time_30s"] / 86400.0, payload["tz_30s"] - 273.15


def run(args):
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "flow_sample_mpc"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    cfg = dict(DEFAULT_ENV_CFG)
    cfg.update(max_episode_length=max(1, int(round(args.days * 86400.0 / cfg["step_period"]))))
    ctx = load_flow(run_dir / "adr_flow.pt", device).sample((args.n_samples,)).detach().cpu().numpy().astype(np.float32)
    data = load_rc5_data()

    rewards, temp_days, tz_c = zip(*(run_one(c, i, args, cfg, data, out_dir) for i, c in enumerate(ctx)))
    rewards = np.stack(rewards, axis=1)
    tz_c = np.stack(tz_c, axis=1)
    days, cum = plot_mean_cum_reward(rewards, cfg["step_period"], out_dir / "mean_cumulative_reward.png", label="MPC")
    header = ",".join(["day", *[f"sample_{i:03d}" for i in range(args.n_samples)], "mean"])
    np.savetxt(out_dir / "cumulative_rewards.csv", np.column_stack([days, cum, cum.mean(axis=1)]), delimiter=",", header=header)
    save_summary(out_dir / "summary.npz", method="MPC", step_period=cfg["step_period"], rewards=rewards, temp_days=temp_days[0], tz_c=tz_c)
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR)
    p.add_argument("-n", "--n-samples", type=int, default=4)
    p.add_argument("--start-day", type=float, default=31.0)
    p.add_argument("--days", type=float, default=7.0)
    p.add_argument("--horizon-hours", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir")
    print(run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
