from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from NOMAD_RC5.backend import DEFAULT_ENV_CFG
from NOMAD_RC5.sim import BASE_SETPOINT, TZ_MAX_K, TZ_MIN_K
from NOMAD_RC5.test.flow_sample_common import DEFAULT_RUN_DIR, load_flow, plot_mean_cum_reward, save_summary
from NOMAD_RC5.test.env import NomadEnv


def normalize(obs, vecnorm):
    eps, clip = float(getattr(vecnorm, "epsilon", 1e-8)), float(vecnorm.clip_obs)
    return {
        k: np.clip((v - vecnorm.obs_rms[k].mean) / np.sqrt(vecnorm.obs_rms[k].var + eps), -clip, clip).astype(np.float32)
        for k, v in obs.items()
    }


def stack_obs(obs):
    return {k: np.stack([o[k] for o in obs]).astype(np.float32) for k in obs[0]}


def run(args):
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "flow_sample_NOMAD"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    flow = load_flow(run_dir / "adr_flow.pt", device)
    model = RecurrentPPO.load(
        run_dir / "model.zip",
        device=device,
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2},
    )
    model.policy.set_training_mode(False)
    with (run_dir / "vecnormalize.pkl").open("rb") as f:
        vecnorm = pickle.load(f)
    vecnorm.training = False

    cfg = dict(DEFAULT_ENV_CFG)
    cfg.update(max_episode_length=max(1, int(round(args.days * 86400.0 / cfg["step_period"]))))
    ctx = flow.sample((args.n_samples,)).detach().cpu().numpy().astype(np.float32)
    env_cfg = {k: v for k, v in cfg.items() if k != "max_dev"}
    envs = [NomadEnv(include_ctx=True, env_id=i, rollout_dir=out_dir, **env_cfg) for i in range(args.n_samples)]
    opts = {"start_time_s": float(args.start_day) * 86400.0}
    obs = [env.reset(seed=args.seed + i, options=opts, ctx=ctx[i])[0] for i, env in enumerate(envs)]

    state = None
    starts = np.ones((args.n_samples,), dtype=bool)
    rewards = []
    for _ in range(cfg["max_episode_length"]):
        action, state = model.predict(normalize(stack_obs(obs), vecnorm), state=state, episode_start=starts, deterministic=True)
        setpoints = np.clip(BASE_SETPOINT + cfg["max_dev"] * np.asarray(action).reshape(-1), TZ_MIN_K, TZ_MAX_K)
        starts[:] = False
        step_rewards, next_obs, done = [], [], []
        for env, setpoint in zip(envs, setpoints):
            o, r, term, trunc, _ = env.step(np.asarray([setpoint], dtype=np.float32))
            next_obs.append(o)
            step_rewards.append(r)
            done.append(term or trunc)
        obs = next_obs
        rewards.append(step_rewards)
        if any(done):
            break

    mean_path = out_dir / "mean_cumulative_reward.png"
    days, cum = plot_mean_cum_reward(rewards, cfg["step_period"], mean_path, label="NOMAD")
    header = ",".join(["day", *[f"sample_{i:03d}" for i in range(args.n_samples)], "mean"])
    np.savetxt(out_dir / "cumulative_rewards.csv", np.column_stack([days, cum, cum.mean(axis=1)]), delimiter=",", header=header)
    episodes = [env._episode_payload() for env in envs]
    save_summary(
        out_dir / "summary.npz",
        method="NOMAD",
        step_period=cfg["step_period"],
        rewards=rewards,
        temp_days=episodes[0]["time_30s"] / 86400.0,
        tz_c=np.stack([ep["tz_30s"] - 273.15 for ep in episodes], axis=1),
    )
    for i, env in enumerate(envs):
        env.save_last_episode(out_dir / f"sample_{i:03d}.npz")
        env.plot_last_episode(out_dir / f"sample_{i:03d}.png")
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR, help="folder containing model.zip, vecnormalize.pkl and adr_flow.pt")
    p.add_argument("-n", "--n-samples", type=int, default=4)
    p.add_argument("--start-day", type=float, default=31.0)
    p.add_argument("--days", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir")
    out_dir = run(p.parse_args(argv))
    print(out_dir)


if __name__ == "__main__":
    main()
