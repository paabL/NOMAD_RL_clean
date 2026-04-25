from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

from .adr import ADRFlows
from .utils import (
    _base_vec_env,
    _env_getattr,
    _env_has_method,
    build_initial_dist,
    build_ppo_kwargs,
    get_resume_paths,
    lr_schedule,
    lock_model_lr,
    merge_dict,
    resolve_resume_dir,
    set_global_seed,
    vecnorm_stats,
)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CFG = {
    "seed": 0,
    "device": "cpu",
    "adr_device": "mps",
    "n_envs": 8,
    "total_timesteps": 10_000_000,
    "save_every_steps": 100_000,
    "save_dir": str(ROOT / "runs" / "default"),
    "resume_dir": None,
    "plot_every_episodes": 100,
    "init_flow_path": None,
    "ppo": {
        "learning_rate_start": 1e-4, #default 1e-4
        "learning_rate_end": 5e-5, #default 5e-5
        "n_steps": 256, #default 128
        "batch_size": 256,
        "n_epochs": 5,
        "verbose": 1,
        "tensorboard_log": str(ROOT / "tensorboard_logs" / "tb"),
        "target_kl": 0.01, #default None
    },
    "vecnorm": {
        "norm_obs": True,
        "norm_reward": True,
        "clip_obs": 10.0,
    },
    "adr": {
        "transforms": 3,
        "bins": 8,
        "hidden": (64, 64),
        "iters": 50,
        "lr": 1e-3,
        "n_sample": 500,
        "refine_steps": 5,
        "refine_lr": 5e-3,
        "ess_bounds": (0.05, 0.2),
        "temp_bounds": (1e-3, 1e3),
        "ret_coef": 2.0,
        "bonus_coef": 1.0,
        "surprise_coef": 5.0, #default 10.0
        "kl_beta": 500, #defualt 20
        "kl_M": 1000, #nb de samples to estimate kl, default 1000
        "update_every_episodes": 100,
    },
}


class ADRUpdateCallback(BaseCallback):
    def __init__(self, adr, update_every_episodes=10, train_device="cpu", verbose=0):
        super().__init__(verbose)
        self.adr = adr
        self.update_every = int(update_every_episodes)
        self.train_device = torch.device(train_device)
        self.episodes_since_update = 0
        self.n_updates = 0

    @staticmethod
    def _fmt(stats, params_std_pct):
        return (
            f"[ADR #{stats['update_id']:03d}] "
            f"steps={stats['timesteps']} "
            f"episodes={stats['episodes']} "
            f"set={stats['set_policy_s']:.3f}s "
            f"upd={stats['update_s']:.2f}s "
            f"dt={stats['dt']:.2f}s | "
            f"ret={stats['ret_mean']:.3f} "
            f"bonus={stats['bonus_mean']:.3f} "
            f"obj={stats['obj_mean']:.3f} | "
            f"entropy={stats['entropy']:.3f} "
            f"ess={stats['ess']:.1f} "
            f"temp={stats['temp']:.3g} "
            f"wmax={stats['weight_max']:.3f} "
            f"flow_std={params_std_pct:.1f}%"
        )

    def _on_step(self):
        self.episodes_since_update += int(np.asarray(self.locals["dones"]).sum())
        if self.episodes_since_update < self.update_every:
            return True
        t0 = time.perf_counter()
        self.adr.set_policy(self.model, obs_norm=vecnorm_stats(self.training_env))
        set_policy_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        stats = self.adr.update()
        update_s = time.perf_counter() - t1
        try:
            self.training_env.env_method("set_sampling_dist", self.adr.get_train_dist_on(self.train_device))
        except Exception:
            pass
        self.logger.record("adr/set_policy_s", float(set_policy_s))
        self.logger.record("adr/update_s", float(update_s))
        for key, value in stats.items():
            self.logger.record(f"adr/{key}", float(value))
        params_std_pct = 0.0
        try:
            if not self.training_env.has_attr("get_rollout_std_pct_mean"):
                raise AttributeError
            vals = self.training_env.env_method("get_rollout_std_pct_mean")
            params_std_pct = float(np.mean(vals))
            self.logger.record("adr/params_std_pct_mean", params_std_pct)
        except Exception:
            pass
        self.n_updates += 1
        print(
            self._fmt(
                {
                    **stats,
                    "update_id": self.n_updates,
                    "timesteps": int(self.num_timesteps),
                    "episodes": int(self.episodes_since_update),
                    "set_policy_s": float(set_policy_s),
                    "update_s": float(update_s),
                },
                params_std_pct,
            ),
            flush=True,
        )
        self.episodes_since_update = 0
        return True


class PeriodicSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_dir, adr, verbose=0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_dir = Path(save_dir)
        self.adr = adr

    def _save(self, target):
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / "model.zip"
        vecnorm_path = target / "vecnormalize.pkl"
        flow_path = target / "adr_flow.pt"
        self.model.save(model_path)
        self.training_env.save(vecnorm_path)
        torch.save(self.adr.state_dict(), flow_path)

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            self._save(self.save_dir)
            self._save(self.save_dir / str(int(self.num_timesteps)))
        return True


class PeriodicPlotCallback(BaseCallback):
    def __init__(self, out_dir, every_episodes=0, verbose=0):
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.every = int(every_episodes)
        self.episodes = 0

    def _on_step(self):
        if self.every <= 0:
            return True
        dones = np.asarray(self.locals["dones"])
        done_idx = np.flatnonzero(dones)
        count = int(done_idx.size)
        if count == 0:
            return True
        self.episodes += count
        if self.episodes % self.every:
            return True
        self.out_dir.mkdir(parents=True, exist_ok=True)
        save_s = 0.0
        plot_s = 0.0
        try:
            if not self.training_env.has_attr("save_last_episode") or not self.training_env.has_attr("plot_last_episode"):
                return True
            for idx in done_idx.tolist():
                plot_path = self.out_dir / f"episode_{self.episodes:06d}_env{idx}.png"
                data_path = self.out_dir / f"episode_{self.episodes:06d}_env{idx}.npz"
                t0 = time.perf_counter()
                self.training_env.env_method("save_last_episode", path=data_path, indices=idx)
                save_s += time.perf_counter() - t0
                t0 = time.perf_counter()
                self.training_env.env_method("plot_last_episode", path=plot_path, indices=idx)
                plot_s += time.perf_counter() - t0
        except Exception:
            return True
        self.logger.record("plot/save_s", float(save_s))
        self.logger.record("plot/png_s", float(plot_s))
        return True


def run_training(backend, cfg=None):
    cfg = merge_dict(DEFAULT_CFG, cfg or {})
    save_dir = Path(cfg["save_dir"])
    rollout_dir = save_dir / "rollouts"
    resume = get_resume_paths(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(int(cfg["seed"]))
    device = cfg["device"]
    adr_device = cfg.get("adr_device") or device
    train_dist = build_initial_dist(cfg, device, backend)
    adr_dist = train_dist if torch.device(adr_device) == torch.device(device) else build_initial_dist(cfg, adr_device, backend)

    factories = [
        lambda i=i: backend.make_train_env(
            sampling_dist=train_dist,
            env_id=i,
            rollout_dir=rollout_dir,
            plot_every_episodes=int(cfg["plot_every_episodes"]),
        )
        for i in range(int(cfg["n_envs"]))
    ]
    #venv = DummyVecEnv(factories)
    base_venv = SubprocVecEnv(factories, start_method="spawn")
    venv = VecNormalize.load(str(resume["vecnorm"]), base_venv) if resume and resume["vecnorm"].exists() else VecNormalize(base_venv, **cfg["vecnorm"])

    policy_spec = backend.policy_spec()
    if resume and resume["model"].exists():
        print(f"Loading checkpoint from {resume['dir']}", flush=True)
        model = RecurrentPPO.load(resume["model"], env=venv, device=device)
        print(f"Resuming PPO with lr={lock_model_lr(model):.3e}", flush=True)
    else:
        model = RecurrentPPO(
            policy_spec.policy,
            venv,
            policy_kwargs=policy_spec.policy_kwargs,
            device=device,
            **build_ppo_kwargs(cfg),
        )
    # Initialize policy to have near-deterministic actions at the start of training, centered around zero
    # ==== MAY HARM EXPLORATION, USE WITH CAUTION ==== 
    # with torch.no_grad():
    #     model.policy.action_net.weight.fill_(0.0)
    #     model.policy.action_net.bias.fill_(0.0)
    #     if hasattr(model.policy, "log_std"):
    #         model.policy.log_std.data.fill_(0.0)

    adr = ADRFlows(
        backend,
        dist=adr_dist,
        device=adr_device,
        iters=cfg["adr"]["iters"],
        lr=cfg["adr"]["lr"],
        n_sample=cfg["adr"]["n_sample"],
        refine_steps=cfg["adr"]["refine_steps"],
        refine_lr=cfg["adr"]["refine_lr"],
        ess_bounds=cfg["adr"]["ess_bounds"],
        temp_bounds=cfg["adr"]["temp_bounds"],
        ret_coef=cfg["adr"]["ret_coef"],
        bonus_coef=cfg["adr"]["bonus_coef"],
        kl_beta=cfg["adr"]["kl_beta"],
        kl_M=cfg["adr"]["kl_M"],
        surprise_coef=cfg["adr"]["surprise_coef"],
    )

    callbacks = [
        ADRUpdateCallback(adr, cfg["adr"]["update_every_episodes"], train_device=device),
        PeriodicSaveCallback(cfg["save_every_steps"], save_dir, adr),
        PeriodicPlotCallback(rollout_dir, cfg["plot_every_episodes"]),
    ]
    model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=callbacks, reset_num_timesteps=not resume)

    model_path = save_dir / "model.zip"
    vecnorm_path = save_dir / "vecnormalize.pkl"
    flow_path = save_dir / "adr_flow.pt"
    model.save(model_path)
    venv.save(vecnorm_path)
    torch.save(adr.state_dict(), flow_path)
    venv.close()
    return {
        "save_dir": save_dir,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path,
        "flow_path": flow_path,
    }
