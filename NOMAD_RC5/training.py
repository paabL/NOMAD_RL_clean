from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from NOMAD.core.adr import ADRFlows
    from NOMAD.core.training import ADRUpdateCallback, MemoryTrimCallback, PeriodicSaveCallback
    from NOMAD.core.utils import build_initial_dist, build_ppo_kwargs, get_resume_paths, lock_model_lr, merge_dict, set_global_seed
    from NOMAD_RC5.backend import DEFAULT_ADR_CFG, DEFAULT_ENV_CFG, DEFAULT_POLICY_CFG, RC5Backend
    from NOMAD_RC5.env import RC5TorchVecEnv
else:
    from NOMAD.core.adr import ADRFlows
    from NOMAD.core.training import ADRUpdateCallback, MemoryTrimCallback, PeriodicSaveCallback
    from NOMAD.core.utils import build_initial_dist, build_ppo_kwargs, get_resume_paths, lock_model_lr, merge_dict, set_global_seed
    from .backend import DEFAULT_ADR_CFG, DEFAULT_ENV_CFG, DEFAULT_POLICY_CFG, RC5Backend
    from .env import RC5TorchVecEnv

ROOT = Path(__file__).resolve().parent
LEGACY_FLOW_PATH = ROOT.parent / "flows" / "collapsed_flow.pt"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else DEFAULT_DEVICE

DEFAULT_CFG = {
    "seed": 0,
    "device": DEFAULT_DEVICE,
    "adr_device": DEFAULT_DEVICE,
    "n_envs": 64,
    "total_timesteps": 20_000_000,
    "save_every_steps": 100_000,
    "resume_dir": None,
    "init_flow_path": str(LEGACY_FLOW_PATH),
    "save_dir": str(ROOT / "runs" / "default"),
    "ppo": {
        "learning_rate_start": 5e-4,
        "learning_rate_end": 1e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 5,
        "verbose": 1,
        "tensorboard_log": str(ROOT / "tensorboard_logs" / "tb"),
    },
    "vecnorm": {"norm_obs": True, "norm_reward": True, "clip_obs": 10.0},
    "env": DEFAULT_ENV_CFG,
    "policy": DEFAULT_POLICY_CFG,
    "adr": {
        "transforms": 3,
        "bins": 8,
        "hidden": (64, 64),
        "iters": 50,
        "lr": 1e-3,
        "n_sample": 512,
        "refine_steps": 5,
        "refine_lr": 5e-3,
        "ess_bounds": (0.05, 0.2),
        "temp_bounds": (1e-3, 1e3),
        "ret_coef": 2.0,
        "bonus_coef": 1.0,
        "surprise_coef": 5.0,
        "kl_beta": 500,
        "kl_M": 1000,
        "update_every_episodes": 100,
        **DEFAULT_ADR_CFG,
    },
}


def _load_cfg(path):
    return json.loads(Path(path).read_text())


def run_training(cfg=None):
    cfg = merge_dict(DEFAULT_CFG, cfg or {})
    save_dir = Path(cfg["save_dir"])
    resume = get_resume_paths(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(int(cfg["seed"]))
    device = cfg["device"]
    adr_device = cfg.get("adr_device") or device
    backend = RC5Backend(env_cfg=cfg["env"], policy_cfg=cfg["policy"], adr_cfg=cfg["adr"])
    train_dist = build_initial_dist(cfg, device, backend)
    adr_dist = train_dist if torch.device(adr_device) == torch.device(device) else build_initial_dist(cfg, adr_device, backend)

    venv = RC5TorchVecEnv(
        data=backend.data,
        sampling_dist=train_dist,
        device=device,
        n_envs=int(cfg["n_envs"]),
        **cfg["env"],
    )
    venv = VecMonitor(venv)
    venv = VecNormalize.load(str(resume["vecnorm"]), venv) if resume and resume["vecnorm"].exists() else VecNormalize(venv, **cfg["vecnorm"])

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
        MemoryTrimCallback(),
    ]
    model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=callbacks, reset_num_timesteps=not resume)

    model_path = save_dir / "model.zip"
    vecnorm_path = save_dir / "vecnormalize.pkl"
    flow_path = save_dir / "adr_flow.pt"
    model.save(model_path)
    venv.save(vecnorm_path)
    torch.save(adr.state_dict(), flow_path)
    venv.close()
    return {"save_dir": save_dir, "model_path": model_path, "vecnorm_path": vecnorm_path, "flow_path": flow_path}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) > 1:
        raise SystemExit("usage: python -m NOMAD_RC5.training [config.json]")
    return run_training(_load_cfg(argv[0]) if argv else None)


if __name__ == "__main__":
    main()
