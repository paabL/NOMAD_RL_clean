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
    from NOMAD.core.training import ADRUpdateCallback, PeriodicSaveCallback, build_initial_dist, build_ppo_kwargs, merge_dict, set_global_seed
    from NOMAD_RC5.backend import DEFAULT_ADR_CFG, DEFAULT_ENV_CFG, DEFAULT_POLICY_CFG, RC5Backend
    from NOMAD_RC5.env import RC5TorchVecEnv
    from NOMAD_RC5.training import DEFAULT_CFG as CPU_DEFAULT_CFG
else:
    from NOMAD.core.adr import ADRFlows
    from NOMAD.core.training import ADRUpdateCallback, PeriodicSaveCallback, build_initial_dist, build_ppo_kwargs, merge_dict, set_global_seed
    from .backend import DEFAULT_ADR_CFG, DEFAULT_ENV_CFG, DEFAULT_POLICY_CFG, RC5Backend
    from .env import RC5TorchVecEnv
    from .training import DEFAULT_CFG as CPU_DEFAULT_CFG

ROOT = Path(__file__).resolve().parent
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else DEFAULT_DEVICE
DEFAULT_N_ENVS = 64
DEFAULT_N_SAMPLE = 128

DEFAULT_CFG = merge_dict(
    CPU_DEFAULT_CFG,
    {
        "device": DEFAULT_DEVICE,
        "adr_device": DEFAULT_DEVICE,
        "n_envs": DEFAULT_N_ENVS,
        "save_dir": str(ROOT / "runs" / "gpu_default"),
        "plot_every_episodes": 0,
        "env": DEFAULT_ENV_CFG,
        "policy": DEFAULT_POLICY_CFG,
        "adr": {"n_sample": DEFAULT_N_SAMPLE, **DEFAULT_ADR_CFG},
    },
)


def _load_cfg(path):
    return json.loads(Path(path).read_text())


def run_training(cfg=None):
    cfg = merge_dict(DEFAULT_CFG, cfg or {})
    save_dir = Path(cfg["save_dir"])
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
        step_period=cfg["env"]["step_period"],
        future_steps=cfg["env"]["future_steps"],
        max_episode_length=cfg["env"]["max_episode_length"],
        base_setpoint=cfg["env"]["base_setpoint"],
        max_dev=cfg["env"]["max_dev"],
        tz_min=cfg["env"]["tz_min"],
        tz_max=cfg["env"]["tz_max"],
        w_energy=cfg["env"]["w_energy"],
        w_comfort=cfg["env"]["w_comfort"],
        comfort_huber_k=cfg["env"]["comfort_huber_k"],
        w_sat=cfg["env"]["w_sat"],
    )
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, **cfg["vecnorm"])

    policy_spec = backend.policy_spec()
    model = RecurrentPPO(
        policy_spec.policy,
        venv,
        policy_kwargs=policy_spec.policy_kwargs,
        device=device,
        **build_ppo_kwargs(cfg),
    )
    with torch.no_grad():
        model.policy.action_net.weight.fill_(0.0)
        model.policy.action_net.bias.fill_(0.0)
        if hasattr(model.policy, "log_std"):
            model.policy.log_std.data.fill_(0.0)

    adr = ADRFlows(
        backend,
        dist=adr_dist,
        device=adr_device,
        iters=cfg["adr"]["iters"],
        lr=cfg["adr"]["lr"],
        n_sample=cfg["adr"]["n_sample"],
        refine_steps=cfg["adr"]["refine_steps"],
        refine_lr=cfg["adr"]["refine_lr"],
        temp_init=cfg["adr"]["temp_init"],
        ret_coef=cfg["adr"]["ret_coef"],
        bonus_coef=cfg["adr"]["bonus_coef"],
        kl_beta=cfg["adr"]["kl_beta"],
        kl_M=cfg["adr"]["kl_M"],
        surprise_coef=cfg["adr"]["surprise_coef"],
    )
    callbacks = [
        ADRUpdateCallback(adr, cfg["adr"]["update_every_episodes"], train_device=device),
        PeriodicSaveCallback(cfg["save_every_steps"], save_dir, adr),
    ]
    model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=callbacks)

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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) > 1:
        raise SystemExit("usage: python -m NOMAD_RC5.training_gpu [config.json]")
    return run_training(_load_cfg(argv[0]) if argv else None)


if __name__ == "__main__":
    main()
