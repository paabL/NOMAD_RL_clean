from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from NOMAD.core.training import merge_dict, run_training as run_core_training, vecnorm_stats
    from NOMAD_RC5.backend import (
        ConvForecastTemporalFuseExtractor,
        DEFAULT_ADR_CFG,
        DEFAULT_ENV_CFG,
        DEFAULT_POLICY_CFG,
        RC5Backend,
        ValueCtxLstmPolicy,
    )
else:
    from NOMAD.core.training import merge_dict, run_training as run_core_training, vecnorm_stats
    from .backend import (
        ConvForecastTemporalFuseExtractor,
        DEFAULT_ADR_CFG,
        DEFAULT_ENV_CFG,
        DEFAULT_POLICY_CFG,
        RC5Backend,
        ValueCtxLstmPolicy,
    )

ROOT = Path(__file__).resolve().parent
LEGACY_FLOW_PATH = ROOT.parent / "last_chance_out_collapsed" / "flow.pt"

__all__ = [
    "ConvForecastTemporalFuseExtractor",
    "DEFAULT_CFG",
    "RC5Backend",
    "ValueCtxLstmPolicy",
    "run_training",
    "vecnorm_stats",
]

DEFAULT_CFG = {
    "seed": 0,
    "device": "cpu",
    "adr_device": "cpu",
    "n_envs": 8,
    "total_timesteps": 10_000_000,
    "save_every_steps": 100_000,
    "init_flow_path": str(LEGACY_FLOW_PATH),
    "save_dir": str(ROOT / "runs" / "default"),
    "plot_every_episodes": 100,
    "ppo": {
        "learning_rate_start": 5e-4,
        "learning_rate_end": 1e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 5,
        "verbose": 1,
        "tensorboard_log": str(ROOT / "tensorboard_logs" / "tb"),
    },
    "vecnorm": {
        "norm_obs": True,
        "norm_reward": True,
        "clip_obs": 10.0,
    },
    "env": DEFAULT_ENV_CFG,
    "policy": DEFAULT_POLICY_CFG,
    "adr": {
        "transforms": 3,
        "bins": 8,
        "hidden": (64, 64),
        "iters": 50,
        "lr": 1e-3,
        "n_sample": 500,
        "refine_steps": 5,
        "refine_lr": 5e-3,
        "temp_init": 1.0,
        "ret_coef": 2.0,
        "bonus_coef": 1.0,
        "surprise_coef": 5.0,
        "kl_beta": 500,
        "kl_M": 1000,
        "update_every_episodes": 100,
        **DEFAULT_ADR_CFG,
    },
}


def _parse_value(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_nested(cfg, key, value):
    keys = [part.replace("-", "_") for part in key.split(".")]
    node = cfg
    for part in keys[:-1]:
        node = node.setdefault(part, {})
    node[keys[-1]] = value


def _parse_cli_overrides(argv):
    cfg = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--"):
            raise SystemExit(f"unexpected argument: {arg}")
        key = arg[2:]
        if "=" in key:
            key, raw = key.split("=", 1)
        else:
            i += 1
            if i >= len(argv):
                raise SystemExit(f"missing value for --{key}")
            raw = argv[i]
        _set_nested(cfg, key, _parse_value(raw))
        i += 1
    return cfg


def run_training(cfg=None):
    cfg = merge_dict(DEFAULT_CFG, cfg or {})
    backend = RC5Backend(env_cfg=cfg["env"], policy_cfg=cfg["policy"], adr_cfg=cfg["adr"])
    core_cfg = {k: v for k, v in cfg.items() if k not in ("env", "policy")}
    return run_core_training(backend, core_cfg)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    run_training(_parse_cli_overrides(argv) if argv else None)


if __name__ == "__main__":
    main()
