from __future__ import annotations

from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from NOMAD.core.training import DEFAULT_CFG as CORE_DEFAULT_CFG
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
    from NOMAD.core.training import DEFAULT_CFG as CORE_DEFAULT_CFG
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

DEFAULT_CFG = merge_dict(
    CORE_DEFAULT_CFG,
    {
        "device": "cpu",
        "adr_device": "cpu",
        "init_flow_path": str(LEGACY_FLOW_PATH),
        "save_dir": str(ROOT / "runs" / "default"),
        "ppo": {"tensorboard_log": str(ROOT / "tensorboard_logs" / "tb")},
        "env": DEFAULT_ENV_CFG,
        "policy": DEFAULT_POLICY_CFG,
        "adr": DEFAULT_ADR_CFG,
    },
)


def run_training(cfg=None):
    cfg = merge_dict(DEFAULT_CFG, cfg or {})
    backend = RC5Backend(env_cfg=cfg["env"], policy_cfg=cfg["policy"], adr_cfg=cfg["adr"])
    core_cfg = {k: v for k, v in cfg.items() if k not in ("env", "policy")}
    core_cfg["adr"] = {k: cfg["adr"][k] for k in CORE_DEFAULT_CFG["adr"]}
    return run_core_training(backend, core_cfg)


def main():
    run_training()


if __name__ == "__main__":
    main()
