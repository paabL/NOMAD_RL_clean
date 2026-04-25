from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from NOMAD.core.training import DEFAULT_CFG as CORE_DEFAULT_CFG
    from NOMAD.core.training import run_training as run_core_training
    from NOMAD.core.utils import merge_dict
    from NOMAD_test1.backend import DEFAULT_ADR_CFG, DEFAULT_ENV_CFG, SwingBackend
else:
    from NOMAD.core.training import DEFAULT_CFG as CORE_DEFAULT_CFG
    from NOMAD.core.training import run_training as run_core_training
    from NOMAD.core.utils import merge_dict
    from .backend import DEFAULT_ADR_CFG, DEFAULT_ENV_CFG, SwingBackend

ROOT = Path(__file__).resolve().parent

DEFAULT_CFG = merge_dict(
    CORE_DEFAULT_CFG,
    {
        "save_dir": str(ROOT / "runs" / "default"),
        "plot_every_episodes": 100,
        "total_timesteps": 100_000,
        "save_every_steps": 10_000,
        "ppo": {"tensorboard_log": str(ROOT / "tensorboard_logs" / "tb")},
        "env": DEFAULT_ENV_CFG,
        "adr": DEFAULT_ADR_CFG,
    },
)


def run_training(cfg=None):
    cfg = merge_dict(DEFAULT_CFG, cfg or {})
    backend = SwingBackend(env_cfg=cfg["env"])
    core_cfg = {k: v for k, v in cfg.items() if k != "env"}
    core_cfg["adr"] = {k: cfg["adr"][k] for k in CORE_DEFAULT_CFG["adr"]}
    return run_core_training(backend, core_cfg)


def main():
    run_training()


if __name__ == "__main__":
    main()
