from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from NOMAD.core.utils import merge_dict
else:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    from NOMAD.core.utils import merge_dict

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs" / "sweep_gpu"
BASE_CFG = {"seed": 0}
GRID = {
    "adr.ret_coef": [1.0, 2.0],
    "adr.bonus_coef": [0.5, 1.0],
    "adr.surprise_coef": [2.5, 5.0],
    "adr.refine_steps": [0, 5],
    "adr.kl_M": [500, 1000, 2000],
    "adr.kl_beta": [100.0, 500.0, 1000.0],
}

# GRID = {
#     "adr.n_sample": [128, 256, 512],
#     "adr.temp_init": [0.5, 1.0, 2.0],
#     "ppo.batch_size": [128, 256],
#     "ppo.n_steps": [128, 256],
#     "env.max_episode_length": [24 * 7, 24 * 21],
# }


MAX_WORKERS = None


def _nested(key, value):
    out = value
    for part in reversed(key.split(".")):
        out = {part: out}
    return out


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _short_key(key):
    return key.split(".")[-1].replace("_coef", "").replace("_steps", "").replace("_", "")


def _short_value(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def _run_name(idx, overrides):
    parts = [f"{_short_key(key)}{_short_value(value)}" for key, value in overrides.items()]
    return f"run_{idx:03d}" + ("__" + "__".join(parts) if parts else "")


def _build_cfg(overrides, run_dir):
    cfg = merge_dict({}, BASE_CFG)
    for key, value in overrides.items():
        cfg = merge_dict(cfg, _nested(key, value))
    cfg["save_dir"] = str(run_dir)
    cfg = merge_dict(cfg, {"ppo": {"tensorboard_log": str(run_dir / "tb")}})
    return cfg


def _prepare_jobs():
    keys = list(GRID)
    combos = product(*(GRID[key] for key in keys))
    jobs = []
    for idx, combo in enumerate(combos):
        overrides = dict(zip(keys, combo))
        run_dir = RUNS_ROOT / _run_name(idx, overrides)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.json"
        cfg_path.write_text(json.dumps(_jsonable(_build_cfg(overrides, run_dir)), indent=2, sort_keys=True) + "\n")
        jobs.append((run_dir, cfg_path, run_dir / "train.log"))
    return jobs


def _run_job(job):
    run_dir, cfg_path, log_path = job
    with log_path.open("w") as log:
        proc = subprocess.run(
            [sys.executable, "-m", "NOMAD_RC5.training_gpu", str(cfg_path)],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode:
        raise RuntimeError(f"{run_dir.name} failed, see {log_path}")
    return run_dir


def main():
    jobs = _prepare_jobs()
    print(f"{len(jobs)} runs -> {RUNS_ROOT}", flush=True)
    workers = len(jobs) if MAX_WORKERS is None else int(MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        run_dirs = list(pool.map(_run_job, jobs))
    return run_dirs


if __name__ == "__main__":
    main()
