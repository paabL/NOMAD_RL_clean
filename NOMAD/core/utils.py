from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import importlib.util
import random

import numpy as np
import torch

from .adr import NormFlowDist


def merge_dict(base, extra):
    out = deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def vecnorm_stats(vecnorm):
    rms = vecnorm.obs_rms
    stats = {"clip_obs": float(vecnorm.clip_obs), "eps": float(getattr(vecnorm, "epsilon", 1e-8))}
    if isinstance(rms, dict):
        for key, value in rms.items():
            stats[f"{key}_mean"] = np.asarray(value.mean, dtype=np.float32)
            stats[f"{key}_var"] = np.asarray(value.var, dtype=np.float32)
    return stats


def lr_schedule(start, end):
    start = float(start)
    end = float(end)

    def schedule(progress_remaining):
        return end + (start - end) * float(progress_remaining)

    return schedule


def build_ppo_kwargs(cfg):
    ppo = dict(cfg["ppo"])
    start = ppo.pop("learning_rate_start", 1e-4)
    end = ppo.pop("learning_rate_end", start)
    if "learning_rate" not in ppo:
        ppo["learning_rate"] = lr_schedule(start, end)
    if ppo.get("tensorboard_log") and importlib.util.find_spec("tensorboard") is None:
        ppo.pop("tensorboard_log", None)
    return ppo


def resolve_resume_dir(path):
    path = Path(path)
    if path.is_file():
        path = path.parent
    best = path if (path / "model.zip").exists() else None
    for child in path.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "model.zip").exists():
            best_step = int(best.name) if best and best.name.isdigit() else -1
            if best is None or int(child.name) > best_step:
                best = child
    if best is None:
        raise FileNotFoundError(f"no checkpoint found in {path}")
    if best != path and (path / "model.zip").exists() and (path / "model.zip").stat().st_mtime > (best / "model.zip").stat().st_mtime:
        return path
    return best


def get_resume_paths(cfg):
    resume_dir = cfg.get("resume_dir")
    if not resume_dir:
        return None
    path = resolve_resume_dir(resume_dir)
    return {
        "dir": path,
        "model": path / "model.zip",
        "vecnorm": path / "vecnormalize.pkl",
        "flow": path / "adr_flow.pt",
    }


def lock_model_lr(model):
    lr = float(model.policy.optimizer.param_groups[0]["lr"])
    model.learning_rate = lr
    model.lr_schedule = lambda _: lr
    return lr


def build_initial_dist(cfg, device, backend):
    resume = get_resume_paths(cfg)
    init_flow_path = resume["flow"] if resume and resume["flow"].exists() else cfg.get("init_flow_path")
    if init_flow_path:
        path = Path(init_flow_path)
        if path.exists():
            state = torch.load(path, map_location=device, weights_only=False)
            low, high = (state["low"], state["high"]) if resume and path == resume["flow"] else backend.flow_bounds(device=device)
            dist = NormFlowDist(
                low,
                high,
                transforms=int(state.get("transforms", cfg["adr"]["transforms"])),
                bins=int(state.get("bins", cfg["adr"]["bins"])),
                hidden=tuple(state.get("hidden", cfg["adr"]["hidden"])),
                device=device,
            )
            dist.load_state_dict(state)
            print(f"Loading flow from {path}", flush=True)
            return dist
    low, high = backend.flow_bounds(device=device)
    return NormFlowDist(
        low,
        high,
        transforms=int(cfg["adr"]["transforms"]),
        bins=int(cfg["adr"]["bins"]),
        hidden=tuple(cfg["adr"]["hidden"]),
        device=device,
    )


def _base_vec_env(vec_env):
    return getattr(vec_env, "venv", vec_env)


def _env_getattr(vec_env, name):
    envs = getattr(_base_vec_env(vec_env), "envs", [])
    if not envs:
        return None
    env = envs[0]
    getter = getattr(env, "get_wrapper_attr", None)
    if getter is not None:
        try:
            return getter(name)
        except AttributeError:
            return None
    return getattr(env, name, None)


def _env_has_method(vec_env, name):
    return callable(_env_getattr(vec_env, name))


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
