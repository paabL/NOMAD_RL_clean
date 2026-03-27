from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class PolicySpec:
    policy: str | type
    policy_kwargs: dict[str, Any]


class ADRBatchEnv(Protocol):
    max_episode_length: int

    def set_ctx(self, ctx_batch: torch.Tensor) -> None: ...

    def reset(self): ...

    def step(self, action): ...


class NomadBackend(Protocol):
    def flow_bounds(self, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]: ...

    def make_train_env(self, *, sampling_dist, env_id: int, rollout_dir: str | Path | None, plot_every_episodes: int): ...

    def make_adr_env(self, *, device: str | torch.device, n_envs: int) -> ADRBatchEnv: ...

    def policy_spec(self) -> PolicySpec: ...
