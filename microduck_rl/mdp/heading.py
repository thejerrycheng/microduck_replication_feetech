"""Heading hold, from Pollen's official 2 m sprint recipe (``microduck-challenges/sprint_2m/env.py``).

The runner is commanded straight ahead and is never told its heading, so nothing in the velocity
recipe prices a slow yaw drift (at the yaw-tracking std a 0.07 rad/s turn keeps 99 % of that reward);
the first official reference turned 3--8 deg/s into the lane wall. ``heading_hold`` prices the yaw
ANGLE against the heading at episode start, and ``heading_error`` gives the critic (only) that angle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg

__all__ = ["heading_hold", "heading_error", "trunk_yaw"]


def trunk_yaw(asset: Articulation) -> torch.Tensor:
    """World yaw of the root link [rad]; Newton stores quaternions as (x, y, z, w)."""
    x, y, z, w = asset.data.root_link_quat_w.torch.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class _HeadingReference(ManagerTermBase):
    """Shared machinery: remember each environment's yaw at episode start."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reference = torch.zeros(env.num_envs, device=env.device)
        self._pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # The reset events have already placed the robot when managers reset, but capture lazily on the
        # first read so the value is the settled post-reset yaw whatever the manager order.
        self._pending[slice(None) if env_ids is None else env_ids] = True

    def error(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        yaw = trunk_yaw(asset)
        if self._pending.any():
            self._reference = torch.where(self._pending, yaw, self._reference)
            self._pending.fill_(False)
        return _wrap(yaw - self._reference)


class heading_hold(_HeadingReference):
    """``exp(-(yaw error / std)^2)`` against the episode's starting heading."""

    def __call__(
        self, env: ManagerBasedRLEnv, std: float = 0.4, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        error = self.error(env, asset_cfg)
        return torch.nan_to_num(torch.exp(-((error / std) ** 2)), nan=0.0)


class heading_error(_HeadingReference):
    """Yaw error against the episode's starting heading [rad], shape (num_envs, 1). Critic only."""

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        return torch.nan_to_num(self.error(env, asset_cfg), nan=0.0).unsqueeze(-1)
