"""Sensor and servo-bus realism for a hobby-servo robot.

The official recipe models the IMU with a fixed misalignment and a 0--1 sample latency, the joint
velocity with a fixed 1-sample lag, and the servo command with a 3--6 physics-step delay. A Feetech
serial bus on a replica is less regular than that: reads arrive late by a varying amount, a read or a
write occasionally fails and the last value stands, and no two servos have exactly the same gain or
friction. These terms add those effects; they are observation/action corruptions and change no reward.

* :class:`dropout_obs` wraps any observation function: with probability ``dropout_prob`` per environment
  and step the previous reading is returned again (a failed bus read); on top of the configured
  ``delay_*`` lag of the term, which models a late read.
* :class:`DropoutJointPositionAction`: with probability ``dropout_prob`` per environment and step the
  servos keep their previous target (a failed bus write).
* :func:`randomize_bam_per_joint`: per-JOINT firmware-gain (``kp_scale``) and friction multipliers, so
  the two legs are never identical servos.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.actuators import BamActuatorCfg
from isaaclab.actuators.newton import read_group_parameter, write_group_parameter
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.mdp.actions import BiasedJointPositionAction, BiasedJointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg

__all__ = [
    "dropout_obs",
    "DropoutJointPositionAction",
    "DropoutJointPositionActionCfg",
    "randomize_bam_per_joint",
]


class dropout_obs(ManagerTermBase):
    """Return ``func(env, **params)``, or the previous reading with probability ``dropout_prob`` (per env, per step).

    ``params`` may hold :class:`SceneEntityCfg` entries; they are resolved here because the observation
    manager only resolves the wrapper's own top-level params. Instances of :class:`ManagerTermBase` are
    supported as ``func`` (they are constructed with the inner config).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        inner_params: dict[str, Any] = cfg.params.get("params", {})
        for value in inner_params.values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)
        func = cfg.params["func"]
        if isinstance(func, type) and issubclass(func, ManagerTermBase):
            inner_cfg = cfg.replace(func=func, params=inner_params)
            func = func(inner_cfg, env)
        self._func: Callable = func
        self._last: torch.Tensor | None = None
        self._pending_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._pending_reset[slice(None) if env_ids is None else env_ids] = True
        if hasattr(self._func, "reset"):
            self._func.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv, func, params: dict, dropout_prob: float) -> torch.Tensor:
        reading = self._func(env, **params)
        if self._last is None or dropout_prob <= 0.0:
            self._last = reading.clone()
            self._pending_reset.fill_(False)
            return reading
        dropped = torch.rand(env.num_envs, device=env.device) < dropout_prob
        dropped &= ~self._pending_reset  # a freshly reset environment always gets a real reading
        output = torch.where(dropped.unsqueeze(-1), self._last, reading)
        self._last = output.clone()
        self._pending_reset.fill_(False)
        return output


class DropoutJointPositionAction(BiasedJointPositionAction):
    """Encoder-bias-compensating joint position action whose bus write fails with probability ``dropout_prob``."""

    cfg: DropoutJointPositionActionCfg

    def __init__(self, cfg: DropoutJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._last_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._has_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._has_target[slice(None) if env_ids is None else env_ids] = False

    def apply_actions(self):
        target = self.processed_actions - self.encoder_bias
        if self.cfg.dropout_prob > 0.0:
            dropped = (torch.rand(self.num_envs, device=self.device) < self.cfg.dropout_prob) & self._has_target
            target = torch.where(dropped.unsqueeze(-1), self._last_target, target)
        self._last_target = target.clone()
        self._has_target[:] = True
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class DropoutJointPositionActionCfg(BiasedJointPositionActionCfg):
    """Configuration for :class:`DropoutJointPositionAction`."""

    class_type: type[DropoutJointPositionAction] = DropoutJointPositionAction
    dropout_prob: float = 0.0
    """Probability per environment and control step that the servos keep their previous target."""


def randomize_bam_per_joint(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    kp_scale_range: tuple[float, float] = (0.85, 1.15),
    friction_scale_range: tuple[float, float] = (0.8, 1.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample a firmware-gain and a friction multiplier PER SERVO (not per robot) for every BAM actuator group.

    The official ``randomize_bam_friction`` draws one friction multiplier per robot and broadcasts it to
    all servos; here each servo gets its own draw, so the two legs differ the way two real servos do.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    for name, actuator_cfg in asset.cfg.actuators.items():
        if not isinstance(actuator_cfg, BamActuatorCfg):
            continue
        num_joints = read_group_parameter(asset.actuators, name, "drive", "kp_scale").shape[1]
        kp = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(*kp_scale_range)
        fr = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(*friction_scale_range)
        write_group_parameter(asset.actuators, name, "drive", "kp_scale", values=kp, env_ids=env_ids)
        write_group_parameter(asset.actuators, name, "drive", "friction_scale", values=fr, env_ids=env_ids)
