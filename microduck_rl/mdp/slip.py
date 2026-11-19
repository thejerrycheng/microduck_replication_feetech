"""Foot-slip terms for slippery-floor training.

The official ``foot_slip`` charges the squared planar speed of every foot in contact at all times (weight -0.1):
a tax on normal stance dynamics (the ankle link rolls through every step) that reads ~0.05 m/s at the median on the
deployed walker. :func:`foot_slip_over_threshold` charges nothing below a threshold and the squared excess above
it, so it only bites when a foot actually slides. Calibrate the threshold against the contact-foot speed
distribution ``scripts/measure_foot_friction.py --policy`` prints (p50/p90/p99 and the fraction over 0.05/0.10/0.20).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _shape_names(model) -> list[str] | None:
    for attr in ("shape_key", "shape_label", "shape_name"):
        if hasattr(model, attr):
            return list(getattr(model, attr))
    return None


def set_sole_contact_priority(env: ManagerBasedEnv, env_ids: torch.Tensor | None, priority: int = 1, name_token: str = "/sole_") -> None:
    """Startup event: give the sole colliders MuJoCo contact priority ``priority`` over the floor.

    Newton's MuJoCo solver resolves a contact's friction as the MAX of the two geoms' friction unless their
    ``geom_priority`` differ, in which case the higher-priority geom's friction (and condim/solref/solimp) is used
    alone. The flat terrain is authored at friction 1.0, so without this the sole randomisation can only RAISE the
    contact friction above 1.0 (measured, ``docs/friction/mu_eff_velstand_asis.json``: every sole contact reads 1.0
    for sole draws 0.2-1.0). The USD route (``mjc:priority`` through ``collision_props``) does not reach these soles
    because the MicroDuck USDs instance them, so the priority is written straight into the solver's geom table,
    once, for the geoms whose shape path contains ``name_token``.
    """
    backend = env.sim.backend_manager.backend
    solver = backend.solver
    model = backend.model
    names = _shape_names(model)
    if names is None or not hasattr(solver, "mjw_model"):
        raise RuntimeError("set_sole_contact_priority needs the Newton MuJoCo solver and shape names")
    sole_shapes = {i for i, k in enumerate(names) if name_token in k}
    if not sole_shapes:
        raise RuntimeError(f"no shape path contains {name_token!r}")
    g2s = solver.mjc_geom_to_newton_shape.numpy()
    g2s = g2s[0] if g2s.ndim == 2 else g2s
    geoms = [g for g in range(len(g2s)) if int(g2s[g]) in sole_shapes]
    if not geoms:
        raise RuntimeError("sole shapes have no MuJoCo geoms")
    prio = wp.to_torch(solver.mjw_model.geom_priority)
    if prio.ndim == 2:
        prio[:, geoms] = priority
    else:
        prio[geoms] = priority
    mj = getattr(model, "mujoco", None)
    gp = getattr(mj, "geom_priority", None) if mj is not None else None
    if gp is not None:
        t = wp.to_torch(gp)
        t.view(-1)[sorted(sole_shapes)] = priority


def foot_slip_over_threshold(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold_m_s: float = 0.15,
) -> torch.Tensor:
    """Sum over feet in contact of ``max(0, |v_xy| - threshold)^2`` (m^2/s^2); zero for a foot that holds."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    speed = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    excess = (speed - threshold_m_s).clamp_min(0.0)
    cost = torch.sum(torch.square(excess) * in_contact, dim=1)
    return torch.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)
