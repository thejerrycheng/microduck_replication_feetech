"""Domain randomisation for the Feetech replica, on top of the official MicroDuck recipe.

The official task randomises the TRUNK mass by +-5 % and its CoM by up to +-15 mm, the
head CoM by up to +-10 mm. A replica built from the kissqy files differs from the modelled
robot in ways that are not in the USD:

* every servo is an HD1910 (23 g measured, 21 +-2 g datasheet) instead of an XL330 (18 g):
  +5 g on each of the 15 servo-bearing bodies, +75 g (~+7 %) on the whole robot;
* the compute (Radxa ZERO 3W), the custom HAT, the custom IMU board and the battery holder
  are different parts in different places, so the trunk and head CoM move by centimetres;
* every print is a different infill and a different filament.

Rather than guess those numbers, the ranges below are wide enough to contain them. They are
applied by :func:`apply_replica_dr` to every task in this package, so all three policies see
the same plant family. Final CoM ranges are reached through the same curriculum the official
task uses (widening every 500 iterations), only ending wider.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.microduck.flat_env_cfg import _TRUNK_BODY_CFG, _schedule

from microduck_rl import mdp

MASS_SCALE_RANGE: tuple[float, float] = (0.90, 1.25)
"""Per-body mass multiplier, every body but the trunk, sampled once per environment at start-up (official: trunk 0.95--1.05)."""

TRUNK_MASS_SCALE_RANGE: tuple[float, float] = (0.90, 1.60)
"""Trunk mass multiplier (179--318 g for the 199 g trunk): the real battery, board, HAT and wiring are heavier than the
modelled ones and have not been weighed (2026-11-18). Narrow it once the duck is on a scale."""

TRUNK_COM_STAGES_M: tuple[float, ...] = (0.005, 0.010, 0.015, 0.025)
"""Trunk CoM offset half-ranges reached at 500, 1000, 1500 and 2000 iterations (official ends at 0.015)."""

HEAD_COM_STAGES_M: tuple[float, ...] = (0.005, 0.010, 0.015)
"""Head-assembly CoM offset half-ranges reached at 500, 1000 and 1500 iterations (official ends at 0.010)."""

FOOT_FRICTION_RANGE: tuple[float, float] = (0.5, 1.3)
"""Superseded by :data:`FOOT_FRICTION_RANGE_WIDE` (2026-11-19); kept for the record of what the first policies drew
(which, with the floor at 1.0 and no sole priority, never produced a contact below 1.0)."""


def _com_dict(half_range: float) -> dict[str, tuple[float, float]]:
    return {axis: (-half_range, half_range) for axis in "xyz"}


def apply_replica_dr(cfg, first_stage_iteration: int = 500, stage_spacing: int = 500) -> None:
    """Widen the official mass/CoM/friction randomisation in place on a MicroDuck env config.

    Args:
        cfg: An env config with the official ``events`` and ``curriculum`` members
            (``mass_inertia``, ``randomize_com``, ``randomize_head_com``, ``foot_friction``).
        first_stage_iteration: Iteration at which the first CoM stage starts (flat: 500, rough: 1000).
        stage_spacing: Iterations between CoM stages.
    """
    trunk = _TRUNK_BODY_CFG.body_names if isinstance(_TRUNK_BODY_CFG.body_names, list) else [_TRUNK_BODY_CFG.body_names]
    cfg.events.mass_inertia.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[rf"^(?!{'|'.join(trunk)}$).*"])
    cfg.events.mass_inertia.params["mass_distribution_params"] = MASS_SCALE_RANGE
    cfg.events.battery_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=trunk), "mass_distribution_params": TRUNK_MASS_SCALE_RANGE, "operation": "scale", "recompute_inertia": True},
    )
    # the soles own the contact friction (the floor is authored at 1.0 and the solver takes the max otherwise) and
    # draw it wide, per sole, at every reset: docs/friction/NOTES.md
    cfg.events.sole_contact_priority = EventTerm(func=mdp.set_sole_contact_priority, mode="startup", params={"priority": 1})
    apply_wide_foot_friction(cfg, FOOT_FRICTION_RANGE_WIDE, per_reset=True)
    cfg.curriculum.com_range = _schedule(
        "events.randomize_com.params.com_range",
        *[
            (first_stage_iteration + i * stage_spacing, _com_dict(half_range))
            for i, half_range in enumerate(TRUNK_COM_STAGES_M)
        ],
    )
    cfg.curriculum.head_com_range = _schedule(
        "events.randomize_head_com.params.com_range",
        *[
            (first_stage_iteration + i * stage_spacing, _com_dict(half_range))
            for i, half_range in enumerate(HEAD_COM_STAGES_M)
        ],
    )


FOOT_FRICTION_RANGE_WIDE: tuple[float, float] = (0.25, 1.20)
"""Sole friction for slippery-floor training. Only meaningful with sole priority over the floor
(:func:`microduck_rl.mdp.set_sole_contact_priority`): without it the contact takes the floor's 1.0 whenever the
sole draws below 1.0, measured in ``docs/friction/mu_eff_velstand_asis.json``."""


def apply_wide_foot_friction(cfg, friction_range: tuple[float, float] = FOOT_FRICTION_RANGE_WIDE, per_reset: bool = True) -> None:
    """Draw the sole friction from ``friction_range`` for every sole shape, re-drawn at every reset when ``per_reset``."""
    cfg.events.foot_friction.params["static_friction_range"] = friction_range
    cfg.events.foot_friction.params["dynamic_friction_range"] = friction_range
    if per_reset:
        cfg.events.foot_friction.mode = "reset"
