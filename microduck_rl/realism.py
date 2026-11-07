"""Apply the servo-bus realism settings to a task config (see :mod:`microduck_rl.mdp.realism`).

Numbers (all per environment unless stated):

* joint positions: read lag 0--2 control steps, redrawn on 3 % of samples; 5 % dropped reads;
  encoder noise +-0.002 rad (HD1910 12-bit encoder: 0.0015 rad per count; official +-0.001);
* joint velocities: read lag 1--3 steps (official fixed 1); 5 % dropped reads; noise +-0.25 rad/s;
* IMU: lag 0--2 samples (official 0--1), misalignment up to 6 deg (official);
* servo targets: 3 % dropped writes, on top of the BAM 3--6 physics-step command delay;
* per-servo firmware gain 0.85--1.15 and friction 0.8--1.2 (replaces the per-robot 0.9--1.1 friction draw).

These are deployment-side effects of a Feetech serial bus on a replica, estimated rather than measured;
measure the real bus (read period jitter, failed-read rate) and tighten them when the numbers exist.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.hydra import PresetCfg, _preset_fields

from microduck_rl.mdp.realism import DropoutJointPositionActionCfg, dropout_obs, randomize_bam_per_joint

JOINT_POS_LAG = (0, 2)
JOINT_VEL_LAG = (1, 3)
LAG_HOLD_PROB = 0.97  # the lag is redrawn on ~3 % of samples: every ~0.7 s at 50 Hz
IMU_LAG = (0, 2)
READ_DROPOUT_PROB = 0.05
WRITE_DROPOUT_PROB = 0.03
JOINT_POS_NOISE_RAD = 0.002
KP_SCALE_RANGE = (0.85, 1.15)
FRICTION_SCALE_RANGE = (0.8, 1.2)


def _wrap_dropout(term, lag: tuple[int, int], noise=None):
    """Wrap an observation term (or each alternative of a preset) in :class:`dropout_obs` with a varying read lag."""
    if isinstance(term, PresetCfg):
        return preset(**{name: _wrap_dropout(alt, lag, noise) for name, alt in _preset_fields(term).items()})
    changes = dict(
        func=dropout_obs,
        params={"func": term.func, "params": term.params, "dropout_prob": READ_DROPOUT_PROB},
        delay_min_lag=lag[0],
        delay_max_lag=lag[1],
        delay_hold_prob=LAG_HOLD_PROB,
    )
    if noise is not None:
        changes["noise"] = noise
    return term.replace(**changes)


def apply_servo_realism(cfg) -> None:
    """Modify a MicroDuck env config in place (actor observations, action term, reset events)."""
    policy = cfg.observations.policy
    policy.joint_pos = _wrap_dropout(
        policy.joint_pos, JOINT_POS_LAG, noise=Unoise(n_min=-JOINT_POS_NOISE_RAD, n_max=JOINT_POS_NOISE_RAD)
    )
    policy.joint_vel = _wrap_dropout(policy.joint_vel, JOINT_VEL_LAG)
    policy.imu.params["min_lag"], policy.imu.params["max_lag"] = IMU_LAG
    cfg.actions.joint_pos = DropoutJointPositionActionCfg(
        **{k: v for k, v in cfg.actions.joint_pos.to_dict().items() if k != "class_type"},
        dropout_prob=WRITE_DROPOUT_PROB,
    )
    cfg.events.randomize_joint_friction = EventTerm(
        func=randomize_bam_per_joint,
        mode="reset",
        params={"kp_scale_range": KP_SCALE_RANGE, "friction_scale_range": FRICTION_SCALE_RANGE},
    )
