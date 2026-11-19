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
READ_DROPOUT_PROB = 0.005  # measured 2026-11-18: 0 missed sync reads in thousands of ticks (was 0.05)
WRITE_DROPOUT_PROB = 0.03
JOINT_POS_NOISE_RAD = 0.002
KP_SCALE_RANGE = (1.0, 1.6)  # firmware P 5-8 (BAM fit at P 5): the replicas' servo P register (decision 2026-11-17)
FRICTION_SCALE_RANGE = (0.8, 1.5)  # the high end is the stalled get-up on the real duck (2026-11-18)
JOINT_VEL_QUANT_RAD_S = 0.077  # the servo's speed register: 1 count = 50 ticks/s (fitted 42.7-49.6, 2026-11-18)
JOINT_VEL_NOISE_RAD_S = 0.04  # residual noise on top of the quantisation (was +-0.25 uniform)
CONTROL_PERIOD_RANGE_MS = (20.0, 40.0)  # loop measured 20.2-21.7 ms cool/hot and 30-38 ms throttled (user: 50 Hz down to 25 Hz)


def _wrap_dropout(term, lag: tuple[int, int], noise=None, quantization: float = 0.0):
    """Wrap an observation term (or each alternative of a preset) in :class:`dropout_obs` with a varying read lag."""
    if isinstance(term, PresetCfg):
        return preset(**{name: _wrap_dropout(alt, lag, noise, quantization) for name, alt in _preset_fields(term).items()})
    changes = dict(
        func=dropout_obs,
        params={"func": term.func, "params": term.params, "dropout_prob": READ_DROPOUT_PROB, "quantization": quantization},
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
    policy.joint_vel = _wrap_dropout(
        policy.joint_vel,
        JOINT_VEL_LAG,
        noise=Unoise(n_min=-JOINT_VEL_NOISE_RAD_S, n_max=JOINT_VEL_NOISE_RAD_S),
        quantization=JOINT_VEL_QUANT_RAD_S,
    )
    policy.imu.params["min_lag"], policy.imu.params["max_lag"] = IMU_LAG
    cfg.actions.joint_pos = DropoutJointPositionActionCfg(
        **{k: v for k, v in cfg.actions.joint_pos.to_dict().items() if k != "class_type"},
        dropout_prob=WRITE_DROPOUT_PROB,
        control_period_range_ms=CONTROL_PERIOD_RANGE_MS,
    )
    cfg.events.randomize_joint_friction = EventTerm(
        func=randomize_bam_per_joint,
        mode="reset",
        params={"kp_scale_range": KP_SCALE_RANGE, "friction_scale_range": FRICTION_SCALE_RANGE},
    )


def coerce_override_ranges(cfg) -> None:
    """Turn list-valued ranges that hydra overrides produce back into tuples (the Newton BAM actuator hashes
    ``vin_range`` and ``vin_drop_gain_range``; a list raises ``unhashable type``). Called from every ``play_mode``,
    which the evaluators run after ``parse_env_cfg`` has applied the overrides."""
    for group in getattr(cfg.scene.robot, "actuators", {}).values():
        for name in ("vin_range", "vin_drop_gain_range"):
            v = getattr(group, name, None)
            if isinstance(v, list):
                setattr(group, name, tuple(v))
    act = getattr(cfg.actions, "joint_pos", None)
    v = getattr(act, "control_period_range_ms", None)
    if isinstance(v, list):
        act.control_period_range_ms = tuple(v)
