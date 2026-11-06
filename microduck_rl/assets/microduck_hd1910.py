"""MicroDuck replica with Feetech HD1910 servos (kissqy/Microduck-kissqy build).

Geometry, masses and inertias come from the official Pollen Robotics USDs shipped with
Isaac Lab (the Dynamixel XL330 build). The servo model is the Rhoban BAM ``m6`` fit of the
HD1910 published by LuwuDynamics/xgoduck_rl (``1910_m6.json``, Apache-2.0), the same file
``Ley-WL/microduck-rl`` and kissqy's training console use for their HD1910 tasks. The
firmware law is Feetech's (``bam.feetech.STS3215Actuator``): no current limiter, a 0.97
duty ceiling and an error gain of 0.166 duty/rad per firmware-P unit, scaled by the fit's
``error_gain_ratio``.

The mass of the replica is NOT modelled explicitly (HD1910 servos are 23 g against 18 g
for the XL330, and the HAT/IMU/compute are different parts in different places); that
difference is covered by the domain randomisation in :mod:`microduck_rl.dr`.
"""

from isaaclab.actuators import BamActuatorCfg, BamMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import clone, replace

from isaaclab_assets.robots.microduck import (  # noqa: F401  (re-exported for the tasks)
    MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG,
    MICRODUCK_ALLCOLLISIONS_CFG,
    MICRODUCK_BACKLASH_CFG,
    MICRODUCK_CFG,
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
)

__all__ = [
    "HD1910_BAM_FIT",
    "HD1910_MOTOR_CFG",
    "HD1910_SERVO_CFG",
    "MICRODUCK_HD1910_CFG",
    "MICRODUCK_HD1910_BACKLASH_CFG",
    "MICRODUCK_HD1910_ALLCOLLISIONS_CFG",
    "MICRODUCK_HD1910_ALLCOLLISIONS_BACKLASH_CFG",
    "MICRODUCK_JOINT_NAMES",
    "MICRODUCK_HEAD_JOINT_NAMES",
    "MICRODUCK_LEG_JOINT_NAMES",
    "with_hd1910_servos",
]

HD1910_BAM_FIT: dict[str, float | str] = {
    "kt": 0.6237611235393989,
    "error_gain_ratio": 1.321738701681577,
    "R": 4.910191564179625,
    "armature": 0.0017819049974475338,
    "q_offset": 0.032748109622048015,
    "command_delay": 0.006005265953265889,
    "friction_base": 0.015803365327573494,
    "friction_stribeck": 0.009590058942953754,
    "load_friction_motor": 0.06851344455161522,
    "load_friction_external": 0.15822186020023804,
    "load_friction_motor_stribeck": 0.12202539397513879,
    "load_friction_external_stribeck": 0.07862542939514451,
    "load_friction_motor_quad": 0.001967982142115035,
    "load_friction_external_quad": 0.0008854091803780388,
    "dtheta_stribeck": 0.946465692532126,
    "alpha": 1.5220506664931566,
    "friction_viscous": 0.0003101648214055195,
    "max_velocity": 100.0,
    "model": "m6",
    "actuator": "sts3215",
}
"""``1910_m6.json`` from LuwuDynamics/xgoduck_rl @ 326d77a1122870bdefa2c36403937502c958e69c, unchanged.

``q_offset`` is an identification bias and must not be copied into a zero table; ``command_delay``
is superseded by the 3--6 physics-step command queue below; ``max_velocity`` = 100 rad/s is the
upstream convention for disabling Feetech's target-position rate limiter (the Isaac Lab BAM drive
has no such limiter, which is the same thing).
"""

FEETECH_ERROR_GAIN_PER_P = 0.166
"""Rhoban BAM ``STS3215Actuator``: duty cycle per rad of error per unit of firmware P (oscilloscope)."""

FEETECH_MAX_PWM = 0.97
"""Rhoban BAM ``STS3215Actuator``: maximum duty cycle."""

HD1910_KP_FW = 5.0
"""Firmware P gain used by every HD1910 replica training setup (LuwuDynamics, Ley-WL, kissqy)."""

HD1910_VIN_RANGE = (7.4, 8.0)
HD1910_VIN_DROP_GAIN_RANGE = (0.0, 0.2)
HD1910_VIN_MIN = 7.0
"""Supply ranges from kissqy's training configs (2S Li-ion pack; ``vin_min`` 7.0 V, Ley-WL uses 4.0 V)."""

HD1910_MOTOR_CFG = BamMotorCfg(
    model="m6",
    kt=HD1910_BAM_FIT["kt"],
    resistance=HD1910_BAM_FIT["R"],
    error_gain=FEETECH_ERROR_GAIN_PER_P * HD1910_BAM_FIT["error_gain_ratio"],
    max_pwm=FEETECH_MAX_PWM,
    # The Feetech firmware model has no current limiter; zero disables it in the Newton drive.
    max_current=0.0,
    friction_base=HD1910_BAM_FIT["friction_base"],
    friction_viscous=HD1910_BAM_FIT["friction_viscous"],
    friction_stribeck=HD1910_BAM_FIT["friction_stribeck"],
    dtheta_stribeck=HD1910_BAM_FIT["dtheta_stribeck"],
    alpha=HD1910_BAM_FIT["alpha"],
    load_friction_motor=HD1910_BAM_FIT["load_friction_motor"],
    load_friction_external=HD1910_BAM_FIT["load_friction_external"],
    load_friction_motor_stribeck=HD1910_BAM_FIT["load_friction_motor_stribeck"],
    load_friction_external_stribeck=HD1910_BAM_FIT["load_friction_external_stribeck"],
    load_friction_motor_quad=HD1910_BAM_FIT["load_friction_motor_quad"],
    load_friction_external_quad=HD1910_BAM_FIT["load_friction_external_quad"],
)
"""HD1910 BAM m6 motor fit. Stall torque at 8.0 V: ``8.0 * kt / R`` = 1.02 N.m (the drive's effort limit)."""

HD1910_SERVO_CFG = BamActuatorCfg(
    joint_names_expr=["^(?!passive_).*"],
    motor=HD1910_MOTOR_CFG,
    kp_fw=HD1910_KP_FW,
    vin=HD1910_VIN_RANGE[0],
    vin_range=HD1910_VIN_RANGE,
    vin_drop_gain_range=HD1910_VIN_DROP_GAIN_RANGE,
    vin_min=HD1910_VIN_MIN,
    min_delay=3,
    max_delay=6,
    armature=HD1910_BAM_FIT["armature"],
)
"""All 14 HD1910 servos: Feetech firmware P control, battery sag, 3--6 physics-step command delay."""


def with_hd1910_servos(cfg: ArticulationCfg) -> ArticulationCfg:
    """Return ``cfg`` with its XL330 BAM servos replaced by the HD1910 fit, keeping everything else.

    The backlash variants keep their ``backlash_joint_template``; only the motor, firmware gain,
    supply, delay and armature change.
    """
    cfg = clone(cfg)
    servos = cfg.actuators["servos"]
    cfg.actuators["servos"] = replace(
        servos,
        motor=HD1910_MOTOR_CFG,
        kp_fw=HD1910_KP_FW,
        vin=HD1910_VIN_RANGE[0],
        vin_range=HD1910_VIN_RANGE,
        vin_drop_gain_range=HD1910_VIN_DROP_GAIN_RANGE,
        vin_min=HD1910_VIN_MIN,
        min_delay=3,
        max_delay=6,
        armature=HD1910_BAM_FIT["armature"],
    )
    return cfg


MICRODUCK_HD1910_CFG = with_hd1910_servos(MICRODUCK_CFG)
"""Walking collision set (soles only), HD1910 servos."""

MICRODUCK_HD1910_BACKLASH_CFG = with_hd1910_servos(MICRODUCK_BACKLASH_CFG)
"""Walking collision set with the +-1 deg gearbox-play hinges, HD1910 servos."""

MICRODUCK_HD1910_ALLCOLLISIONS_CFG = with_hd1910_servos(MICRODUCK_ALLCOLLISIONS_CFG)
"""Whole-body collisions (trunk, hips, head), HD1910 servos. Used by sit/stand."""

MICRODUCK_HD1910_ALLCOLLISIONS_BACKLASH_CFG = with_hd1910_servos(MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)
"""Whole-body collisions with gearbox play, HD1910 servos."""
