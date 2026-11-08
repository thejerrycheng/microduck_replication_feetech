"""Get-up from any pose + robust standing hold for the HD1910 replica (port of Pollen's official StandUp,
``alpha_stand``, widened start distribution, pushes during the hold). See ``microduck_rl/mdp/standup.py``.

Runtime role: when the walking/running policy falls (trunk tilt past ~45 deg or trunk below 0.09 m), the
controller switches to this policy with an all-zero command; once upright and still it hands back.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import (
    _HEAD_JOINT_CFG,
    _HEAD_PLAY_JOINT_CFG,
    _SERVO_JOINT_CFG,
    _TRUNK_BODY_CFG,
    _backlash_preset,
    _schedule,
)

from microduck_rl import mdp
from microduck_rl.tasks.roulade_env_cfg import CommandsCfg as _RouladeCommandsCfg
from microduck_rl.tasks.sitstand_env_cfg import (
    LEG_JOINT_INDICES,
    SITTING_TARGET_OVERRIDES,
    STAND_Z,
    EventsCfg as _SitStandEventsCfg,
    MicroDuckHD1910SitStandFlatEnvCfg,
)
from microduck_rl.viz import play_visualizer_cfg

EPISODE_LENGTH_S = 6.0
_ROBOT = {"asset_cfg": SceneEntityCfg("robot")}
_LEGS = {"asset_cfg": _SERVO_JOINT_CFG, "joint_indices": LEG_JOINT_INDICES}


@configclass
class CommandsCfg(_RouladeCommandsCfg):
    """Twist noise, head pose live (ranges widened by curriculum, as upstream), body zero."""

    head_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0), ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
    )


@configclass
class RewardsCfg:
    """Upstream's stand-up stack at its /4 scale. ``*_l1`` and ``gentle_rise`` return <= 0: positive weights."""

    pose_stand_legs = RewTerm(func=mdp.pose_stand_legs, weight=2.0, params={**_LEGS, "std": 0.5})
    pose_stand_l1 = RewTerm(func=mdp.pose_stand_l1, weight=1.25, params=_LEGS)
    height_stand = RewTerm(func=mdp.height_stand_gaussian, weight=1.0, params={**_ROBOT, "target_height": STAND_Z, "std": 0.04})
    height_stand_sharp = RewTerm(func=mdp.height_stand_gaussian, weight=1.0, params={**_ROBOT, "target_height": STAND_Z, "std": 0.015})
    height_stand_l1 = RewTerm(func=mdp.height_stand_l1, weight=7.5, params={**_ROBOT, "target_height": STAND_Z})
    com_upward_velocity = RewTerm(func=mdp.com_upward_velocity, weight=0.75, params={**_ROBOT, "max_height": 0.125})
    gentle_rise = RewTerm(func=mdp.trunk_vertical_accel_penalty, weight=0.005, params={"asset_cfg": _TRUNK_BODY_CFG})
    arrival_damping = RewTerm(
        func=mdp.body_ang_vel_at_height,
        weight=0.0,
        params={"asset_cfg": _TRUNK_BODY_CFG, "height_low": 0.09, "height_high": 0.11, "tilt_full_deg": 20.0, "tilt_zero_deg": 45.0},
    )
    upright_linear = RewTerm(func=mdp.upright_linear, weight=1.5, params=_ROBOT)
    upright_sharp = RewTerm(func=mdp.upright_sharp, weight=1.5, params={**_ROBOT, "std": 0.3, "z_low": 0.06, "z_high": STAND_Z})
    standing_composite = RewTerm(
        func=mdp.standing_composite_score,
        weight=3.75,
        params={**_LEGS, "target_height": STAND_Z, "height_std": 0.04, "upright_std": 0.40, "pose_std": 0.40},
    )
    head_pose_tracking = _backlash_preset(
        RewTerm(func=mdp.head_pose_tracking, weight=0.75, params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG}),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )
    head_pose_bias = _backlash_preset(
        RewTerm(func=mdp.head_pose_bias_penalty, weight=0.0, params={"command_name": "head_pose", "tau_s": 1.0, "asset_cfg": _HEAD_JOINT_CFG}),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=0.0, params={"asset_cfg": _SERVO_JOINT_CFG})
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    dof_pos_limits = _backlash_preset(RewTerm(func=mdp.joint_pos_limits_finite, weight=-1.0), asset_cfg=_SERVO_JOINT_CFG)
    self_collisions = RewTerm(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("self_collision")})
    track = RewTerm(func=mdp.track_standup, weight=0.0, params=_ROBOT)  # bookkeeping only (returns zeros)


@configclass
class TerminationsCfg:
    """The robot starts fallen: NaN guard and time-out only."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.robot_state_is_nan, time_out=False, params={"sensor_names": ("contact_forces",)})


@configclass
class EventsCfg(_SitStandEventsCfg):
    """Sit/stand DR stack; the fallen-state spawner; pushes ramped by curriculum (robust hold)."""

    set_ground_state = None
    seated_head_jerk = None
    seated_tip = None
    set_fallen_state = EventTerm(
        func=mdp.reset_fallen_state,
        mode="reset",
        params={
            "asset_cfg": _SERVO_JOINT_CFG,
            "standing_prob": 0.40,
            "sitting_prob": 0.40,
            "prone_prob": 0.20,
            "supine_prob": 0.0,
            "random_prob": 0.0,
            "bank_prob": 0.0,
            "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std": 0.12,
            "tilt_noise_deg": 10.0,
            "ground_z_range": (0.05, 0.09),
            "standing_z_range": (0.11, 0.12),
            "face_up_roll_max_deg": 90.0,
            "random_joint_frac": 0.6,
            "random_z_range": (0.06, 0.10),
            "fall_bank_path": "docs/fall_bank.json",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


def _mix(stand, sit, prone, supine, rand, bank):
    return {"standing_prob": stand, "sitting_prob": sit, "prone_prob": prone, "supine_prob": supine, "random_prob": rand, "bank_prob": bank}


@configclass
class CurriculumCfg:
    """Upstream's easy->hard start mix, extended with the random-orientation and fall-bank families."""

    # Start-family mix (one schedule per key: a schedule on the whole params dict would drop asset_cfg).
    mix_standing = _schedule("events.set_fallen_state.params.standing_prob", (600, 0.25), (1500, 0.2), (2500, 0.15), (4000, 0.15), (6000, 0.15))
    mix_sitting = _schedule("events.set_fallen_state.params.sitting_prob", (600, 0.3), (1500, 0.25), (2500, 0.2), (4000, 0.15), (6000, 0.1))
    mix_prone = _schedule("events.set_fallen_state.params.prone_prob", (600, 0.35), (1500, 0.3), (2500, 0.25), (4000, 0.2), (6000, 0.15))
    mix_supine = _schedule("events.set_fallen_state.params.supine_prob", (600, 0.1), (1500, 0.25), (2500, 0.25), (4000, 0.2), (6000, 0.2))
    mix_random = _schedule("events.set_fallen_state.params.random_prob", (600, 0.0), (1500, 0.0), (2500, 0.1), (4000, 0.2), (6000, 0.25))
    mix_bank = _schedule("events.set_fallen_state.params.bank_prob", (600, 0.0), (1500, 0.0), (2500, 0.05), (4000, 0.1), (6000, 0.15))
    head_pose_range = _schedule(
        "commands.head_pose.ranges",
        (500, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
        (1000, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
        (1500, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.2, 0.2))),
        (2000, ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-0.31, 0.31))),
    )
    com_range = _schedule(
        "events.randomize_com.params.com_range",
        (500, {a: (-0.005, 0.005) for a in "xyz"}),
        (1000, {a: (-0.010, 0.010) for a in "xyz"}),
        (1500, {a: (-0.015, 0.015) for a in "xyz"}),
        (3000, {a: (-0.025, 0.025) for a in "xyz"}),
    )
    head_com_range = _schedule(
        "events.randomize_head_com.params.com_range",
        (500, {a: (-0.005, 0.005) for a in "xyz"}),
        (1000, {a: (-0.010, 0.010) for a in "xyz"}),
        (2500, {a: (-0.015, 0.015) for a in "xyz"}),
    )
    # Robust standing: the official ladder stops at +-0.3; this one keeps climbing once the get-up exists.
    push_magnitude = _schedule(
        "events.push_robot.params.velocity_range",
        (500, {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}),
        (1000, {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
        (3000, {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}),
        (5000, {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}),
        (7000, {"x": (-1.3, 1.3), "y": (-1.3, 1.3)}),
    )
    action_rate_weight = _schedule("rewards.action_rate_l2.weight", (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0))
    # Anti-violence terms only after the flips exist (upstream: timing, not magnitude).
    arrival_damping_weight = _schedule("rewards.arrival_damping.weight", (3000, -0.025), (4000, -0.05))
    head_pose_bias_weight = _schedule("rewards.head_pose_bias.weight", (3000, 0.5), (4000, 1.5))
    torque_rate_weight = _schedule("rewards.joint_torque_rate_l2.weight", (3000, -1e-3))
    height_sharp_weight = _schedule("rewards.height_stand_sharp.weight", (3000, 0.5), (4000, 0.2))
    upright_sharp_weight = _schedule("rewards.upright_sharp.weight", (3000, 1.0), (4000, 0.5))
    composite_weight = _schedule("rewards.standing_composite.weight", (3000, 2.5), (4000, 1.5))
    # Diagnostics (Curriculum/...): standing at episode end per start family, time to stand, peak |omega|.
    stood_prone = CurrTerm(func=mdp.stood_prone)
    stood_supine = CurrTerm(func=mdp.stood_supine)
    stood_random = CurrTerm(func=mdp.stood_random)
    stood_bank = CurrTerm(func=mdp.stood_bank)
    stood_sitting = CurrTerm(func=mdp.stood_sitting)
    still_standing = CurrTerm(func=mdp.still_standing)
    time_to_stand_s = CurrTerm(func=mdp.time_to_stand_s)
    peak_omega = CurrTerm(func=mdp.peak_omega)


@configclass
class MicroDuckHD1910StandUpFlatEnvCfg(MicroDuckHD1910SitStandFlatEnvCfg):
    """Get-up from any pose and hold a pushed stand, HD1910 servos, Newton MJWarp."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S

    def play_mode(self):
        """Evaluation picks its own start family; default to the hard ones, no pushes, no curriculum."""
        super(MicroDuckHD1910SitStandFlatEnvCfg, self).play_mode()
        self.events.push_robot = None
        self.events.set_fallen_state.params.update(_mix(0.0, 0.0, 0.3, 0.3, 0.4, 0.0))
        for name in ("mix_standing", "mix_sitting", "mix_prone", "mix_supine", "mix_random", "mix_bank", "push_magnitude", "stood_prone", "stood_supine", "stood_random", "stood_bank", "stood_sitting", "still_standing", "time_to_stand_s", "peak_omega"):
            setattr(self.curriculum, name, None)
        self.sim.visualizer_cfgs = play_visualizer_cfg()
