"""One-shot standing jump for the HD1910 replica (see ``microduck_rl/mdp/jump.py`` for the recipe's provenance).

The target scale is NOT a cap: the visible-height frontier is uncapped and the durable-height jackpot is
logarithmic, so every extra millimetre stays valuable. The scale (12 mm) is the community's measured open-loop
ceiling on the XL330; this servo is slower, so the first training run IS the measurement of what the HD1910
can do -- read ``Curriculum/rise_mm`` and the deterministic evaluator before believing any number.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import _SERVO_JOINT_CFG, _TRUNK_BODY_CFG, _backlash_preset, _schedule

from microduck_rl import mdp
from microduck_rl.tasks.roulade_env_cfg import MicroDuckRouladeSceneCfg
from microduck_rl.tasks.sitstand_env_cfg import STAND_Z, EventsCfg as _SitStandEventsCfg, MicroDuckHD1910SitStandFlatEnvCfg
from microduck_rl.viz import play_visualizer_cfg

EPISODE_LENGTH_S = 3.5
LAUNCH_WINDOW_S = 0.75
VISIBLE_HEIGHT_SCALE_M = 0.012
LAUNCH_VELOCITY_SCALE_M_S = 0.60
# Foot-flat crouch (upstream's CPU MuJoCo feasibility sweep), servo-index keyed, same joint conventions as HOME.
CROUCH_OVERRIDES: dict[str, float] = {"2": 0.4188, "3": 1.3776, "4": 0.9588, "11": -0.4188, "12": -1.3776, "13": -0.9588}


@configclass
class CommandsCfg:
    """One-shot launch flag in the twist slot; head and body commands zero (61-wide parity)."""

    base_velocity = mdp.JumpOneShotCommandCfg(launch_window_s=LAUNCH_WINDOW_S, debug_vis=False)
    head_pose = mdp.UniformPoseDeltaCommandCfg(resampling_time_range=(2.0, 5.0), ranges=((0.0, 0.0),) * 4)
    body_pose = mdp.UniformPoseDeltaCommandCfg(resampling_time_range=(2.0, 5.0), ranges=((0.0, 0.0),) * 6)


@configclass
class RewardsCfg:
    jump_launch_velocity = RewTerm(func=mdp.jump_launch_velocity_progress, weight=2.0, params={"target_velocity": LAUNCH_VELOCITY_SCALE_M_S})
    jump_takeoff = RewTerm(func=mdp.jump_takeoff_bonus, weight=1.0)
    jump_visible_height = RewTerm(func=mdp.jump_visible_height_progress, weight=6.0, params={"target_height": VISIBLE_HEIGHT_SCALE_M})
    jump_airtime = RewTerm(func=mdp.jump_airtime_progress, weight=1.0, params={"target_airtime": 0.10})
    jump_touchdown = RewTerm(func=mdp.jump_touchdown_quality, weight=3.0)
    jump_recovery = RewTerm(func=mdp.jump_recovery_progress, weight=2.0, params={"stand_height": STAND_Z})
    jump_settle = RewTerm(func=mdp.jump_settle_progress, weight=5.0, params={"stand_height": STAND_Z})
    jump_stable_landing = RewTerm(func=mdp.jump_stable_landing_bonus, weight=6.0)
    jump_durable_landing = RewTerm(func=mdp.jump_durable_landing_bonus, weight=10.0)
    jump_durable_visible_height = RewTerm(func=mdp.jump_durable_visible_height_bonus, weight=24.0, params={"target_height": VISIBLE_HEIGHT_SCALE_M})
    failed_landing = RewTerm(func=mdp.jump_failed_recovery_cost, weight=-8.0)
    post_touchdown_pose = RewTerm(func=mdp.jump_post_touchdown_pose_cost, weight=-0.75)
    post_touchdown_head_pose = RewTerm(func=mdp.jump_post_touchdown_head_pose_cost, weight=-1.0)
    post_touchdown_stance_width = RewTerm(func=mdp.jump_post_touchdown_stance_width_cost, weight=-1.0)
    post_touchdown_motion = RewTerm(func=mdp.jump_post_touchdown_motion_cost, weight=-0.4)
    nonfoot_impact = RewTerm(func=mdp.jump_nonfoot_impact_cost, weight=-0.12)
    airborne_tilt = RewTerm(func=mdp.jump_airborne_tilt_cost, weight=-0.4)
    airborne_drift = RewTerm(func=mdp.jump_horizontal_drift_cost, weight=-0.5)
    dof_pos_limits = _backlash_preset(RewTerm(func=mdp.joint_pos_limits_finite, weight=-1.0), asset_cfg=_SERVO_JOINT_CFG)
    self_collisions = RewTerm(func=mdp.self_collision_cost, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("self_collision")})
    # Smoothness is introduced only after takeoff/landing exist (discovery lesson, all three jump references).
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=0.0, params={"asset_cfg": _SERVO_JOINT_CFG})
    gentle_landing = RewTerm(func=mdp.trunk_vertical_accel_penalty, weight=0.0, params={"asset_cfg": _TRUNK_BODY_CFG})


@configclass
class TerminationsCfg:
    """No success termination (upstream V4 hid delayed falls); failed recovery ends the episode with its cost."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.robot_state_is_nan, time_out=False, params={"sensor_names": ("contact_forces",)})
    jump_failed_recovery = DoneTerm(func=mdp.jump_failed_recovery, time_out=False)


@configclass
class EventsCfg(_SitStandEventsCfg):
    set_ground_state = None
    push_robot = None  # a push mid-flight is incoherent
    seated_head_jerk = None
    seated_tip = None
    set_jump_state = EventTerm(
        func=mdp.reset_jump_state,
        mode="reset",
        params={
            "asset_cfg": _SERVO_JOINT_CFG,
            "standing_prob": 0.35,
            "crouch_prob": 0.35,
            "descending_prob": 0.30,
            "standing_z_range": (0.112, 0.115),
            "crouch_z_range": (0.068, 0.071),
            "descending_z_range": (0.125, 0.150),
            "descending_vz_range": (-0.35, -0.05),
            "tilt_max_deg": 2.0,
            "crouch_overrides": CROUCH_OVERRIDES,
            "joint_noise_std": 0.03,
        },
    )


@configclass
class CurriculumCfg:
    """Iteration-based stand-ins for upstream's performance-gated schedules; conservative timing."""

    spawn_standing = _schedule("events.set_jump_state.params.standing_prob", (1500, 0.50), (3000, 0.65))
    spawn_crouch = _schedule("events.set_jump_state.params.crouch_prob", (1500, 0.40), (3000, 0.35))
    spawn_descending = _schedule("events.set_jump_state.params.descending_prob", (1500, 0.10), (3000, 0.0))
    com_range = _schedule(
        "events.randomize_com.params.com_range",
        (750, {a: (-0.005, 0.005) for a in "xyz"}),
        (1500, {a: (-0.010, 0.010) for a in "xyz"}),
        (2500, {a: (-0.015, 0.015) for a in "xyz"}),
        (3500, {a: (-0.025, 0.025) for a in "xyz"}),
    )
    head_com_range = _schedule(
        "events.randomize_head_com.params.com_range",
        (750, {a: (-0.005, 0.005) for a in "xyz"}),
        (1500, {a: (-0.010, 0.010) for a in "xyz"}),
        (2500, {a: (-0.015, 0.015) for a in "xyz"}),
    )
    action_rate_weight = _schedule("rewards.action_rate_l2.weight", (1000, -0.005), (2000, -0.01), (3000, -0.02))
    torque_rate_weight = _schedule("rewards.joint_torque_rate_l2.weight", (2500, -2.5e-4), (3500, -5e-4))
    gentle_landing_weight = _schedule("rewards.gentle_landing.weight", (2500, 0.001), (3500, 0.003))
    # Diagnostics on the ending episodes (Curriculum/...).
    valid_takeoff_rate = CurrTerm(func=mdp.jump_valid_takeoff_rate)
    rise_mm = CurrTerm(func=mdp.jump_rise_mm)
    airtime_ms = CurrTerm(func=mdp.jump_airtime_ms)
    stable_landing_rate = CurrTerm(func=mdp.jump_stable_landing_rate)
    durable_rate = CurrTerm(func=mdp.jump_durable_rate)
    body_contact_rate = CurrTerm(func=mdp.jump_body_contact_rate)


@configclass
class MicroDuckHD1910JumpFlatEnvCfg(MicroDuckHD1910SitStandFlatEnvCfg):
    """One-shot standing jump on flat ground, HD1910 servos, Newton MJWarp."""

    scene: MicroDuckRouladeSceneCfg = MicroDuckRouladeSceneCfg(num_envs=4096, env_spacing=2.0)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S
        self.scene.head_ground.update_period = self.sim.dt
        self.scene.robot_ground.update_period = self.sim.dt

    def play_mode(self):
        """Standing starts only, no spawn curriculum: a checkpoint is judged on the real task."""
        super(MicroDuckHD1910SitStandFlatEnvCfg, self).play_mode()
        p = self.events.set_jump_state.params
        p.update({"standing_prob": 1.0, "crouch_prob": 0.0, "descending_prob": 0.0, "standing_z_range": (STAND_Z, STAND_Z), "tilt_max_deg": 0.0, "joint_noise_std": 0.0})
        for name in ("spawn_standing", "spawn_crouch", "spawn_descending", "valid_takeoff_rate", "rise_mm", "airtime_ms", "stable_landing_rate", "durable_rate", "body_contact_rate"):
            setattr(self.curriculum, name, None)
        self.sim.visualizer_cfgs = play_visualizer_cfg()
