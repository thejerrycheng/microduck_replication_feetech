"""Forward roll over the head -- the duck's FRONT FLIP -- ported from Pollen's official
``microduck_roulade_env_cfg.py`` (attempt 3, run 5) onto the HD1910 replica and Isaac Lab / Newton.

Why this and not an aerial flip: the community's most careful attempt at a ballistic flip on the stock
XL330 duck (Lulzx/microduck-backflip, 90 experiments) never achieved an unassisted standing-start flip, and
the HD1910 is a slower servo (~12 rad/s no-load against the XL330's 20), so an aerial front flip is out of
reach of these legs. The roulade is the forward flip the real duck performs: it rolls over the flat top of
its head with the ground as the pivot and lands back on its feet. Measured on this plant: open-loop leg
extension from the deepest balanced crouch never leaves the ground (``docs/jump_ceiling*.json``).

Episodic policy, same 61 -> 14 interface as walking: a policy switch starts the roll (no phase clock, no
reference motion). The twist command is a near-zero noise, head and body commands are zero.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import _SERVO_JOINT_CFG, _TRUNK_BODY_CFG, _backlash_preset, _schedule

from microduck_rl import mdp
from microduck_rl.tasks.sitstand_env_cfg import (
    _HEAD_BODY_CFG,
    LEG_JOINT_INDICES,
    STAND_Z,
    EventsCfg as _SitStandEventsCfg,
    MicroDuckHD1910SitStandFlatEnvCfg,
    MicroDuckSitStandSceneCfg,
)
from microduck_rl.viz import play_visualizer_cfg

EPISODE_LENGTH_S = 5.0
"""A controlled roll ~2 s + rise ~1.5 s + settle (upstream run 3: 4 s left no room for the rise)."""

# Tuck anchor by servo index (string keys: Hydra rejects int keys). Upstream's crouch anchor plus the chin
# tuck (neck -1 / head +1 puts the flat head top on the floor). Same joint conventions as our HOME.
TUCK_OVERRIDES: dict[str, float] = {
    "2": -1.15,  # left hip_pitch
    "3": 1.25,  # left knee
    "4": 1.05,  # left ankle
    "5": -1.0,  # neck_pitch (chin tuck)
    "6": 1.0,  # head_pitch (chin tuck)
    "11": 1.15,  # right hip_pitch
    "12": -1.25,  # right knee
    "13": -1.05,  # right ankle
}
MIDROLL_PITCH = (math.radians(50.0), math.radians(340.0))
MIDROLL_OMEGA = (0.0, 3.0)
LANDING_GATE = (math.radians(260.0), math.radians(330.0))
RISE_GATE = (math.radians(180.0), math.radians(260.0))
GROUND_SHAPE = "/World/ground/terrain/GroundPlane/CollisionPlane"
"""Newton SHAPE label of the plane (the ground has no body, so filters must be shape expressions)."""
_ROBOT = {"asset_cfg": SceneEntityCfg("robot")}
_SERVOS = {"asset_cfg": _SERVO_JOINT_CFG}


@configclass
class MicroDuckRouladeSceneCfg(MicroDuckSitStandSceneCfg):
    """Sit/stand scene plus the two contact sensors the roll accounting reads (names are load-bearing)."""

    # Head pivot: the body that carries the head shell geometry, against the ground only.
    head_ground = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/jaw_soft",
        filter_shape_prim_expr=[GROUND_SHAPE],
    )
    # Support gate: any part of the robot on the ground (a roulade never leaves the floor).
    robot_ground = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*)?",
        filter_shape_prim_expr=[GROUND_SHAPE],
    )


@configclass
class CommandsCfg:
    """Near-zero twist noise and zero head/body commands: 61-wide observation parity, nothing to track."""

    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(EPISODE_LENGTH_S, 2 * EPISODE_LENGTH_S),
        rel_standing_envs=0.0,
        rel_forward_envs=0.0,
        rel_turn_in_place_envs=0.0,
        debug_vis=False,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(lin_vel_x=(-0.01, 0.01), lin_vel_y=(-0.01, 0.01), ang_vel_z=(-0.05, 0.05)),
    )
    head_pose = mdp.UniformPoseDeltaCommandCfg(resampling_time_range=(2.0, 5.0), ranges=((0.0, 0.0),) * 4)
    body_pose = mdp.UniformPoseDeltaCommandCfg(resampling_time_range=(2.0, 5.0), ranges=((0.0, 0.0),) * 6)


@configclass
class RewardsCfg:
    """Upstream's run-5 roulade stack; penalty functions that return <= 0 carry POSITIVE weights."""

    roulade_progress = RewTerm(func=mdp.roulade_progress, weight=8.0, params={**_ROBOT, "target_angle": 2 * math.pi, "max_paid_rate": 5.0})
    roulade_overspeed = RewTerm(func=mdp.roulade_overspeed_penalty, weight=-0.1, params={**_ROBOT, "omega_max": 7.0})
    roulade_head_pivot = RewTerm(
        func=mdp.roulade_head_pivot,
        weight=0.5,
        params={**_ROBOT, "angle_lo": math.radians(30.0), "angle_hi": math.radians(240.0), "rate_norm": 2.0},
    )
    roulade_landing_composite = RewTerm(
        func=mdp.roulade_landing_composite,
        weight=4.0,
        params={
            **_SERVOS,
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": LEG_JOINT_INDICES,
            "gate_lo": LANDING_GATE[0],
            "gate_hi": LANDING_GATE[1],
        },
    )
    roulade_upright_after_roll = RewTerm(
        func=mdp.roulade_upright_after_roll, weight=1.5, params={**_ROBOT, "gate_lo": LANDING_GATE[0], "gate_hi": LANDING_GATE[1]}
    )
    roulade_height_after_roll = RewTerm(
        func=mdp.roulade_height_after_roll,
        weight=1.0,
        params={**_ROBOT, "target_height": STAND_Z, "std": 0.04, "gate_lo": LANDING_GATE[0], "gate_hi": LANDING_GATE[1]},
    )
    roulade_landing_sharp = RewTerm(
        func=mdp.roulade_landing_sharp,
        weight=2.0,
        params={**_ROBOT, "target_height": STAND_Z, "height_std": 0.015, "upright_std": 0.3, "gate_lo": LANDING_GATE[0], "gate_hi": LANDING_GATE[1]},
    )
    roulade_stand_tax = RewTerm(
        func=mdp.roulade_stand_tax, weight=5.0, params={**_ROBOT, "target_height": STAND_Z, "gate_lo": LANDING_GATE[0], "gate_hi": LANDING_GATE[1]}
    )
    roulade_rise_velocity = RewTerm(
        func=mdp.roulade_rise_velocity, weight=0.75, params={**_ROBOT, "max_height": STAND_Z + 0.01, "gate_lo": RISE_GATE[0], "gate_hi": RISE_GATE[1]}
    )
    roulade_sagittal = RewTerm(func=mdp.roulade_sagittal_penalty, weight=-0.1, params=_ROBOT)
    roulade_lateral_vel = RewTerm(func=mdp.roulade_lateral_velocity_penalty, weight=-0.5, params=_ROBOT)
    roulade_flatness = RewTerm(func=mdp.roulade_flatness_penalty, weight=-0.5, params=_ROBOT)
    # Sim-to-real regularisers: motion blockers near zero during discovery, polish ramped in by curriculum.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=0.0, params={"asset_cfg": _SERVO_JOINT_CFG})
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.002, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.001)
    arrival_damping = RewTerm(
        func=mdp.body_ang_vel_at_height,
        weight=0.0,
        params={"asset_cfg": _TRUNK_BODY_CFG, "height_low": 0.09, "height_high": 0.11, "tilt_full_deg": 20.0, "tilt_zero_deg": 45.0},
    )
    gentle_landing = RewTerm(func=mdp.trunk_vertical_accel_penalty, weight=0.002, params={"asset_cfg": _TRUNK_BODY_CFG})
    dof_pos_limits = _backlash_preset(RewTerm(func=mdp.joint_pos_limits_finite, weight=-1.0), asset_cfg=_SERVO_JOINT_CFG)
    # Light: a tucked roll needs knees against the trunk.
    self_collisions = RewTerm(func=mdp.self_collision_cost, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("self_collision")})


@configclass
class TerminationsCfg:
    """Falling over is the task: NaN guard and time-out only."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.robot_state_is_nan, time_out=False, params={"sensor_names": ("contact_forces",)})


@configclass
class EventsCfg(_SitStandEventsCfg):
    """Sit/stand DR (replica mass, CoM, friction, servo, armature, encoder bias) with the roll's own spawns."""

    set_ground_state = None
    push_robot = None  # a push mid-roll is incoherent
    seated_head_jerk = None
    seated_tip = None
    set_roulade_state = EventTerm(
        func=mdp.reset_roulade_state,
        mode="reset",
        params={
            "asset_cfg": _SERVO_JOINT_CFG,
            "standing_prob": 0.5,
            "midroll_prob": 0.5,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
            "standing_tilt_max": math.radians(5.0),
            "forward_vel_range": (0.0, 0.0),
            "midroll_pitch_min": MIDROLL_PITCH[0],
            "midroll_pitch_max": MIDROLL_PITCH[1],
            "midroll_z_min": 0.05,
            "midroll_z_max": 0.10,
            "midroll_omega_range": MIDROLL_OMEGA,
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_factor_range": (0.3, 1.0),
            "joint_noise_std": 0.08,
        },
    )


@configclass
class CurriculumCfg:
    """Upstream's run-5 schedules (iterations of 24 steps)."""

    spawn_standing = _schedule("events.set_roulade_state.params.standing_prob", (3000, 0.65), (6000, 0.80))
    spawn_midroll = _schedule("events.set_roulade_state.params.midroll_prob", (3000, 0.35), (6000, 0.20))
    com_range = _schedule(
        "events.randomize_com.params.com_range",
        (500, {a: (-0.005, 0.005) for a in "xyz"}),
        (1000, {a: (-0.010, 0.010) for a in "xyz"}),
        (1500, {a: (-0.015, 0.015) for a in "xyz"}),
        (2500, {a: (-0.025, 0.025) for a in "xyz"}),
    )
    head_com_range = _schedule(
        "events.randomize_head_com.params.com_range",
        (500, {a: (-0.005, 0.005) for a in "xyz"}),
        (1000, {a: (-0.010, 0.010) for a in "xyz"}),
        (2000, {a: (-0.015, 0.015) for a in "xyz"}),
    )
    action_rate_weight = _schedule("rewards.action_rate_l2.weight", (1500, -0.2), (3000, -0.4))
    arrival_damping_weight = _schedule("rewards.arrival_damping.weight", (2500, -0.025), (3500, -0.05))
    torque_rate_weight = _schedule("rewards.joint_torque_rate_l2.weight", (2500, -5e-4), (3500, -1e-3))
    gentle_landing_weight = _schedule("rewards.gentle_landing.weight", (2500, 0.005))
    # Diagnostics logged as Curriculum/...: computed on the episodes ending at each reset.
    frontier_deg = CurrTerm(func=mdp.roulade_frontier_deg)
    completed_frac = CurrTerm(func=mdp.roulade_completed_frac)
    completed_standing_frac = CurrTerm(func=mdp.roulade_standing_frac)


@configclass
class MicroDuckHD1910RouladeFlatEnvCfg(MicroDuckHD1910SitStandFlatEnvCfg):
    """Forward roll over the head on flat ground, HD1910 servos, Newton MJWarp."""

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
        """Standing starts only: a clip shows the whole roll from the feet."""
        ManagerBasedRLEnvCfgPlay = super(MicroDuckHD1910SitStandFlatEnvCfg, self)  # skip the sit/stand hook (push_robot is gone)
        ManagerBasedRLEnvCfgPlay.play_mode()
        self.events.set_roulade_state.params["standing_prob"] = 1.0
        self.events.set_roulade_state.params["midroll_prob"] = 0.0
        self.curriculum.spawn_standing = None
        self.curriculum.spawn_midroll = None
        self.curriculum.frontier_deg = None
        self.curriculum.completed_frac = None
        self.curriculum.completed_standing_frac = None
        self.sim.visualizer_cfgs = play_visualizer_cfg()


@configclass
class MicroDuckHD1910RouladeHeadFlatEnvCfg(MicroDuckHD1910RouladeFlatEnvCfg):
    """Roll + stand STRAIGHT afterwards. The recipe scores only the trunk after the roll; checkpoint 6000 lands
    standing in 97 % of rolls with the neck/head still where the tuck put them. This adds a landing-gated term
    paying for the four head joints at their default pose (fine-tune from 6000, standing spawns only)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.roulade_head_home = RewTerm(
            func=mdp.roulade_head_home_after_roll,
            # std 0.25 paid 0.05/s at the 1.0 rad error the warm start lands with (exp(-16): no gradient); 0.8 reads
            # 0.21 there and still 0.86 at 0.3 rad, so the term pulls the head home from where the policy is.
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["neck_pitch", "head_pitch", "head_yaw", "head_roll"]),
                "gate_lo": LANDING_GATE[0],
                "gate_hi": LANDING_GATE[1],
                "std": 0.8,
            },
        )
        self.rewards.roulade_head_l1 = RewTerm(
            func=mdp.roulade_head_l1_after_roll,
            weight=-3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["neck_pitch", "head_pitch", "head_yaw", "head_roll"]),
                "gate_lo": LANDING_GATE[0],
                "gate_hi": LANDING_GATE[1],
            },
        )
        # the warm start already rolls from a stand: train on standing spawns from the first iteration
        self.events.set_roulade_state.params["standing_prob"] = 0.8
        self.curriculum.spawn_standing = None
