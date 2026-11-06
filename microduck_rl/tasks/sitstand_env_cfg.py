"""Commanded sit <-> stand for the HD1910 replica, ported from ``microduck_sitstand_env_cfg.py`` (v1.5).

One posture-conditioned recipe serves three policies:

* ``SitStand``: the upstream task, command flips mid-episode, 50/50 seated/standing starts;
* ``Sit``: always starts standing and is always commanded SIT (stand -> sit, then hold);
* ``Stand``: always starts seated and is always commanded STAND (sit -> stand, then hold).

The policy interface is the walking one (61 observations: the posture flag sits in the twist slot,
head-pose command live, body-pose command zero), so the runtime drives it by writing 0/1 into the
``vx`` slot. Full-collision robot, no fall termination, gentleness enforced by descent/rise speed
caps and a vertical-shock penalty, a slewed target the rewards track (the anti-crash mechanism),
seated fall-back hardening (head jerks and backward kicks once the sit exists), and the walking
task's sim-to-real randomisation plus the replica's wider mass/CoM ranges.

Trunk heights (``STAND_Z``/``SIT_Z``) are upstream's measured MuJoCo rest heights; re-measure with
``scripts/measure_rest_heights.py`` before trusting them on a changed model.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.microduck.flat_env_cfg import (
    _FOOT_SENSOR_CFG,
    _HEAD_JOINT_CFG,
    _HEAD_PLAY_JOINT_CFG,
    _SERVO_JOINT_CFG,
    _TRUNK_BODY_CFG,
    MICRODUCK_FOOT_BODY_NAMES,
    ActionsCfg,
    ObservationsCfg,
    _backlash_preset,
    _schedule,
)
from isaaclab_tasks.utils import preset

from microduck_rl import mdp
from microduck_rl.assets import (
    MICRODUCK_HD1910_ALLCOLLISIONS_BACKLASH_CFG,
    MICRODUCK_HD1910_ALLCOLLISIONS_CFG,
    MICRODUCK_JOINT_NAMES,
)
from microduck_rl.dr import FOOT_FRICTION_RANGE, HEAD_COM_STAGES_M, MASS_SCALE_RANGE, TRUNK_COM_STAGES_M

# ── Keyframes (upstream, stability-verified 2026-07-27; verify TILT in sim before changing) ──
SITTING_TARGET_OVERRIDES: dict[str, float] = {
    "1": 0.0,  # left hip_roll (HOME -0.0873)
    "2": -0.4079,  # left hip_pitch (HOME -0.4579; +0.05 = slight forward lean)
    "3": 1.35,  # left knee (HOME -0.0049)
    "4": 0.0,  # left ankle (HOME +0.4530)
    "10": 0.0,  # right hip_roll
    "11": 0.4079,  # right hip_pitch
    "12": -1.35,  # right knee
    "13": 0.0,  # right ankle
}
"""SIT keyframe by servo index (string keys: Hydra/``class_to_dict`` reject int keys); neck/head omitted."""

LEG_JOINT_INDICES = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
"""Servo indices of the 10 leg joints in :data:`MICRODUCK_JOINT_NAMES` order."""

STAND_Z = 0.115
SIT_Z = 0.060
"""Trunk rest heights [m], measured upstream in MuJoCo."""
STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z = 0.075
POSTURE_RAMP_S = 2.0
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED = 0.08
EPISODE_LENGTH_S = 12.0
POSTURE_DWELL_S = (3.5, 6.5)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)

_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=["neck", "neck_pitch(_[0-9]+)?", "yaw_roll_motion", "jaw_soft"])
_ROBOT_PRESET = dict(
    default=MICRODUCK_HD1910_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    backlash=MICRODUCK_HD1910_ALLCOLLISIONS_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
)


@configclass
class MicroDuckSitStandSceneCfg(InteractiveSceneCfg):
    """Full-collision HD1910 robot on a plane, foot contacts and a left-leg/right-leg self-collision sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False,
    )
    robot = preset(**_ROBOT_PRESET)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )
    # Upstream penalises every robot-robot contact; here the left leg chain against the right one,
    # which is where a folding sit crosses itself.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll(/.*)?",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll(/.*)?"],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9))
    )


@configclass
class CommandsCfg:
    """Posture flag in the twist slot, live head pose, zero body pose (61-wide observation parity)."""

    base_velocity = mdp.SitStandCommandCfg(
        asset_name="robot",
        resampling_time_range=POSTURE_DWELL_S,
        sit_prob=0.5,
        ramp_s=POSTURE_RAMP_S,
        sit_z=SIT_Z,
        stand_z=STAND_Z,
        debug_vis=False,
    )
    head_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0), ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
    )
    body_pose = mdp.UniformPoseDeltaCommandCfg(resampling_time_range=(2.0, 5.0), ranges=((0.0, 0.0),) * 6)


_POSTURE = {"command_name": "base_velocity", "asset_cfg": _SERVO_JOINT_CFG}
_POSE = {**_POSTURE, "sit_overrides": SITTING_TARGET_OVERRIDES, "joint_indices": LEG_JOINT_INDICES}
_HEIGHT = {**_POSTURE, "sit_z": SIT_Z, "stand_z": STAND_Z}


@configclass
class RewardsCfg:
    """Upstream's posture-conditioned stack with its weights (penalty functions are self-negating)."""

    posture_pose_legs = RewTerm(func=mdp.posture_pose_match, weight=4.0, params={**_POSE, "std": 0.5})
    head_pose_tracking = _backlash_preset(
        RewTerm(
            func=mdp.head_pose_tracking,
            weight=0.75,
            params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
        ),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )
    posture_pose_l1 = RewTerm(func=mdp.posture_pose_l1, weight=1.0, params=_POSE)
    posture_height = RewTerm(func=mdp.posture_height_gaussian, weight=1.0, params={**_HEIGHT, "std": 0.04})
    posture_height_sharp = RewTerm(func=mdp.posture_height_gaussian, weight=1.0, params={**_HEIGHT, "std": 0.015})
    posture_height_l1 = RewTerm(func=mdp.posture_height_l1, weight=6.0, params=_HEIGHT)
    rise_bootstrap = RewTerm(
        func=mdp.posture_rise_bootstrap,
        weight=0.75,
        params={**_POSTURE, "max_height": 0.125, "max_vz": MAX_RISE_SPEED},
    )
    # Gentleness: these return <= 0, so the weights are positive (upstream run 7ev90yd9 had them negative
    # and trained a crash-sitting policy). Descent cap on from step 0, rise cap by curriculum.
    descent_speed = RewTerm(
        func=mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={"max_down_vel": MAX_DESCENT_SPEED, "asset_cfg": _TRUNK_BODY_CFG},
    )
    rise_speed = RewTerm(
        func=mdp.trunk_upward_velocity_penalty,
        weight=0.0,
        params={"max_up_vel": MAX_RISE_SPEED, "asset_cfg": _TRUNK_BODY_CFG},
    )
    gentle_motion = RewTerm(func=mdp.trunk_vertical_accel_penalty, weight=0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    upright_linear = RewTerm(func=mdp.body_upright_linear, weight=2.5, params={"asset_cfg": _TRUNK_BODY_CFG})
    upright_while_tall = RewTerm(
        func=mdp.upright_while_tall,
        weight=1.5,
        params={"height_low": SIT_UPRIGHT_Z, "height_high": STAND_UPRIGHT_Z, "asset_cfg": _TRUNK_BODY_CFG},
    )
    posture_stillness = RewTerm(
        func=mdp.posture_stillness,
        weight=2.0,
        params={
            **_HEIGHT,
            "band_full": 0.012,
            "band_zero": 0.03,
            "vel_std": 0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )
    posture_composite = RewTerm(
        func=mdp.posture_composite,
        weight=3.0,
        params={
            **_POSE,
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "head_cfg": _HEAD_JOINT_CFG,
            "height_std": 0.03,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "head_std": 0.40,
        },
    )
    # Sim-to-real regularisers at the walking task's weights.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=0.0, params={"asset_cfg": _SERVO_JOINT_CFG})
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    dof_pos_limits = _backlash_preset(RewTerm(func=mdp.joint_pos_limits_finite, weight=-1.0), asset_cfg=_SERVO_JOINT_CFG)
    neck_strain_leaned = RewTerm(
        func=mdp.neck_torque_when_tilted,
        weight=-0.5,
        params={
            "asset_cfg": _HEAD_JOINT_CFG,
            "command_name": "base_velocity",
            "tilt_start_deg": 12.0,
            "tilt_full_deg": 25.0,
            "torque_scale": 0.1,
        },
    )
    self_collisions = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("self_collision")}
    )


@configclass
class TerminationsCfg:
    """No fall termination: wobbles and tips must play out so the policy pays the upright/impact costs."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.robot_state_is_nan, time_out=False, params={"sensor_names": ("contact_forces",)})


@configclass
class EventsCfg:
    """Walking-task sim-to-real DR (replica ranges), seated/standing start mix, delayed pushes, seated hardening."""

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES),
            "static_friction_range": FOOT_FRICTION_RANGE,
            "dynamic_friction_range": FOOT_FRICTION_RANGE,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="startup", params={"bias_range": (-0.015, 0.015)})
    mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "mass_distribution_params": MASS_SCALE_RANGE,
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    # 50 % standing / 50 % seated starts x an independent 50/50 command: all four cases are trained.
    set_ground_state = EventTerm(
        func=mdp.set_random_ground_state,
        mode="reset",
        params={
            "asset_cfg": _SERVO_JOINT_CFG,
            "sitting_prob": 0.5,
            "standing_prob": 0.5,
            "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std": 0.10,
            "sitting_tilt_max": math.radians(8),
            "sitting_z_min": 0.06,
            "sitting_z_max": 0.075,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={"asset_cfg": _TRUNK_BODY_CFG, "com_range": {axis: (-0.003, 0.003) for axis in "xyz"}},
    )
    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={"asset_cfg": _HEAD_BODY_CFG, "com_range": {axis: (-0.003, 0.003) for axis in "xyz"}},
    )
    randomize_joint_friction = EventTerm(func=mdp.randomize_bam_friction, mode="reset", params={"scale_range": (0.9, 1.1)})
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # Pushes start at zero and ramp by curriculum: pushes mid-descent before the transition exists
    # made the upstream sit policy unlearn sitting.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )
    seated_head_jerk = EventTerm(
        func=mdp.seated_head_command_jerk,
        mode="interval",
        interval_range_s=(0.3, 0.8),
        params={"prob": 0.0, "flip_frac": 0.5},
    )
    seated_tip = EventTerm(
        func=mdp.seated_backward_tip,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"ang_vel_range": (0.0, 0.0), "prob": 0.5, "lateral_frac": 0.3, "asset_cfg": _TRUNK_BODY_CFG},
    )


def _com_dict(half_range: float) -> dict[str, tuple[float, float]]:
    return {axis: (-half_range, half_range) for axis in "xyz"}


@configclass
class CurriculumCfg:
    """Upstream's schedules (in iterations of 24 steps), with the replica's wider CoM end points."""

    head_pose_range = _schedule(
        "commands.head_pose.ranges",
        (500, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
        (1000, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
        (1500, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.2, 0.2))),
        (2000, ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-0.31, 0.31))),
    )
    com_range = _schedule(
        "events.randomize_com.params.com_range",
        *[(500 + 500 * i, _com_dict(r)) for i, r in enumerate(TRUNK_COM_STAGES_M)],
    )
    head_com_range = _schedule(
        "events.randomize_head_com.params.com_range",
        *[(500 + 500 * i, _com_dict(r)) for i, r in enumerate(HEAD_COM_STAGES_M)],
    )
    push_magnitude = _schedule(
        "events.push_robot.params.velocity_range",
        (1000, {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}),
        (1500, {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}),
        (2000, {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}),
        (2500, {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}),
    )
    action_rate_weight = _schedule(
        "rewards.action_rate_l2.weight", (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)
    )
    descent_speed_weight = _schedule("rewards.descent_speed.weight", (500, 20.0))
    # The rise cap is introduced only after the rise exists (a motion tax during discovery kills it).
    rise_speed_weight = _schedule("rewards.rise_speed.weight", (1500, 5.0), (2500, 10.0))
    seated_head_jerk_prob = _schedule("events.seated_head_jerk.params.prob", (1500, 0.2), (2000, 0.4))
    seated_tip_range = _schedule(
        "events.seated_tip.params.ang_vel_range", (1500, (1.0, 2.5)), (2000, (1.5, 3.25)), (2500, (2.0, 4.0))
    )
    torque_rate_weight = _schedule("rewards.joint_torque_rate_l2.weight", (750, -5e-4), (1250, -1e-3))


@configclass
class MicroDuckHD1910SitStandFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Commanded sit <-> stand on flat ground, HD1910 servos, Newton MJWarp."""

    ngpus: int | None = None
    nnodes: int | None = None

    sim: SimulationCfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                # Seated contact (trunk + folded legs + head on the ground) overflows the walking task's
                # contact budget; upstream raised nconmax to 200 for the same reason.
                njmax=1024,
                nconmax=200,
                iterations=100,
                ls_iterations=50,
                cone="pyramidal",
                impratio=1.0,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2500000),
            default_shape_cfg=NewtonShapeCfg(margin=0.0),
            num_substeps=1,
        )
    )
    scene: MicroDuckSitStandSceneCfg = MicroDuckSitStandSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 0.005
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.self_collision.update_period = self.sim.dt
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))

    def play_mode(self):
        """Inference defaults; keep the pushes off so a clip shows the commanded behaviour."""
        super().play_mode()
        self.events.push_robot = None


@configclass
class MicroDuckHD1910SitFlatEnvCfg(MicroDuckHD1910SitStandFlatEnvCfg):
    """Stand -> sit, then hold: standing starts, SIT always commanded."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.sit_prob = 1.0
        self.events.set_ground_state.params["sitting_prob"] = 0.0
        self.events.set_ground_state.params["standing_prob"] = 1.0


@configclass
class MicroDuckHD1910StandFlatEnvCfg(MicroDuckHD1910SitStandFlatEnvCfg):
    """Sit -> stand, then hold: seated starts, STAND always commanded."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.sit_prob = 0.0
        self.events.set_ground_state.params["sitting_prob"] = 1.0
        self.events.set_ground_state.params["standing_prob"] = 0.0
