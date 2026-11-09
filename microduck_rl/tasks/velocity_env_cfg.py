"""Velocity tracking (walk / run) for the HD1910 replica.

The official Isaac Lab MicroDuck recipe (``isaaclab_tasks.contrib.microduck``), with the servo
swapped for the HD1910 BAM fit and the replica domain randomisation widened
(:mod:`microduck_rl.dr`). Observation (61), action (14) and the reward stack are unchanged, so
an exported policy drops into the Microduck runtime like the official one.
"""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import (
    _HEAD_JOINT_CFG,
    _HEAD_PLAY_JOINT_CFG,
    MicroDuckSceneCfg,
    MicroDuckVelocityFlatEnvCfg,
    _backlash_preset,
    _schedule,
)
from isaaclab_tasks.contrib.microduck.rough_env_cfg import MicroDuckRoughSceneCfg, MicroDuckVelocityRoughEnvCfg
from isaaclab_tasks.utils import preset

from microduck_rl import mdp
from microduck_rl.assets import (
    MICRODUCK_HD1910_ALLCOLLISIONS_BACKLASH_CFG,
    MICRODUCK_HD1910_ALLCOLLISIONS_CFG,
    MICRODUCK_HD1910_BACKLASH_CFG,
    MICRODUCK_HD1910_CFG,
)
from microduck_rl.dr import apply_replica_dr
from microduck_rl.realism import apply_servo_realism
from microduck_rl.viz import clip_visualizer_cfg, play_visualizer_cfg

_ROBOT_PRESET = dict(
    default=MICRODUCK_HD1910_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    backlash=MICRODUCK_HD1910_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
)


@configclass
class MicroDuckHD1910SceneCfg(MicroDuckSceneCfg):
    """The official flat walking scene with the HD1910 robot."""

    robot = preset(**_ROBOT_PRESET)


@configclass
class MicroDuckHD1910RoughSceneCfg(MicroDuckRoughSceneCfg):
    """The official rough walking scene with the HD1910 robot."""

    robot = preset(**_ROBOT_PRESET)


@configclass
class MicroDuckHD1910VelocityFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """Flat velocity walking, HD1910 servos, replica DR."""

    scene: MicroDuckHD1910SceneCfg = MicroDuckHD1910SceneCfg(num_envs=4096, env_spacing=2.0)

    def __post_init__(self):
        super().__post_init__()
        apply_replica_dr(self, first_stage_iteration=500)
        self.sim.visualizer_cfgs = clip_visualizer_cfg(self.scene.num_envs)
        apply_servo_realism(self)
        # Wider forward range than the official +-0.4 so the walking policy also covers a brisk pace.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.6)

    def play_mode(self):
        super().play_mode()
        self.sim.visualizer_cfgs = play_visualizer_cfg()


@configclass
class MicroDuckHD1910VelocityRoughEnvCfg(MicroDuckVelocityRoughEnvCfg):
    """Rough velocity walking, HD1910 servos, replica DR."""

    scene: MicroDuckHD1910RoughSceneCfg = MicroDuckHD1910RoughSceneCfg(num_envs=16384, env_spacing=2.0)

    def __post_init__(self):
        super().__post_init__()
        apply_replica_dr(self, first_stage_iteration=1000)
        self.sim.visualizer_cfgs = clip_visualizer_cfg(self.scene.num_envs)
        apply_servo_realism(self)

    def play_mode(self):
        super().play_mode()
        self.sim.visualizer_cfgs = play_visualizer_cfg()


@configclass
class MicroDuckHD1910RunFlatEnvCfg(MicroDuckHD1910VelocityFlatEnvCfg):
    """Running: a port of Pollen's official 2 m sprint recipe (``microduck-challenges/sprint_2m``).

    Straight-ahead commands over the Arena's sweep (0.20--1.20 m/s), no sideways or turning command;
    ``heading_hold`` prices yaw drift against the spawn heading and the critic sees that error; the
    "previous" tracking terms (weight 2.0, std^2 0.1 / 0.5), the posture reward active while moving, the
    head command held at its small range (no widening curriculum), the action-rate ramp to -1.0 by 1500.
    Official budget: 3000 iterations; the reference policy's best checkpoint was 1000 (0.55 m/s).
    """

    def __post_init__(self):
        super().__post_init__()
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.20, 1.20)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_forward_envs = 0.0
        cmd.rel_turn_in_place_envs = 0.0
        # Previous (sprint) tracking recipe.
        self.rewards.track_lin_vel = RewTerm(
            func=mdp.track_linear_velocity, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.1)}
        )
        self.rewards.track_ang_vel = RewTerm(
            func=mdp.track_angular_velocity, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
        )
        self.rewards.pose.params["standing_only"] = False
        self.rewards.heading_hold = RewTerm(func=mdp.heading_hold, weight=1.0, params={"std": 0.4})
        self.observations.critic.heading_error = ObsTerm(func=mdp.heading_error)
        # The sprint holds the head command small for the whole run and ramps the action-rate penalty earlier.
        self.curriculum.head_pose_range = None
        self.curriculum.action_rate_weight = _schedule(
            "rewards.action_rate_l2.weight", (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)
        )


@configclass
class MicroDuckHD1910RunV2FlatEnvCfg(MicroDuckHD1910RunFlatEnvCfg):
    """Running, second attempt: the sprint recipe with a SPEED CURRICULUM and the forward knob at 4.0.

    Measured on run v1 (``docs/run_speed_*.json``): achieved speed fell with the command (0.23 m/s at 0.3,
    0.02 m/s at 1.1, zero falls) because the Gaussian tracking kernel (std^2 0.1) carries no gradient once the
    error exceeds ~0.5 m/s, so the fast commands paid nothing from iteration 0. The command range therefore
    starts at 0.2--0.5 m/s and widens in four stages to the Arena sweep 0.2--1.2, and the sprint's own
    ``forward_reward`` knob is set to 4.0 (its documented range is 0.5--5.0).
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.50)
        self.rewards.track_lin_vel.weight = 4.0
        self.curriculum.speed_range = _schedule(
            "commands.base_velocity.ranges.lin_vel_x", (750, (0.2, 0.7)), (1500, (0.2, 0.9)), (2250, (0.2, 1.2))
        )


# ── Full-collision variants ───────────────────────────────────────────────────────────────────────────────
# The walking USD enables sole colliders only. On the full-collision model (shells, legs, head: the one the
# sit/stand, roll and jump policies live on, and the one closest to the real robot's shells) the policies
# trained on the reduced model are fragile: measured 2026-11-08, walk 0.3 m/s falls in 2 of 16 within 4 s and
# the run in 2 of 16, with 21 N of trunk self-contact at HOME. These variants fine-tune them where they will
# be deployed next to the other policies; the self-collision sensor keeps the official foot-against-foot pair.
_ALLCOLL_ROBOT_PRESET = dict(
    default=MICRODUCK_HD1910_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    backlash=MICRODUCK_HD1910_ALLCOLLISIONS_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
)


@configclass
class MicroDuckHD1910AllCollSceneCfg(MicroDuckSceneCfg):
    robot = preset(**_ALLCOLL_ROBOT_PRESET)


@configclass
class MicroDuckHD1910VelocityFlatAllCollEnvCfg(MicroDuckHD1910VelocityFlatEnvCfg):
    """Walking on the full-collision model (fine-tune target)."""

    scene: MicroDuckHD1910AllCollSceneCfg = MicroDuckHD1910AllCollSceneCfg(num_envs=4096, env_spacing=2.0)


@configclass
class MicroDuckHD1910RunV2FlatAllCollEnvCfg(MicroDuckHD1910RunV2FlatEnvCfg):
    """Running (speed-curriculum recipe, curriculum pinned at its final band) on the full-collision model."""

    scene: MicroDuckHD1910AllCollSceneCfg = MicroDuckHD1910AllCollSceneCfg(num_envs=4096, env_spacing=2.0)

    def __post_init__(self):
        super().__post_init__()
        # A fine-tune starts from a policy that already runs the full band: no speed curriculum.
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 1.20)
        self.curriculum.speed_range = None


@configclass
class MicroDuckHD1910RunV3FlatAllCollEnvCfg(MicroDuckHD1910RunV2FlatAllCollEnvCfg):
    """Run v3 environment: the full-collision run WITH a (compressed) speed curriculum again.

    Measured on the v2 -> full-collision fine-tune without a curriculum (``docs/run_allcoll_speed_4498_*``):
    0.26 / 0.44 / 0.35 / 0.00 / 0.00 m/s for 0.3 / 0.5 / 0.7 / 0.9 / 1.1 commanded -- inside 1500 iterations on
    the new plant the policy gave up the fast commands, exactly the run-v1 failure. The band therefore
    reopens from 0.2-0.7 to 0.2-1.2 over the first half of the run (iterations are the warm start's own count,
    continued: ``_schedule`` runs on the global step counter, so stages are placed past the loaded 2999)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.70)
        self.curriculum.speed_range = _schedule(
            "commands.base_velocity.ranges.lin_vel_x", (3000 + 500, (0.2, 0.9)), (3000 + 1000, (0.2, 1.2))
        )
        # The head is a dynamic body (~38 % of the mass): let the runner use it (lean forward, counter-swing)
        # instead of pinning it to the zero command at weight 2.0 / std 0.5 as the walking recipe does.
        self.rewards.head_pose_tracking = _backlash_preset(
            RewTerm(func=mdp.head_pose_tracking, weight=0.25, params={"command_name": "head_pose", "std": 1.0, "asset_cfg": _HEAD_JOINT_CFG}),
            backlash_cfg=_HEAD_PLAY_JOINT_CFG,
        )
        # Pushes stay at the official +-0.3 for the runner: the +-0.6 ladder of the first v3 attempt made it stop
        # running (see the walk v3 note); robustness to big shoves is the stand-up policy's job.
        self.events.push_robot.interval_range_s = (3.0, 6.0)
        self.curriculum.push_magnitude = None
    def play_mode(self):
        """The base play mode removes the push event; drop the schedule that addresses it or play crashes."""
        super().play_mode()
        self.curriculum.push_magnitude = None



@configclass
class MicroDuckHD1910VelocityFlatAllCollV3EnvCfg(MicroDuckHD1910VelocityFlatAllCollEnvCfg):
    """Walk v3 (full-collision fine-tune): no squat, no limp.

    Measured on the walk policy: at zero command the trunk drops to 0.067-0.092 m (a squat) and a turn starts
    with the same dip; the official recipe trains standing still in only 2 % of environments and has no trunk
    height term (``pose`` pays only while standing). Here 15 % of environments stand and 15 % turn in place,
    an always-on trunk-height cost holds the body at standing height through stops and turns, and the
    mirror loss (with the data augmentation) addresses the air-time asymmetry (L/R 0.93)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_turn_in_place_envs = 0.15
        self.rewards.trunk_height = RewTerm(
            func=mdp.trunk_height_error_l1, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.115}
        )
        # push robustness ladder (warm start continues the walk's counter at 16498: stages must sit past it).
        # Standing balance must be the most robust policy, then walking, then running: walk takes pushes to
        # +-0.8 m/s every 2-4 s, the official recipe stopped at +-0.3 every 5-10 s.
        # Measured on run v3: pushes of +-0.6 m/s every 3-5 s made the runner STOP (tracking error 0.16 -> 0.30,
        # falls 2.7 % -> 0.1 %, achieved 0.1 m/s): past ~0.4 m/s a shove is survived by standing still, so big
        # pushes belong to the stand-up policy, not to locomotion. Walk: +-0.4 every 3-5 s.
        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.curriculum.push_magnitude = _schedule(
            "events.push_robot.params.velocity_range", (16498 + 300, {"x": (-0.4, 0.4), "y": (-0.4, 0.4)})
        )
    def play_mode(self):
        """The base play mode removes the push event; drop the schedule that addresses it or play crashes."""
        super().play_mode()
        self.curriculum.push_magnitude = None



@configclass
class MicroDuckHD1910VelocityFlatAllCollV4EnvCfg(MicroDuckHD1910VelocityFlatAllCollV3EnvCfg):
    """Walk v4: v3 plus a height cost that actually bites. Measured on walk v3 (`docs/squat_walk_v3_17997.json`):
    trunk 0.121 m walking, 0.073 m at zero command, 0.072 m at turn onset -- a 42 mm squat, unchanged from the
    pre-v3 walk, under a -3.0 L1 cost that logged -0.03/s. Here: deadbanded L1 at -50 (a 42 mm squat costs
    ~1.7/s, the walking height costs 0), and the upstream default-pose term tripled so standing still means
    standing at the default (0.115 m) pose. Fine-tune from walk v3; pushes stay at +-0.4."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.trunk_height = RewTerm(
            func=mdp.trunk_height_deadband_l1,
            weight=-50.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.115, "deadband_m": 0.008},
        )
        self.rewards.pose.weight = 3.0


def _with_vin(robot_cfg, vin_range):
    """Copy of a robot cfg whose servo actuator samples its supply in ``vin_range`` (the preset wrapper cannot be
    edited after construction, so each alternative is rebuilt here)."""
    from dataclasses import replace as _dc_replace

    cfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actuators = dict(cfg.actuators)
    cfg.actuators["servos"] = _dc_replace(cfg.actuators["servos"], vin=vin_range[0], vin_range=vin_range)
    return cfg


@configclass
class MicroDuckHD1910AllCollSprintSceneCfg(MicroDuckSceneCfg):
    """Full-collision robot with the battery sampled up to a fully charged 8.4 V (the floor stays 7.4 V)."""

    robot = preset(
        default=_with_vin(MICRODUCK_HD1910_ALLCOLLISIONS_CFG, (7.4, 8.4)),
        backlash=_with_vin(MICRODUCK_HD1910_ALLCOLLISIONS_BACKLASH_CFG, (7.4, 8.4)),
    )


@configclass
class MicroDuckHD1910SprintMaxAllCollEnvCfg(MicroDuckHD1910RunV3FlatAllCollEnvCfg):
    """Sprint-max: find the hardware's top speed.

    Measured on run v2 at the 0.7 m/s command (0.49 m/s achieved): knee/ankle peaks 11.3-11.4 rad/s against the
    HD1910 no-load 11.9 rad/s at 7.4 V, torque peaks at stall -- the gait already rides the servo wall. This
    variant pays forward speed DIRECTLY (clipped at 2 m/s) on top of a weakened tracking term, pins the command
    at the top of the band (so the observation says "as fast as you can"), samples the battery up to a fully
    charged 8.4 V (the one knob that raises the wall, ~13 %), and logs how often the joints sit above 10 rad/s.
    Keep straight-line symmetry and fall terms; expect tens of percent, not a doubling."""

    scene: MicroDuckHD1910AllCollSprintSceneCfg = MicroDuckHD1910AllCollSprintSceneCfg(num_envs=4096, env_spacing=2.0)

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.4)
        self.curriculum.speed_range = None
        # Measured on the first attempt (weight 3.0, 700 iterations): the speed term paid ~0.8/s against air_time 2.4/s
        # and upright 1.8/s and stayed flat; the gait did not get faster. At 15 a 0.6 m/s run pays ~9/s and the term
        # dominates the stack.
        self.rewards.forward_speed = RewTerm(func=mdp.forward_speed_reward, weight=15.0, params={"asset_cfg": SceneEntityCfg("robot"), "cap": 2.0})
        self.rewards.track_lin_vel.weight = 1.0
        self.curriculum.joint_speed_saturation = CurrTerm(
            func=mdp.joint_speed_saturation_frac, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[r"^(?!passive_).*"]), "limit_rad_s": 10.0}
        )
