"""Velocity tracking (walk / run) for the HD1910 replica.

The official Isaac Lab MicroDuck recipe (``isaaclab_tasks.contrib.microduck``), with the servo
swapped for the HD1910 BAM fit and the replica domain randomisation widened
(:mod:`microduck_rl.dr`). Observation (61), action (14) and the reward stack are unchanged, so
an exported policy drops into the Microduck runtime like the official one.
"""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckSceneCfg, MicroDuckVelocityFlatEnvCfg, _schedule
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
