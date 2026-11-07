"""Velocity tracking (walk / run) for the HD1910 replica.

The official Isaac Lab MicroDuck recipe (``isaaclab_tasks.contrib.microduck``), with the servo
swapped for the HD1910 BAM fit and the replica domain randomisation widened
(:mod:`microduck_rl.dr`). Observation (61), action (14) and the reward stack are unchanged, so
an exported policy drops into the Microduck runtime like the official one.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckSceneCfg, MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.contrib.microduck.rough_env_cfg import MicroDuckRoughSceneCfg, MicroDuckVelocityRoughEnvCfg
from isaaclab_tasks.utils import preset

from microduck_rl.assets import MICRODUCK_HD1910_BACKLASH_CFG, MICRODUCK_HD1910_CFG
from microduck_rl.dr import apply_replica_dr
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

    def play_mode(self):
        super().play_mode()
        self.sim.visualizer_cfgs = play_visualizer_cfg()


@configclass
class MicroDuckHD1910RunFlatEnvCfg(MicroDuckHD1910VelocityFlatEnvCfg):
    """Flat velocity tracking with a faster forward command range (the "run" policy).

    Same recipe as walking; only the command distribution asks for more speed. The air-time window
    is shortened to reward the quicker stepping a faster gait needs. Experimental: the official
    recipe is tuned for the 0.4 m/s range.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.8)
        self.commands.base_velocity.forward_min_speed = 0.4
        self.commands.base_velocity.rel_forward_envs = 0.4
        self.rewards.air_time.params["threshold_min"] = 0.1
        self.rewards.air_time.params["threshold_max"] = 0.25
