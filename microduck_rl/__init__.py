"""Microduck HD1910 (Feetech) replica tasks for Isaac Lab 3.0 + Newton MJWarp.

Importing this package registers the tasks below. Add ``presets=backlash`` to train the robot with
gearbox-play hinges (what the official Microduck policies deploy with).

* ``Microduck-Velocity-Flat-HD1910``  walking / velocity tracking (official recipe)
* ``Microduck-Velocity-Rough-HD1910`` the same on rough terrain
* ``Microduck-Run-Flat-HD1910``       running: port of Pollen's official sprint recipe
* ``Microduck-RunV2-Flat-HD1910``     running with a speed curriculum (see velocity_env_cfg)
* ``Microduck-Sit-Flat-HD1910``       stand -> sit
* ``Microduck-Stand-Flat-HD1910``     sit -> stand
* ``Microduck-SitStand-Flat-HD1910``  commanded sit <-> stand, one policy
"""

import gymnasium as gym

_TASKS = "microduck_rl.tasks"
_ENV = "isaaclab.envs:ManagerBasedRLEnv"


def _register(task_id: str, env_cfg: str, agent_cfg: str, symmetry_cfg: str | None = None) -> None:
    kwargs = {
        "env_cfg_entry_point": f"{_TASKS}.{env_cfg}",
        "rsl_rl_cfg_entry_point": f"{_TASKS}.agents:{agent_cfg}",
    }
    if symmetry_cfg is not None:
        kwargs["rsl_rl_with_symmetry_cfg_entry_point"] = f"{_TASKS}.agents:{symmetry_cfg}"
    gym.register(id=task_id, entry_point=_ENV, disable_env_checker=True, kwargs=kwargs)


_register(
    "Microduck-Velocity-Flat-HD1910",
    "velocity_env_cfg:MicroDuckHD1910VelocityFlatEnvCfg",
    "WalkFlatPPORunnerCfg",
    "WalkFlatSymmetricPPORunnerCfg",
)
_register(
    "Microduck-Velocity-Rough-HD1910",
    "velocity_env_cfg:MicroDuckHD1910VelocityRoughEnvCfg",
    "WalkRoughPPORunnerCfg",
    "WalkRoughPPORunnerCfg",
)
_register(
    "Microduck-Run-Flat-HD1910", "velocity_env_cfg:MicroDuckHD1910RunFlatEnvCfg", "RunFlatPPORunnerCfg", "RunFlatPPORunnerCfg"
)
_register(
    "Microduck-RunV2-Flat-HD1910", "velocity_env_cfg:MicroDuckHD1910RunV2FlatEnvCfg", "RunFlatPPORunnerCfg", "RunFlatPPORunnerCfg"
)
_register(
    "Microduck-Roulade-Flat-HD1910", "roulade_env_cfg:MicroDuckHD1910RouladeFlatEnvCfg", "RouladePPORunnerCfg", "RouladePPORunnerCfg"
)
_register(
    "Microduck-Velocity-Flat-HD1910-AllColl", "velocity_env_cfg:MicroDuckHD1910VelocityFlatAllCollEnvCfg", "WalkFlatAllCollPPORunnerCfg", "WalkFlatAllCollPPORunnerCfg"
)
_register(
    "Microduck-RunV2-Flat-HD1910-AllColl", "velocity_env_cfg:MicroDuckHD1910RunV2FlatAllCollEnvCfg", "RunFlatAllCollPPORunnerCfg", "RunFlatAllCollPPORunnerCfg"
)
_register(
    "Microduck-RunV3-Flat-HD1910-AllColl", "velocity_env_cfg:MicroDuckHD1910RunV3FlatAllCollEnvCfg", "RunFlatAllCollMirrorPPORunnerCfg", "RunFlatAllCollMirrorPPORunnerCfg"
)
_register(
    "Microduck-WalkV3-Flat-HD1910-AllColl", "velocity_env_cfg:MicroDuckHD1910VelocityFlatAllCollV3EnvCfg", "WalkFlatAllCollMirrorPPORunnerCfg", "WalkFlatAllCollMirrorPPORunnerCfg"
)
_register("Microduck-StandUp-Flat-HD1910", "standup_env_cfg:MicroDuckHD1910StandUpFlatEnvCfg", "StandUpPPORunnerCfg")
_register(
    "Microduck-WalkV4-Flat-HD1910-AllColl", "velocity_env_cfg:MicroDuckHD1910VelocityFlatAllCollV4EnvCfg", "WalkV4AllCollPPORunnerCfg", "WalkV4AllCollPPORunnerCfg"
)
_register(
    "Microduck-VelStand-Flat-HD1910-AllColl", "velstand_env_cfg:MicroDuckHD1910VelStandAllCollEnvCfg", "VelStandPPORunnerCfg", "VelStandPPORunnerCfg"
)
_register(
    "Microduck-SprintMax-Flat-HD1910-AllColl", "velocity_env_cfg:MicroDuckHD1910SprintMaxAllCollEnvCfg", "SprintMaxPPORunnerCfg", "SprintMaxPPORunnerCfg"
)
_register("Microduck-Jump-Flat-HD1910", "jump_env_cfg:MicroDuckHD1910JumpFlatEnvCfg", "JumpPPORunnerCfg", "JumpPPORunnerCfg")
_register("Microduck-Sit-Flat-HD1910", "sitstand_env_cfg:MicroDuckHD1910SitFlatEnvCfg", "SitPPORunnerCfg")
_register("Microduck-Stand-Flat-HD1910", "sitstand_env_cfg:MicroDuckHD1910StandFlatEnvCfg", "StandPPORunnerCfg")
_register(
    "Microduck-SitStand-Flat-HD1910", "sitstand_env_cfg:MicroDuckHD1910SitStandFlatEnvCfg", "SitStandPPORunnerCfg"
)

TASK_IDS = [
    "Microduck-Velocity-Flat-HD1910",
    "Microduck-Velocity-Rough-HD1910",
    "Microduck-Run-Flat-HD1910",
    "Microduck-RunV2-Flat-HD1910",
    "Microduck-Sit-Flat-HD1910",
    "Microduck-Roulade-Flat-HD1910",
    "Microduck-Jump-Flat-HD1910",
    "Microduck-Velocity-Flat-HD1910-AllColl",
    "Microduck-RunV2-Flat-HD1910-AllColl",
    "Microduck-RunV3-Flat-HD1910-AllColl",
    "Microduck-WalkV3-Flat-HD1910-AllColl",
    "Microduck-StandUp-Flat-HD1910",
    "Microduck-SprintMax-Flat-HD1910-AllColl",
    "Microduck-WalkV4-Flat-HD1910-AllColl",
    "Microduck-VelStand-Flat-HD1910-AllColl",
    "Microduck-Stand-Flat-HD1910",
    "Microduck-SitStand-Flat-HD1910",
]
