"""RSL-RL PPO runners: the official MicroDuck recipe with per-task experiment names."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg, MicroDuckSymmetricPPORunnerCfg
from isaaclab_tasks.utils import preset


def _names(stem: str):
    return preset(default=f"{stem}_hd1910", backlash=f"{stem}_hd1910_backlash")


@configclass
class WalkFlatPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = _names("microduck_walk_flat")


@configclass
class WalkFlatSymmetricPPORunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    experiment_name = _names("microduck_walk_flat_symmetry")


@configclass
class WalkRoughPPORunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    experiment_name = _names("microduck_walk_rough")


@configclass
class RunFlatPPORunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    """Sprint budget (official: 3000 iterations, best checkpoint often early) with symmetry augmentation."""

    experiment_name = _names("microduck_run_flat")
    max_iterations = 3000


@configclass
class SitStandPPORunnerCfg(MicroDuckPPORunnerCfg):
    """Upstream: same PPO, no symmetry, 15k iterations, checkpoint every 250."""

    experiment_name = _names("microduck_sitstand_flat")
    max_iterations = 15000


@configclass
class SitPPORunnerCfg(SitStandPPORunnerCfg):
    experiment_name = _names("microduck_sit_flat")


@configclass
class StandPPORunnerCfg(SitStandPPORunnerCfg):
    experiment_name = _names("microduck_stand_flat")


@configclass
class RouladePPORunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    """Forward roll: upstream trains 10k iterations at entropy 0.01 with the mirror loss on (the roll is
    sagittal; symmetry fights the shoulder-roll cheat of upstream run 5). Entropy 0.001 here: the sit stage
    showed a behaviour initiated by exploration noise is not one the deployed mean action performs."""

    experiment_name = _names("microduck_roulade_flat")
    max_iterations = 8000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.001
