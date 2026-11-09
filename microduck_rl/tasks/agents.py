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


@configclass
class JumpPPORunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    """One-shot jump: upstream v6 ran 5000 iterations at entropy 0.001 with a bounded std (init 0.6, floor 0.05);
    here the mirror loss is on (the jump is sagittal) and the std starts at 0.6."""

    experiment_name = _names("microduck_jump_flat")
    max_iterations = 5000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.001
        self.actor.distribution_cfg.init_std = 0.6


@configclass
class WalkFlatAllCollPPORunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    """Walk fine-tune on the full-collision model (warm start from the symmetric walk; entropy as the walk used)."""

    experiment_name = _names("microduck_walk_flat_allcoll")
    max_iterations = 1500


@configclass
class RunFlatAllCollPPORunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    """Run fine-tune on the full-collision model (warm start from run v2)."""

    experiment_name = _names("microduck_run_flat_allcoll")
    max_iterations = 1500


@configclass
class RunFlatAllCollMirrorPPORunnerCfg(RunFlatAllCollPPORunnerCfg):
    """Run v3: the full-collision run fine-tune with the MIRROR LOSS on top of data augmentation.
    Measured on run v2 (docs/run_v2_speed_2999_vx*.json): knee amplitude L/R 1.00 but air time L/R 0.89-0.93
    and stance L/R 1.05-1.10 -- a timing asymmetry (one leg stays on the ground longer) that reads as a limp.
    Augmentation alone did not remove it; the mirror loss penalises the actor's left/right difference directly."""

    experiment_name = _names("microduck_run_flat_allcoll_mirror")
    max_iterations = 3000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg.use_mirror_loss = True
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 0.5


@configclass
class WalkFlatAllCollMirrorPPORunnerCfg(WalkFlatAllCollPPORunnerCfg):
    """Walk v3 fine-tune: mirror loss 0.5 on top of the data augmentation."""

    experiment_name = _names("microduck_walk_flat_allcoll_mirror")
    max_iterations = 1500

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg.use_mirror_loss = True
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 0.5


@configclass
class StandUpPPORunnerCfg(SitStandPPORunnerCfg):
    """Get-up + robust stand: upstream's recipe (no symmetry, entropy 0.01; deployed checkpoint came at 9750 of
    15000). Trained two-phase by scripts/train_stand_two_phase.sh (entropy 0.001 for the last 3000)."""

    experiment_name = _names("microduck_standup_flat")
    max_iterations = 10000


@configclass
class SprintMaxPPORunnerCfg(RunFlatAllCollMirrorPPORunnerCfg):
    """Sprint-max fine-tune from run v3 (mirror loss kept)."""

    experiment_name = _names("microduck_sprint_max_allcoll")
    max_iterations = 2000
