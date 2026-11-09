"""VelStand: walking + balance at zero speed + getting up after a fall, ONE policy (HD1910 replica).

Port of Pollen's ``Mjlab-VelStand-Flat-MicroDuck`` recipe (their deployed policy), on top of walk v4:

* warm start from the walk; the velocity recipe's curricula have already run their course at that
  iteration count, so only VelStand's own ramps move (they are keyed off ``VELSTAND_START``);
* a share of the resets start FALLEN -- face down, face up / on a side (roll to +-90 deg), random orientation
  with random joints, and the bank of real fall end-states captured from the pushed walk/run policies
  (`docs/fall_bank.json`); the standing share keeps the walk's own randomised reset;
* a second push event with kicks far beyond the walking randomisation (ramped +-0.6 -> +-1.2 m/s) so falls are
  frequent on-policy data and not rare accidents;
* no fall termination: a fallen duck has to get up, and the episode runs on;
* recovery terms from the stand-up stack that are neutral while walking (upright, upward CoM velocity), the
  deadbanded trunk-height cost already in walk v4 (a lying trunk pays it until it rises), and the action-rate
  tax cut to a tenth while fallen (their run-1 lesson: a full tax teaches lying still);
* the recovery SKILL is distilled from the stand-up expert and the walk is anchored to its own checkpoint by
  ``PpoWithExpertBc`` (`microduck_rl/mdp/distill.py`), because their measurement says PPO alone never finds it.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import _schedule

from microduck_rl import mdp
from microduck_rl.tasks.velocity_env_cfg import MicroDuckHD1910VelocityFlatAllCollV4EnvCfg

VELSTAND_START = 19497
"""Iteration count carried in by the warm start (walk v4's final checkpoint); the ramps below are relative to it.
A warm start a few iterations off (a resumed walk v4) shifts them by the same few iterations, which is harmless."""

_SERVOS = SceneEntityCfg("robot", joint_names=[r"^(?!passive_).*"])


@configclass
class MicroDuckHD1910VelStandAllCollEnvCfg(MicroDuckHD1910VelocityFlatAllCollV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        S = VELSTAND_START
        # fallen spawns on top of the walk's standing reset (the standing share is left to that reset)
        self.events.set_fallen_state = EventTerm(
            func=mdp.reset_fallen_state,
            mode="reset",
            params={
                "asset_cfg": _SERVOS,
                "standing_keep": True,
                "standing_prob": 0.55,
                "sitting_prob": 0.0,
                "prone_prob": 0.12,
                "supine_prob": 0.13,
                "random_prob": 0.10,
                "bank_prob": 0.10,
                "face_up_roll_max_deg": 90.0,
                "fall_bank_path": "docs/fall_bank.json",
            },
        )
        # topple pushes: far beyond the walking randomisation, so falls are on-policy data
        self.events.topple_push = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
        )
        self.curriculum.topple_magnitude = _schedule(
            "events.topple_push.params.velocity_range",
            (S + 300, {"x": (-0.9, 0.9), "y": (-0.9, 0.9)}),
            (S + 800, {"x": (-1.2, 1.2), "y": (-1.2, 1.2)}),
        )
        # a fallen duck gets up; the episode runs on
        self.terminations.fell_over = None
        # recovery terms neutral while walking
        self.rewards.upright_linear = RewTerm(func=mdp.upright_linear, weight=1.5, params={"asset_cfg": SceneEntityCfg("robot")})
        self.rewards.com_upward_velocity = RewTerm(
            func=mdp.com_upward_velocity, weight=0.75, params={"asset_cfg": SceneEntityCfg("robot"), "max_height": 0.125}
        )
        # the walk keeps its smoothness tax; recovery attempts pay a tenth of it
        self.rewards.action_rate_l2 = RewTerm(
            func=mdp.action_rate_l2_fallen_scaled,
            weight=self.rewards.action_rate_l2.weight,
            params={"asset_cfg": SceneEntityCfg("robot"), "fallen_tilt_deg": 40.0, "fallen_scale": 0.1},
        )

    def play_mode(self):
        super().play_mode()
        self.events.topple_push = None
        self.curriculum.topple_magnitude = None
        # Evaluations and demos start upright; the get-up battery sets the fallen families explicitly.
        self.events.set_fallen_state.params.update(
            {"standing_prob": 1.0, "sitting_prob": 0.0, "prone_prob": 0.0, "supine_prob": 0.0, "random_prob": 0.0, "bank_prob": 0.0}
        )
