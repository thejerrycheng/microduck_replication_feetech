"""RunStand: the run v3 policy with fall recovery folded in, ONE policy (HD1910 replica).

The VelStand recipe (`velstand_env_cfg.py`, Pollen's deployed walk) applied to the full-collision run v3 env:

* warm start from run v3 (checkpoint 5999; the run's own speed and action-rate curricula have run their course
  at that count, so only RunStand's ramps move, keyed off ``RUNSTAND_START``);
* a share of the resets start FALLEN (face down / face up / side / random / the real fall bank, which holds
  171 run falls); the standing share keeps the run's randomised reset;
* a second push event far beyond the run's +-0.3 m/s (ramped +-0.6 -> +-1.2 m/s) so falls are on-policy data.
  Run v3 take 1 learned to STOP under a +-0.6 ladder; here the upright anchor to the run checkpoint
  (``PpoWithExpertBc``) holds the run while the expert teaches the get-up, which is the whole point of the
  distillation;
* no fall termination: a fallen runner has to get up and resume the command, and the episode runs on;
* recovery terms neutral while running (upright, upward CoM velocity), the action-rate tax cut to a tenth
  while fallen.

Every command here is non-zero (0.2-1.2 m/s straight ahead), so the zero-twist clause of the distillation never
fires and every upright frame is anchored to the run.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import _schedule

from microduck_rl import mdp
from microduck_rl.mdp import gate
from microduck_rl.tasks.velocity_env_cfg import MicroDuckHD1910RunV3FlatAllCollEnvCfg

RUNSTAND_START = 6000
"""Iteration count carried in by the warm start (run v3 take 2's final checkpoint, 5999); the ramps are relative to it."""

_SERVOS = SceneEntityCfg("robot", joint_names=[r"^(?!passive_).*"])


@configclass
class MicroDuckHD1910RunStandAllCollEnvCfg(MicroDuckHD1910RunV3FlatAllCollEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        S = RUNSTAND_START
        # the warm start is past the run's speed curriculum: pin the full band (the schedule's last stage)
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 1.20)
        self.curriculum.speed_range = None
        # fallen spawns on top of the run's standing reset (the standing share is left to that reset)
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
        # topple pushes: far beyond the run's +-0.3, so falls are on-policy data
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
        # a fallen runner gets up; the episode runs on
        self.terminations.fell_over = None
        # recovery terms neutral while running
        self.rewards.upright_linear = RewTerm(func=mdp.upright_linear, weight=1.5, params={"asset_cfg": SceneEntityCfg("robot")})
        self.rewards.com_upward_velocity = RewTerm(
            func=mdp.com_upward_velocity, weight=0.75, params={"asset_cfg": SceneEntityCfg("robot"), "max_height": 0.125}
        )
        # the run keeps its smoothness tax; recovery attempts pay a tenth of it
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


@configclass
class GateObsCfg(ObsGroup):
    """Teacher-gate quantities (never fed to the networks; see `microduck_rl/mdp/gate.py`)."""

    trunk_height = ObsTerm(func=gate.trunk_height_obs)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class MicroDuckHD1910RunStandV2AllCollEnvCfg(MicroDuckHD1910RunStandAllCollEnvCfg):
    """RunStand v2: the expert also teaches UPRIGHT frames with a LOW trunk.

    Measured on v1 at checkpoint 6500 (`docs/standup_runstand_6500_crouch.json`): every fallen duck rises, but
    31-36 % of them park in a level squat -- trunk 0.040 m (standing 0.115), tilt 3 deg, knees +-0.9 rad, moving
    at 0.04 m/s -- and stay there. The v1 gate hands over at 35 deg of tilt, the anchor then clones run v3's action
    on a squat that policy never saw, and the run env has no height cost to pull it up. VelStand never showed this
    because its zero-command upright frames were cloned from the expert; the run env has no zero command. Here the
    trunk height rides along in a ``gate`` observation group and frames below ``gate_trunk_m`` (0.085, midway
    between the squat and the running trunk) go to the expert whatever their tilt."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.gate = GateObsCfg()
