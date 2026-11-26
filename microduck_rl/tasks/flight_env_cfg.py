"""Flying duck tasks (HD1910 replica + ducted-fan jetpack, docs/duck_flight.tex).

* ``Microduck-WalkPack-Flat-HD1910-AllColl``  walk v4 carrying the flight hardware, fans off (fine-tune of walk v4)
* ``Microduck-RunPack-Flat-HD1910-AllColl``   run v3 carrying it (fine-tune of run v3)
* ``Microduck-Takeoff-Flat-HD1910-AllColl``   the full cycle: run, take off on command, track a velocity
  command with a vertical component (vx, vy, vz, wz), land with a flare, and walk on: the legs' 14 joint
  targets plus 4 fan throttles from one policy. The action size differs from the run policy's, so this one trains
  from scratch (and without the mirror loss, whose joint map does not know the fans).

The deployed walk and run policies fall 6/6 with the 685 g pack in MuJoCo (docs/flight_rollout_results.json): the
first two tasks exist because of that measurement.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import MICRODUCK_FOOT_BODY_NAMES

from microduck_rl import mdp
from microduck_rl.mdp import flight
from microduck_rl.tasks.velocity_env_cfg import (
    MicroDuckHD1910RunV3FlatAllCollEnvCfg,
    MicroDuckHD1910VelocityFlatAllCollV4EnvCfg,
)

_TRUNK = SceneEntityCfg("robot", body_names=["trunk_base"])
_FEET = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)


def add_flight_pack(cfg) -> None:
    """Weld the flight hardware's inertial properties onto the trunk (after the trunk's own mass randomisation)."""
    cfg.events.flight_pack = EventTerm(func=flight.attach_flight_pack, mode="startup", params={"asset_cfg": _TRUNK})


@configclass
class MicroDuckHD1910WalkPackAllCollEnvCfg(MicroDuckHD1910VelocityFlatAllCollV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_flight_pack(self)


@configclass
class MicroDuckHD1910RunPackAllCollEnvCfg(MicroDuckHD1910RunV3FlatAllCollEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_flight_pack(self)
        # the pack adds 84 % to the duck's mass: the top of the run band is not expected to survive
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.70)
        self.curriculum.speed_range = None


@configclass
class MicroDuckHD1910TakeoffAllCollEnvCfg(MicroDuckHD1910RunPackAllCollEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 16.0     # walk 2-5 s, fly 4-7 s, land, walk again
        self.actions.fans = flight.FanThrottleActionCfg()
        self.commands.takeoff = flight.TakeoffCommandCfg()
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.60)

        for group in (self.observations.policy, self.observations.critic):
            group.takeoff_command = ObsTerm(func=mdp.generated_commands, params={"command_name": flight.TAKEOFF_CMD})
            group.fan_state = ObsTerm(func=flight.fan_state)
        self.observations.policy.base_height = ObsTerm(func=flight.base_height_noisy, params={"noise_m": 0.01})
        self.observations.critic.base_height = ObsTerm(func=flight.base_height_noisy, params={"noise_m": 0.0})

        r = self.rewards
        # flight
        r.flight_vz = RewTerm(func=flight.flight_vz_tracking, weight=3.0, params={"std": 0.22, "asset_cfg": SceneEntityCfg("robot")})
        r.flight_airborne = RewTerm(func=flight.flight_airborne, weight=1.5, params={"clearance_m": 0.05})
        r.flight_feet_off = RewTerm(func=flight.flight_feet_off_ground, weight=0.5, params={"sensor_cfg": _FEET})
        r.flight_upright = RewTerm(func=flight.flight_upright, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot")})
        r.fan_ground = RewTerm(func=flight.fan_throttle_ground, weight=-0.5)
        r.fan_rate = RewTerm(func=flight.fan_throttle_rate, weight=-0.5)
        r.landing_impact = RewTerm(func=flight.landing_impact, weight=-8.0, params={"asset_cfg": SceneEntityCfg("robot")})
        # gait terms mean nothing in the air: switch them off after the take-off flag
        for name in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip"):
            term = getattr(r, name, None)
            if term is not None:
                setattr(r, name, flight.gated(term, on_ground=True))
        self.terminations.ceiling = DoneTerm(func=flight.above_ceiling, params={"ceiling_m": 1.6})

    def play_mode(self):
        super().play_mode()
        self.commands.takeoff.no_takeoff_prob = 0.0
        self.commands.takeoff.takeoff_time_range_s = (3.0, 3.0)
        self.commands.takeoff.climb_vz_range = (0.5, 0.5)
        self.commands.takeoff.hover_prob = 1.0
        self.commands.takeoff.no_landing_prob = 0.0
        self.commands.takeoff.flight_time_range_s = (5.0, 5.0)
