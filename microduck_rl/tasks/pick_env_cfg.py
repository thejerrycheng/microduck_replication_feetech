"""Pick-up task: walk to a randomised object and lift it with the beak (``Microduck-Pick-Flat-HD1910-AllColl``).

Built on walk v4 (the VelStand parent, so velstand_21495 warm-starts it with the same 61-float observation and
14-float action): the ``base_velocity`` command is replaced by :class:`~microduck_rl.mdp.pick.ObjectSeekCommand`,
the two zero command slots (head pose 4, body pose 6) carry the camera-style object estimate, and four pick rewards
are added on top of the walk's tracking and regularisation terms. See :mod:`microduck_rl.mdp.pick` for the sensing
and grasp model and ``docs/pick_env.md`` for the real-robot side of the contract.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import RigidBodyMaterialBaseCfg
from isaaclab.sim.schemas import MassPropertiesCfg, UsdPhysicsCollisionCfg, UsdPhysicsMeshCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.utils import configclass

from microduck_rl import mdp
from microduck_rl.mdp import pick
from microduck_rl.tasks.velocity_env_cfg import MicroDuckHD1910AllCollSceneCfg, MicroDuckHD1910VelocityFlatAllCollV4EnvCfg

OBJECT_MASS_RANGE_KG = (0.005, 0.040)
"""Per-reset object mass. 10-40 g is the mouth payload kissqy's GroundPick trains on; 5 g adds near-weightless items."""

_OBJ = dict(
    rigid_props=UsdPhysicsRigidBodyCfg(),
    mass_props=MassPropertiesCfg(mass=0.02),
    # convex hulls on purpose: Newton types a convex-hull mesh as a mesh, which is the only geom kind the MJWarp
    # solver lets differ between environments (the spawner's default bounding-shape approximation becomes a box/sphere)
    collision_props=[UsdPhysicsCollisionCfg(), UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")],
    physics_material=RigidBodyMaterialBaseCfg(static_friction=0.8, dynamic_friction=0.8),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.1)),
)

OBJECT_SHAPES = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[
        sim_utils.MeshCuboidCfg(size=(0.03, 0.03, 0.03), **_OBJ),
        sim_utils.MeshCuboidCfg(size=(0.05, 0.03, 0.025), **_OBJ),
        sim_utils.MeshCuboidCfg(size=(0.04, 0.04, 0.02), **_OBJ),
        sim_utils.MeshCuboidCfg(size=(0.025, 0.025, 0.045), **_OBJ),
        sim_utils.MeshSphereCfg(radius=0.015, **_OBJ),
        sim_utils.MeshSphereCfg(radius=0.02, **_OBJ),
        sim_utils.MeshSphereCfg(radius=0.025, **_OBJ),
        sim_utils.MeshCylinderCfg(radius=0.015, height=0.04, **_OBJ),
        sim_utils.MeshCylinderCfg(radius=0.02, height=0.025, **_OBJ),
        sim_utils.MeshCapsuleCfg(radius=0.012, height=0.03, **_OBJ),
    ],
    random_choice=True,
)
"""One shape per environment, drawn at spawn: 2.5-5 cm items a 9 cm beak can cover. Mesh variants on purpose: the
MJWarp solver only lets MESH geoms differ between environments (primitive types must match across worlds)."""


@configclass
class MicroDuckHD1910PickSceneCfg(MicroDuckHD1910AllCollSceneCfg):
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=OBJECT_SHAPES,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.03)),
    )


@configclass
class MicroDuckHD1910PickAllCollEnvCfg(MicroDuckHD1910VelocityFlatAllCollV4EnvCfg):
    scene: MicroDuckHD1910PickSceneCfg = MicroDuckHD1910PickSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        old = self.commands.base_velocity
        self.commands.base_velocity = pick.ObjectSeekCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 5.0),
            ranges=old.ranges,
            heading_command=False,
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
        )
        # the two zero command slots carry the object estimate (policy) / the true object state (critic)
        for group, fn in ((self.observations.policy, pick.object_obs), (self.observations.critic, pick.object_obs_true)):
            group.head_pose_commands = ObsTerm(func=fn, params={"start": 0, "stop": 4})
            group.body_pose_commands = ObsTerm(func=fn, params={"start": 4, "stop": 10})
        for term in (self.observations.policy.head_pose_commands, self.observations.policy.body_pose_commands):
            term.delay_min_lag, term.delay_max_lag, term.delay_hold_prob = 1, 3, 0.97  # camera + detector latency
        # the head is the gripper now: no home-pose pull on it
        for name in ("head_pose_tracking", "head_pose_bias"):
            if getattr(self.rewards, name, None) is not None:
                getattr(self.rewards, name).weight = 0.0
        # object: mass per reset, placed ahead of the robot after its own reset
        self.events.object_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": OBJECT_MASS_RANGE_KG, "operation": "abs"},
        )
        self.events.reset_object = EventTerm(
            func=pick.reset_pick_object, mode="reset", params={"dist_range": (0.25, 0.9), "bearing_deg": 60.0}
        )
        # pick rewards; pick_grasp first: it advances the shared state the others read
        self.rewards.pick_grasp = RewTerm(func=pick.pick_grasp, weight=2.0)
        self.rewards.object_progress = RewTerm(func=pick.object_progress, weight=10.0)
        self.rewards.beak_to_object = RewTerm(func=pick.beak_to_object, weight=2.0, params={"std": 0.04, "gate_xy_m": 0.3})
        self.rewards.object_lift = RewTerm(func=pick.object_lift, weight=3.0, params={"height_m": 0.06})
        self.episode_length_s = 15.0


@configclass
class MicroDuckHD1910PickAllCollEnvCfg_PLAY(MicroDuckHD1910PickAllCollEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
