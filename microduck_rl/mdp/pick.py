"""Pick-up with the beak: object spawn, camera-realistic object estimate, seek command, grasp model, rewards.

Design (2026-11-17). The policy never sees pixels. It sees the object's position in the base-yaw frame the way the
real duck will estimate it from its single head camera: a detector gives the object's pixel, and because the object
rests on the floor the ray through that pixel meets the ground plane at a known 3D point (camera pose from the head
servo angles + the trunk's IMU tilt). No depth sensor, no point cloud, no pixel-level sim-to-real gap. In simulation
that estimate is the true position passed through the same limitations the real pipeline has: the object is only
seen inside the head camera's field of view, detections drop out, the position error grows with distance, the
estimate is held when the object is not in view, and the observation is delayed 1-3 control steps.

The grasp is a passive beak (no jaw servo on the 14-servo replica): when the beak tip is on top of the object and
pointing down, the object is held (kinematically attached to the beak) and its mass is added to the head body, so
the servos carry the real load. The object's shape is drawn per environment at spawn (cuboids, spheres, cylinders,
capsules, 2.5-5 cm) and its mass per reset (5-40 g, the range kissqy's GroundPick recipe trains the mouth payload on).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from isaaclab_tasks.contrib.microduck.mdp.commands import MicroDuckVelocityCommand, MicroDuckVelocityCommandCfg

__all__ = [
    "BEAK_TIP_OFFSET",
    "HEAD_FORWARD",
    "HEAD_UP",
    "HEAD_CAMERA_OFFSET",
    "HEAD_CAMERA_TILT_DEG",
    "ObjectSeekCommand",
    "ObjectSeekCommandCfg",
    "beak_to_object",
    "object_lift",
    "object_obs",
    "object_obs_true",
    "object_progress",
    "pick_grasp",
    "reset_pick_object",
]

HEAD_FORWARD = (0.0, 0.0, -1.0)
HEAD_UP = (1.0, 0.0, 0.0)
"""The ``jaw_soft`` (head) body frame is NOT world-aligned: at the zero pose its +x points up and its -z points
forward (USD rotation rows [[0,0,1],[0,1,0],[-1,0,0]]). Every head-frame offset below uses that convention."""

BEAK_TIP_OFFSET = (-0.009, 0.0, -0.078)
"""Beak tip in the head frame [m]: the USD beak meshes span x -0.016..-0.007 and z -0.044..-0.078 in that frame, and
kissqy's HD1910 MJCF puts its ``mouth_tip`` site at (-0.0083, 0, -0.0777) in the same frame."""

HEAD_CAMERA_OFFSET = (0.012, 0.0, -0.070)
"""Head camera in the head frame [m]. ASSUMED from kissqy's ``head_camera`` site (0.01175, 0, -0.0735): just above
and behind the beak tip. Measure it on the robot before deploying."""

HEAD_CAMERA_TILT_DEG = 20.0
"""Camera optical axis below the head's forward axis [deg]. ASSUMED; measure."""

CAMERA_HFOV_DEG, CAMERA_VFOV_DEG, CAMERA_RANGE_M = 62.2, 48.8, 1.5
"""Raspberry Pi camera v2 (IMX219) field of view; detection range for a 3 cm object at 8 MP binned to 640 px."""


class PickState:
    """Per-environment pick state, created on first use and kept on the env."""

    def __init__(self, env: ManagerBasedRLEnv):
        n, d = env.num_envs, env.device
        robot: Articulation = env.scene["robot"]
        self.grasped = torch.zeros(n, dtype=torch.bool, device=d)
        self.est = torch.zeros(n, 3, device=d)  # object estimate, base-yaw frame
        self.seen = torch.zeros(n, dtype=torch.bool, device=d)
        self.spawn_z = torch.zeros(n, device=d)
        self.dist_prev = torch.zeros(n, device=d)
        self.tip_dist = torch.zeros(n, device=d)
        self.jaw_idx = robot.find_bodies("jaw_soft")[0][0]
        self.jaw_mass0 = robot.data.body_mass.torch[:, self.jaw_idx].clone()
        self.radius, self.half_h = self._object_extents(env)
        self.step_updated = -1

    @staticmethod
    def _object_extents(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
        """Half-width (xy) and half-height (z) of each environment's spawned object, read from its USD prim."""
        n, d = env.num_envs, env.device
        radius = torch.full((n,), 0.02, device=d)
        half_h = torch.full((n,), 0.02, device=d)
        try:
            from pxr import Usd, UsdGeom

            from isaaclab.sim.utils.stage import get_current_stage

            stage = get_current_stage()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            for i, env_path in enumerate(env.scene.env_prim_paths):
                prim = stage.GetPrimAtPath(f"{env_path}/Object")
                if not prim.IsValid():
                    continue
                r = cache.ComputeLocalBound(prim).ComputeAlignedRange()
                lo, hi = r.GetMin(), r.GetMax()
                radius[i] = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
                half_h[i] = 0.5 * (hi[2] - lo[2])
        except Exception as exc:  # noqa: BLE001  (a missing stage leaves the default extents)
            print(f"[pick] object extents fall back to 2 cm: {exc!r}")
        return radius, half_h


def _state(env: ManagerBasedRLEnv) -> PickState:
    st = getattr(env, "_pick_state", None)
    if st is None:
        st = PickState(env)
        env._pick_state = st
    return st


def _frames(env: ManagerBasedRLEnv):
    """Base position/yaw quaternion, head (jaw_soft) pose, object centre, all in the world frame."""
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    st = _state(env)
    base_p = robot.data.root_link_pos_w.torch
    base_qy = yaw_quat(robot.data.root_link_quat_w.torch)
    jaw_p = robot.data.body_link_pos_w.torch[:, st.jaw_idx]
    jaw_q = robot.data.body_link_quat_w.torch[:, st.jaw_idx]
    obj_p = obj.data.root_link_pos_w.torch
    return st, robot, obj, base_p, base_qy, jaw_p, jaw_q, obj_p


def _beak_tip(jaw_p: torch.Tensor, jaw_q: torch.Tensor) -> torch.Tensor:
    off = torch.tensor(BEAK_TIP_OFFSET, device=jaw_p.device).expand_as(jaw_p)
    return jaw_p + quat_apply(jaw_q, off)


def _camera_sees(jaw_p: torch.Tensor, jaw_q: torch.Tensor, obj_p: torch.Tensor) -> torch.Tensor:
    """True where the object centre lies inside the head camera's frustum and range."""
    cam = jaw_p + quat_apply(jaw_q, torch.tensor(HEAD_CAMERA_OFFSET, device=jaw_p.device).expand_as(jaw_p))
    v = quat_apply_inverse(jaw_q, obj_p - cam)  # head frame: forward = -z, up = +x, left = +y
    fwd, up, y = -v[:, 2], v[:, 0], v[:, 1]
    t = math.radians(HEAD_CAMERA_TILT_DEG)
    # camera frame: optical axis = forward tilted down by t
    x = fwd * math.cos(t) - up * math.sin(t)
    z = fwd * math.sin(t) + up * math.cos(t)
    rng = v.norm(dim=-1)
    az = torch.atan2(y, x.clamp_min(1e-6))
    el = torch.atan2(z, x.clamp_min(1e-6))
    return (x > 0.02) & (rng < CAMERA_RANGE_M) & (az.abs() < math.radians(CAMERA_HFOV_DEG / 2)) & (el.abs() < math.radians(CAMERA_VFOV_DEG / 2))


def _rel_b(base_p: torch.Tensor, base_qy: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(base_qy, p - base_p)


# ----------------------------------------------------------------------------------------------- events


def reset_pick_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    dist_range: tuple[float, float] = (0.25, 0.9),
    bearing_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Drop the object on the floor ahead of the (already reset) robot and clear the pick state."""
    st, robot, obj, base_p, base_qy, jaw_p, jaw_q, obj_p = _frames(env)
    d = env.device
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=d)[env_ids if env_ids is not None else slice(None)]
    env_ids = torch.as_tensor(env_ids, device=d)
    n = len(env_ids)
    # release: restore the head mass
    robot.set_masses_index(masses=st.jaw_mass0[env_ids].unsqueeze(1), body_ids=[st.jaw_idx], env_ids=env_ids)
    st.grasped[env_ids] = False
    r = torch.empty(n, device=d).uniform_(*dist_range)
    b = torch.empty(n, device=d).uniform_(-math.radians(bearing_deg), math.radians(bearing_deg))
    local = torch.stack((r * torch.cos(b), r * torch.sin(b), torch.zeros_like(r)), dim=-1)
    pos = base_p[env_ids] + quat_apply(base_qy[env_ids], local)
    pos[:, 2] = env.scene.env_origins[env_ids, 2] + st.half_h[env_ids] + 0.003
    yaw = torch.empty(n, device=d).uniform_(-math.pi, math.pi)
    quat = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)), dim=-1)
    obj.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=d), env_ids=env_ids)
    st.spawn_z[env_ids] = pos[:, 2]
    rel = _rel_b(base_p[env_ids], base_qy[env_ids], pos)
    st.est[env_ids] = rel + 0.02 * torch.randn_like(rel)  # the detector's first fix, before the first frame
    st.seen[env_ids] = False
    st.dist_prev[env_ids] = rel[:, :2].norm(dim=-1)


# ----------------------------------------------------------------------------------------------- state update (reward slot)


def pick_grasp(
    env: ManagerBasedRLEnv,
    tip_tol_m: float = 0.015,
    beak_down_min: float = 0.25,
    dropout_prob: float = 0.10,
    noise_m: float = 0.01,
    noise_per_m: float = 0.03,
) -> torch.Tensor:
    """Advance the pick state once per step and pay 1 while the object is held.

    Keep this the FIRST pick term in the rewards config: the other pick terms, the seek command and the
    observations read what it writes. (A zero-weight reward term is skipped by the manager, so this term carries
    the hold reward rather than sitting at weight 0.)
    """
    st, robot, obj, base_p, base_qy, jaw_p, jaw_q, obj_p = _frames(env)
    if st.step_updated == env.common_step_counter:
        return st.grasped.float()
    st.step_updated = env.common_step_counter
    tip = _beak_tip(jaw_p, jaw_q)
    top = obj_p.clone()
    top[:, 2] += st.half_h
    st.tip_dist = (tip - top).norm(dim=-1)
    beak_axis = quat_apply(jaw_q, torch.tensor(HEAD_FORWARD, device=env.device).expand_as(jaw_p))
    new = (~st.grasped) & (st.tip_dist < tip_tol_m) & (-beak_axis[:, 2] > beak_down_min)
    if new.any():
        ids = new.nonzero(as_tuple=False).squeeze(-1)
        st.grasped[ids] = True
        m_obj = obj.data.body_mass.torch[ids, 0]
        robot.set_masses_index(masses=(st.jaw_mass0[ids] + m_obj).unsqueeze(1), body_ids=[st.jaw_idx], env_ids=ids)
    if st.grasped.any():
        ids = st.grasped.nonzero(as_tuple=False).squeeze(-1)
        held = tip[ids].clone()
        held[:, 2] -= st.half_h[ids]
        obj.write_root_pose_to_sim_index(root_pose=torch.cat((held, jaw_q[ids]), dim=-1), env_ids=ids)
        vel = torch.zeros(len(ids), 6, device=env.device)
        vel[:, :3] = robot.data.body_link_lin_vel_w.torch[ids, st.jaw_idx]
        obj.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
        obj_p = obj.data.root_link_pos_w.torch
    # camera-limited estimate
    seen = _camera_sees(jaw_p, jaw_q, obj_p) & (torch.rand(env.num_envs, device=env.device) > dropout_prob)
    rel = _rel_b(base_p, base_qy, obj_p)
    dist = rel.norm(dim=-1)
    noisy = rel + torch.randn_like(rel) * (noise_m + noise_per_m * dist).unsqueeze(-1)
    st.est = torch.where(seen.unsqueeze(-1), noisy, st.est)
    st.seen = seen
    return st.grasped.float()


# ----------------------------------------------------------------------------------------------- observations


def object_obs(env: ManagerBasedRLEnv, start: int = 0, stop: int = 10) -> torch.Tensor:
    """Camera-style object estimate, 10 floats: xyz in the base-yaw frame, seen, held, size (2 x half-width), 4 spare."""
    st = _state(env)
    out = torch.zeros(env.num_envs, 10, device=env.device)
    out[:, 0:3] = st.est
    out[:, 3] = st.seen.float()
    out[:, 4] = st.grasped.float()
    out[:, 5] = 2.0 * st.radius
    return out[:, start:stop]


def object_obs_true(env: ManagerBasedRLEnv, start: int = 0, stop: int = 10) -> torch.Tensor:
    """Privileged version for the critic: true position, beak-tip distance, half-height."""
    st, robot, obj, base_p, base_qy, jaw_p, jaw_q, obj_p = _frames(env)
    out = torch.zeros(env.num_envs, 10, device=env.device)
    out[:, 0:3] = _rel_b(base_p, base_qy, obj_p)
    out[:, 3] = st.seen.float()
    out[:, 4] = st.grasped.float()
    out[:, 5] = 2.0 * st.radius
    out[:, 6] = st.tip_dist
    out[:, 7] = st.half_h
    return out[:, start:stop]


# ----------------------------------------------------------------------------------------------- rewards


def object_progress(env: ManagerBasedRLEnv, clip_m: float = 0.05) -> torch.Tensor:
    """Metres the base closed on the object this step (zero once held, zero when standing still)."""
    st, robot, obj, base_p, base_qy, jaw_p, jaw_q, obj_p = _frames(env)
    dist = (obj_p - base_p)[:, :2].norm(dim=-1)
    prog = (st.dist_prev - dist).clamp(-clip_m, clip_m)
    st.dist_prev = dist
    return torch.where(st.grasped, torch.zeros_like(prog), prog)


def beak_to_object(env: ManagerBasedRLEnv, std: float = 0.04, gate_xy_m: float = 0.3) -> torch.Tensor:
    """exp(-d/std) of beak tip to object top, only once the object is within ``gate_xy_m`` of the base and not held."""
    st, robot, obj, base_p, base_qy, jaw_p, jaw_q, obj_p = _frames(env)
    near = (obj_p - base_p)[:, :2].norm(dim=-1) < gate_xy_m
    r = torch.exp(-st.tip_dist / std)
    return torch.where(near & ~st.grasped, r, torch.zeros_like(r))


def object_lift(env: ManagerBasedRLEnv, height_m: float = 0.06) -> torch.Tensor:
    """Fraction of ``height_m`` the held object has been raised above where it lay."""
    st, robot, obj, base_p, base_qy, jaw_p, jaw_q, obj_p = _frames(env)
    lift = ((obj_p[:, 2] - st.spawn_z) / height_m).clamp(0.0, 1.0)
    return torch.where(st.grasped, lift, torch.zeros_like(lift))


# ----------------------------------------------------------------------------------------------- seek command


class ObjectSeekCommand(MicroDuckVelocityCommand):
    """Twist toward the object ESTIMATE (the same estimate the policy sees): forward speed proportional to the
    remaining distance, yaw rate proportional to the bearing, zero once the object is under the beak or held.
    Fills the ``base_velocity`` command slot, so the walk's tracking rewards and the 61-float observation contract
    are unchanged; on the robot the same law runs on the camera estimate."""

    def _resample_command(self, env_ids: Sequence[int]):
        self.vel_command_b[env_ids] = 0.0
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False

    def _update_command(self):
        cfg: ObjectSeekCommandCfg = self.cfg
        st = _state(self._env)
        rel = st.est
        d = rel[:, :2].norm(dim=-1)
        b = torch.atan2(rel[:, 1], rel[:, 0])
        vx = (cfg.k_v * (d - cfg.stop_dist_m)).clamp(0.0, cfg.max_vx) * torch.cos(b).clamp_min(0.0)
        wz = (cfg.k_w * b).clamp(-cfg.max_wz, cfg.max_wz)
        arrived = (d < cfg.stop_dist_m + 0.02) & (b.abs() < math.radians(15.0))
        stop = st.grasped | arrived
        self.vel_command_b[:, 0] = torch.where(stop, torch.zeros_like(vx), vx)
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = torch.where(stop, torch.zeros_like(wz), wz)
        self.is_standing_env[:] = stop


@configclass
class ObjectSeekCommandCfg(MicroDuckVelocityCommandCfg):
    class_type: type = ObjectSeekCommand
    k_v: float = 1.5
    k_w: float = 2.0
    stop_dist_m: float = 0.13
    """Base-to-object distance at which the beak reaches the object with the neck down."""
    max_vx: float = 0.4
    max_wz: float = 1.0
