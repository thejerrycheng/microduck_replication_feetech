"""Flying duck: the ducted-fan jetpack as Isaac Lab terms.

* :func:`attach_flight_pack` (startup event) adds the flight hardware's mass, centre of mass and inertia
  (``microduck_rl/flight/flight_pack.json``, built by ``scripts/tools/flight_model.py``) to the trunk body, with a
  per-environment draw of its mass and position. The official USD is not edited.
* :class:`FanThrottleAction` (4 actions) is the plant: command delay, the grey-box fan + pack model of
  :mod:`microduck_rl.flight.fan_model` stepped at the physics rate, and the resulting wrench on the trunk
  (momentum thrust with inflow, inlet momentum drag, swirl / spin-up reaction, rotor gyroscopics).
  Fan and pack parameters are redrawn per environment at reset.
* :class:`TakeoffCommand` raises a take-off flag at a random time and then commands a vertical velocity: with the
  walk's (vx, vy, wz) command the policy tracks a full (vx, vy, vz, wz) velocity command.
* observation and reward terms for the take-off task, and :func:`gated` to switch a ground-only reward off in flight.

Nothing here has been identified on hardware: the parameter ranges are the priors of ``scripts/tools/fan_sysid.py``.
"""

from __future__ import annotations

import inspect
import json
import math
import os
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.utils.configclass import configclass

from microduck_rl.flight import fan_model as F

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = [
    "FLIGHT_PACK",
    "attach_flight_pack",
    "FanThrottleAction",
    "FanThrottleActionCfg",
    "TakeoffCommand",
    "TakeoffCommandCfg",
    "fan_state",
    "base_height_noisy",
    "flight_vz_tracking",
    "flight_airborne",
    "flight_feet_off_ground",
    "flight_upright",
    "fan_throttle_ground",
    "fan_throttle_rate",
    "above_ceiling",
    "gated",
]

_PACK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flight", "flight_pack.json")
with open(_PACK_PATH) as _f:
    FLIGHT_PACK: dict = json.load(_f)
"""Mass properties of the flight hardware and fan geometry in the ``trunk_base`` frame."""

TRUNK = "trunk_base"
FANS_TERM = "fans"
TAKEOFF_CMD = "takeoff"
STAND_Z = 0.12


def _fans(env) -> "FanThrottleAction":
    return env.action_manager.get_term(FANS_TERM)


def _height(env, asset_name: str = "robot") -> torch.Tensor:
    z = env.scene[asset_name].data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.nan_to_num(z, nan=STAND_Z)


def flight_flag(env) -> torch.Tensor:
    """1 once the take-off command is raised, else 0. Shape (num_envs,)."""
    return env.command_manager.get_command(TAKEOFF_CMD)[:, 0]


# ------------------------------------------------------------------------------------------------ pack on the trunk
def attach_flight_pack(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[TRUNK]),
    mass_scale_range: tuple[float, float] = (0.95, 1.08),
    com_offset_m: float = 0.006,
):
    """Add the flight hardware to the trunk: composite mass, CoM and inertia (parallel-axis), per environment.

    Run as a ``startup`` event AFTER the mass randomisation of the trunk (terms added in ``__post_init__`` come last),
    so the randomised scale applies to the bare trunk and not to the pack; the CoM randomisation, which stores its
    defaults on its first reset, then draws around the composite CoM.
    """
    asset = env.scene[asset_cfg.name]
    b = asset_cfg.body_ids
    b = [int(b[0])] if not isinstance(b, slice) else [int(asset.find_bodies(TRUNK)[0][0])]   # plain ints: the setters build warp arrays
    dev = asset.device
    n = env.num_envs
    m0 = asset.data.body_mass.torch[:, b[0]].clone()                       # (N,)
    c0 = asset.data.body_com_pos_b.torch[:, b[0], :3].clone()              # (N, 3)
    I0 = asset.data.body_inertia.torch[:, b[0]].clone().reshape(n, 3, 3)   # about the trunk CoM, body axes

    mp = FLIGHT_PACK["mass"] * torch.empty(n, device=dev).uniform_(*mass_scale_range)
    cp = torch.tensor(FLIGHT_PACK["com"], device=dev).repeat(n, 1) + torch.empty(n, 3, device=dev).uniform_(-com_offset_m, com_offset_m)
    Ip = torch.tensor(FLIGHT_PACK["inertia"], device=dev).repeat(n, 1, 1) * (mp / FLIGHT_PACK["mass"])[:, None, None]

    m = m0 + mp
    c = (m0[:, None] * c0 + mp[:, None] * cp) / m[:, None]
    eye = torch.eye(3, device=dev)

    def shift(mass, d):  # parallel-axis term of a point mass at offset d
        return mass[:, None, None] * ((d * d).sum(-1)[:, None, None] * eye - d[:, :, None] * d[:, None, :])

    inertia = I0 + shift(m0, c0 - c) + Ip + shift(mp, cp - c)
    asset.set_masses_index(masses=m[:, None], body_ids=b)
    asset.set_coms_index(coms=c[:, None, :], body_ids=b)
    asset.set_inertias_index(inertias=inertia.reshape(n, 1, 9), body_ids=b)


# ------------------------------------------------------------------------------------------------ fan plant
class FanThrottleAction(ActionTerm):
    """Four ducted fans on one LiPo pack: throttle commands in, a wrench on the trunk out.

    The raw action ``a`` maps to throttle ``u = clip(offset + scale * a, 0, 1)``. :meth:`apply_actions` runs every
    physics step (or once per control step when the physics owns the decimation, as Newton does): it advances the
    command delay line and the rotor and battery states by the physics steps that elapsed, and writes the net force
    and torque (body frame, about the trunk's CoM) into the articulation's permanent wrench composer.
    """

    cfg: "FanThrottleActionCfg"

    def __init__(self, cfg: "FanThrottleActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self._trunk = int(self._asset.find_bodies(TRUNK)[0][0])
        fans = FLIGHT_PACK["fans"]
        self._pos = torch.tensor([f["pos"] for f in fans], device=dev)            # (4, 3) trunk frame
        self._axis = torch.tensor([f["axis"] for f in fans], device=dev)          # (4, 3)
        self._spin = torch.tensor([float(f["spin"]) for f in fans], device=dev)   # (4,)
        fp, bp = FLIGHT_PACK["fan"], FLIGHT_PACK["battery"]
        base = F.calibrate(fp["bench_thrust_n"], fp["bench_v"], fp["bench_i"],
                           F.FanParams(kv_rpm_per_v=fp["kv_rpm_per_v"], i_max=fp["i_limit"], area=fp["area"]))
        self._base = replace(base, eta_inst=fp["eta_inst"])
        self._bat = F.BatteryParams(n_s=bp["n_s"], q_ah=bp["q_ah"], r_cell=bp["r_cell"])
        self.w_ref = math.sqrt(FLIGHT_PACK["mass"] * 0 + 4.0 / self._base.kf)      # rotor speed at 4 N static thrust
        # per-environment parameters (tensors broadcast inside fan_model): thrust gain per fan, torque coefficient,
        # rotor inertia, cell resistance, command delay
        self._eta = torch.full((n, 4), self._base.eta_inst, device=dev)
        self._kq = torch.full((n, 1), self._base.kq, device=dev)
        self._J = torch.full((n, 1), self._base.J, device=dev)
        self._r_cell = torch.full((n,), self._bat.r_cell, device=dev)
        self._delay = torch.zeros(n, dtype=torch.long, device=dev)
        self._queue = torch.zeros(n, cfg.max_delay_steps + 1, 4, device=dev)
        # states
        self.w = torch.zeros(n, 4, device=dev)
        self.soc = torch.full((n,), 0.9, device=dev)
        self.v1 = torch.zeros(n, device=dev)
        self.thrust = torch.zeros(n, 4, device=dev)
        self.v_bus = torch.full((n,), self._bat.n_s * 4.1, device=dev)
        self.i_bus = torch.zeros(n, device=dev)
        self._raw = torch.zeros(n, 4, device=dev)
        self._u = torch.zeros(n, 4, device=dev)
        self._u_prev = torch.zeros(n, 4, device=dev)
        self._all = torch.arange(n, device=dev)
        self._last_counter = int(getattr(env, "_sim_step_counter", 0))
        self._randomize(self._all)

    # -- ActionTerm interface
    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._u

    @property
    def throttle_rate(self) -> torch.Tensor:
        """Change of the throttle command over the last control step. Shape (num_envs, 4)."""
        return self._u - self._u_prev

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._u_prev[:] = self._u
        self._u[:] = torch.clamp(self.cfg.offset + self.cfg.scale * torch.nan_to_num(actions, nan=0.0), 0.0, 1.0)

    def apply_actions(self):
        # With physics-owned decimation (Newton) this is called once per control step, covering `decimation` physics
        # steps; otherwise once per physics step. Advance the delay line and the fan model by the steps that elapsed.
        dt = self._env.physics_dt
        counter = int(self._env._sim_step_counter)
        n_steps = max(counter - self._last_counter, 1)
        self._last_counter = counter
        p = replace(self._base, eta_inst=self._eta, kq=self._kq, J=self._J)
        bat = replace(self._bat, r_cell=self._r_cell)
        w_start = self.w
        for _ in range(n_steps):
            self._queue = torch.roll(self._queue, 1, dims=1)
            self._queue[:, 0] = self._u
            u = self._queue[self._all, self._delay]
            self.w, self.soc, self.v1, aux = F.step(self.w, self.soc, self.v1, u, dt, p, bat, torch, substeps=1)
        aux["w_dot"] = (self.w - w_start) / (n_steps * dt)
        self.soc = torch.clamp(self.soc, 0.0, 1.0)
        self.v_bus, self.i_bus = aux["v_bus"], aux["i_bus"]

        data = self._asset.data
        v = torch.nan_to_num(data.root_com_lin_vel_b.torch, nan=0.0)                       # (N, 3) trunk frame
        om = torch.nan_to_num(data.root_com_ang_vel_b.torch, nan=0.0)
        com = data.body_com_pos_b.torch[:, self._trunk, :3]
        r = self._pos[None] - com[:, None]                                                  # (N, 4, 3)
        axis = self._axis[None].expand_as(r)
        v_pt = v[:, None] + torch.cross(om[:, None].expand_as(r), r, dim=-1)
        v_ax = (v_pt * axis).sum(-1)
        thrust = F.thrust(self.w, v_ax, p, torch)                                          # (N, 4)
        mdot = F.mass_flow(self.w, self._base)
        force = thrust[..., None] * axis - mdot[..., None] * (v_pt - v_ax[..., None] * axis)
        spin = self._spin[None, :]
        react = -spin * (self._base.kappa * self._kq * self.w * self.w + self._J * aux["w_dot"])
        gyro = -torch.cross(om[:, None].expand_as(r), (spin * self._J * self.w)[..., None] * axis, dim=-1)
        torque = torch.cross(r, force, dim=-1) + react[..., None] * axis + gyro
        self.thrust = thrust
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force.sum(1, keepdim=True), torques=torque.sum(1, keepdim=True), body_ids=[self._trunk], is_global=False
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            ids = self._all
        elif isinstance(env_ids, slice):
            ids = self._all[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
        self._raw[ids] = 0.0
        self._u[ids] = 0.0
        self._u_prev[ids] = 0.0
        self._queue[ids] = 0.0
        self.w[ids] = 0.0
        self.v1[ids] = 0.0
        self.thrust[ids] = 0.0
        self._randomize(ids)

    def _randomize(self, ids: torch.Tensor):
        c, k, dev = self.cfg, len(ids), self.device

        def U(lo_hi, *shape):
            return torch.empty(*shape, device=dev).uniform_(*lo_hi)

        self._eta[ids] = self._base.eta_inst * U(c.thrust_scale_range, k, 1) * U(c.per_fan_thrust_range, k, 4)
        self._kq[ids] = self._base.kq * U(c.torque_coeff_range, k, 1)
        self._J[ids] = self._base.J * U(c.rotor_inertia_range, k, 1)
        self._r_cell[ids] = self._bat.r_cell * U(c.cell_resistance_range, k)
        self.soc[ids] = U(c.soc_range, k)
        self._delay[ids] = torch.randint(c.min_delay_steps, c.max_delay_steps + 1, (k,), device=dev)


@configclass
class FanThrottleActionCfg(ActionTermCfg):
    """Configuration of the fan plant. Ranges are multipliers on the calibrated prior unless noted."""

    class_type: type[ActionTerm] = FanThrottleAction
    asset_name: str = "robot"
    offset: float = 0.0
    """Throttle at zero action: the fans are off until the policy asks for them."""
    scale: float = 0.5
    thrust_scale_range: tuple[float, float] = (0.80, 1.20)
    """All four fans together: seller thrust is unverified and the install loss is a guess."""
    per_fan_thrust_range: tuple[float, float] = (0.93, 1.07)
    torque_coeff_range: tuple[float, float] = (0.85, 1.15)
    rotor_inertia_range: tuple[float, float] = (0.7, 1.8)
    cell_resistance_range: tuple[float, float] = (0.6, 1.6)
    soc_range: tuple[float, float] = (0.45, 0.98)
    min_delay_steps: int = 1
    max_delay_steps: int = 4
    """Command delay in physics steps (5 ms each): DShot + ESC + FC loop."""


# ------------------------------------------------------------------------------------------------ take-off command
class TakeoffCommand(CommandTerm):
    """(flag, vertical velocity): the walk's velocity command gets its missing z component.

    The flag rises at a per-episode time (take-off). From then on the term commands a vertical velocity [m/s]: a climb
    for ``climb_s``, then a value redrawn every ``resample_s`` (zero with probability ``hover_prob``). Outside the
    height band the command is overridden so the duck comes back into it. Forward, sideways and yaw-rate commands stay
    with ``base_velocity``; together the policy tracks (vx, vy, vz, wz).
    """

    cfg: "TakeoffCommandCfg"

    def __init__(self, cfg: "TakeoffCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self._cmd = torch.zeros(n, 2, device=dev)
        self._t_go = torch.full((n,), 1.0e9, device=dev)
        self._vz_climb = torch.zeros(n, device=dev)
        self._vz = torch.zeros(n, device=dev)
        self._next = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        k = len(env_ids)
        t = torch.empty(k, device=self.device).uniform_(*self.cfg.takeoff_time_range_s)
        stay = torch.rand(k, device=self.device) < self.cfg.no_takeoff_prob
        self._t_go[env_ids] = torch.where(stay, torch.full_like(t, 1.0e9), t)
        self._vz_climb[env_ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.climb_vz_range)
        self._vz[env_ids] = 0.0
        self._next[env_ids] = 0.0
        self._cmd[env_ids] = 0.0

    def _update_command(self):
        c = self.cfg
        t = self._env.episode_length_buf.float() * self._env.step_dt
        since = t - self._t_go
        flag = (since >= 0.0).float()
        # after the climb, redraw the vertical velocity every resample_s
        due = (since >= c.climb_s) & (t >= self._next)
        if due.any():
            ids = due.nonzero(as_tuple=False).squeeze(-1)
            vz = torch.empty(len(ids), device=self.device).uniform_(*c.vz_range)
            vz = torch.where(torch.rand(len(ids), device=self.device) < c.hover_prob, torch.zeros_like(vz), vz)
            self._vz[ids] = vz
            self._next[ids] = t[ids] + c.resample_s
        vz = torch.where(since < c.climb_s, self._vz_climb, self._vz)
        z = _height(self._env)
        vz = torch.where(z > c.height_band_m[1], torch.clamp(vz, max=-0.3), vz)
        vz = torch.where((z < c.height_band_m[0]) & (since > c.climb_s), torch.clamp(vz, min=0.2), vz)
        self._cmd[:, 0] = flag
        self._cmd[:, 1] = flag * vz


@configclass
class TakeoffCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = TakeoffCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """The take-off time is drawn once per episode (at reset)."""
    takeoff_time_range_s: tuple[float, float] = (2.0, 5.0)
    climb_vz_range: tuple[float, float] = (0.3, 0.6)
    climb_s: float = 1.2
    vz_range: tuple[float, float] = (-0.4, 0.6)
    hover_prob: float = 0.4
    resample_s: float = 2.0
    height_band_m: tuple[float, float] = (0.25, 1.0)
    no_takeoff_prob: float = 0.15


# ------------------------------------------------------------------------------------------------ observations
def fan_state(env: ManagerBasedEnv) -> torch.Tensor:
    """What the ESC telemetry gives: four rotor speeds (bidirectional DShot eRPM) and the bus voltage, normalised."""
    f = _fans(env)
    return torch.cat((f.w / f.w_ref, (f.v_bus / (f._bat.n_s * 4.2))[:, None]), dim=-1)


def base_height_noisy(env: ManagerBasedEnv, noise_m: float = 0.01) -> torch.Tensor:
    """Trunk height over the floor as a downward range sensor would report it."""
    z = _height(env)
    return (z + noise_m * torch.randn_like(z))[:, None]


# ------------------------------------------------------------------------------------------------ rewards
def flight_vz_tracking(env: ManagerBasedRLEnv, std: float = 0.22, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vertical-velocity tracking after the take-off flag: exp(-(vz - vz_cmd)^2 / std^2)."""
    cmd = env.command_manager.get_command(TAKEOFF_CMD)
    vz = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return cmd[:, 0] * torch.exp(-torch.square((vz - cmd[:, 1]) / std))


def flight_airborne(env: ManagerBasedRLEnv, clearance_m: float = 0.05) -> torch.Tensor:
    """After the flag: being clear of the floor at all (the velocity term alone is satisfied by standing when vz_cmd = 0)."""
    return flight_flag(env) * (_height(env) > STAND_Z + clearance_m).float()


def flight_feet_off_ground(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    off = (torch.nan_to_num(f, nan=0.0).norm(dim=-1) < threshold).all(dim=-1).float()
    return flight_flag(env) * off


def flight_upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    g = torch.nan_to_num(env.scene[asset_cfg.name].data.projected_gravity_b.torch, nan=0.0)
    return flight_flag(env) * torch.exp(-torch.sum(torch.square(g[:, :2]), dim=-1) / 0.05)


def fan_throttle_ground(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared throttle before the take-off flag: the fans stay quiet while the legs run."""
    return (1.0 - flight_flag(env)) * torch.sum(torch.square(_fans(env).processed_actions), dim=-1)


def fan_throttle_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(_fans(env).throttle_rate), dim=-1)


def above_ceiling(env: ManagerBasedRLEnv, ceiling_m: float = 1.2) -> torch.Tensor:
    """Termination: the duck left the training volume."""
    return _height(env) > ceiling_m


def gated(term: RewardTermCfg, on_ground: bool = True) -> RewardTermCfg:
    """A copy of ``term`` that is zeroed in flight (``on_ground``) or on the ground (``not on_ground``).

    The wrapper keeps the wrapped callable's signature (the managers validate ``params`` against it) and
    instantiates class-based terms itself.
    """
    inner = term.func
    is_cls = inspect.isclass(inner)
    sig = inspect.signature(inner.__call__ if is_cls else inner)
    params = list(sig.parameters.values())
    if not is_cls:
        params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + params

    class _Gated(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._inner = inner(cfg, env) if is_cls else inner

        def reset(self, env_ids=None):
            if is_cls:
                self._inner.reset(env_ids)

        def __call__(self, env, *args, **kwargs):
            flag = flight_flag(env)
            return self._inner(env, *args, **kwargs) * ((1.0 - flag) if on_ground else flag)

    _Gated.__call__.__signature__ = sig.replace(parameters=params)
    name = f"gated_{'ground' if on_ground else 'air'}_{getattr(inner, '__name__', 'term')}"
    _Gated.__name__ = _Gated.__qualname__ = name
    _Gated.__module__ = __name__
    globals()[name] = _Gated      # configs are stored as 'module:name' strings and resolved again: the class must be importable
    return term.replace(func=_Gated)
