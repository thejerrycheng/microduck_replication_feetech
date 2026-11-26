"""Grey-box model of N electric ducted fans on one LiPo pack. Backend-agnostic: pass ``numpy`` or ``torch`` as ``xp``;
every array has a trailing fan axis ``(..., N)`` (battery quantities ``(...,)``).

Per fan (state: rotor speed ``w`` [rad/s]; input: throttle ``u`` in [0, 1] after the command delay):

    i     = clip((u V_b - Ke w) / R, i_min, i_max)              phase current, inductance neglected
    J w'  = Kt i - kq w^2 - Kt I0 - b w                         rotor
    v_e   = kv w                                                jet exit speed (fixed-geometry fan)
    mdot  = rho A v_e
    T     = eta_inst mdot (v_e - v_ax)+                         momentum thrust, v_ax = airspeed along the thrust axis
    F_ram = -mdot v_perp                                        inlet momentum drag, applied at the duct lip
    tau   = -s (kappa kq w^2 + J w') z  -  Omega x (s J w z)    residual swirl + spin-up reaction + gyroscopic

``Ke = Kt = 60 / (2 pi Kv)``. Static thrust is ``kf w^2`` with ``kf = eta_inst rho A kv^2``.

Battery (state: state of charge ``soc``, RC voltage ``v1``), ``n_s`` cells in series:

    I_b   = sum_k u_k i_k + I_avionics                          bus current (ESC as an ideal buck stage)
    V_b   = n_s OCV(soc) - R0 I_b - v1 ,   v1' = -v1/(R1 C1) + I_b/C1 ,   soc' = -I_b / (3600 Q)

``V_b`` and ``I_b`` are coupled linearly through ``i``; :func:`bus_voltage` solves that loop in closed form (then
re-applies the current clip once). Parameters are priors until identified: see scripts/tools/fan_sysid.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

RHO = 1.225

# per-cell open-circuit voltage of a LiPo cell against state of charge (rested cell, 25 C; generic LiCoO2/graphite)
_OCV_SOC = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
_OCV_V = (3.27, 3.61, 3.69, 3.73, 3.77, 3.80, 3.84, 3.87, 3.95, 4.02, 4.11, 4.20)


@dataclass(frozen=True)
class FanParams:
    kv_rpm_per_v: float = 4600.0   # motor velocity constant
    R: float = 0.030               # phase-to-bus equivalent winding + ESC resistance [ohm]
    J: float = 6.0e-6              # rotor inertia (fan + bell) [kg m^2]
    kq: float = 1.5e-9             # aerodynamic torque coefficient [N m s^2]
    kv_exit: float = 0.0108        # exit speed per rotor speed [m]
    area: float = 1.432e-3         # fan annulus [m^2]
    i0: float = 2.5                # no-load current [A]
    b: float = 0.0                 # viscous rotor friction [N m s]
    i_max: float = 55.0            # ESC current limit [A]
    i_min: float = 0.0             # 0: no regenerative braking; negative: damped-light braking limit
    kappa: float = 0.2             # share of rotor torque left as swirl after the stators (airframe reaction)
    eta_inst: float = 1.0          # installed / bench thrust
    u_idle: float = 0.0            # ESC dead band

    @property
    def ke(self) -> float:
        return 60.0 / (2.0 * math.pi * self.kv_rpm_per_v)

    @property
    def kf(self) -> float:
        return self.eta_inst * RHO * self.area * self.kv_exit ** 2


@dataclass(frozen=True)
class BatteryParams:
    n_s: int = 4
    q_ah: float = 1.3
    r_cell: float = 0.0035         # ohmic resistance per cell incl. tabs [ohm]
    r_lead: float = 0.004          # connector + leads [ohm]
    r1: float = 0.004              # polarisation resistance, whole pack [ohm]
    tau1: float = 8.0              # polarisation time constant [s]
    i_avionics: float = 0.3

    @property
    def r0(self) -> float:
        return self.n_s * self.r_cell + self.r_lead


def ocv_cell(soc, xp):
    s = xp.clip(soc, 0.0, 1.0)
    v = s * 0.0 + _OCV_V[0]
    for k in range(len(_OCV_SOC) - 1):
        a, b_ = _OCV_SOC[k], _OCV_SOC[k + 1]
        seg = xp.clip((s - a) / (b_ - a), 0.0, 1.0)
        v = v + seg * (_OCV_V[k + 1] - _OCV_V[k])
    return v


def phase_current(u, w, v_bus, p: FanParams, xp):
    return xp.clip((u * v_bus[..., None] - p.ke * w) / p.R, p.i_min, p.i_max)


def bus_voltage(u, w, soc, v1, p: FanParams, bat: BatteryParams, xp):
    """Terminal voltage with the fans loading the pack. Returns (V_b, I_b, i_phase)."""
    v_oc = bat.n_s * ocv_cell(soc, xp) - v1
    a = xp.sum(u * u, axis=-1) / p.R                 # dI_b/dV_b
    c = xp.sum(u * p.ke * w, axis=-1) / p.R
    v_b = (v_oc + bat.r0 * (c - bat.i_avionics)) / (1.0 + bat.r0 * a)
    for _ in range(2):                                # re-solve with the clipped currents
        i = phase_current(u, w, v_b, p, xp)
        i_b = xp.sum(u * i, axis=-1) + bat.i_avionics
        v_b = v_oc - bat.r0 * i_b
    i = phase_current(u, w, v_b, p, xp)
    return v_b, xp.sum(u * i, axis=-1) + bat.i_avionics, i


def thrust(w, v_ax, p: FanParams, xp):
    v_e = p.kv_exit * w
    return p.eta_inst * RHO * p.area * v_e * xp.clip(v_e - v_ax, 0.0, None)


def mass_flow(w, p: FanParams):
    return RHO * p.area * p.kv_exit * w


def step(w, soc, v1, u, dt, p: FanParams, bat: BatteryParams, xp, substeps: int = 4):
    """Advance rotor speeds and battery by ``dt``. Returns (w, soc, v1, aux) with aux = dict(v_bus, i_bus, i_phase, w_dot)."""
    u = xp.clip(u, 0.0, 1.0)
    u = xp.where(u < p.u_idle, u * 0.0, u)
    h = dt / substeps
    w0 = w
    for _ in range(substeps):
        v_b, i_b, i = bus_voltage(u, w, soc, v1, p, bat, xp)
        f = (p.ke * i - p.kq * w * w - p.ke * p.i0 * (w > 1.0) - p.b * w) / p.J
        dfdw = -(p.ke * p.ke / p.R * u * 0.0 + p.ke * p.ke / p.R + 2.0 * p.kq * w + p.b) / p.J
        w = xp.clip(w + h * f / (1.0 - h * dfdw), 0.0, None)   # linearly implicit Euler: stable for any dt
        soc = soc - h * i_b / (3600.0 * bat.q_ah)
        v1 = v1 + h * (-v1 / bat.tau1 + i_b * bat.r1 / bat.tau1)
    v_b, i_b, i = bus_voltage(u, w, soc, v1, p, bat, xp)
    return w, soc, v1, dict(v_bus=v_b, i_bus=i_b, i_phase=i, w_dot=(w - w0) / dt)


def steady_speed(u, v_bus, p: FanParams):
    """Rotor speed at constant throttle and bus voltage (scalar closed form, current clip ignored)."""
    a, b_, c = p.kq, p.ke * p.ke / p.R + p.b, -(p.ke * u * v_bus / p.R - p.ke * p.i0)
    return max((-b_ + math.sqrt(max(b_ * b_ - 4 * a * c, 0.0))) / (2 * a), 0.0)


def steady_state(u, n_fans, soc, p: FanParams, bat: BatteryParams, v1: float = 0.0):
    """Scalar steady state of n identical fans at throttle u: dict(w, thrust, i_phase, i_bus, v_bus, p_elec)."""
    v_oc = bat.n_s * (lambda s: float(ocv_cell(_np().asarray(s), _np())))(soc) - v1
    v_b = v_oc
    for _ in range(60):
        w = steady_speed(u, v_b, p)
        i = min(max((u * v_b - p.ke * w) / p.R, p.i_min), p.i_max)
        if i >= p.i_max:  # current-limited: torque balance at the limit
            w = math.sqrt(max(p.ke * (p.i_max - p.i0), 0.0) / p.kq)
        i_b = n_fans * u * i + bat.i_avionics
        v_new = v_oc - bat.r0 * i_b
        if abs(v_new - v_b) < 1e-6:
            break
        v_b = 0.5 * (v_b + v_new)
    return dict(w=w, thrust=p.kf * w * w, i_phase=i, i_bus=i_b, v_bus=v_b, p_elec=v_b * i_b)


def throttle_for_thrust(t_each, n_fans, soc, p: FanParams, bat: BatteryParams):
    """Throttle that makes each of n fans produce ``t_each`` in steady state, or None if u = 1 is not enough."""
    if steady_state(1.0, n_fans, soc, p, bat)["thrust"] < t_each:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if steady_state(mid, n_fans, soc, p, bat)["thrust"] < t_each:
            lo = mid
        else:
            hi = mid
    return hi


def calibrate(t_ref_n: float, v_ref: float, i_ref: float, base: FanParams = FanParams()) -> FanParams:
    """Priors from one bench point (static thrust, voltage at the ESC, current): sets kv_exit and kq.
    R and i0 stay at their motor-class priors; the implied fan efficiency is returned by :func:`fan_efficiency`."""
    w = (v_ref - i_ref * base.R) / base.ke
    kq = base.ke * (i_ref - base.i0) / (w * w)
    kv_exit = math.sqrt(t_ref_n / (base.eta_inst * RHO * base.area)) / w
    return replace(base, kq=kq, kv_exit=kv_exit)


def fan_efficiency(p: FanParams) -> float:
    """Ideal jet power over shaft power at static conditions (independent of speed in this model)."""
    return (0.5 * RHO * p.area * p.kv_exit ** 3) / p.kq


def _np():
    import numpy
    return numpy
