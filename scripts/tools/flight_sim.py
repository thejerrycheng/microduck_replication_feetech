"""MuJoCo side of the flying duck: the fan/battery plant applied to the trunk and a model-based flight controller.

Used by scripts/tools/flight_rollout.py (run with the app's environment: ~/.venvs/microduck_live). The plant is
microduck_rl/flight/fan_model.py, the same code the Isaac Lab task steps; the controller here is a classical baseline
(cascaded PD + allocation), not the learned policy.
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "microduck_rl", "flight"))
import fan_model as F

PACK = json.load(open(os.path.join(ROOT, "microduck_rl", "flight", "flight_pack.json")))
G = 9.81


def prior_params(thrust_scale=1.0):
    f = PACK["fan"]
    p = F.calibrate(f["bench_thrust_n"], f["bench_v"], f["bench_i"], F.FanParams(kv_rpm_per_v=f["kv_rpm_per_v"], i_max=f["i_limit"], area=f["area"]))
    b = PACK["battery"]
    return F.replace(p, eta_inst=f["eta_inst"] * thrust_scale), F.BatteryParams(n_s=b["n_s"], q_ah=b["q_ah"], r_cell=b["r_cell"])


class FanPlant:
    """Four fans + pack acting on the trunk through ``qfrc_applied``. Call :meth:`substep` once per physics step."""

    def __init__(self, model, data, params=None, battery=None, cmd_delay_steps=2, soc0=0.95):
        self.m, self.d = model, data
        self.p, self.bat = (params, battery) if params is not None else prior_params()
        self.sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fan_" + f["name"]) for f in PACK["fans"]]
        self.spin = np.array([f["spin"] for f in PACK["fans"]], dtype=float)
        self.body = model.site_bodyid[self.sites[0]]
        self.w = np.zeros((1, 4)); self.soc = np.array([soc0]); self.v1 = np.zeros(1)
        self.queue = [np.zeros(4) for _ in range(cmd_delay_steps + 1)]
        self.u = np.zeros(4)
        self.aux = dict(v_bus=np.array([self.bat.n_s * 4.15]), i_bus=np.zeros(1), i_phase=np.zeros((1, 4)), w_dot=np.zeros((1, 4)))
        self.thrust = np.zeros(4)
        self.lip = 0.5 * PACK["duct_len"]

    def command(self, u):
        self.u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)

    def substep(self):
        m, d, p = self.m, self.d, self.p
        self.queue.append(self.u.copy())
        u = self.queue.pop(0)
        dt = m.opt.timestep
        self.w, self.soc, self.v1, self.aux = F.step(self.w, self.soc, self.v1, u[None], dt, p, self.bat, np, substeps=2)
        vel = np.zeros(6)
        for k, s in enumerate(self.sites):
            R = d.site_xmat[s].reshape(3, 3)
            axis, pos = R[:, 2], d.site_xpos[s]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, s, vel, 0)       # world frame (rot, lin)
            v_ax = float(vel[3:] @ axis)
            wk = self.w[0, k]
            T = float(F.thrust(np.array(wk), v_ax, p, np))
            mdot = F.mass_flow(wk, p)
            v_perp = vel[3:] - v_ax * axis
            force = T * axis - mdot * v_perp                                          # thrust + inlet momentum drag
            # residual swirl + spin-up reaction about the axis; gyroscopic torque of the rotor
            tau = -self.spin[k] * (p.kappa * p.kq * wk * wk + p.J * self.aux["w_dot"][0, k]) * axis
            tau += -np.cross(vel[:3], self.spin[k] * p.J * wk * axis)
            mujoco.mj_applyFT(m, d, force, tau, pos + self.lip * axis * 0.0, self.body, d.qfrc_applied)
            self.thrust[k] = T


class FlightController:
    """Cascaded PD on altitude / horizontal velocity / attitude, then thrust allocation and an inverse fan map.
    Uses the simulator's true state (a real one needs the FC's estimator)."""

    def __init__(self, model, data, plant, inertia_diag, att_wn=(9.0, 8.0, 3.0), att_zeta=1.1):
        # bandwidth is set by the fans: ~45 ms rotor lag + 10 ms command delay leave ~25 deg of phase at 8 rad/s
        self.m, self.d, self.plant = model, data, plant
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.I = np.asarray(inertia_diag)
        wn = np.asarray(att_wn)
        self.kp, self.kd = self.I * wn ** 2, 2 * att_zeta * self.I * wn
        self.mass = float(model.body_subtreemass[self.trunk])
        self.pos_fan = np.array([f["pos"] for f in PACK["fans"]])
        self.axis = np.array([f["axis"] for f in PACK["fans"]])
        self.yaw_ref = None
        self.com_bias = np.zeros(3)      # controller's CoM belief minus the true CoM (body frame): model error for tests
        self.int_e = np.zeros(3)
        self.int_z = 0.0
        self.int_v = np.zeros(2)
        self.ki = np.array([2.0, 2.0, 0.0]) * self.kp   # slow integral on roll/pitch error: trims a CoM / thrust imbalance
        self.dt = model.opt.timestep

    def allocation(self):
        d = self.d
        R = d.xmat[self.trunk].reshape(3, 3)
        com_b = R.T @ (d.subtree_com[self.trunk] - d.xpos[self.trunk]) + self.com_bias
        B = np.zeros((4, 4))
        for k in range(4):
            B[0, k] = self.axis[k, 2]
            B[1:, k] = np.cross(self.pos_fan[k] - com_b, self.axis[k])
        return B

    def throttle(self, T):
        """Inverse static fan map at the present bus voltage (feed-forward; the rotor lag is left to the plant)."""
        p = self.plant.p
        T = np.clip(T, 0.0, None)
        w = np.sqrt(T / p.kf)
        i = p.i0 + p.kq * w * w / p.ke
        u = (p.ke * w + p.R * i) / max(float(self.plant.aux["v_bus"][0]), 6.0)
        return np.clip(np.where(T > 0, u, 0.0), 0.0, 1.0)

    def ground_assist(self, lift, gain=1.0, extra=0.0):
        """On the ground: carry ``lift`` newtons applied through the flight hardware's own CoM (so the legs see the
        bare duck's weight distribution), plus ``extra`` newtons through the all-up CoM (the take-off ramp: no pitching
        moment), and hold the trunk level in roll and pitch with the fans; yaw is left free."""
        d = self.d
        R = d.xmat[self.trunk].reshape(3, 3)
        com_b = R.T @ (d.subtree_com[self.trunk] - d.xpos[self.trunk]) + self.com_bias
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.trunk, vel, 0)
        om_b = R.T @ vel[:3]
        e_b = R.T @ np.cross(R[:, 2], [0.0, 0.0, 1.0])
        tau = np.cross(np.asarray(PACK["com"]) - com_b, [0.0, 0.0, lift]) + gain * (self.kp * e_b - self.kd * om_b) * np.array([1.0, 1.0, 0.0])
        T = np.linalg.solve(self.allocation(), np.r_[lift + extra, tau[0], tau[1], 0.0])
        return self.throttle(T)

    def update(self, z_ref, vz_ref, v_xy_ref, t_max_each):
        d, p = self.d, self.plant.p
        R = d.xmat[self.trunk].reshape(3, 3)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.trunk, vel, 0)
        v, om_b = vel[3:], R.T @ vel[:3]
        yaw = math.atan2(R[1, 0], R[0, 0])
        if self.yaw_ref is None:
            self.yaw_ref = yaw
        ez = z_ref - d.xpos[self.trunk][2]
        self.int_z = float(np.clip(self.int_z + np.clip(ez, -0.08, 0.08) * self.dt, -0.5, 0.5))   # absorbs a thrust-scale error
        az = 6.0 * ez + 4.0 * (vz_ref - v[2]) + 6.0 * self.int_z
        ev = np.asarray(v_xy_ref) - v[:2]
        self.int_v = np.clip(self.int_v + ev * self.dt, -1.5, 1.5)                # trims a steady tilt bias (CoM error)
        a_xy = np.clip(1.5 * ev + 2.0 * self.int_v, -3.0, 3.0)
        zb = np.array([a_xy[0], a_xy[1], G + max(az, -0.6 * G)]); zb /= np.linalg.norm(zb)
        xc = np.array([math.cos(self.yaw_ref), math.sin(self.yaw_ref), 0.0])
        yb = np.cross(zb, xc); yb /= np.linalg.norm(yb)
        Rd = np.c_[np.cross(yb, zb), yb, zb]
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        e_R = np.array([E[2, 1], E[0, 2], E[1, 0]])
        self.int_e = np.clip(self.int_e + e_R * self.dt, -0.1, 0.1)
        tau = -self.kp * e_R - self.kd * om_b - self.ki * self.int_e
        Fz = self.mass * (G + az) / max(R[2, 2], 0.5)
        B = self.allocation()
        T = np.linalg.solve(B, np.r_[Fz, tau])
        lo, hi = 0.05 * t_max_each, t_max_each
        if T.max() > hi or T.min() < lo:          # keep the torques, give up collective first
            Tm = np.linalg.solve(B, np.r_[0.0, tau])
            Tc = np.linalg.solve(B, np.r_[1.0, 0, 0, 0])
            c_hi = np.min((hi - Tm) / np.maximum(Tc, 1e-9)); c_lo = np.max((lo - Tm) / np.maximum(Tc, 1e-9))
            T = Tm + Tc * np.clip(Fz, c_lo, max(c_hi, c_lo))
        T = np.clip(T, lo, hi)
        return self.throttle(T), dict(T_cmd=T, e_R=e_R, Fz=Fz)


def tilt_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
