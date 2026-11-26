#!/usr/bin/env python3
"""System identification of the ducted-fan power train (microduck_rl/flight/fan_model.py).

    python scripts/tools/fan_sysid.py                       # validate the pipeline on synthetic bench data
    python scripts/tools/fan_sysid.py --stand stand.csv --step step.csv --pack pack.csv --out docs/fan_ident.json

Three bench experiments, one fan, pack on the bench (no flight needed):

  A  static sweep on a thrust stand: throttle staircase, steady samples of (u, V_bus, I_bus, rpm, thrust)
       u V = Ke w + R i            -> Ke, R        (i = (I_bus - I_idle) / u, linear least squares)
       Ke i = kq w^2 + Ke I0 + b w -> kq, I0       (b fixed at 0 unless --fit-b)
       T = kf w^2                  -> kf (kv_exit)
     then a joint weighted Gauss-Newton refinement on (V, I, T) residuals and a bootstrap over samples.
  B  throttle steps logged at >= 500 Hz with bidirectional-DShot eRPM: simulate the model, fit rotor inertia J
     and the command delay t_d to the rpm trace (spin-up and spin-down both constrain J).
  C  pack pulse test (two current levels, rests between): fit R0, R1, tau1 of the Thevenin model to V_bus.

CSV columns:  stand: u,v_bus,i_bus,rpm,thrust_g   step: t,u,v_bus,rpm   pack: t,i_bus,v_bus  (SI otherwise).
The bootstrap intervals are what the simulator's domain randomisation should draw from (written to the output JSON);
with no real logs yet, the synthetic run only shows that the procedure recovers known parameters under sensor noise.
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "microduck_rl", "flight"))
import flight_config as C
import fan_model as F

RPM = 60.0 / (2 * math.pi)
I_IDLE = 0.3


def prior():
    p = F.calibrate(C.FAN_BENCH_THRUST_G * 1e-3 * C.G, C.FAN_BENCH_V, C.FAN_BENCH_I,
                    F.FanParams(kv_rpm_per_v=C.FAN_KV, i_max=C.FAN_I_LIMIT, area=math.pi / 4 * (0.050 ** 2 - 0.026 ** 2)))
    return p, F.BatteryParams(q_ah=C.BATT_REF_AH, r_cell=C.CELL_MOHM_AH * 1e-3 / C.BATT_REF_AH, i_avionics=I_IDLE)


# ------------------------------------------------------------------ synthetic bench -------------
def simulate(p, bat, t, u_cmd, delay=0.0, n_fans=1, soc0=0.95):
    dt = t[1] - t[0]
    w = np.zeros((1, n_fans)); soc = np.array([soc0]); v1 = np.zeros(1)
    nd = int(round(delay / dt))
    out = np.zeros((len(t), 4))
    for k in range(len(t)):
        u = u_cmd[max(k - nd, 0)]
        w, soc, v1, aux = F.step(w, soc, v1, np.full((1, n_fans), u), dt, p, bat, np, substeps=1)
        out[k] = (w[0, 0], aux["v_bus"][0], aux["i_bus"][0], F.thrust(w, 0.0, p, np)[0, 0])
    return out


def synth(seed=0):
    """Ground truth deliberately off the priors; sensor noise of a hobby stand (HX711 load cell, AM32 telemetry)."""
    rng = np.random.default_rng(seed)
    p0, b0 = prior()
    true_p = F.replace(p0, R=p0.R * 1.30, kq=p0.kq * 0.88, kv_exit=p0.kv_exit * 1.06, J=p0.J * 1.45, i0=3.4,
                       kv_rpm_per_v=p0.kv_rpm_per_v * 0.97)
    true_b = F.replace(b0, r_cell=b0.r_cell * 1.35, r1=0.006, tau1=5.0)
    true_delay = 0.012
    noise = dict(thrust=0.05, rpm_rel=0.003, i=0.4, v=0.02)   # N, -, A, V
    # A: staircase 10 %..100 %, 3 s per level, last second averaged; pack sags as it would on the bench
    levels = np.linspace(0.15, 1.0, 12)
    dt = 0.002
    t = np.arange(0, 3.0 * len(levels), dt)
    u = levels[np.minimum((t / 3.0).astype(int), len(levels) - 1)]
    y = simulate(true_p, true_b, t, u)
    rows = []
    for k, lv in enumerate(levels):
        m = (t > 3.0 * k + 2.0) & (t < 3.0 * k + 3.0)
        for _ in range(8):   # eight 0.125 s averages per level
            rows.append((lv, y[m, 1].mean() + rng.normal(0, noise["v"]), y[m, 2].mean() + rng.normal(0, noise["i"] / 3),
                         y[m, 0].mean() * RPM * (1 + rng.normal(0, noise["rpm_rel"])), (y[m, 3].mean() + rng.normal(0, noise["thrust"])) / C.G * 1e3))
    stand = np.array(rows)
    # B: steps 0.3 -> 0.7 -> 0.4 -> 0.9 -> 0.3 at 1 kHz
    dt = 0.001
    t = np.arange(0, 3.0, dt)
    u = np.select([t < 0.4, t < 1.0, t < 1.6, t < 2.2], [0.3, 0.7, 0.4, 0.9], 0.3)
    y = simulate(true_p, true_b, t, u, delay=true_delay)
    step = np.c_[t, u, y[:, 1] + rng.normal(0, noise["v"], len(t)), y[:, 0] * RPM * (1 + rng.normal(0, noise["rpm_rel"], len(t)))]
    # C: pack pulses with four fans as the load (bus current is what matters)
    t = np.arange(0, 24.0, 0.01)
    u = np.select([t < 2, t < 7, t < 11, t < 16, t < 20], [0.0, 0.55, 0.0, 0.85, 0.0], 0.0)
    y = simulate(true_p, true_b, t, u, n_fans=4)
    pack = np.c_[t, y[:, 2] + rng.normal(0, noise["i"], len(t)), y[:, 1] + rng.normal(0, noise["v"], len(t))]
    truth = dict(ke=true_p.ke, R=true_p.R, kq=true_p.kq, i0=true_p.i0, kf=true_p.kf, J=true_p.J, delay=true_delay,
                 r0=true_b.r0, r1=true_b.r1, tau1=true_b.tau1)
    return stand, step, pack, truth


# ------------------------------------------------------------------ stage A ---------------------
def fit_static(stand, p0, n_boot=400, seed=1):
    u, v, ib, rpm, tg = stand.T
    w = rpm / RPM
    i = (ib - I_IDLE) / u
    T = tg * 1e-3 * C.G

    def solve(idx):
        ke, R = np.linalg.lstsq(np.c_[w[idx], i[idx]], (u * v)[idx], rcond=None)[0]
        kq, c0 = np.linalg.lstsq(np.c_[w[idx] ** 2, np.ones(len(idx))], ke * i[idx], rcond=None)[0]
        kf = float(np.sum(T[idx] * w[idx] ** 2) / np.sum(w[idx] ** 4))
        x0 = np.array([ke, R, kq, c0 / ke, kf])

        def resid(x):  # output error: predict (rpm, I_bus, thrust) from the well-known inputs (u, V_bus)
            ke_, R_, kq_, i0_, kf_ = x
            bq = ke_ * ke_ / R_
            cq = ke_ * (u[idx] * v[idx] / R_ - i0_)
            w_ = (-bq + np.sqrt(np.maximum(bq * bq + 4 * kq_ * cq, 0.0))) / (2 * kq_)
            i_ = (u[idx] * v[idx] - ke_ * w_) / R_
            lim = i_ > p0.i_max                      # ESC current limit (a configured value, not fitted)
            w_ = np.where(lim, np.sqrt(np.maximum(ke_ * (p0.i_max - i0_), 0.0) / kq_), w_)
            i_ = np.where(lim, p0.i_max, i_)
            return np.r_[(w_ - w[idx]) / (0.003 * w[idx]), (u[idx] * i_ + I_IDLE - ib[idx]) / 0.13, (kf_ * w_ ** 2 - T[idx]) / 0.05]
        return least_squares(resid, x0, x_scale=np.abs(x0) + 1e-12).x

    idx = np.arange(len(u))
    x = solve(idx)
    rng = np.random.default_rng(seed)
    boot = np.array([solve(rng.choice(idx, len(idx))) for _ in range(n_boot)])
    names = ("ke", "R", "kq", "i0", "kf")
    est = {n: float(x[k]) for k, n in enumerate(names)}
    ci = {n: [float(np.percentile(boot[:, k], 2.5)), float(np.percentile(boot[:, k], 97.5))] for k, n in enumerate(names)}
    kv_exit = math.sqrt(est["kf"] / (F.RHO * p0.area))
    p = F.replace(p0, kv_rpm_per_v=60 / (2 * math.pi * est["ke"]), R=est["R"], kq=est["kq"], i0=est["i0"], kv_exit=kv_exit, eta_inst=1.0)
    pred = dict(w=w, T=T, T_fit=est["kf"] * w ** 2, i=i, i_fit=np.minimum((u * v - est["ke"] * w) / est["R"], p0.i_max))
    rms = dict(thrust_g=float(np.sqrt(np.mean((pred["T_fit"] - T) ** 2)) / C.G * 1e3), i_a=float(np.sqrt(np.mean((pred["i_fit"] - i) ** 2))))
    return p, est, ci, pred, rms


# ------------------------------------------------------------------ stage B ---------------------
def fit_dynamic(step, p, bat):
    t, u, v, rpm = step.T
    dt = t[1] - t[0]
    w_meas = rpm / RPM

    def sim(J, delay):
        pj = F.replace(p, J=J)
        nd = delay / dt
        ud = np.interp(t - delay, t, u, left=u[0])
        w = np.zeros(len(t)); wk = np.array([[w_meas[:20].mean()]])
        for k in range(len(t)):
            i = F.phase_current(np.array([[ud[k]]]), wk, np.array([v[k]]), pj, np)   # measured bus voltage drives the model
            f = (pj.ke * i - pj.kq * wk * wk - pj.ke * pj.i0 - pj.b * wk) / pj.J
            dfdw = -(pj.ke ** 2 / pj.R + 2 * pj.kq * wk + pj.b) / pj.J
            wk = np.clip(wk + dt * f / (1 - dt * dfdw), 0, None)
            w[k] = wk[0, 0]
        return w

    def resid(x):
        return (sim(x[0] * 1e-6, x[1] * 1e-3) - w_meas) / (0.003 * w_meas.mean())

    sol = least_squares(resid, [p.J * 1e6, 5.0], bounds=([0.5, 0.0], [60.0, 60.0]), diff_step=[1e-3, 0.2])
    J, delay = sol.x[0] * 1e-6, sol.x[1] * 1e-3
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * np.mean(sol.fun ** 2)   # residuals autocorrelated: a lower bound
    sd = np.sqrt(np.diag(cov))
    w_fit = sim(J, delay)
    return dict(J=float(J), delay=float(delay), J_sd=float(sd[0] * 1e-6), delay_sd=float(sd[1] * 1e-3),
                rms_rpm=float(np.sqrt(np.mean((w_fit - w_meas) ** 2)) * RPM)), (t, w_meas, w_fit, u)


# ------------------------------------------------------------------ stage C ---------------------
def fit_pack(pack, bat):
    t, i, v = pack.T
    dt = t[1] - t[0]
    q0 = 0.95

    def sim(r0, r1, tau1):
        soc = q0 - np.cumsum(i) * dt / (3600 * bat.q_ah)
        v1 = np.zeros(len(t))
        for k in range(1, len(t)):
            v1[k] = v1[k - 1] + dt * (-v1[k - 1] / tau1 + i[k] * r1 / tau1)
        return bat.n_s * F.ocv_cell(soc, np) - r0 * i - v1

    sol = least_squares(lambda x: (sim(x[0] * 1e-3, x[1] * 1e-3, x[2]) - v) / 0.02, [bat.r0 * 1e3, 4.0, 8.0],
                        bounds=([1, 0, 0.5], [120, 60, 60]))
    r0, r1, tau1 = sol.x[0] * 1e-3, sol.x[1] * 1e-3, sol.x[2]
    cov = np.linalg.inv(sol.jac.T @ sol.jac) * np.mean(sol.fun ** 2)
    sd = np.sqrt(np.diag(cov))
    v_fit = sim(r0, r1, tau1)
    return dict(r0=float(r0), r1=float(r1), tau1=float(tau1), r0_sd=float(sd[0] * 1e-3), r1_sd=float(sd[1] * 1e-3), tau1_sd=float(sd[2]),
                rms_v=float(np.sqrt(np.mean((v_fit - v) ** 2)))), (t, v, v_fit, i)


# ------------------------------------------------------------------ main ------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--stand"); ap.add_argument("--step"); ap.add_argument("--pack")
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "fan_sysid_results.json"))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
p0, b0 = prior()
real = bool(args.stand)
if real:
    stand = np.loadtxt(args.stand, delimiter=",", skiprows=1)
    step = np.loadtxt(args.step, delimiter=",", skiprows=1) if args.step else None
    pack = np.loadtxt(args.pack, delimiter=",", skiprows=1) if args.pack else None
    truth = None
else:
    stand, step, pack, truth = synth(args.seed)
    print("SYNTHETIC bench data (no hardware yet): truth is the prior with R x1.30, kq x0.88, kv_exit x1.06, J x1.45, Kv x0.97, pack R x1.35")

p_fit, est, ci, pred, rmsA = fit_static(stand, p0)
res = dict(synthetic=not real, prior=dict(ke=p0.ke, R=p0.R, kq=p0.kq, i0=p0.i0, kf=p0.kf / p0.eta_inst, J=p0.J, r0=b0.r0), static=dict(est=est, ci95=ci, rms=rmsA))
rows = []


def report(name, unit, scale, value, lo, hi, pri):
    tv = truth[name] * scale if truth else float("nan")
    err = (value * scale - tv) / tv * 100 if truth else float("nan")
    perr = (pri * scale - tv) / tv * 100 if truth and pri is not None else float("nan")
    rows.append((name, unit, pri * scale if pri is not None else float("nan"), tv, value * scale, lo * scale, hi * scale, perr, err))
    print(f"  {name:6s} [{unit:>10s}] prior {pri*scale if pri is not None else float('nan'):9.4g}  truth {tv:9.4g}  fit {value*scale:9.4g}  "
          f"95% [{lo*scale:.4g}, {hi*scale:.4g}]  prior err {perr:+6.1f} %  fit err {err:+5.2f} %")


print(f"\nA  static sweep: {len(stand)} samples, fit rms {rmsA['thrust_g']:.1f} g thrust, {rmsA['i_a']:.2f} A")
for n, unit, sc in (("ke", "mV s/rad", 1e3), ("R", "mOhm", 1e3), ("kq", "nN m s^2", 1e9), ("i0", "A", 1.0), ("kf", "1e-7 N s^2", 1e7)):
    report(n, unit, sc, est[n], ci[n][0], ci[n][1], res["prior"][n])
if step is not None:
    dyn, tr = fit_dynamic(step, p_fit, b0)
    res["dynamic"] = dyn
    print(f"B  steps: fit rms {dyn['rms_rpm']:.0f} rpm")
    report("J", "g cm^2", 1e7, dyn["J"], dyn["J"] - 2 * dyn["J_sd"], dyn["J"] + 2 * dyn["J_sd"], p0.J)
    report("delay", "ms", 1e3, dyn["delay"], dyn["delay"] - 2 * dyn["delay_sd"], dyn["delay"] + 2 * dyn["delay_sd"], 0.005)
if pack is not None:
    pk, tp = fit_pack(pack, b0)
    res["pack"] = pk
    print(f"C  pack pulses: fit rms {pk['rms_v']*1e3:.0f} mV")
    report("r0", "mOhm", 1e3, pk["r0"], pk["r0"] - 2 * pk["r0_sd"], pk["r0"] + 2 * pk["r0_sd"], b0.r0)
    report("r1", "mOhm", 1e3, pk["r1"], pk["r1"] - 2 * pk["r1_sd"], pk["r1"] + 2 * pk["r1_sd"], b0.r1)
    report("tau1", "s", 1.0, pk["tau1"], pk["tau1"] - 2 * pk["tau1_sd"], pk["tau1"] + 2 * pk["tau1_sd"], b0.tau1)
res["table"] = [dict(zip(("name", "unit", "prior", "truth", "fit", "lo", "hi", "prior_err_pct", "fit_err_pct"), r)) for r in rows]

# what the mistake costs in flight terms: hover throttle and full-throttle thrust, prior vs identified vs truth
if truth:
    tp_ = F.replace(p0, kv_rpm_per_v=60 / (2 * math.pi * truth["ke"]), R=truth["R"], kq=truth["kq"], i0=truth["i0"],
                    kv_exit=math.sqrt(truth["kf"] / (F.RHO * p0.area)), eta_inst=1.0)
    tb_ = F.replace(b0, r_cell=(truth["r0"] - b0.r_lead) / b0.n_s)
    fb_ = F.replace(b0, r_cell=(res["pack"]["r0"] - b0.r_lead) / b0.n_s)
    m = json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))["primary"]["mass"]
    gap = {}
    for lab, (pp, bb) in dict(prior=(F.replace(p0, eta_inst=1.0), b0), identified=(p_fit, fb_), truth=(tp_, tb_)).items():
        full = F.steady_state(1.0, 4, 0.575, pp, bb)
        uh = F.throttle_for_thrust(m * C.G / 4, 4, 0.575, pp, bb)
        gap[lab] = dict(t_full_g=full["thrust"] / C.G * 1e3, u_hover=uh)
    res["gap"] = gap
    # thrust delivered if the controller commands the model's hover throttle on the true plant
    for lab in ("prior", "identified"):
        got = F.steady_state(gap[lab]["u_hover"], 4, 0.575, tp_, tb_)["thrust"] * 4 / (m * C.G)
        gap[lab]["hover_thrust_ratio_on_truth"] = got
    print(f"\nflight consequence (4 fans, {m*1e3:.0f} g): full-throttle thrust per fan prior {gap['prior']['t_full_g']:.0f} g, identified "
          f"{gap['identified']['t_full_g']:.0f} g, truth {gap['truth']['t_full_g']:.0f} g; commanding the model's hover throttle on the true plant gives "
          f"{gap['prior']['hover_thrust_ratio_on_truth']*100:.1f} % of weight with the prior, {gap['identified']['hover_thrust_ratio_on_truth']*100:.1f} % after identification")
json.dump(res, open(args.out, "w"), indent=1)

fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.0))
o = np.argsort(pred["w"])
axs[0, 0].plot(pred["w"] * RPM / 1e3, pred["T"] / C.G * 1e3, ".", ms=3, color="#555", label="stand samples")
axs[0, 0].plot(pred["w"][o] * RPM / 1e3, pred["T_fit"][o] / C.G * 1e3, color="#d1603d", label="$k_f\\omega^2$ fit")
axs[0, 0].plot(pred["w"][o] * RPM / 1e3, p0.kf / p0.eta_inst * pred["w"][o] ** 2 / C.G * 1e3, "--", color="#2a6f97", label="prior")
axs[0, 0].set_xlabel("rotor speed [krpm]"); axs[0, 0].set_ylabel("thrust [g]"); axs[0, 0].legend(fontsize=7); axs[0, 0].set_title("A: static thrust", fontsize=9)
axs[0, 1].plot(pred["i"], pred["i_fit"], ".", ms=3, color="#d1603d"); axs[0, 1].plot([0, pred["i"].max()], [0, pred["i"].max()], "k:", lw=0.8)
axs[0, 1].set_xlabel("measured phase current [A]"); axs[0, 1].set_ylabel("model [A]"); axs[0, 1].set_title("A: electrical equation", fontsize=9)
if step is not None:
    t, wm, wf, u = tr
    axs[1, 0].plot(t, wm * RPM / 1e3, color="#999", lw=0.8, label="eRPM"); axs[1, 0].plot(t, wf * RPM / 1e3, color="#d1603d", lw=1, label="model")
    axs[1, 0].set_xlabel("time [s]"); axs[1, 0].set_ylabel("rotor speed [krpm]"); axs[1, 0].legend(fontsize=7)
    axs[1, 0].set_title(f"B: steps, J = {dyn['J']*1e7:.1f} g cm$^2$, delay {dyn['delay']*1e3:.1f} ms", fontsize=9)
if pack is not None:
    t, v, vf, i = tp
    axs[1, 1].plot(t, v, color="#999", lw=0.8, label="measured"); axs[1, 1].plot(t, vf, color="#d1603d", lw=1, label="Thevenin fit")
    axs[1, 1].set_xlabel("time [s]"); axs[1, 1].set_ylabel("pack voltage [V]"); axs[1, 1].legend(fontsize=7)
    axs[1, 1].set_title(f"C: pack, R0 = {pk['r0']*1e3:.1f} mOhm", fontsize=9)
for a in axs.flat:
    a.grid(alpha=0.3)
fig.suptitle("Fan identification" + (" on SYNTHETIC bench data" if not real else ""), fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "figures", "fan_sysid.pdf"))
if not real:
    open(os.path.join(ROOT, "docs", "duck_flight_sysid_table.tex"), "w").write(
        "% generated by scripts/tools/fan_sysid.py (synthetic validation)\n" + "\n".join(
            f"{r[0]} & {r[1]} & {r[2]:.3g} & {r[3]:.3g} & {r[4]:.3g} & [{r[5]:.3g}, {r[6]:.3g}] & {r[7]:+.0f} & {r[8]:+.1f} \\\\" for r in rows).replace("_", "\\_").replace("^", "\\^{}") + "\n")
print("wrote", os.path.relpath(args.out, ROOT), "docs/figures/fan_sysid.pdf")
