#!/usr/bin/env python3
"""MuJoCo experiments for the flying duck (classical baseline, existing leg policies).

    ~/.venvs/microduck_live/bin/python scripts/tools/flight_rollout.py

E1  carrying the flight hardware with the fans off: the deployed walk / run policies with and without the 685 g pack
E2  thrust-assisted run -> take-off -> hover -> land with the model-based controller; trajectory saved for the
    Isaac Sim replay (on the ground the fans carry the flight hardware's weight, because of E1)
E3  robustness of E2 to thrust and centre-of-mass error
Writes docs/flight_rollout_results.json, logs/flight/rollout_baseline.npz, docs/figures/flight_rollout*.pdf.
"""
import json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "app")); sys.path.insert(0, HERE)
import engine as E
import flight_sim as S
import fan_model as F

FLIGHT_XML = os.path.join(ROOT, "app", "assets", "microduck", "scene_flight_backlash.xml")
PLAIN_XML = E.SCENE_XML
res = {}
I_DIAG = json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))["layout"]["I_diag"]


def make(xml, **kw):
    E.SCENE_XML = xml
    return E.Duck(**kw)


# ---- E1 ---------------------------------------------------------------------------------------------
def carry(xml, run, seconds=6.0, lag=4, vin=7.4):
    duck = make(xml, lag=lag, vin=vin)
    for _ in range(50):
        duck.step()
    x0 = duck.data.qpos[duck.free_qadr]
    fell, tmax = False, 0.0
    for k in range(int(seconds / duck.control_dt)):
        duck.set_stick(0.0, 1.0, 0.0, run=run)
        duck.step()
        t = S.tilt_deg(duck.data.xmat[duck.trunk].reshape(3, 3))
        tmax = max(tmax, t)
        if t > 60 or duck.data.xpos[duck.trunk][2] < 0.06:
            fell = True; break
    return dict(speed=(duck.data.qpos[duck.free_qadr] - x0) / ((k + 1) * duck.control_dt), fell=fell, max_tilt=tmax, t=(k + 1) * duck.control_dt)


def run_e1():
    print("E1  carrying the pack, fans off (6 s forward command; 6 servo-lag / supply variants each)")
    variants = [(lag, vin) for lag in (3, 5, 7) for vin in (6.8, 7.8)]
    res["carry"] = {}
    for label, xml in (("no pack", PLAIN_XML), ("with pack", FLIGHT_XML)):
        for gait, run in (("walk", False), ("run", True)):
            rr = [carry(xml, run, lag=l, vin=v) for l, v in variants]
            sp = [r["speed"] for r in rr if not r["fell"]]
            res["carry"][f"{label}|{gait}"] = dict(falls=sum(r["fell"] for r in rr), n=len(rr), speed_mean=float(np.mean(sp)) if sp else None,
                                                 speed_min=float(np.min(sp)) if sp else None, speed_max=float(np.max(sp)) if sp else None,
                                                 max_tilt=float(max(r["max_tilt"] for r in rr)))
            c = res["carry"][f"{label}|{gait}"]
            print(f"  {label:9s} {gait:4s}: falls {c['falls']}/{c['n']}, speed " + (f"{c['speed_mean']:.2f} m/s [{c['speed_min']:.2f}, {c['speed_max']:.2f}]" if sp else "-") + f", max tilt {c['max_tilt']:.0f} deg")




# ---- E2 / E3 ----------------------------------------------------------------------------------------
def rollout(thrust_scale=1.0, com_dx=0.0, model_scale=1.0, z_hover=0.55, record=False, t_run=1.0, t_spool=3.0, t_takeoff=3.5, t_land=8.5, t_end=12.5,
            assist=None, run=True, ki_scale=None, verbose=False, tuck_frac=0.0):
    """thrust_scale scales the PLANT's thrust; com_dx is the controller's CoM belief error; the controller keeps the nominal fan model."""
    duck = make(FLIGHT_XML)
    m, d = duck.model, duck.data
    # com_dx: the controller's belief of the all-up CoM is off by this much (the plant is untouched, so the leg policy
    # sees the same robot: moving the real CoM by 7.5 mm makes the deployed stand policy fall before take-off)
    p_true, bat = S.prior_params(thrust_scale)
    p_ctrl, _ = S.prior_params(model_scale)
    plant = S.FanPlant(m, d, p_true, bat)
    ctrl_plant = type("P", (), dict(p=p_ctrl, aux=plant.aux))()
    ctrl = S.FlightController(m, d, plant, I_DIAG)
    ctrl.plant = plant
    ctrl.com_bias = np.array([-com_dx, 0.0, 0.0])
    if assist is None:
        assist = 1.2
    if ki_scale is not None:
        ctrl.ki = ki_scale * ctrl.kp
    orig = duck.apply_perturb
    duck.apply_perturb = lambda: (orig(), plant.substep())
    log = {k: [] for k in ("t", "pos", "quat", "vel", "u", "w", "thrust", "v_bus", "i_bus", "soc", "tilt", "mode", "xpos", "xquat", "z_ref", "qpos")}
    mode, z_ref, landed_t, t_off = "ground", None, None, None
    z_stand = None
    hold = E.DEFAULT_POSE.astype(np.float64)
    sit = m.key_qpos[duck.key_sit][duck.servos.qpos_idx]
    tuck = hold + tuck_frac * (sit - hold)        # legs part-way to the seated pose
    tuck[5:9] = hold[5:9]                         # head stays put
    ext_f = 0.0
    n = int(t_end / duck.control_dt)
    for k in range(n):
        t = k * duck.control_dt
        z = d.xpos[duck.trunk][2]
        if z_stand is None and t >= 0.8:
            z_stand = z
        if z_stand is None:
            z_ref_dummy = z
        full = F.steady_state(1.0, 4, float(plant.soc[0]), p_ctrl, bat)["thrust"]
        if mode in ("ground", "landed"):
            go = mode == "ground" and t >= t_run
            duck.set_stick(0.0, 1.0 if (go and run) else 0.0, 0.0, run=run and go)
            # thrust assist: the deployed policies fall with the pack (E1), also when its weight alone is cancelled
            # (the trunk's inertia doubles). So on the ground the fans carry `assist` x the hardware's weight through
            # its own CoM and hold the trunk level; from t_spool the lift ramps to 85 % of the all-up weight.
            lift = (assist if mode == "ground" else 1.2) * S.PACK["mass"] * S.G * min(t / 0.5, 1.0)
            extra = 0.0
            if mode == "ground" and t >= t_spool:   # pre-load: up to 80 % of the all-up weight on the fans, legs still running
                extra = max(0.80 * ctrl.mass * S.G - lift, 0.0) * min((t - t_spool) / max(t_takeoff - t_spool, 1e-3), 1.0)
            if mode == "landed" and t > landed_t + 2.0:
                lift *= max(1.0 - (t - landed_t - 2.0) / 1.0, 0.0)
            plant.command(ctrl.ground_assist(lift, extra=extra) if lift + extra > 0 else np.zeros(4))
            duck.step()
            # decisive lift-off: at t_takeoff the flight controller takes over with a height step, so the feet clear the
            # floor within a few ticks (a slow ramp leaves them dragging and trips the duck); legs ease to the hold pose
            if mode == "ground" and t >= t_takeoff:
                mode, z_ref, v_ref = "flight", z + 0.10, d.qvel[duck.free_dadr:duck.free_dadr + 2].copy()
                ctrl.yaw_ref = None; ctrl.int_e[:] = 0; ctrl.int_z = 0.0; ctrl.int_v[:] = 0
                t_off = t
                leg0 = (E.DEFAULT_POSE + duck.last_action).astype(np.float64)
        elif mode in ("flight", "descend"):
            if mode == "flight":
                z_ref = min(z_ref + 0.5 * duck.control_dt, z_hover); vz_ref = 0.5 if z_ref < z_hover else 0.0
                if t >= t_off + (t_land - t_takeoff):
                    mode = "descend"
            else:
                z_ref = max(z_ref - 0.25 * duck.control_dt, 0.10); vz_ref = -0.25
            v_ref = v_ref * math.exp(-duck.control_dt / 0.8)                 # carry the run speed, then brake to hover
            for _ in range(duck.decimation):
                u, info = ctrl.update(z_ref, vz_ref, v_ref, full)
                plant.command(u)
                # legs: tuck at lift-off so the feet leave the floor at once, stay tucked in flight, extend for landing
                bl = min((t - t_off) / 0.15, 1.0)
                ext = 1.0 if (mode == "descend" and z_ref < 0.30) else 0.0
                ext_f = ext_f + (ext - ext_f) * min(m.opt.timestep / 0.25, 1.0)
                pose_air = (1 - ext_f) * tuck + ext_f * hold
                duck.servos.command((1 - bl) * leg0 + bl * pose_air)
                duck.apply_perturb()
                duck.servos.substep()
            duck.measure()
            if mode == "descend" and z < 0.128:
                mode, landed_t = "landed", t
        R = d.xmat[duck.trunk].reshape(3, 3)
        if verbose and t > t_takeoff - 0.2 and t < t_takeoff + 1.5 and k % 5 == 0:
            print(f"   t {t:.2f} {mode:7s} z {z:.3f} zref {z_ref if z_ref else 0:.3f} tilt {S.tilt_deg(R):5.1f} v {d.qvel[duck.free_dadr:duck.free_dadr+3].round(2)} T {plant.thrust.round(1)} u {plant.u.round(2)} V {float(plant.aux['v_bus'][0]):.1f}")
        for key, val in (("t", t), ("pos", d.xpos[duck.trunk].copy()), ("quat", d.xquat[duck.trunk].copy()), ("vel", d.qvel[duck.free_dadr:duck.free_dadr + 3].copy()),
                         ("u", plant.u.copy()), ("w", plant.w[0].copy()), ("thrust", plant.thrust.copy()), ("v_bus", float(plant.aux["v_bus"][0])),
                         ("i_bus", float(plant.aux["i_bus"][0])), ("soc", float(plant.soc[0])), ("tilt", S.tilt_deg(R)),
                         ("mode", ["ground", "flight", "descend", "landed"].index(mode)), ("z_ref", z_ref if z_ref is not None else z)):
            log[key].append(val)
        if record:
            log["xpos"].append(d.xpos.copy()); log["xquat"].append(d.xquat.copy()); log["qpos"].append(d.qpos.copy())
    L = {k: np.array(v) for k, v in log.items() if len(v)}
    fl = L["mode"] == 1
    hov = fl & (L["t"] > (t_off or t_takeoff) + 2.5)
    air = (L["mode"] == 1) | (L["mode"] == 2)
    out = dict(max_tilt_flight=float(L["tilt"][air].max()), max_tilt_all=float(L["tilt"].max()), assist_share=float(assist), z_peak=float(L["pos"][:, 2].max()),
               z_err_hover=float(np.abs(L["pos"][hov, 2] - z_hover).mean()) if hov.any() else None,
               hover_speed=float(np.linalg.norm(L["vel"][hov, :2], axis=1).mean()) if hov.any() else None,
               run_speed=float(L["vel"][(L["t"] > t_run + 1.0) & (L["t"] < t_spool), 0].mean()), t_handover=t_off,
               v_at_liftoff=float(L["vel"][L["mode"] == 1][0, 0]) if fl.any() else None,
               u_hover=float(L["u"][hov].mean()) if hov.any() else None, i_hover=float(L["i_bus"][hov].mean()) if hov.any() else None,
               v_hover=float(L["v_bus"][hov].mean()) if hov.any() else None, soc_used=float(L["soc"][0] - L["soc"][-1]),
               landed=landed_t is not None, upright_end=bool(L["tilt"][(L["t"] >= landed_t) & (L["t"] < landed_t + 2.0)].max() < 25) if landed_t else False,
               t_liftoff=float(L["t"][np.argmax(L["pos"][:, 2] > L["pos"][int(t_run / duck.control_dt), 2] + 0.03)]),
               x_travel=float(L["pos"][-1, 0] - L["pos"][0, 0]))
    out["success"] = bool(out["z_peak"] > z_hover - 0.08 and out["max_tilt_flight"] < 35 and out["landed"] and out["upright_end"])
    return out, L, (m, duck)


def run_e23():
    print("\nE2  run -> take-off -> hover -> land (nominal plant)")
    out, L, (m, duck) = rollout(record=True)
    res["baseline"] = out
    print("  " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()))
    os.makedirs(os.path.join(ROOT, "logs", "flight"), exist_ok=True)
    names = [m.body(i).name for i in range(m.nbody)]
    np.savez_compressed(os.path.join(ROOT, "logs", "flight", "rollout_baseline.npz"), t=L["t"], xpos=L["xpos"], xquat=L["xquat"], body_names=np.array(names),
                        fan_w=L["w"], fan_u=L["u"], thrust=L["thrust"], mode=L["mode"], v_bus=L["v_bus"], i_bus=L["i_bus"])

    fig, axs = plt.subplots(2, 2, figsize=(8.8, 5.8), sharex=True)
    t = L["t"]
    for a in axs.flat:
        for mm, c in ((1, "#e8f0f7"), (2, "#f7efe2")):
            sel = L["mode"] == mm
            if sel.any():
                a.axvspan(t[sel][0], t[sel][-1], color=c, lw=0)
        a.grid(alpha=0.3)
    axs[0, 0].plot(t, L["pos"][:, 2], color="#2a6f97", label="trunk height"); axs[0, 0].plot(t, L["z_ref"], "k:", lw=0.8, label="reference")
    axs[0, 0].plot(t, L["vel"][:, 0], color="#d1603d", label="forward speed [m/s]"); axs[0, 0].set_ylabel("m, m/s"); axs[0, 0].legend(fontsize=7)
    axs[0, 1].plot(t, L["tilt"], color="#2a6f97"); axs[0, 1].set_ylabel("trunk tilt [deg]")
    axs[1, 0].plot(t, L["thrust"] / S.G * 1e3); axs[1, 0].set_ylabel("thrust per fan [g]"); axs[1, 0].set_xlabel("time [s]")
    axs[1, 1].plot(t, L["v_bus"], color="#2a6f97", label="bus voltage [V]"); axs[1, 1].plot(t, L["i_bus"] / 5, color="#d1603d", label="bus current / 5 [A]")
    axs[1, 1].legend(fontsize=7); axs[1, 1].set_xlabel("time [s]")
    fig.suptitle("MuJoCo baseline: run, take-off (blue), descent (tan), landing; model-based controller, deployed run policy", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_rollout.pdf"))

    print("\nE3  plant error against the nominal controller model (take-off from a stand: isolates the flight controller)")
    res["robust"] = {}
    grid_s, grid_c = (0.80, 0.90, 1.00, 1.10, 1.20), (-0.015, -0.0075, 0.0, 0.0075, 0.015)
    succ = np.zeros((len(grid_s), len(grid_c))); tilt = np.zeros_like(succ)
    for i, s in enumerate(grid_s):
        for j, c in enumerate(grid_c):
            o, _, _ = rollout(thrust_scale=s, com_dx=c, run=False, assist=0.0)
            succ[i, j], tilt[i, j] = o["success"], o["max_tilt_flight"]
            res["robust"][f"thrust{s:.2f}_com{c*1e3:+.1f}mm"] = o
        print(f"  thrust x{s:.2f}: " + "  ".join(f"com {c*1e3:+5.1f} mm {'ok ' if succ[i,j] else 'FAIL'} tilt {tilt[i,j]:4.1f}" for j, c in enumerate(grid_c)))
    res["robust_success_rate"] = float(succ.mean())
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    im = ax.imshow(tilt, origin="lower", cmap="viridis", aspect="auto")
    for i in range(len(grid_s)):
        for j in range(len(grid_c)):
            ax.text(j, i, f"{tilt[i,j]:.0f}" + ("" if succ[i, j] else "\nfail"), ha="center", va="center", color="w", fontsize=7)
    ax.set_xticks(range(len(grid_c))); ax.set_xticklabels([f"{c*1e3:+.1f}" for c in grid_c]); ax.set_xlabel("centre-of-mass error, x [mm]")
    ax.set_yticks(range(len(grid_s))); ax.set_yticklabels([f"{s:.2f}" for s in grid_s]); ax.set_ylabel("true thrust / modelled thrust")
    plt.colorbar(im, label="max tilt in flight [deg]"); ax.set_title(f"Baseline controller, {int(succ.sum())}/{succ.size} runs succeed", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_rollout_robust.pdf"))
    json.dump(res, open(os.path.join(ROOT, "docs", "flight_rollout_results.json"), "w"), indent=1)
    print("wrote docs/flight_rollout_results.json, logs/flight/rollout_baseline.npz, docs/figures/flight_rollout*.pdf")


if __name__ == "__main__":
    run_e1()
    run_e23()
