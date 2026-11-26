#!/usr/bin/env python3
"""Flying-duck design study: how many ducted fans, which fan, where they go, and whether a printed
frame carries them. Regenerates every figure and number of docs/duck_flight.tex.

    python scripts/tools/flight_design.py            # from the repo root

Deterministic (no random numbers). Writes docs/flight_results.json, docs/duck_flight_*.tex
fragments and docs/figures/flight_*.pdf. Reads docs/flight_cad_results.json (printed-part masses
from cad/flight/build_parts.py) when it exists.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "microduck_rl", "flight"))
import flight_config as C
import fan_model as F


def prior_fan():
    """Grey-box fan calibrated on the seller bench point (derated), with the ESC current limit."""
    f = C.FANS[C.PRIMARY]
    p = F.calibrate(C.FAN_BENCH_THRUST_G * 1e-3 * C.G, C.FAN_BENCH_V, C.FAN_BENCH_I,
                    F.FanParams(kv_rpm_per_v=C.FAN_KV, i_max=C.FAN_I_LIMIT, area=annulus(f)))
    return F.replace(p, eta_inst=C.FAN_INSTALL_ETA)


SOC_MID = 0.5 * (C.MISSION_SOC[0] + C.MISSION_SOC[1])

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})


# ------------------------------------------------------------------ duck model ----------------
def duck_model():
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, C.MODEL_XML))
    d = mujoco.MjData(m)
    out = {}
    for k in range(m.nkey):
        mujoco.mj_resetDataKeyframe(m, d, k)
        mujoco.mj_forward(m, d)
        com = d.subtree_com[1].copy()
        I = np.zeros((3, 3))
        boxes = {}
        for b in range(1, m.nbody):
            Rb = d.ximat[b].reshape(3, 3)
            r = d.xipos[b] - com
            I += Rb @ np.diag(m.body_inertia[b]) @ Rb.T + m.body_mass[b] * ((r @ r) * np.eye(3) - np.outer(r, r))
            mn, mx = np.full(3, 1e9), np.full(3, -1e9)
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                mn, mx = np.minimum(mn, w.min(0)), np.maximum(mx, w.max(0))
            boxes[m.body(b).name] = (mn, mx)
        out[m.key(k).name] = dict(com=com, I=I, boxes=boxes, trunk=d.xpos[1].copy())
    return float(m.body_subtreemass[1]), out


# ------------------------------------------------------------------ fan physics ---------------
def annulus(f):
    return math.pi / 4 * ((f["d_mm"] * 1e-3) ** 2 - (f["hub_mm"] * 1e-3) ** 2)


def ideal_power(T, A):
    """Momentum theory, ducted fan with exit area = fan annulus (no diffuser): P = T^1.5 / sqrt(4 rho A)."""
    return T ** 1.5 / math.sqrt(4 * C.RHO_AIR * A)


def fan_point(name):
    """Usable max thrust [N] and electrical power [W] at that thrust, plus figure of merit."""
    f = C.FANS[name]
    T_list = f["thrust_g"] * 1e-3 * C.G
    fom_list = ideal_power(T_list, annulus(f)) / f["power_w"] if f["power_w"] else None
    T = T_list * (C.CLAIM_DERATE if f["source"] == "claimed" else 1.0) * C.INSTALL_DERATE
    P = f["power_w"] if f["power_w"] else ideal_power(T_list, annulus(f)) / C.FOM_DEFAULT
    return T, P, fom_list


def struct_g(n):
    p = os.path.join(ROOT, "docs", "flight_cad_results.json")
    if n == 4 and os.path.exists(p):
        return json.load(open(p))["frame_mass_g"]["PLA"], "CAD"   # PLA, 15 % infill
    return C.STRUCT_G_FALLBACK[n], "estimate"


def budget(name, n, duck_kg, batt_ah=C.BATT_REF_AH, derate_scale=1.0):
    f = C.FANS[name]
    T_max, P_max, fom = fan_point(name)
    T_max *= derate_scale
    v_nom = 3.7 * f["cells"]
    batt_wh = batt_ah * 14.8  # packs compared at equal energy: capacity quoted as 4S-equivalent Ah
    items = {
        "duck": duck_kg * 1e3,
        "fans": n * f["mass_g"],
        "ESC": n * f["esc_a"] * C.ESC_G_PER_A,
        "flight controller": C.FC_G,
        "battery": batt_wh / C.BATT_WH_PER_KG * 1e3,
        "wiring": C.WIRING_G,
        "printed frame": struct_g(n)[0],
    }
    if n == 2:
        items["tilt servos"] = C.TILT_SERVO_G
    m = sum(items.values()) * 1e-3
    T_h = m * C.G / n
    if name == C.PRIMARY:
        # sag-aware: the coupled fan + pack model at mid charge (a tether is modelled as a stiff 2.6 Ah source)
        q = batt_ah if batt_ah > 0 else 2.6
        bat = F.BatteryParams(q_ah=q, r_cell=C.CELL_MOHM_AH * 1e-3 / q)
        fan = prior_fan()
        fan = F.replace(fan, eta_inst=fan.eta_inst * derate_scale)
        full = F.steady_state(1.0, n, SOC_MID, fan, bat)
        T_max = full["thrust"]
        u = F.throttle_for_thrust(T_h, n, SOC_MID, fan, bat)
        if u is None:
            return dict(items=items, mass=m, tw=n * T_max / (m * C.G), T_max=T_max, T_hover=T_h, hover_frac=1.0, P_hover=float("nan"),
                        I_hover=float("nan"), I_max=full["i_bus"], endurance_s=0.0, fom=fom, batt_wh=batt_wh, P_hover_listed=float("nan"),
                        endurance_listed_s=0.0, u_hover=None, v_full=full["v_bus"])
        hov = F.steady_state(u, n, SOC_MID, fan, bat)
        t_h = (C.MISSION_SOC[0] - C.MISSION_SOC[1]) * batt_ah * 3600 / hov["i_bus"]
        nod = F.replace(fan, eta_inst=fan.eta_inst / C.CLAIM_DERATE)  # optimistic bracket: listing thrust taken at face value
        u2 = F.throttle_for_thrust(T_h, n, SOC_MID, nod, bat)
        h2 = F.steady_state(u2, n, SOC_MID, nod, bat)
        return dict(items=items, mass=m, tw=n * T_max / (m * C.G), T_max=T_max, T_hover=T_h, hover_frac=T_h / T_max,
                    P_hover=hov["p_elec"], I_hover=hov["i_bus"], I_max=full["i_bus"], endurance_s=t_h, fom=fom, batt_wh=batt_wh,
                    P_hover_listed=h2["p_elec"], endurance_listed_s=(C.MISSION_SOC[0] - C.MISSION_SOC[1]) * batt_ah * 3600 / h2["i_bus"],
                    u_hover=u, v_full=full["v_bus"], v_hover=hov["v_bus"])
    P_h = n * P_max * (T_h / T_max) ** 1.5  # P ~ T^1.5 along the fan's own curve
    T_list = f["thrust_g"] * 1e-3 * C.G
    P_opt = n * P_max * (T_h / T_list) ** 1.5  # optimistic bracket: the listing's thrust at the listing's power
    return dict(items=items, mass=m, tw=n * T_max / (m * C.G), T_max=T_max, T_hover=T_h,
                P_hover_listed=P_opt, endurance_listed_s=batt_wh * C.BATT_USABLE * 3600 / P_opt,
                hover_frac=T_h / T_max, P_hover=P_h, I_hover=P_h / v_nom,
                I_max=n * P_max / v_nom, endurance_s=batt_wh * C.BATT_USABLE * 3600 / P_h,
                fom=fom, batt_wh=batt_wh)


# ------------------------------------------------------------------ all-up rigid body ---------
def all_up(duck_kg, duck, bud, x_c):
    """CoM and inertia with the flight hardware added as point masses (fans as 4 points)."""
    f = C.FANS[C.PRIMARY]
    pts = [(duck_kg, duck["com"])]
    frame_and_wire = (bud["items"]["printed frame"] + bud["items"]["wiring"]) * 1e-3
    for sx in (1, -1):
        for sy in (1, -1):
            p = np.array([x_c + sx * C.FAN_LX, sy * C.FAN_LY, C.FAN_Z])
            pts.append((f["mass_g"] * 1e-3 + 0.6 * frame_and_wire / 4, p))
    pts.append((0.4 * frame_and_wire, np.array([x_c, 0, C.FAN_Z])))
    pts.append((bud["items"]["battery"] * 1e-3, np.array(C.BATT_POS)))
    pts.append(((bud["items"]["ESC"] + bud["items"]["flight controller"]) * 1e-3, np.array(C.FC_POS)))
    M = sum(p[0] for p in pts)
    com = sum(p[0] * p[1] for p in pts) / M
    I = duck["I"] * (duck_kg / DUCK_MODEL_KG)
    r = duck["com"] - com
    I = I + duck_kg * ((r @ r) * np.eye(3) - np.outer(r, r))
    for mi, p in pts[1:]:
        r = p - com
        I = I + mi * ((r @ r) * np.eye(3) - np.outer(r, r))
    return M, com, I


def solve_centroid(duck_kg, duck, bud):
    x_c = 0.0
    for _ in range(20):  # thrust centroid on the CoM (fixed point; converges in 2-3 iterations)
        M, com, I = all_up(duck_kg, duck, bud, x_c)
        x_c = com[0]
    return x_c, M, com, I


# ------------------------------------------------------------------ printed-arm statics -------
def arm_checks(T_max, tip_mass):
    b, h, t = C.ARM_W_MM * 1e-3, C.FRAME_H_MM * 1e-3, C.ARM_WALL_MM * 1e-3
    Ixx = (b * h ** 3 - (b - 2 * t) * (h - 2 * t) ** 3) / 12  # bending about the horizontal axis (thrust)
    half_w = C.TRUNK_WID_MM * 0.5e-3 + (C.COLLAR_FIT_MM + C.COLLAR_WALL_MM) * 1e-3
    L = math.hypot(C.FAN_LY - half_w, max(C.FAN_LX - C.TRUNK_LEN_MM * 0.5e-3, 0.0)) - (C.DUCT_OD_MM * 0.5e-3)
    rows = {}
    for mat, p in C.MATERIALS.items():
        Mb = T_max * (L + C.DUCT_OD_MM * 0.5e-3)  # thrust acts at the fan axis
        sigma = Mb * (h / 2) / Ixx
        k = 3 * p["E"] * Ixx / (L + C.DUCT_OD_MM * 0.5e-3) ** 3
        rows[mat] = dict(L_mm=L * 1e3, I_mm4=Ixx * 1e12, sigma_mpa=sigma / 1e6,
                         sf_yield=p["yield_pa"] / sigma, defl_mm=T_max / k * 1e3,
                         f_hz=math.sqrt(k / tip_mass) / (2 * math.pi))
    return rows


# ------------------------------------------------------------------ main ----------------------
DUCK_MODEL_KG, KEYS = duck_model()
DUCK_KG = DUCK_MODEL_KG + C.SERVO_EXTRA_KG
duck = KEYS["STAND"] if "STAND" in KEYS else KEYS["INIT"]
res = {"duck_model_kg": DUCK_MODEL_KG, "duck_kg": DUCK_KG, "struct_source": struct_g(4)[1]}

# ---- 1. trade study: fan x count ---------------------------------------------------------------
print(f"\nDuck: model {DUCK_MODEL_KG*1e3:.0f} g + servos {C.SERVO_EXTRA_KG*1e3:.0f} g = {DUCK_KG*1e3:.0f} g (not yet weighed)")
print(f"Battery {C.BATT_REF_AH} Ah 4S-equivalent ({C.BATT_REF_AH*14.8:.1f} Wh); frame mass from {struct_g(4)[1]}")
hdr = f"{'fan':30s} {'n':>1s} {'src':>8s} {'FoM':>5s} {'all-up g':>8s} {'T/W':>5s} {'hover %':>7s} {'P_hov W':>7s} {'I_hov A':>7s} {'t_hov s':>7s}"
print(hdr + "\n" + "-" * len(hdr))
trade = {}
for name in C.FANS:
    for n in (2, 4):
        b = budget(name, n, DUCK_KG)
        trade[f"{name}|{n}"] = b
        fom = f"{b['fom']:.2f}" if b["fom"] else "  -  "
        hov = f"{b['hover_frac']*100:7.0f} {b['P_hover']:7.0f} {b['I_hover']:7.1f} {b['endurance_s']:7.0f}" if b["tw"] > 1.02 else "   cannot hover"
        print(f"{name:30s} {n:1d} {C.FANS[name]['source']:>8s} {fom:>5s} {b['mass']*1e3:8.0f} {b['tw']:5.2f} {hov}")
res["trade"] = {k: {kk: vv for kk, vv in v.items() if kk != "items"} for k, v in trade.items()}

fig, ax = plt.subplots(figsize=(6.4, 3.6))
names = list(C.FANS)
x = np.arange(len(names))
for i, n in enumerate((2, 4)):
    tw = [trade[f"{nm}|{n}"]["tw"] for nm in names]
    bars = ax.bar(x + (i - 0.5) * 0.38, tw, 0.38, label=f"{n} fans", color=["#9aa7b5", "#d1603d"][i])
    for xi, v in zip(x, tw):
        ax.text(xi + (i - 0.5) * 0.38, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
ax.axhline(1.0, color="k", lw=0.8)
ax.axhline(1.5, color="k", lw=0.8, ls="--")
ax.text(len(names) - 0.45, 1.52, "T/W = 1.5 (control margin)", ha="right", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=6.5)
ax.set_ylabel("usable thrust / all-up weight")
ax.set_title(f"Derated static thrust-to-weight, duck {DUCK_KG*1e3:.0f} g, {C.BATT_REF_AH*14.8:.0f} Wh pack")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "flight_tw_trade.pdf"))
plt.close(fig)

# ---- 2. chosen configuration --------------------------------------------------------------------
bud = budget(C.PRIMARY, 4, DUCK_KG)
res["primary"] = dict(name=C.PRIMARY, items=bud["items"], **{k: v for k, v in bud.items() if k != "items"})
print(f"\nChosen: 4 x {C.PRIMARY}")
for k, v in bud["items"].items():
    print(f"  {k:20s} {v:7.1f} g")
print(f"  {'ALL-UP':20s} {bud['mass']*1e3:7.1f} g   usable thrust {4*bud['T_max']/C.G*1e3:.0f} g   T/W {bud['tw']:.2f}")
print(f"  hover: {bud['hover_frac']*100:.0f} % of usable thrust, {bud['P_hover']:.0f} W, {bud['I_hover']:.1f} A, "
      f"{bud['endurance_s']:.0f} s on {bud['batt_wh']:.1f} Wh; full throttle {bud['I_max']:.0f} A "
      f"({bud['I_max']/C.BATT_REF_AH:.0f} C on {C.BATT_REF_AH} Ah)")
print(f"  if the listing's thrust and power were exact: hover {bud['P_hover_listed']:.0f} W, {bud['endurance_listed_s']:.0f} s")
tb = budget(C.PRIMARY, 4, DUCK_KG, batt_ah=0.0)
res["tethered"] = dict(mass=tb["mass"], tw=tb["tw"], P_hover=tb["P_hover"], I_hover=tb["I_hover"])
print(f"  tethered (no pack on board): all-up {tb['mass']*1e3:.0f} g, T/W {tb['tw']:.2f}, hover {tb['P_hover']:.0f} W = {tb['I_hover']:.0f} A at 14.8 V")

# sensitivity: duck mass x thrust derate
dm = np.linspace(*C.DUCK_MASS_RANGE_KG, 26)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for s, c in zip((1.25, 1.0, 0.85), ("#2a6f97", "#d1603d", "#7a7a7a")):
    ax.plot(dm * 1e3, [budget(C.PRIMARY, 4, m_, derate_scale=s)["tw"] for m_ in dm], color=c,
            label=f"{s*C.CLAIM_DERATE*100:.0f} % of listed thrust")
ax.axvline(DUCK_KG * 1e3, color="k", lw=0.6, ls=":")
ax.axhline(1.5, color="k", lw=0.8, ls="--")
ax.set_xlabel("bare duck mass [g]")
ax.set_ylabel("thrust / weight, 4 fans")
ax.set_title(C.PRIMARY)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "flight_tw_sensitivity.pdf"))
plt.close(fig)
res["sensitivity"] = {f"duck{m_:.2f}_scale{s:.2f}": budget(C.PRIMARY, 4, m_, derate_scale=s)["tw"]
                      for m_ in C.DUCK_MASS_RANGE_KG for s in (1.25, 1.0, 0.85)}

# battery sweep
ah = np.linspace(0.65, 3.0, 48)
bb = [budget(C.PRIMARY, 4, DUCK_KG, a) for a in ah]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(ah * 1e3, [b["endurance_s"] for b in bb], color="#2a6f97")
ax.set_xlabel("4S pack capacity [mAh]")
ax.set_ylabel("hover time to 20 % [s]", color="#2a6f97")
ax2 = ax.twinx()
ax2.plot(ah * 1e3, [b["tw"] for b in bb], color="#d1603d")
ax2.set_ylabel("thrust / weight", color="#d1603d")
ax2.grid(False)
for a in (0.85, 1.05, 1.3, 1.8, 2.2):
    b = budget(C.PRIMARY, 4, DUCK_KG, a)
    ax.plot(a * 1e3, b["endurance_s"], "o", color="#2a6f97", ms=4)
    res.setdefault("battery_sweep", {})[f"{int(a*1e3)}mAh"] = dict(
        mass_g=b["items"]["battery"], all_up_g=b["mass"] * 1e3, tw=b["tw"], hover_s=b["endurance_s"],
        I_hover=b["I_hover"], c_hover=b["I_hover"] / a, c_max=b["I_max"] / a)
ax.set_title("Pack size: hover time against thrust margin")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "flight_battery.pdf"))
plt.close(fig)
print("\nBattery sweep (4S):")
for k, v in res["battery_sweep"].items():
    print(f"  {k:8s} pack {v['mass_g']:4.0f} g  all-up {v['all_up_g']:5.0f} g  T/W {v['tw']:.2f}  hover {v['hover_s']:4.0f} s"
          f"  {v['I_hover']:.0f} A = {v['c_hover']:.0f} C hover, {v['c_max']:.0f} C full throttle")

# ---- 3. placement, CoM, control authority -------------------------------------------------------
x_c, M, com, I = solve_centroid(DUCK_KG, duck, bud)
res["layout"] = dict(x_c=x_c, com=com.tolist(), I_diag=np.diag(I).tolist(), mass=M,
                     fans=[[x_c + sx * C.FAN_LX, sy * C.FAN_LY, C.FAN_Z] for sx in (1, -1) for sy in (1, -1)])
print(f"\nLayout: thrust centroid x = {x_c*1e3:+.1f} mm, all-up CoM = {np.round(com*1e3,1)} mm, "
      f"fan plane z = {C.FAN_Z*1e3:.0f} mm ({(C.FAN_Z-com[2])*1e3:+.1f} mm from CoM)")
print(f"  inertia about CoM  Ixx {I[0,0]*1e3:.2f}  Iyy {I[1,1]*1e3:.2f}  Izz {I[2,2]*1e3:.2f}  g.m^2  (bare duck "
      f"{duck['I'][0,0]*1e3:.2f} / {duck['I'][1,1]*1e3:.2f} / {duck['I'][2,2]*1e3:.2f})")
T_h, T_m = bud["T_hover"], bud["T_max"]
dT = min(T_m - T_h, T_h)  # per-fan differential available about hover
cant = math.radians(C.FAN_CANT_DEG)
r_arm = math.hypot(C.FAN_LX, C.FAN_LY)
tau = dict(roll=4 * dT * C.FAN_LY, pitch=4 * dT * C.FAN_LX, yaw=4 * dT * math.sin(cant) * r_arm)
acc = dict(roll=tau["roll"] / I[0, 0], pitch=tau["pitch"] / I[1, 1], yaw=tau["yaw"] / I[2, 2])
res["authority"] = dict(dT=dT, tau=tau, acc=acc, cant_thrust_loss=1 - math.cos(cant))
for a in ("roll", "pitch", "yaw"):
    print(f"  {a:5s} torque {tau[a]:.3f} N.m -> {acc[a]:5.0f} rad/s^2")
# posture CoM travel relative to the trunk: the trim moment the fans must hold
shift = {}
for k, v in KEYS.items():
    dx = ((v["com"] - v["trunk"]) - (duck["com"] - duck["trunk"]))[0]
    mom = DUCK_KG * C.G * dx
    shift[k] = dict(dx_mm=dx * 1e3, moment=mom, frac_pitch=abs(mom) / tau["pitch"])
    print(f"  posture {k:6s} CoM dx {dx*1e3:+5.1f} mm -> trim moment {mom:+.3f} N.m ({abs(mom)/tau['pitch']*100:.0f} % of pitch authority)")
res["posture_shift"] = shift
# jet / leg clearance
jet_in = C.FAN_LY - C.FANS[C.PRIMARY]["d_mm"] * 0.5e-3
clr = {}
for k, v in KEYS.items():
    ymax = max(v["boxes"][b][1][1] for b in ("upper_leg_left", "leg", "ankle_left"))
    clr[k] = (jet_in - ymax) * 1e3
    print(f"  posture {k:6s} leg outer edge y = {ymax*1e3:.0f} mm, jet inner edge {jet_in*1e3:.0f} mm -> clearance {clr[k]:+.0f} mm")
res["jet_clearance_mm"] = clr
def leg_y_at_roll(q):
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, C.MODEL_XML)); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[m.jnt_qposadr[m.joint("left_hip_roll").id]] = q
    mujoco.mj_forward(m, d)
    y = -1e9
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name in ("upper_leg_left", "leg", "ankle_left") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            y = max(y, (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 1].max())
    return y
qs = np.radians(np.arange(-22, 22.1, 1.0))
ys = np.array([leg_y_at_roll(q) for q in qs])
worst = int(np.argmax(ys))
ok = qs[ys < jet_in]
res["hip_roll"] = dict(worst_deg=math.degrees(qs[worst]), worst_y_mm=ys[worst] * 1e3,
                       clear_range_deg=[math.degrees(ok.min()), math.degrees(ok.max())] if len(ok) else None)
print(f"  left hip roll sweep +-22 deg: leg reaches y = {ys[worst]*1e3:.0f} mm at {math.degrees(qs[worst]):+.0f} deg; "
      f"clear of the jet for roll in {res['hip_roll']['clear_range_deg']} deg")
head = duck["boxes"]["jaw_soft"]
res["head_clearance_mm"] = (C.FAN_LY - C.DUCT_OD_MM * 0.5e-3 - head[1][1]) * 1e3

# layout figure
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.9))
col = {"trunk_base": "#c9a227", "jaw_soft": "#e9c46a", "neck": "#e9c46a"}
for ax, (i, j, lab) in zip(axs, ((1, 0, ("y [mm]", "x forward [mm]", "Top view")), (0, 2, ("x forward [mm]", "z [mm]", "Side view")))):
    for b, (mn, mx) in duck["boxes"].items():
        ax.add_patch(plt.Rectangle((mn[i] * 1e3, mn[j] * 1e3), (mx[i] - mn[i]) * 1e3, (mx[j] - mn[j]) * 1e3,
                                   fc=col.get(b, "#b8b8b8"), ec="k", lw=0.4, alpha=0.55))
    for p in res["layout"]["fans"]:
        if j == 0:
            ax.add_patch(plt.Circle((p[1] * 1e3, p[0] * 1e3), C.DUCT_OD_MM / 2, fc="#333", ec="k", alpha=0.8))
            ax.add_patch(plt.Circle((p[1] * 1e3, p[0] * 1e3), 13, fc="#888", ec="k", lw=0.3))
            ax.plot([np.sign(p[1]) * 34, p[1] * 1e3], [np.clip(p[0] * 1e3, -40, 30), p[0] * 1e3], color="#d1603d", lw=3, zorder=0)
        else:
            ax.add_patch(plt.Rectangle((p[0] * 1e3 - C.DUCT_OD_MM / 2, p[2] * 1e3 - C.DUCT_LEN_MM / 2), C.DUCT_OD_MM, C.DUCT_LEN_MM, fc="#333", ec="k", alpha=0.35))
    bp, bs = np.array(C.BATT_POS) * 1e3, np.array(C.BATT_SIZE) * 1e3
    ax.add_patch(plt.Rectangle((bp[i] - bs[i] / 2, bp[j] - bs[j] / 2), bs[i], bs[j], fc="#2a6f97", ec="k", lw=0.5, alpha=0.8))
    fp = np.array(C.FC_POS) * 1e3
    ax.add_patch(plt.Rectangle((fp[i] - 18, fp[j] - (18 if j == 0 else 7)), 36, 36 if j == 0 else 14, fc="#3a7d44", ec="k", lw=0.5, alpha=0.8))
    ax.plot(com[i] * 1e3, com[j] * 1e3, "r+", ms=12, mew=2, label="all-up CoM")
    ax.plot(duck["com"][i] * 1e3, duck["com"][j] * 1e3, "kx", ms=6, label="bare-duck CoM")
    ax.set_aspect("equal")
    ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1]); ax.set_title(lab[2])
axs[0].set_xlim(-150, 150); axs[0].set_ylim(-110, 110)
axs[1].set_xlim(-110, 110); axs[1].set_ylim(0, 290); axs[1].axhline(0, color="k", lw=1)
axs[1].legend(fontsize=7, loc="upper left")
fig.suptitle("Fans (black), flight pack (blue), FC/ESC stack (green) on the standing duck; boxes are per-body mesh bounds", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "flight_layout.pdf"))
plt.close(fig)

# ---- 4. printed arm ------------------------------------------------------------------------------
tip = C.FANS[C.PRIMARY]["mass_g"] * 1e-3 + 0.008
T_list_max = C.FANS[C.PRIMARY]["thrust_g"] * 1e-3 * C.G  # size the arm for the listed thrust, no derating
arms = arm_checks(T_list_max, tip)
res["arm"] = arms
print(f"\nArm (box {C.ARM_W_MM:.0f} x {C.FRAME_H_MM:.0f} mm, wall {C.ARM_WALL_MM} mm), load = listed max thrust {T_list_max:.1f} N:")
for mat, a in arms.items():
    print(f"  {mat}: free length {a['L_mm']:.0f} mm, I {a['I_mm4']:.0f} mm^4, root stress {a['sigma_mpa']:.2f} MPa "
          f"(SF {a['sf_yield']:.0f} on yield), tip deflection {a['defl_mm']:.2f} mm, first bending mode {a['f_hz']:.0f} Hz")

# ---- 5. open-rotor comparison (momentum theory only) --------------------------------------------
A_edf = annulus(C.FANS[C.PRIMARY])
A_3in = math.pi / 4 * 0.0762 ** 2
ratio = math.sqrt(4 * A_edf) / math.sqrt(2 * A_3in)  # P_open/P_ducted at equal thrust and equal FoM
res["open_rotor"] = dict(A_edf_cm2=A_edf * 1e4, A_3in_cm2=A_3in * 1e4, ideal_power_ratio_3in_vs_edf=ratio)
print(f"\nMomentum theory, equal thrust and figure of merit: a bare 3-inch propeller needs {ratio*100:.0f} % of the 50 mm fan's power "
      f"(disc {A_3in*1e4:.1f} vs annulus {A_edf*1e4:.1f} cm^2)")

# ---- outputs --------------------------------------------------------------------------------------
json.dump(res, open(os.path.join(ROOT, "docs", "flight_results.json"), "w"), indent=1, default=float)


def tex(name, body):
    open(os.path.join(ROOT, "docs", name), "w").write("% generated by scripts/tools/flight_design.py - do not edit\n" + body)


rows = []
for name in C.FANS:
    f = C.FANS[name]
    b2, b4 = trade[f"{name}|2"], trade[f"{name}|4"]
    fom = f"{b4['fom']:.2f}" if b4["fom"] else "--"
    hov = f"{b4['P_hover']:.0f} & {b4['endurance_s']:.0f}" if b4["tw"] > 1.02 else "-- & --"
    rows.append(f"{name} & {f['mass_g']} & {f['thrust_g']} ({f['source'][0]}) & {fom} & {b2['mass']*1e3:.0f} & {b2['tw']:.2f} & "
                f"{b4['mass']*1e3:.0f} & {b4['tw']:.2f} & {hov} \\\\")
tex("duck_flight_trade_table.tex", "\n".join(rows) + "\n")
tex("duck_flight_budget_table.tex", "\n".join(f"{k} & {v:.0f} \\\\" for k, v in bud["items"].items())
    + f"\n\\midrule\nall-up & {bud['mass']*1e3:.0f} \\\\\n")
tex("duck_flight_battery_table.tex", "\n".join(
    f"{k.replace('mAh','')} & {v['mass_g']:.0f} & {v['all_up_g']:.0f} & {v['tw']:.2f} & {v['hover_s']:.0f} & {v['I_hover']:.0f} & {v['c_hover']:.0f} & {v['c_max']:.0f} \\\\"
    for k, v in res["battery_sweep"].items()) + "\n")
N = {
    "DuckModelG": f"{DUCK_MODEL_KG*1e3:.0f}", "DuckG": f"{DUCK_KG*1e3:.0f}", "AllUpG": f"{bud['mass']*1e3:.0f}",
    "ThrustG": f"{4*bud['T_max']/C.G*1e3:.0f}", "TW": f"{bud['tw']:.2f}", "HoverPct": f"{bud['hover_frac']*100:.0f}",
    "HoverW": f"{bud['P_hover']:.0f}", "HoverA": f"{bud['I_hover']:.0f}", "HoverS": f"{bud['endurance_s']:.0f}",
    "MaxA": f"{bud['I_max']:.0f}", "Xc": f"{x_c*1e3:+.1f}", "ComX": f"{com[0]*1e3:+.1f}", "ComZ": f"{com[2]*1e3:.0f}",
    "Ixx": f"{I[0,0]*1e3:.1f}", "Iyy": f"{I[1,1]*1e3:.1f}", "Izz": f"{I[2,2]*1e3:.1f}",
    "AccRoll": f"{acc['roll']:.0f}", "AccPitch": f"{acc['pitch']:.0f}", "AccYaw": f"{acc['yaw']:.0f}",
    "TauRoll": f"{tau['roll']:.2f}", "TauPitch": f"{tau['pitch']:.2f}", "TauYaw": f"{tau['yaw']:.3f}",
    "ClrStand": f"{clr.get('STAND', clr['INIT']):.0f}", "HeadClr": f"{res['head_clearance_mm']:.0f}",
    "SitPct": f"{shift['SIT']['frac_pitch']*100:.0f}", "FoldPct": f"{shift['FOLD']['frac_pitch']*100:.0f}",
    "SitDx": f"{shift['SIT']['dx_mm']:+.1f}", "FoldDx": f"{shift['FOLD']['dx_mm']:+.1f}",
    "ArmL": f"{arms['ABS']['L_mm']:.0f}", "ArmI": f"{arms['ABS']['I_mm4']:.0f}",
    "ArmSigABS": f"{arms['ABS']['sigma_mpa']:.1f}", "ArmSFABS": f"{arms['ABS']['sf_yield']:.0f}", "ArmSFPLA": f"{arms['PLA']['sf_yield']:.0f}",
    "ArmDeflABS": f"{arms['ABS']['defl_mm']:.2f}", "ArmDeflPLA": f"{arms['PLA']['defl_mm']:.2f}",
    "ArmHzABS": f"{arms['ABS']['f_hz']:.0f}", "ArmHzPLA": f"{arms['PLA']['f_hz']:.0f}",
    "OpenRatioPct": f"{ratio*100:.0f}", "TwLow": f"{min(res['sensitivity'].values()):.2f}", "TwHigh": f"{max(res['sensitivity'].values()):.2f}",
    "TwoFanTW": f"{max(trade[k]['tw'] for k in trade if k.endswith('|2') and '50mm' in k):.2f}", "TwoFanBigTW": f"{trade['64mm 11b 3500KV (4S)|2']['tw']:.2f}",
    "HoverWOpt": f"{bud['P_hover_listed']:.0f}", "HoverSOpt": f"{bud['endurance_listed_s']:.0f}",
    "TetherG": f"{tb['mass']*1e3:.0f}", "TetherTW": f"{tb['tw']:.2f}", "TetherA": f"{tb['I_hover']:.0f}", "TetherW": f"{tb['P_hover']:.0f}",
    "RollWorstDeg": f"{res['hip_roll']['worst_deg']:+.0f}", "RollWorstY": f"{res['hip_roll']['worst_y_mm']:.0f}",
    "RollClearLo": f"{res['hip_roll']['clear_range_deg'][0]:+.0f}", "RollClearHi": f"{res['hip_roll']['clear_range_deg'][1]:+.0f}",
    "BattAh": f"{C.BATT_REF_AH}", "BattG": f"{bud['items']['battery']:.0f}", "MaxC": f"{bud['I_max']/C.BATT_REF_AH:.0f}", "HoverC": f"{bud['I_hover']/C.BATT_REF_AH:.0f}",
    "FrameG": f"{bud['items']['printed frame']:.0f}", "FrameSrc": struct_g(4)[1],
}
tex("duck_flight_numbers.tex", "".join(f"\\newcommand{{\\fl{k}}}{{{v}}}\n" for k, v in N.items()))
print("\nwrote docs/flight_results.json, docs/duck_flight_*.tex, docs/figures/flight_*.pdf")
