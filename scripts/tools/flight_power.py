#!/usr/bin/env python3
"""Flight-pack optimisation with the coupled fan + battery model (voltage sag, I^2 R heating, energy).

    python scripts/tools/flight_power.py        # after flight_design.py (needs the no-pack mass)

For each 4S capacity: all-up mass, full-throttle thrust on the sagging pack, hover current, cell temperature rise
and state of charge used over the design mission. The smallest capacity meeting every constraint is the optimum.
Writes docs/flight_power_results.json, docs/duck_flight_power_numbers.tex, docs/figures/flight_pack_opt.pdf.
"""
import json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "microduck_rl", "flight"))
import flight_config as C
import fan_model as F

res_d = json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))
M_NOPACK = res_d["tethered"]["mass"]
FAN = F.calibrate(C.FAN_BENCH_THRUST_G * 1e-3 * C.G, C.FAN_BENCH_V, C.FAN_BENCH_I,
                  F.FanParams(kv_rpm_per_v=C.FAN_KV, i_max=C.FAN_I_LIMIT, eta_inst=1.0, area=math.pi / 4 * (0.050 ** 2 - 0.026 ** 2)))
FAN = F.replace(FAN, eta_inst=C.FAN_INSTALL_ETA)


def pack(q_ah):
    return F.BatteryParams(q_ah=q_ah, r_cell=C.CELL_MOHM_AH * 1e-3 / q_ah)


def evaluate(q_ah, mission_s=C.MISSION_S):
    m_pack = q_ah * 14.8 / C.BATT_WH_PER_KG
    m = M_NOPACK + m_pack
    b = pack(q_ah)
    soc_mid = 0.5 * (C.MISSION_SOC[0] + C.MISSION_SOC[1])
    full = F.steady_state(1.0, 4, soc_mid, FAN, b)
    tw = 4 * full["thrust"] / (m * C.G)
    u = F.throttle_for_thrust(m * C.G / 4, 4, soc_mid, FAN, b)
    out = dict(q_ah=q_ah, pack_g=m_pack * 1e3, all_up_g=m * 1e3, tw_sag=tw, v_full=full["v_bus"], i_full=full["i_bus"])
    if u is None:
        return dict(out, ok=False, why="cannot hover")
    hov = F.steady_state(u, 4, soc_mid, FAN, b)
    heat = hov["i_bus"] ** 2 * (b.n_s * b.r_cell)
    dT = heat * mission_s / (max(m_pack - C.PACK_OVERHEAD_G * 1e-3, 1e-3) * C.CELL_CP)
    soc_used = hov["i_bus"] * mission_s / 3600 / q_ah
    t_energy = (C.MISSION_SOC[0] - C.MISSION_SOC[1]) * q_ah * 3600 / hov["i_bus"]
    lim = dict(thrust=tw >= C.TW_MIN_SAG, thermal=dT <= C.MISSION_DT_MAX_K, energy=soc_used <= C.MISSION_SOC[0] - C.MISSION_SOC[1])
    return dict(out, u_hover=u, i_hover=hov["i_bus"], p_hover=hov["p_elec"], v_hover=hov["v_bus"], dT=dT, soc_used=soc_used,
                c_hover=hov["i_bus"] / q_ah, hover_s=t_energy, ok=all(lim.values()), limits=lim)


print(f"fan prior: kf {FAN.kf:.3e} N s^2, kq {FAN.kq:.3e}, R {FAN.R} ohm, fan efficiency {F.fan_efficiency(FAN):.2f}; no-pack mass {M_NOPACK*1e3:.0f} g")
qs = np.linspace(0.45, 2.6, 87)
ev = [evaluate(q) for q in qs]
res = dict(fan=FAN.__dict__, m_nopack=M_NOPACK, optimum={}, commercial={})
for t in (15, 30, 45, 60):
    okq = [q for q in qs if evaluate(q, t)["ok"]]
    res["optimum"][str(t)] = dict(q_ah=float(min(okq)) if okq else None,
                                  pack_g=float(min(okq) * 14.8 / C.BATT_WH_PER_KG * 1e3) if okq else None)
    print(f"mission {t:2d} s: smallest feasible pack {min(okq)*1e3:.0f} mAh ({min(okq)*14.8/C.BATT_WH_PER_KG*1e3:.0f} g)" if okq else f"mission {t} s: none")
print(f"\n{'mAh':>5s} {'pack g':>6s} {'all-up':>6s} {'T/W sag':>7s} {'V full':>6s} {'hover A':>7s} {'C':>4s} {'dT K':>5s} {'SoC used':>8s} {'hover s':>7s}  limits")
for mah in C.COMMERCIAL_4S_MAH:
    e = evaluate(mah / 1e3)
    res["commercial"][str(mah)] = e
    if "u_hover" not in e:
        print(f"{mah:5d} {e['pack_g']:6.0f} {e['all_up_g']:6.0f} {e['tw_sag']:7.2f}  cannot hover"); continue
    bad = ",".join(k for k, v in e["limits"].items() if not v) or "ok"
    print(f"{mah:5d} {e['pack_g']:6.0f} {e['all_up_g']:6.0f} {e['tw_sag']:7.2f} {e['v_full']:6.2f} {e['i_hover']:7.1f} {e['c_hover']:4.0f} {e['dT']:5.1f} {e['soc_used']:8.2f} {e['hover_s']:7.0f}  {bad}")
choice = next(m for m in C.COMMERCIAL_4S_MAH if res["commercial"][str(m)].get("ok"))
res["choice_mah"] = choice
print(f"\nsmallest commercial pack meeting the {C.MISSION_S:.0f} s mission: 4S {choice} mAh")

# ---- KV variants of the same 50 mm fan on the chosen pack: listed voltage is not flight voltage ----
print(f"\nKV variants on 4S {choice} mAh at mid charge (derated {C.CLAIM_DERATE:.0%} / listing at face value):")
vrows = []
for name, (kv, tg, v, i) in C.FAN_VARIANTS.items():
    row = [name, f"{tg} g at {v} V, {i} A"]
    for der in (C.CLAIM_DERATE, 1.0):
        pv = F.calibrate(der * tg * 1e-3 * C.G, v, i, F.FanParams(kv_rpm_per_v=kv, i_max=C.FAN_I_LIMIT, area=FAN.area))
        pv = F.replace(pv, eta_inst=C.FAN_INSTALL_ETA)
        b = pack(choice / 1e3)
        m = M_NOPACK + choice / 1e3 * 14.8 / C.BATT_WH_PER_KG
        full = F.steady_state(1.0, 4, 0.575, pv, b)
        u = F.throttle_for_thrust(m * C.G / 4, 4, 0.575, pv, b)
        hov = F.steady_state(u, 4, 0.575, pv, b)["i_bus"] if u else float("nan")
        row += [f"{4*full['thrust']/(m*C.G):.2f}", f"{full['v_bus']:.1f}", f"{full['i_phase']:.0f}", f"{hov:.0f}" if u else "--"]
        res.setdefault("variants", {})[f"{name}|{der}"] = dict(tw=4 * full["thrust"] / (m * C.G), v_full=full["v_bus"], i_phase=full["i_phase"], i_hover=hov)
    vrows.append(row)
    print("  " + "  ".join(f"{c:>22s}" if k < 2 else f"{c:>5s}" for k, c in enumerate(row)))
open(os.path.join(ROOT, "docs", "duck_flight_variant_table.tex"), "w").write(
    "% generated by scripts/tools/flight_power.py\n" + "\n".join(" & ".join(r) + " \\\\" for r in vrows) + "\n")

fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
g = [e for e in ev if "u_hover" in e]
x = [e["q_ah"] * 1e3 for e in g]
for ax, key, lim, lab in zip(axs, ("tw_sag", "dT", "soc_used"), (C.TW_MIN_SAG, C.MISSION_DT_MAX_K, C.MISSION_SOC[0] - C.MISSION_SOC[1]),
                             ("full-throttle thrust / weight", f"cell temperature rise, {C.MISSION_S:.0f} s [K]", f"charge used, {C.MISSION_S:.0f} s")):
    ax.plot(x, [e[key] for e in g], color="#2a6f97")
    ax.axhline(lim, color="#d1603d", ls="--", lw=1)
    ax.axvline(choice, color="k", ls=":", lw=0.8)
    ax.set_xlabel("4S capacity [mAh]"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
fig.suptitle("Pack sizing with sag and heating; dashed = limit, dotted = chosen pack", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_pack_opt.pdf"))
json.dump(res, open(os.path.join(ROOT, "docs", "flight_power_results.json"), "w"), indent=1, default=float)
c = res["commercial"][str(choice)]
N = dict(PackMah=choice, PackG=f"{c['pack_g']:.0f}", PackTW=f"{c['tw_sag']:.2f}", PackVfull=f"{c['v_full']:.1f}", PackHoverA=f"{c['i_hover']:.0f}",
         PackHoverW=f"{c['p_hover']:.0f}", PackC=f"{c['c_hover']:.0f}", PackDT=f"{c['dT']:.0f}", PackSoc=f"{c['soc_used']*100:.0f}", PackHoverS=f"{c['hover_s']:.0f}",
         PackAllUp=f"{c['all_up_g']:.0f}", FanKf=f"{FAN.kf*1e7:.2f}", FanKq=f"{FAN.kq*1e9:.2f}", FanEta=f"{F.fan_efficiency(FAN):.2f}",
         PackUhover=f"{c['u_hover']*100:.0f}")
open(os.path.join(ROOT, "docs", "duck_flight_power_numbers.tex"), "w").write(
    "% generated by scripts/tools/flight_power.py\n" + "".join(f"\\newcommand{{\\fp{k}}}{{{v}}}\n" for k, v in N.items()))
