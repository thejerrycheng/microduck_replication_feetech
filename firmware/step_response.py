"""Step response of one HD1910 against its position-loop gains. Reproduces docs/figures/servo_step_response.pdf.

    ~/.venvs/microduck_live/bin/python firmware/step_response.py            # the servo on the bus, +-10 deg steps

Gains are written to RAM only (registers 50 / 51) and the servo's own are restored at the end, torque off.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import feetech
from feetech import KP_RAM, KD_RAM

ap = argparse.ArgumentParser()
ap.add_argument("--id", type=int)
ap.add_argument("--step", type=int, default=114, help="ticks (114 = 10 deg)")
ap.add_argument("--P", type=int, nargs="+", default=[5, 10, 20, 32, 50, 80])
ap.add_argument("--D", type=int, nargs="+", default=[0, 20, 40])
ap.add_argument("--hold", type=float, default=0.5, help="seconds recorded after each step")
a = ap.parse_args()
DEG = 360 / 4096
ROOT = Path(__file__).resolve().parent.parent

bus = feetech.Bus(feetech.ports()[0])
sid = a.id or bus.scan()[0]
p0, d0 = bus.gains(sid)
home = bus.feedback(sid)["pos"]
runs = []


def record(target):
    t0, T, X = time.perf_counter(), [], []
    bus.goto(sid, target)
    while (t := time.perf_counter() - t0) < a.hold:
        fb = bus.feedback(sid)
        if fb is None:
            continue
        if abs(fb["pos"] - home) > 4 * a.step or fb["temp"] > 60:
            raise RuntimeError(f"abort: pos {fb['pos']}, {fb['temp']} C")
        T.append(t); X.append(fb["pos"])
    return np.array(T), np.array(X)


try:
    bus.goto(sid, home)
    bus.torque(sid, True)
    for d in a.D:
        for p in a.P:
            bus.write(sid, KP_RAM, p); bus.write(sid, KD_RAM, d)
            assert bus.gains(sid) == (p, d)
            record(home)                                   # settle at home under these gains
            x0 = float(np.median(record(home)[1][-10:]))
            T, X = record(home + a.step)
            y = (X - x0) / a.step                           # 0 -> 1
            fin = float(np.median(y[-10:]))
            t10 = next((t for t, v in zip(T, y) if v >= 0.1), None)
            t90 = next((t for t, v in zip(T, y) if v >= 0.9), None)
            band = [t for t, v in zip(T, y) if abs(v - 1) > 0.05]
            runs.append(dict(P=p, D=d, t=T.round(4).tolist(), deg=((X - x0) * DEG).round(3).tolist(),
                             rise_ms=None if t90 is None or t10 is None else round((t90 - t10) * 1e3, 1),
                             t90_ms=None if t90 is None else round(t90 * 1e3, 1), overshoot_pct=round((y.max() - 1) * 100, 1),
                             settle_ms=round((band[-1] if band else 0) * 1e3, 1), final_err_deg=round((1 - fin) * a.step * DEG, 2),
                             hz=round(len(T) / a.hold)))
            r = runs[-1]
            print(f"P {p:>3} D {d:>3}  t90 {r['t90_ms']} ms  rise {r['rise_ms']} ms  overshoot {r['overshoot_pct']:>5} %  "
                  f"settle(5%) {r['settle_ms']} ms  final err {r['final_err_deg']} deg  {r['hz']} Hz")
            record(home)
finally:
    bus.write(sid, KP_RAM, p0); bus.write(sid, KD_RAM, d0)
    bus.goto(sid, home); time.sleep(0.3)
    bus.torque(sid, False)
    print("restored gains", bus.gains(sid), "torque off")
    bus.close()

out = dict(id=sid, step_deg=round(a.step * DEG, 2), restored=[p0, d0], runs=runs)
(ROOT / "docs/servo_step_response.json").write_text(json.dumps(out))
fig, axes = plt.subplots(1, len(a.D), figsize=(4.2 * len(a.D), 3.4), sharey=True)
for ax, d in zip(np.atleast_1d(axes), a.D):
    for r in (r for r in runs if r["D"] == d):
        ax.plot(np.array(r["t"]) * 1e3, r["deg"], lw=1.4, label=f"P {r['P']}")
    ax.axhline(out["step_deg"], color="0.6", lw=0.8, ls="--")
    ax.set_title(f"D = {d}"); ax.set_xlabel("time after step (ms)"); ax.grid(alpha=0.25)
np.atleast_1d(axes)[0].set_ylabel("position (deg)"); np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
fig.suptitle(f"HD1910 ID {sid}: {out['step_deg']} deg step, unloaded", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/servo_step_response.pdf")
print("wrote docs/figures/servo_step_response.pdf, docs/servo_step_response.json")
