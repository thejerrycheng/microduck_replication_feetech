"""Pack the policies Duck Live drives with (exports/*.onnx, the simulator's DEFAULTS) into the one small file the
duck runs, and check that the duck's arithmetic gives the ONNX answers.

    ~/.venvs/microduck_live/bin/python firmware/policy_pack.py [--out policies.npz]

The pack holds, per role (walk, run, sprint, sitstand, jump, roulade, recovery), the normaliser and the four layers
as 16-bit floats, compressed. On the duck they are run by policy_net.py with numpy: no ONNX runtime to install.

The check drives the simulator (app/engine.py, ONNX policies) through walk, turn, run, sprint, stop, sit, stand,
jump and flip, and records every observation with the ONNX action. It then reports, per role, how far the pack's
actions are from those (worst and RMS, rad; an action is a joint-angle offset), and drives the same scenario again
with the pack's networks in the loop: same falls, and how far the duck ends from where ONNX took it.
Writes docs/policy_pack_results.json.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
import policy_net

SCENARIO = [("stand", 1.0), ("up", 3.0), ("up+left", 2.0), ("stand", 1.0), ("run", 3.0), ("stand", 1.5), ("lvl5", 0), ("run", 2.5),
            ("stand", 2.0), ("s", 0), ("stand", 2.0), ("s", 0), ("stand", 2.0), ("down", 2.0), ("right", 2.0), ("stand", 1.0),
            ("space", 0), ("stand", 3.0), ("r", 0), ("stand", 4.0), ("b", 0), ("stand", 2.0)]


def pack(dtype=np.float16):
    """{role/name: array} from the ONNX exports the simulator uses by default."""
    import onnx
    from onnx import numpy_helper
    import engine
    out = {}
    for role, name in engine.DEFAULTS.items():
        init = {i.name: numpy_helper.to_array(i) for i in onnx.load(os.path.join(engine.EXPORTS, name)).graph.initializer}
        out[f"{role}/mean"], out[f"{role}/scale"] = init["obs_normalizer._mean"][0].astype(np.float32), init["add"][0].astype(np.float32)
        for i, k in enumerate((0, 2, 4, 6)):
            out[f"{role}/w{i}"], out[f"{role}/b{i}"] = init[f"mlp.{k}.weight"].astype(dtype), init[f"mlp.{k}.bias"].astype(dtype)
    return out


def drive(nets=None, seed=0):
    """Run the scenario. Returns (records [(role, obs, action)], falls, final xy, the duck)."""
    import engine
    duck = engine.Duck()
    if nets:
        duck.policies = {r: nets[r] for r in duck.policies}
    rec = []
    for role in list(duck.policies):
        f = duck.policies[role]
        duck.policies[role] = (lambda obs, f=f, role=role: (lambda a: (rec.append((role, obs.copy(), np.array(a))), a)[1])(f(obs)))
    for what, secs in SCENARIO:
        if what in ("s", "space", "r", "b"):
            duck.press(what)
        elif what == "lvl5":
            duck.press("d"); duck.press("d")
        else:
            keys = set() if what == "stand" else {"up"} if what == "run" else set(what.split("+"))
            duck.set_keys(keys, meta=what == "run")
        for _ in range(round(secs / engine.CONTROL_DT)):
            duck.step()
    return rec, duck.falls, duck.data.qpos[duck.free_qadr:duck.free_qadr + 2].copy(), duck


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(HERE / "policies.npz"))
    a = ap.parse_args()
    arrays = pack()
    np.savez_compressed(a.out, **arrays)
    nets = policy_net.load(a.out)
    rec, falls, end, _ = drive()
    res = dict(when=time.strftime("%Y-%m-%d %H:%M"), pack_bytes=os.path.getsize(a.out), ticks=len(rec), roles={})
    print(f"{a.out}: {res['pack_bytes'] / 1e6:.2f} MB for {len(nets)} policies; scenario {len(rec)} ticks ({len(rec) / 50:.0f} s), ONNX falls {falls}")
    print(f"{'role':<10}{'ticks':>7}{'worst |da| rad':>16}{'RMS rad':>10}{'largest |a| rad':>17}")
    for role in nets:
        rows = [(o, x) for r, o, x in rec if r == role]
        if not rows:
            res["roles"][role] = dict(ticks=0)
            print(f"{role:<10}{0:>7}   (not reached in the scenario)")
            continue
        d = np.array([nets[role](o) - x for o, x in rows])
        res["roles"][role] = dict(ticks=len(rows), worst_rad=float(np.abs(d).max()), rms_rad=float(np.sqrt((d ** 2).mean())),
                                  largest_action_rad=float(np.abs([x for _, x in rows]).max()))
        print(f"{role:<10}{len(rows):>7}{np.abs(d).max():>16.5f}{np.sqrt((d ** 2).mean()):>10.5f}{np.abs([x for _, x in rows]).max():>17.2f}")
    rec2, falls2, end2, _ = drive(nets)
    t0 = time.perf_counter()
    for _, o, _ in rec[:2000]:
        nets["walk"](o)
    res.update(onnx=dict(falls=int(falls), end=[round(float(v), 3) for v in end]), pack=dict(falls=int(falls2), end=[round(float(v), 3) for v in end2]),
               end_apart_m=round(float(np.linalg.norm(end - end2)), 3), mac_ms_per_call=round((time.perf_counter() - t0) / min(2000, len(rec)) * 1e3, 3))
    print(f"closed loop with the pack's networks: falls {falls2} (ONNX {falls}); ends {res['end_apart_m']} m from where ONNX ended; "
          f"{res['mac_ms_per_call']} ms per call on this Mac")
    (HERE.parent / "docs/policy_pack_results.json").write_text(json.dumps(res, indent=1))
