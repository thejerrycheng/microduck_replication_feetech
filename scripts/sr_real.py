"""Success-rate table for the real-duck grades: python scripts/sr_real.py <policy> [dir]  (default docs/real/<policy>_ckpts).
Prints, per checkpoint and condition, the metrics that define the 90 % target and whether it passes."""
import glob, json, os, sys

name = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 else f"docs/real/{name}_ckpts"
rows = {}
for f in sorted(glob.glob(f"{d}/*.json")):
    b = os.path.basename(f)[:-9]  # strip _n96.json
    tag, ck, cond, suffix = b.split("_", 3)
    j = json.load(open(f)); s = j.get("summary", j)
    r = rows.setdefault((int(ck), tag, cond), {})
    if "families" in j:
        fam = j["families"]
        r["getup_min_stand"] = min(v["standing_at_end"] for k, v in fam.items() if k != "standing_pushed")
        r["getup_max_refall"] = max(v["fell_after_standing"] for k, v in fam.items() if k != "standing_pushed")
        r["pushed_stand"] = fam.get("standing_pushed", {}).get("standing_at_end")
    elif "vx_achieved_mean" in s:
        sp = suffix.replace("gait", ""); r[f"vx{sp}"] = s["vx_achieved_mean"]; r[f"falls{sp}"] = s["fallen_frac"]
    elif "completed_and_standing_frac" in s:
        r["flip_ok"] = s["completed_and_standing_frac"]; r["flip_stand"] = s["final_standing_frac"]
    elif "stable_landing_rate" in s:
        r["takeoff"] = s["valid_takeoff_rate"]; r["landing"] = s["stable_landing_rate"]; r["durable"] = s["durable_rate"]
    elif "final" in j:
        r["standing"] = j["final"]["standing_frac"]; r["seated"] = j["final"]["seated_upright_frac"]; r["fallen"] = j["final"]["fallen_frac"]

def passes(name, r):
    if name in ("velstand", "standup"):
        ok = r.get("getup_min_stand", 0) >= 0.9 and r.get("pushed_stand", 0) >= 0.9
        if name == "velstand":
            ok = ok and all(r.get(k, 1) <= 0.1 for k in ("falls15", "falls03", "falls04")) and r.get("vx04", 0) >= 0.3
        return ok
    if name == "run_v3": return r.get("falls05", 1) <= 0.1 and r.get("falls09", 1) <= 0.1 and r.get("vx05", 0) >= 0.4
    if name == "sprint_max": return r.get("falls12", 1) <= 0.1 and r.get("vx12", 0) >= 0.9
    if name == "roulade_head": return r.get("flip_ok", 0) >= 0.9
    if name == "jump": return r.get("landing", 0) >= 0.9
    if name == "sitstand_fast": return r.get("fallen", 1) <= 0.1
    return False

for (ck, tag, cond), r in sorted(rows.items(), key=lambda kv: (kv[0][2], kv[0][0])):
    print(f"{tag:4s} {ck:6d} {cond:7s} {'PASS' if passes(name, r) else 'fail'}  " + "  ".join(f"{k}={v:.3f}" for k, v in r.items() if v is not None))
