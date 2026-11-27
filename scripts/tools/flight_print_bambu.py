#!/usr/bin/env python3
"""Sliced Bambu Studio projects (.gcode.3mf) of the flight frame, one per printer.

    python3 scripts/tools/flight_print_orient.py && python3 scripts/tools/flight_print_bambu.py [--printers "A2L,X2D"]

Uses the installed Bambu Studio's own system presets (machine, 0.20 mm Standard process, Bambu PLA Basic), flattened,
with the frame's print settings on top: 3 walls, 15 % gyroid, tree supports where needed, brim on the tall parts.
Writes cad/flight/print/microduck_flight_<printer>.3mf (all plates sliced) and prints time and filament per plate.
"""
import argparse, json, os, re, subprocess, sys, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
APP = "/Applications/BambuStudio.app/Contents"
PROF = os.path.join(APP, "Resources", "profiles", "BBL")
OUT = os.path.join(ROOT, "cad", "flight", "print")
OVERRIDE = dict(wall_loops="3", sparse_infill_density="15%", sparse_infill_pattern="gyroid", top_shell_layers="5", bottom_shell_layers="4",
                enable_support="1", support_type="tree(auto)", support_threshold_angle="35", brim_type="auto_brim")
PARTS = [("backpack", 1), ("bottom_cap", 1), ("wing_left", 1), ("wing_right", 1), ("nacelle_left", 1), ("nacelle_right", 1), ("fan_collar", 4)]


def flatten(kind, name):
    d = json.load(open(os.path.join(PROF, kind, name + ".json")))
    if d.get("inherits"):
        base = flatten(kind, d["inherits"]); base.update(d); d = base
    return d


ap = argparse.ArgumentParser(); ap.add_argument("--printers", default="A2L,X2D"); a = ap.parse_args()
for pr in a.printers.split(","):
    tmp = os.path.join(OUT, "_presets_" + pr); os.makedirs(tmp, exist_ok=True)
    machine = flatten("machine", f"Bambu Lab {pr} 0.4 nozzle")
    process = flatten("process", f"0.20mm Standard @BBL {pr}"); process.update(OVERRIDE)
    fil = flatten("filament", f"Bambu PLA Basic @BBL {pr} 0.4 nozzle" if os.path.exists(os.path.join(PROF, "filament", f"Bambu PLA Basic @BBL {pr} 0.4 nozzle.json")) else f"Bambu PLA Basic @BBL {pr}")
    for d in (machine, process, fil):
        d.pop("inherits", None); d["from"] = "system"
    paths = {}
    for k, d in (("machine", machine), ("process", process), ("filament", fil)):
        paths[k] = os.path.join(tmp, k + ".json"); json.dump(d, open(paths[k], "w"), indent=1)
    stls = [os.path.join(OUT, "stl", n + ".stl") for n, q in PARTS for _ in range(q)]
    dst = os.path.join(OUT, f"microduck_flight_{pr}.3mf")
    n_ext = len(machine.get("nozzle_diameter", ["0.4"]))
    cmd = [os.path.join(APP, "MacOS", "BambuStudio"), "--load-settings", paths["machine"] + ";" + paths["process"], "--load-filaments", ";".join([paths["filament"]] * n_ext),
           "--arrange", "1", "--orient", "0", "--slice", "0", "--debug", "1", "--export-3mf", dst, *stls]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    print(f"== {pr}: exit {r.returncode}"); print((r.stdout + r.stderr)[-1500:] if r.returncode else "")
    if r.returncode == 0 and os.path.exists(dst):
        z = zipfile.ZipFile(dst)
        info = z.read("Metadata/slice_info.config").decode()
        for plate in re.findall(r"<plate>.*?</plate>", info, re.S):
            g = lambda k: (re.search(rf'key="{k}" value="([^"]*)"', plate) or [None, "?"])[1]
            used = sum(float(x) for x in re.findall(r'used_g="([\d.]+)"', plate))
            print(f"   plate {g('index')}: {int(float(g('prediction'))) // 3600} h {int(float(g('prediction'))) % 3600 // 60:02d} min, {used:.0f} g filament incl. supports, {plate.count('<object ')} objects")
        print(f"   wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
