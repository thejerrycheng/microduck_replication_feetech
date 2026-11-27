#!/usr/bin/env python3
"""Build the web assembly guide of the flying-duck frame into the personal website checkout.

    ~/.venvs/microduck_live/bin/python scripts/tools/flight_assembly_export.py      # geometry (docs/assembly/)
    python3 scripts/tools/flight_assembly_page.py [--site ~/Desktop/thejerrycheng.github.io]

Writes SITE/microduck_assembly.html (nav and footer taken from SITE/microduck.html), copies the geometry to
SITE/assets/projects/microduck/assembly/ and zips the printable files (STL, STEP) next to it.
Screw and insert positions are the CAD's (cad/flight/build_parts.py, millimetres in the duck's standing frame).
"""
import argparse, json, os, re, shutil, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import flight_config as C

ap = argparse.ArgumentParser(); ap.add_argument("--site", default=os.path.expanduser("~/Desktop/thejerrycheng.github.io"))
a = ap.parse_args()
cad = json.load(open(os.path.join(ROOT, "docs", "flight_cad_results.json")))
XC = json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))["layout"]["x_c"] * 1e3
bay = cad["bay"]
W, BAY_Y = C.PACK_WALL_MM, bay["half_w"]
POD_BOT = cad["pod_z_mm"][0]
g = cad["wing"]
bay_x0 = bay["x"][0]
bay_z0 = C.BATT_POS[2] * 1e3 - C.BATT_SIZE[2] * 1e3 / 2 - 0.5
mm = lambda p: [round(v * 1e-3, 5) for v in p]
SCREWS = dict(
    root=[dict(p=mm((g["root_x"], s * (bay["head_half_w"] + W), z)), d=[0, s, 0]) for s in (1, -1) for z in g["root_z"]],          # insert in the wing root, screw from the bay
    tongue=[dict(p=mm((XC, s * y, POD_BOT)), d=[0, 0, 1]) for s in (1, -1) for y in g["tongue_y"]],
    side=[dict(p=mm((x, s * g["side_land_y"], g["side_z"])), d=[0, s, 0], grip=.007) for s in (1, -1) for x in g["side_x"]],   # insert in the nacelle's flank, screw through the wing tip
    stack=[dict(p=mm((x, y, bay["stack_z"])), d=[0, 0, 1], grip=.022) for x, y in bay["stack_xy"]],
    lid=[dict(p=mm((x, y, bay["lid_z"])), d=[0, 0, -1], grip=.0032) for x, y in bay["lid_xy"]],                     # insert in a rear lug, screw down through the lid                # insert in a roof boss, long screw up through the stack                   # insert under the nacelle, screw from below
    cap=[dict(p=mm((bay_x0 - W / 2 + 4.5, s * (BAY_Y - 4.0), bay_z0)), d=[0, 0, 1]) for s in (1, -1)])              # insert in the backpack's rim, screw from below
PARTS = [("backpack", 1, "top face down"), ("top_lid", 1, "top face down"), ("bottom_cap", 1, "flat"), ("wing_left", 1, "root face down"), ("wing_right", 1, "root face down"),
         ("nacelle_left", 1, "top face down"), ("nacelle_right", 1, "top face down"), ("fan_collar", 4, "flat")]
mass = {n: cad["parts"][n] for n, _, _ in PARTS}

dst = os.path.join(a.site, "assets", "projects", "microduck", "assembly"); os.makedirs(dst, exist_ok=True)
for f in ("parts.bin", "parts.json"):
    shutil.copy(os.path.join(ROOT, "docs", "assembly", f), os.path.join(dst, f))
for ext in ("stl", "step"):
    with zipfile.ZipFile(os.path.join(dst, f"microduck_flight_frame_v8_{ext}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for n, q, _ in PARTS:
            z.write(os.path.join(ROOT, "cad", "flight", "out", f"{n}.{ext}"), f"{n}{'_x%d' % q if q > 1 else ''}.{ext}")
        if ext == "step":
            z.write(os.path.join(ROOT, "cad", "flight", "out", "flight_frame_assembly.step"), "flight_frame_assembly.step")

src = open(os.path.join(a.site, "microduck.html")).read()
nav = src[src.index('<nav class="nav"'):src.index('<main')]
nav = nav.replace('<a href="projects.html" aria-current="page">', '<a href="projects.html" aria-current="page">')
footer = src[src.index('<footer'):src.index('</footer>') + 9]
css_v = re.search(r'gotham\.css\?v=(\w+)', src).group(1)
frame_g = cad["frame_mass_g"]["PLA"]
rows = "".join(f"<tr><td>{n.replace('_', ' ').replace('fan collar', 'fan ring').replace('top lid', 'lid').replace('bottom cap', 'cap')}</td><td>{q}</td><td>{o}</td></tr>" for n, q, o in PARTS)

page = r"""<!DOCTYPE html>
<html lang="en" data-theme="day">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MicroDuck flight pack: assembly — Jerry (Qilong) Cheng</title>
  <meta name="description" content="Step-by-step assembly of the MicroDuck flight pack: eleven printed parts, four 50 mm fans, 24 screws.">
  <meta name="author" content="Jerry Cheng">
  <link rel="canonical" href="https://thejerrycheng.github.io/microduck_assembly.html">
  <link rel="icon" href="assets/images/favicon.v4.svg" type="image/svg+xml">
  <link rel="shortcut icon" href="assets/images/favicon.v4.ico">
  <meta name="theme-color" content="#F4EEE2">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400..700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="assets/css/gotham.css?v=@CSSV@">
  <script>document.documentElement.classList.add('js');</script>
  <script>(function(){try{var t=localStorage.getItem("theme-v2");if(t==="day"||t==="night")document.documentElement.setAttribute("data-theme",t);}catch(e){}})();</script>
  <style>
    .as { --stage: #1c1e22; --hot: #ff671f; }
    .as h1, .as h2 { font-family: var(--font-display); font-weight: 600; letter-spacing: -0.03em; line-height: 1.04; }
    .as-wrap { max-width: 1080px; margin: 0 auto; padding: 0 20px; }
    .as-head { padding-top: clamp(36px, 7vw, 84px); }
    .as-head .eyebrow { font-size: 14px; color: var(--ash); text-decoration: none; }
    .as-head h1 { margin: 10px 0 0; font-size: clamp(40px, 7vw, 76px); }
    .as-head p { margin-top: 14px; font-size: clamp(17px, 1.8vw, 21px); color: var(--ash); }
    .as-links { margin-top: 22px; display: flex; flex-wrap: wrap; gap: 10px; }
    .as-links a { display: inline-flex; align-items: center; height: 42px; padding: 0 18px; border-radius: 999px; background: var(--panel); border: 1px solid var(--line); color: var(--ink); font-size: 14.5px; font-weight: 500; text-decoration: none; }
    .as-links a:hover { background: var(--panel-2); }
    .as-links a.main { background: var(--ink); color: var(--paper); border-color: var(--ink); }

    .as-need { margin-top: clamp(36px, 6vw, 64px); display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
    .as-need div { padding: 14px 16px; border-radius: 18px; background: var(--panel); border: 1px solid var(--line); }
    .as-need b { display: block; font-size: 26px; font-weight: 600; letter-spacing: -0.03em; }
    .as-need span { font-size: 13.5px; color: var(--ash); }

    .as-guide { margin-top: clamp(36px, 6vw, 64px); display: grid; grid-template-columns: 1fr; gap: 12px; width: min(1720px, 100vw - 32px); margin-left: 50%; transform: translateX(-50%); }
    .as-stage { position: relative; border-radius: 28px; overflow: hidden; background: radial-gradient(120% 100% at 50% 0%, #3a3f4b, var(--stage) 70%); height: clamp(460px, calc(100vh - 250px), 1100px); }
    .as-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; touch-action: none; cursor: grab; }
    .as-stage canvas:active { cursor: grabbing; }
    .as-hint { position: absolute; left: 16px; bottom: 14px; font-size: 12.5px; color: rgba(255,255,255,.55); pointer-events: none; }
    .as-replay { position: absolute; right: 12px; bottom: 10px; height: 34px; padding: 0 14px; border-radius: 999px; border: 1px solid rgba(255,255,255,.24); background: rgba(255,255,255,.12); color: #fff; font: inherit; font-size: 13px; cursor: pointer; }
    .as-load { position: absolute; inset: 0; display: grid; place-items: center; color: rgba(255,255,255,.7); font-size: 14px; }
    .as-card { display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, 380px); gap: 8px 32px; align-items: center; padding: clamp(18px, 2.4vw, 26px) clamp(20px, 3vw, 34px); border-radius: 28px; background: var(--panel); border: 1px solid var(--line); }
    .as-card p { max-width: 62ch; }
    .as-n { font-size: 13px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--hot); }
    .as-card h2 { margin: 10px 0 0; font-size: clamp(28px, 3.4vw, 40px); }
    .as-card p { margin-top: 12px; font-size: 17px; line-height: 1.45; color: var(--ink-soft); }
    .as-chips { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
    .as-chips span { display: inline-flex; align-items: center; height: 32px; padding: 0 13px; border-radius: 999px; border: 1px solid var(--line); font-size: 13.5px; font-weight: 500; background: var(--panel-2); }
    .as-chips span.warn { border-color: var(--hot); color: var(--hot); background: transparent; }
    .as-ctl { display: flex; align-items: center; gap: 10px; }
    .as-ctl button { height: 46px; border-radius: 999px; border: 1px solid var(--line); background: transparent; color: var(--ink); font: inherit; font-size: 15px; font-weight: 500; padding: 0 20px; cursor: pointer; }
    .as-ctl button.next { background: var(--ink); color: var(--paper); border-color: var(--ink); flex: 1; }
    .as-ctl button:disabled { opacity: .35; cursor: default; }
    .as-dots { margin-top: 16px; display: flex; gap: 6px; }
    .as-dots button { flex: 1; height: 6px; padding: 0; border: 0; border-radius: 3px; background: var(--line); cursor: pointer; }
    .as-dots button.done { background: var(--ash-light); }
    .as-dots button.on { background: var(--hot); }

    .as-sec { margin-top: clamp(56px, 9vw, 104px); }
    .as-sec h2 { font-size: clamp(28px, 4vw, 44px); }
    .as-table { margin-top: 20px; width: 100%; border-collapse: collapse; font-size: 15.5px; }
    .as-table th, .as-table td { text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--line); }
    .as-table th { font-size: 12.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--ash); }
    .as-table td:nth-child(2) { font-variant-numeric: tabular-nums; }
    .as-note { margin-top: 14px; font-size: 13.5px; color: var(--ash); max-width: 70ch; }
    .as-check { margin-top: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
    .as-check div { padding: 18px; border-radius: 18px; border: 1px solid var(--line); background: var(--panel); font-size: 15.5px; line-height: 1.4; }
    .as-check b { display: block; margin-bottom: 4px; }
    .as-end { height: clamp(64px, 10vw, 120px); }
    @media (max-width: 860px) { .as-stage { height: clamp(380px, 68vh, 720px); } .as-card { grid-template-columns: 1fr; } }
  </style>
</head>
<body>

@NAV@
<main class="as">
<div class="as-wrap">

  <header class="as-head">
    <a class="eyebrow" href="microduck.html#flight">MicroDuck &rsaquo; Flight</a>
    <h1>Assemble the flight pack</h1>
    <p>Print. Press in the fans. 24 screws. Hook it on.</p>
    <div class="as-links">
      <a class="main" href="assets/projects/microduck/assembly/microduck_flight_frame_v8_stl.zip" download>Download STL</a>
      <a href="assets/projects/microduck/assembly/microduck_flight_frame_v8_step.zip" download>STEP</a>
      <a href="https://github.com/thejerrycheng/microduck_replication_feetech/blob/duck-studio/docs/duck_flight_bom.md" target="_blank" rel="noopener">Where to buy</a>
    </div>
  </header>

  <div class="as-need" aria-label="What you need">
    <div><b>4</b><span>50 mm fans: 2 CW, 2 CCW</span></div>
    <div><b>1</b><span>SpeedyBee F405 V4 stack: flight controller + 60 A 4-in-1 ESC</span></div>
    <div><b>1</b><span>4S 1300 mAh pack</span></div>
    <div><b>20</b><span>M3 heat-set inserts</span></div>
    <div><b>12</b><span>M3 &times; 8 screws</span></div>
    <div><b>4</b><span>M3 &times; 12 screws, wing tips</span></div>
    <div><b>4</b><span>M3 &times; 25 screws, for the stack</span></div>
    <div><b>4</b><span>M2.5 grub screws</span></div>
  </div>

  <section class="as-guide" aria-label="Assembly steps">
    <div class="as-stage">
      <canvas id="asCanvas" aria-label="3D view of the current assembly step. Drag to turn."></canvas>
      <div class="as-load" id="asLoad">Loading model&hellip;</div>
      <div class="as-hint">Drag to turn</div>
      <button class="as-replay" id="asReplay" type="button">Replay</button>
    </div>
    <div class="as-card" aria-live="polite">
      <div>
        <div class="as-n" id="asN"></div>
        <h2 id="asT"></h2>
        <p id="asP"></p>
        <div class="as-chips" id="asC"></div>
      </div>
      <div>
      <div class="as-ctl">
        <button type="button" id="asPrev">Back</button>
        <button type="button" class="next" id="asNext">Next</button>
      </div>
      <div class="as-dots" id="asDots"></div>
      </div>
    </div>
  </section>

  <section class="as-sec" id="print">
    <h2>Print</h2>
    <table class="as-table">
      <thead><tr><th>Part</th><th>Qty</th><th>On the bed</th></tr></thead>
      <tbody>@ROWS@</tbody>
    </table>
    <p class="as-note">PLA, 3 walls, 15 % gyroid: @FRAME@ g for the set. Orientations are suggestions; no part has been printed yet.</p>
  </section>

  <section class="as-sec" id="wire">
    <h2>Wire</h2>
    <table class="as-table">
      <thead><tr><th>From</th><th>To</th><th>How</th></tr></thead>
      <tbody>
        <tr><td>Battery, XT60</td><td>ESC power pads</td><td>12 AWG. Capacitor across the pads.</td></tr>
        <tr><td>4 fans</td><td>ESC motor pads</td><td>3 wires each. Swap any two to reverse a fan.</td></tr>
        <tr><td>ESC</td><td>Flight controller</td><td>The 8-pin cable in the box.</td></tr>
        <tr><td>Flight controller T3 / R3 / GND</td><td>Duck computer pin 16 / 18 / 20</td><td>3 wires, 3.3 V serial. T3 &rarr; RX, R3 &rarr; TX.</td></tr>
        <tr><td>Radio receiver</td><td>Flight controller T2 / R2</td><td>Arm and kill switch stay on the radio.</td></tr>
      </tbody>
    </table>
    <p class="as-note">The duck computer is a Radxa ZERO 3W; pins 16 and 18 are its UART4. Check that the robot HAT leaves them free, or use a USB-C cable from the flight controller instead. No 5 V wire between the boards: ground only. Not bench-tested.</p>
  </section>

  <section class="as-sec" id="check">
    <h2>Before power</h2>
    <div class="as-check">
      <div><b>Current limit</b>Keep each fan under 45 A: cap motor output in Betaflight.</div>
      <div><b>Spin</b>Front-left and rear-right one way. The other two the opposite way.</div>
      <div><b>Tether</b>First spin-up with the duck tied down.</div>
    </div>
    <p class="as-note">Status: design and simulation. This frame has not been built or flown.</p>
  </section>
  <div class="as-end"></div>
</div>
</main>

@FOOTER@

<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

const BASE = "assets/projects/microduck/assembly/";
const SCREWS = @SCREWS@;
const FANS = ["FL", "RL", "FR", "RR"];
const AXES = @AXES@;
const fan = k => FANS.map(f => k + "_" + f);
const PRINTED = ["backpack", "top_lid", "bottom_cap", "wing_left", "wing_right", "nacelle_left", "nacelle_right", ...fan("ring")];
// explode offsets (m) for the two "loose parts" steps
const EXPL = { backpack: [-.02, 0, -.05], top_lid: [-.02, 0, .03], bottom_cap: [-.02, 0, -.12], wing_left: [0, .02, -.01], wing_right: [0, -.02, -.01], nacelle_left: [.02, .07, .05], nacelle_right: [.02, -.07, .05],
  ring_FL: [.05, .07, -.05], ring_RL: [-.05, .07, -.05], ring_FR: [.05, -.07, -.05], ring_RR: [-.05, -.07, -.05] };
const side = (n, v) => (/_left|_FL|_RL/.test(n) ? v : [v[0], -v[1], v[2]]);
// each step: show = parts already in place, add = parts that arrive {from: offset}, screws/inserts, camera [pos, target]
const FANSALL = [...fan("shroud"), ...fan("rotor"), ...fan("motor"), ...fan("ring")];
const WINGSET = ["nacelle_left", "nacelle_right", "wing_left", "wing_right", ...FANSALL];
const STEPS = [
  { t: "Print 11 parts", p: "PLA. 3 walls. 15 % gyroid.", chips: ["1 backpack", "1 lid", "1 cap", "2 wings", "2 nacelles", "4 rings"],
    add: PRINTED, expl: true, cam: [[.42, -.48, .42], [0, 0, .17]] },
  { t: "Melt in 20 inserts", p: "Soldering iron, 220 °C. Flush with the surface.", chips: ["20× M3 insert", "2 per wing root", "4 per nacelle", "4 in the backpack", "4 in the lid"],
    show: PRINTED.filter(n => !n.startsWith("ring") && n !== "bottom_cap"), expl: true, inserts: ["root", "tongue", "side", "cap", "stack", "lid"], cam: [[-.30, -.50, .05], [0, 0, .17]] },
  { t: "Push the fans up", p: "From below, until they stop. Wires point inboard.", chips: ["4 fans", "same spin on diagonals"],
    show: ["nacelle_left", "nacelle_right"], add: [...fan("shroud"), ...fan("rotor"), ...fan("motor")], from: [0, 0, -.09], cam: [[.34, -.42, .12], [0, 0, .21]] },
  { t: "Lock with the rings", p: "Slide a ring onto each fan. One grub screw each.", chips: ["4× M2.5 grub"],
    show: ["nacelle_left", "nacelle_right", ...fan("shroud"), ...fan("rotor"), ...fan("motor")], add: fan("ring"), from: [0, 0, -.07], cam: [[.30, -.40, .06], [0, 0, .21]] },
  { t: "Wings under the nacelles", p: "Motor wires down the wing. Two screws from below, each side.", chips: ["4× M3×8"],
    show: ["nacelle_left", "nacelle_right", ...FANSALL], add: ["wing_left", "wing_right"], from: [0, 0, -.07], screws: ["tongue"], focus: [1, .075], cam: [[-.25, -.30, -.42], [0, 0, .19]] },
  { t: "Lock the sides", p: "Two screws through the wing tip, straight into the nacelle.", chips: ["4× M3×12", "2.5 mm hex key"],
    show: WINGSET, screws: ["side"], focus: [1, .06], cam: [[-.22, -.62, .46], [0, .05, .21]] },
  { t: "Wings onto the backpack", p: "Motor wires through the side holes. Two screws from inside, each side.", chips: ["4× M3×8", "reach in from the top"],
    show: WINGSET, add: ["backpack"], from: [-.02, 0, -.10], screws: ["root"], ghost: ["backpack"], cam: [[-.44, -.34, .30], [-.03, 0, .17]] },
  { t: "Stack onto the lid", p: "On the bench, lid upside down. Flight controller first, ESC on it. Four long screws.", chips: ["4× M3×25", "rubber grommets"],
    show: ["top_lid"], add: ["stack"], from: [0, 0, -.08], screws: ["stack"], cam: [[-.30, -.36, -.22], [-.07, 0, .16]] },
  { t: "Wire, then close the lid", p: "Solder the four fans to the ESC. Tab in at the front, two screws at the rear.", chips: ["2× M3×8", "see Wire below"], warn: "Battery unplugged",
    show: [...WINGSET, "backpack"], add: ["top_lid", "stack"], from: [-.03, 0, .09], screws: ["lid"], focus: [0, .07], ghost: ["backpack"], cam: [[-.46, -.30, .34], [-.04, 0, .16]] },
  { t: "Battery in", p: "From below, up under the stack. Foam on the sides.", chips: ["4S pack"], warn: "Unplugged",
    show: [...WINGSET, "backpack", "top_lid", "stack"], add: ["battery"], from: [0, 0, -.13], ghost: ["backpack"], cam: [[-.40, -.36, .0], [-.04, 0, .15]] },
  { t: "Hook onto the duck", p: "Lower it onto the back. The hook grabs the roof edge.", chips: ["no screws"],
    show: ["duck"], add: [...WINGSET, "backpack", "top_lid", "stack", "battery"], from: [-.07, 0, .11], cam: [[-.46, -.46, .34], [-.01, 0, .14]] },
  { t: "Close the cap", p: "It locks under the trunk. Two screws from below.", chips: ["2× M3×8"],
    show: ["duck", ...WINGSET, "backpack", "top_lid", "stack", "battery"], add: ["bottom_cap"], from: [-.03, 0, -.08], screws: ["cap"], focus: [0, .08], cam: [[-.34, -.26, -.30], [-.04, 0, .11]] },
  { t: "Done", p: "Check the three points below before the first spin-up.", chips: ["@ALLUP@ g all-up"],
    show: ["duck", "bottom_cap", ...WINGSET, "backpack", "top_lid", "stack", "battery"], spin: true, cam: [[.44, -.50, .36], [0, 0, .14]] },
];

const $ = id => document.getElementById(id);
const canvas = $("asCanvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, .01, 10); camera.up.set(0, 0, 1);
scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, 1.35));
const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(.5, -.8, 1.2); scene.add(key);
const rim = new THREE.DirectionalLight(0xffd9b8, 1.0); rim.position.set(-.8, .6, .3); scene.add(rim);
const controls = new OrbitControls(camera, canvas);
controls.enablePan = false; controls.enableDamping = true; controls.minDistance = .06; controls.maxDistance = 2.2;
const HOT = new THREE.Color("#ff671f");
const parts = {}, extras = new THREE.Group(); scene.add(extras);
let cur = 0, t0 = 0, camFrom = null, camTo = null, running = true, userMoved = false;
controls.addEventListener("start", () => { userMoved = true; });

function fit() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false); camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
}
new ResizeObserver(fit).observe(canvas.parentElement);

function fastener(kind, s, travel) {            // insert (brass sleeve) or screw (shank + head), axis along s.d, tip at s.p
  const grp = new THREE.Group(), d = new THREE.Vector3(...s.d);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
  if (kind === "insert") {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(.0026, .0026, .0057, 20), new THREE.MeshStandardMaterial({ color: "#d9a441", metalness: .8, roughness: .35, emissive: "#6b4a10" }));
    m.position.copy(d).multiplyScalar(.00285); m.quaternion.copy(q); grp.add(m);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(.0052, .0007, 8, 28), new THREE.MeshBasicMaterial({ color: HOT }));
    ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), d); grp.add(ring);
  } else {
    const mat = new THREE.MeshStandardMaterial({ color: "#c9ccd1", metalness: .9, roughness: .3, emissive: "#40200a" });
    const grip = s.grip || .005;
    const shank = new THREE.Mesh(new THREE.CylinderGeometry(.0015, .0015, grip + .003, 14), mat); shank.position.copy(d).multiplyScalar((.003 - grip) / 2); shank.quaternion.copy(q); grp.add(shank);
    const head = new THREE.Mesh(new THREE.CylinderGeometry(.00275, .00275, .003, 18), mat); head.position.copy(d).multiplyScalar(-grip - .0015); head.quaternion.copy(q); grp.add(head);
  }
  grp.userData = { home: new THREE.Vector3(...s.p), dir: d, travel };
  return grp;
}

function setStep(i, replay) {
  cur = i; const S = STEPS[i];
  $("asN").textContent = "Step " + (i + 1) + " of " + STEPS.length;
  $("asT").textContent = S.t; $("asP").textContent = S.p;
  $("asC").innerHTML = (S.warn ? '<span class="warn">' + S.warn + "</span>" : "") + S.chips.map(c => "<span>" + c + "</span>").join("");
  $("asPrev").disabled = i === 0; $("asNext").textContent = i === STEPS.length - 1 ? "Start over" : "Next";
  [...$("asDots").children].forEach((b, k) => { b.className = k === i ? "on" : k < i ? "done" : ""; });
  const add = new Set(S.add || []), show = new Set(S.show || []);
  for (const [n, o] of Object.entries(parts)) {
    o.visible = add.has(n) || show.has(n);
    const e = S.expl && EXPL[n] ? EXPL[n] : [0, 0, 0];
    o.userData.rest = new THREE.Vector3(...e);
    o.userData.from = add.has(n) && !S.expl ? new THREE.Vector3(...side(n, S.from || [0, 0, 0])) : null;
    o.userData.isNew = add.has(n) && n !== "duck";
    const ghost = (S.ghost || []).includes(n);
    o.material.transparent = ghost || !!o.userData.from; o.material.opacity = ghost ? .28 : 1; o.material.depthWrite = !ghost;
    o.material.emissive.copy(o.userData.isNew && !S.expl ? HOT : new THREE.Color(0)); o.material.emissiveIntensity = .12;
    o.position.copy(o.userData.rest);
  }
  // frame the visible parts: keep the step's view direction, fit the bounding sphere
  const box = new THREE.Box3();
  for (const o of Object.values(parts)) if (o.visible) {
    const m = o.userData.inner || o; if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
    box.union(m.geometry.boundingBox.clone().translate(m === o ? o.userData.rest : m.position.clone().add(o.userData.rest)));
  }
  if (S.focus) {                              // screw steps: frame the screws going in on one side, with some of the parts around them
    box.makeEmpty();
    for (const k of S.screws) for (const s of SCREWS[k]) if (s.p[1] * S.focus[0] >= 0) box.expandByPoint(new THREE.Vector3(...s.p));
    box.expandByScalar(S.focus[1]);
  }
  // fit the box's eight corners in the view along the step's view direction (tighter than a bounding sphere)
  const ctr = box.getCenter(new THREE.Vector3());
  const vfov = camera.fov * Math.PI / 360, hfov = Math.atan(Math.tan(vfov) * camera.aspect);
  const dir = new THREE.Vector3(...S.cam[0]).sub(new THREE.Vector3(...S.cam[1])).normalize();
  const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), dir).normalize(), up = new THREE.Vector3().crossVectors(dir, right);
  let dist = 0;
  for (let i = 0; i < 8; i++) {
    const c = new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(ctr);
    dist = Math.max(dist, Math.abs(c.dot(right)) / Math.tan(hfov) + c.dot(dir), Math.abs(c.dot(up)) / Math.tan(vfov) + c.dot(dir));
  }
  dist *= 1.10;
  camTo = { p: ctr.clone().addScaledVector(dir, dist), t: ctr.clone() };
  extras.clear();
  for (const k of S.inserts || []) for (const s of SCREWS[k]) {
    const owner = k === "root" ? (s.d[1] > 0 ? "wing_left" : "wing_right") : (k === "tongue" || k === "side") ? (s.p[1] > 0 ? "nacelle_left" : "nacelle_right") : k === "stack" ? "top_lid" : "backpack";
    const f = fastener("insert", s, .03); f.userData.home.add(new THREE.Vector3(...EXPL[owner])); extras.add(f);
  }
  for (const k of S.screws || []) for (const s of SCREWS[k]) extras.add(fastener("screw", s, .045));
  if (!replay || !userMoved) { camFrom = { p: camera.position.clone(), t: controls.target.clone() }; userMoved = false; } else camFrom = null;
  t0 = performance.now();
}

const ease = x => 1 - Math.pow(1 - Math.min(Math.max(x, 0), 1), 3);
function frame(now) {
  requestAnimationFrame(frame);
  if (!running) return;
  const S = STEPS[cur], e = (now - t0) / 1000;
  const a = ease((e - .25) / 1.1), b = ease((e - 1.2) / .9);
  for (const o of Object.values(parts)) {
    if (!o.visible) continue;
    if (o.userData.from) { o.position.copy(o.userData.from).multiplyScalar(1 - a).add(o.userData.rest); o.material.opacity = Math.min(1, .15 + a); }
    if (o.userData.isNew && !S.expl) o.material.emissiveIntensity = .2 - .14 * ease((e - 1.8) / 1.2);
    if (o.userData.spin && S.spin) o.rotateOnAxis(o.userData.axis, o.userData.spin * .5);
  }
  for (const f of extras.children) {
    const u = f.userData, k = S.screws ? b : ease((e - .3) / 1.0);
    f.position.copy(u.home).addScaledVector(u.dir, -(1 - k) * u.travel);
    f.visible = k > .01;
  }
  if (camFrom) {
    const k = ease(e / .9);
    camera.position.lerpVectors(camFrom.p, camTo.p, k);
    controls.target.lerpVectors(camFrom.t, camTo.t, k);
    if (k >= 1) camFrom = null;
  }
  controls.update(); renderer.render(scene, camera);
}

async function load() {
  const [man, buf] = await Promise.all([fetch(BASE + "parts.json?v=@GEOV@").then(r => r.json()), fetch(BASE + "parts.bin?v=@GEOV@").then(r => r.arrayBuffer())]);
  for (const p of man.parts) {
    let g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buf, p.v_off, p.v_n * 3), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(buf, p.f_off, p.f_n * 3), 1));
    if (p.name === "duck") g.computeVertexNormals(); else g = toCreasedNormals(g, Math.PI / 5);
    const metal = /shroud|motor/.test(p.name);
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: p.color, roughness: metal ? .5 : .62, metalness: metal ? .35 : .02 }));
    if (p.name.startsWith("rotor")) {           // spin about the fan axis in the last step: re-centre the geometry on the hub
      const c = new THREE.Vector3(...p.center); g.translate(-c.x, -c.y, -c.z);
      const grp = new THREE.Group(); m.position.copy(c); grp.add(m);
      grp.material = m.material; grp.userData.inner = m;
      const f = p.name.slice(-2), sgn = (f === "FL" || f === "RR") ? -1 : 1;
      m.userData.spin = sgn; m.userData.axis = new THREE.Vector3(...AXES[f]).normalize();
      const tick = () => { if (STEPS[cur].spin) m.rotateOnAxis(m.userData.axis, sgn * .45); requestAnimationFrame(tick); }; tick();
      scene.add(grp); parts[p.name] = grp; continue;
    }
    scene.add(m); parts[p.name] = m;
  }
  STEPS.forEach((_, k) => { const b = document.createElement("button"); b.type = "button"; b.setAttribute("aria-label", "Step " + (k + 1)); b.onclick = () => setStep(k); $("asDots").appendChild(b); });
  $("asLoad").remove(); fit();
  camera.position.set(...STEPS[0].cam[0]); controls.target.set(...STEPS[0].cam[1]);
  setStep(0); requestAnimationFrame(frame);
}
$("asPrev").onclick = () => setStep(Math.max(cur - 1, 0));
$("asNext").onclick = () => setStep(cur === STEPS.length - 1 ? 0 : cur + 1);
$("asReplay").onclick = () => setStep(cur, true);
addEventListener("keydown", e => { if (e.target.closest("input,textarea")) return; if (e.key === "ArrowRight") $("asNext").click(); if (e.key === "ArrowLeft") $("asPrev").click(); });
new IntersectionObserver(es => { running = es[0].isIntersecting; }).observe(canvas);
load().catch(err => { $("asLoad").textContent = "The 3D view could not load. The steps still work."; console.error(err); STEPS.forEach((_, k) => { const b = document.createElement("button"); b.type = "button"; b.onclick = () => setStep(k); $("asDots").appendChild(b); }); setStep(0); });
</script>
<script src="assets/js/site.js?v=@CSSV@"></script>
</body>
</html>
"""
allup = json.load(open(os.path.join(ROOT, "microduck_rl", "flight", "flight_pack.json")))
allup_g = round(json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))["battery_sweep"]["%dmAh" % round(C.BATT_REF_AH * 1e3)]["all_up_g"])
page = (page.replace("@NAV@", nav).replace("@FOOTER@", footer).replace("@CSSV@", css_v).replace("@ROWS@", rows).replace("@FRAME@", f"{frame_g:.0f}")
        .replace("@SCREWS@", json.dumps(SCREWS)).replace("@AXES@", json.dumps({f["name"]: [round(v, 4) for v in f["axis"]] for f in allup["fans"]})).replace("@ALLUP@", str(allup_g)).replace("@GEOV@", __import__("hashlib").md5(open(os.path.join(ROOT, "docs", "assembly", "parts.bin"), "rb").read()).hexdigest()[:8]))
open(os.path.join(a.site, "microduck_assembly.html"), "w").write(page)
print("wrote", os.path.join(a.site, "microduck_assembly.html"), "and", dst)
print("screws/inserts [m]:", json.dumps(SCREWS))
