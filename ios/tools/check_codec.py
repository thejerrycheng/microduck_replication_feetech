#!/usr/bin/env python3
"""Is the phone decoding the duck's camera and microphone correctly?

Camera: compiles the app's own decoder (ios/Duck/Pixels.swift) for the Mac and compares it, pixel by pixel, with an
independent floating-point NV12 decoder (full range, BT.601, 4:2:0 nearest chroma, a quarter turn clockwise) on
synthetic frames (colour bars, ramps, seeded noise, odd sizes) and on frames taken from the real duck just now.
Also checks the dark-frame lift: monotonic, never darkens, leaves bright frames alone.
Microphone: reads the duck's readings and checks the numbers are self-consistent (64 samples, within -1..1, the
level in dBFS in line with the samples' own RMS).

    ~/.venvs/microduck_live/bin/python ios/tools/check_codec.py [duck host, default radxa-zero3.local; "none" = synthetic only]
    -> docs/figures/duck_ios_codec.pdf, docs/duck_ios_codec_table.tex, a table on stdout
"""
import json, math, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HOST = sys.argv[1] if len(sys.argv) > 1 else "radxa-zero3.local"
tmp = Path(tempfile.mkdtemp())
exe = tmp / "check_codec"
(tmp / "main.swift").write_text((ROOT / "ios/tools/check_codec.swift").read_text())
subprocess.run(["swiftc", "-O", "-o", str(exe), str(ROOT / "ios/Duck/Pixels.swift"), str(tmp / "main.swift")], check=True)


def app(nv, w, h, lift):
    (tmp / "in").write_bytes(nv.tobytes())
    subprocess.run([str(exe), str(tmp / "in"), str(w), str(h), str(int(lift)), str(tmp / "out")], check=True)
    return np.frombuffer((tmp / "out").read_bytes(), np.uint8).reshape(w, h, 4)[..., :3]


def ref(nv, w, h):
    y = nv[:w * h].reshape(h, w).astype(float)
    uv = nv[w * h:].reshape(h // 2, w // 2, 2).astype(float) - 128
    u, v = (uv[..., k].repeat(2, 0).repeat(2, 1) for k in (0, 1))
    rgb = np.stack([y + 1.402 * v, y - 0.344 * u - 0.714 * v, y + 1.772 * u], -1)
    return np.rot90(np.clip(np.rint(rgb), 0, 255), -1).astype(np.uint8)        # a quarter turn clockwise


def frame(y, u, v):
    return np.concatenate([y.astype(np.uint8).ravel(), np.stack([u, v], -1).astype(np.uint8).ravel()])


rng = np.random.default_rng(0)
W, H = 320, 240
cases = {}
bars = [(235, 128, 128), (210, 16, 146), (170, 166, 16), (145, 54, 34), (106, 202, 222), (81, 90, 240), (41, 240, 110), (16, 128, 128)]
yb = np.zeros((H, W)); ub = np.zeros((H // 2, W // 2)); vb = np.zeros((H // 2, W // 2))
for k, (yy, uu, vv) in enumerate(bars):
    yb[:, k * W // 8:(k + 1) * W // 8] = yy; ub[:, k * W // 16:(k + 1) * W // 16] = uu; vb[:, k * W // 16:(k + 1) * W // 16] = vv
cases["colour bars"] = (frame(yb, ub, vb), W, H)
cases["luma ramp"] = (frame(np.tile(np.linspace(0, 255, W), (H, 1)), np.full((H // 2, W // 2), 128), np.full((H // 2, W // 2), 128)), W, H)
cases["chroma ramps"] = (frame(np.full((H, W), 128), np.tile(np.linspace(0, 255, W // 2), (H // 2, 1)), np.tile(np.linspace(0, 255, H // 2)[:, None], (1, W // 2))), W, H)
cases["noise (seed 0)"] = (rng.integers(0, 256, W * H * 3 // 2, dtype=np.uint8), W, H)
cases["noise, 64x48"] = (rng.integers(0, 256, 64 * 48 * 3 // 2, dtype=np.uint8), 64, 48)
real = []
if HOST != "none":
    for i in range(5):
        d = urllib.request.urlopen(f"http://{HOST}:8770/sense/camera", timeout=8).read()
        if len(d) == W * H * 3 // 2:
            real.append(np.frombuffer(d, np.uint8)); cases[f"real duck, frame {len(real)}"] = (real[-1], W, H)
        time.sleep(0.4)

rows, bad = [], 0
for name, (nv, w, h) in cases.items():
    a, r = app(nv, w, h, False).astype(int), ref(nv, w, h).astype(int)
    err = np.abs(a - r)
    rows.append((name, f"{w}x{h}", int(err.max()), float(err.mean()), float((err > 0).mean() * 100)))
    bad += err.max() > 1
print(f"{'frame':24} {'size':8} {'max |err|':>9} {'mean |err|':>10} {'pixels off %':>12}")
for r in rows:
    print(f"{r[0]:24} {r[1]:8} {r[2]:9d} {r[3]:10.4f} {r[4]:12.2f}")

# the lift: grey frames of every level; output luma must rise with input, never fall below it, and be untouched when bright
grey = lambda g: frame(np.full((H, W), g), np.full((H // 2, W // 2), 128), np.full((H // 2, W // 2), 128))
levels = np.arange(0, 256, 5)
out = np.array([app(grey(g), W, H, True)[0, 0, 0] for g in levels])
lift_ok = bool(np.all(np.diff(out) >= 0) and np.all(out >= levels) and np.all(out[levels >= 90] == levels[levels >= 90]))
print(f"lift: monotonic, never darker, bright frames untouched: {lift_ok};  grey 20 -> {out[4]}, 40 -> {out[8]}, 90 -> {out[18]}")
bad += not lift_ok

mic = []
if HOST != "none":
    for i in range(40):
        j = json.load(urllib.request.urlopen(f"http://{HOST}:8770/sense/mic", timeout=8))
        if j: mic.append(j)
        time.sleep(0.1)
if mic:
    n = {len(j["wave"]) for j in mic}; peak = max(max(abs(x) for x in j["wave"]) for j in mic)
    lv = np.array([j["level"] for j in mic])
    own = np.array([20 * math.log10(max(np.std(j["wave"]), 1e-9)) for j in mic])       # the 64 samples' own RMS, mean removed
    mic_ok = n == {64} and peak <= 1 and all(j["live"] for j in mic) and abs(float(np.mean(lv - own))) < 3
    print(f"mic: {len(mic)} readings, samples per reading {n}, peak {peak:.4f} of full scale, level {lv.min():.1f}..{lv.max():.1f} dBFS "
          f"(mean {lv.mean():.1f}); level minus the samples' own RMS: {np.mean(lv - own):+.2f} dB (sd {np.std(lv - own):.2f}); consistent: {mic_ok}")
    bad += not mic_ok

fig, ax = plt.subplots(1, 4, figsize=(9, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]))
show = real[-1] if real else cases["colour bars"][0]
for a, img, t in zip(ax, (ref(show, W, H), app(show, W, H, False), app(show, W, H, True)), ("reference decoder", "app decoder", "app, dark frame lifted")):
    a.imshow(img); a.set_title(t, fontsize=8); a.axis("off")
if mic:
    for j in mic[:6]: ax[3].plot(np.array(j["wave"]) * 1000, lw=.8)
    ax[3].set_title(f"microphone, 6 readings ({lv.mean():.0f} dBFS)", fontsize=8); ax[3].set_xlabel("sample"); ax[3].set_ylabel(r"$10^{-3}$ of full scale"); ax[3].grid(alpha=.25)
else: ax[3].axis("off")
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/duck_ios_codec.pdf")
with open(ROOT / "docs/duck_ios_codec_table.tex", "w") as f:
    for r in rows: f.write(f"{r[0]} & ${r[1].replace('x', chr(92) + 'times ')}$ & {r[2]} & {r[3]:.3f} & {r[4]:.1f} \\\\\n")
    f.write("\\bottomrule\n")
sys.exit(1 if bad else 0)
