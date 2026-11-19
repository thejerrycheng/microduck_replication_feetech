"""Pack Duck Live's emotion clips (app/animations) into the small file the duck keeps and plays by itself.

    ~/.venvs/microduck_live/bin/python firmware/anim_pack.py [out.zip]      # prints what went in; needs numpy

The Mac studio builds this pack when it connects and sends it to the duck if the duck's differs (`studio.py`,
`duck_agent.py`); the duck stores it on its always-writable partition (/data/animations) and plays from there, so
the animations work with no Mac around and survive power cuts and the read-only lock.

What is kept, and how small:
  motion   the head's four joints and the beak, as the clip's own keyframes (linear in between): time in ms, the
           four head offsets from the default pose in mrad (Pollen's body bow folded into head pitch, as Duck Live
           does), the beak 0..100. One JSON for all clips.
  sound    8-bit mu-law at 24 kHz (ITU-T G.711's curve; from 16-bit 48 kHz), one file per sound. The duck's
           speaker is a 20 mm driver: nothing above 12 kHz is lost that it could play.
  dropped  what needs a policy on the duck: the walk commands (wiggle, side-step), the jump / flip / sit skills
           and play_dead's leg poses. Those clips play their head, beak and sound; `needs` lists what is missing.
"""
import hashlib, io, json, sys, wave, zipfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "app/animations"
RATE = 24000
BODY_PITCH_TO_HEAD = 0.6            # as app/animation.py


def mulaw(pcm16):
    """16-bit samples -> 8-bit mu-law bytes (G.711: bias 0x84, clip 32635, the bits inverted)."""
    x = pcm16.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    mag = np.minimum(np.abs(x), 32635) + 0x84
    exp = np.floor(np.log2(mag)).astype(np.int32) - 7
    man = (mag >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | man) & 0xFF).astype(np.uint8).tobytes()


def sound(path):
    with wave.open(str(path)) as f:
        assert f.getsampwidth() == 2 and f.getnchannels() == 1 and f.getframerate() == 2 * RATE, path
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float64)
    taps = np.sinc((np.arange(63) - 31) / 2) * np.hamming(63) / 2       # half-band low-pass, then every other sample
    return mulaw(np.clip(np.round(np.convolve(x, taps, "same")[::2]), -32768, 32767))


def pack():
    """The pack as zip bytes, the same bytes for the same clips (fixed order and dates)."""
    man, clips, files = json.loads((SRC / "manifest.json").read_text()), [], {}
    for c in man["clips"]:
        keys = json.loads((SRC / c["keyframes"]).read_text())["keyframes"]
        g = lambda k, n: float(k.get(n) or 0.0)
        rows = [[round(k["t"] * 1000), round(1000 * g(k, "neck")),
                 round(1000 * (g(k, "head_pitch") + BODY_PITCH_TO_HEAD * g(k, "body_pitch"))), round(1000 * g(k, "head_yaw")),
                 round(1000 * g(k, "head_roll")), round(100 * min(1.0, max(0.0, g(k, "mouth"))))] for k in keys]
        rows = [r for i, r in enumerate(rows)                       # a keyframe on the line between its neighbours says nothing
                if i in (0, len(rows) - 1) or any(abs((rows[i + 1][j] - rows[i - 1][j]) * (r[0] - rows[i - 1][0])
                                                      - (r[j] - rows[i - 1][j]) * (rows[i + 1][0] - rows[i - 1][0])) > 1e-9
                                                  for j in range(1, 6)) or r[0] - rows[i - 1][0] > 400]
        needs = sorted({k["skill"] for k in keys if k.get("skill")}
                       | ({"walk"} if any(any(k.get("twist") or [0]) for k in keys) else set())
                       | ({"legs"} if c.get("pose_joints") else set()))
        snd = []
        for s in c.get("sounds", []):
            name = Path(s).stem + ".ul"
            files[name] = sound(SRC / s)
            snd.append(name)
        clips.append(dict(name=c["name"], label=c.get("label") or c["name"].replace("_", " ").capitalize(), emoji=c.get("emoji", ""),
                          group=c.get("group", ""), T=c["duration_s"], needs=needs, snd=snd, snd_at=c.get("sound_at_s", 0.0), k=rows))
    files["clips.json"] = json.dumps(dict(v=1, rate=RATE, clips=clips), separators=(",", ":")).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            z.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), files[name], zipfile.ZIP_DEFLATED, 9)
    return buf.getvalue(), clips, files


if __name__ == "__main__":
    data, clips, files = pack()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_bytes(data)
    src = sum(f.stat().st_size for f in SRC.rglob("*") if f.is_file())
    motion, audio = len(files["clips.json"]), sum(len(v) for k, v in files.items() if k.endswith(".ul"))
    print(f"{len(clips)} clips, {sum(len(c['k']) for c in clips)} keyframes, {sum(c['T'] for c in clips):.1f} s")
    print(f"source {src / 1e6:.2f} MB -> motion {motion / 1e3:.1f} kB + sound {audio / 1e6:.2f} MB on the duck; pack to send {len(data) / 1e6:.2f} MB")
    print("sha256", hashlib.sha256(data).hexdigest()[:16])
