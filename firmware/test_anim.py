"""Does the real head follow the duck's onboard animation clips? Run on the Mac with the studio up and the duck in Setup:

    ~/.venvs/microduck_live/bin/python firmware/test_anim.py [clip ...]        # default: yes hello curious

For each clip: ask the duck to play it (studio API), read servos 30-34 about 20 times a second straight from the
bridge while it plays, and compare each joint's measured angle with the clip's commanded angle. The command's clock is
the duck's, so the comparison is made at the delay that fits best, and that delay is reported: time from sending the request to
the head being where the clip says, so the request's own latency plus the servo's lag. Reported per clip: delay, RMS and worst tracking error over the four head
joints and the beak [deg], the largest commanded swing [deg], and how far from the rest pose the head ends [deg].
Writes docs/anim_results.json and docs/figures/anim_tracking.pdf (commanded and measured, every joint, every clip).
"""
import json, math, sys, time, urllib.request, zipfile, io
from pathlib import Path

import numpy as np

import anim_pack
import duck_agent

HERE = Path(__file__).resolve().parent
URL, BUS = "http://127.0.0.1:8770", f"http://127.0.0.1:8772"
P = duck_agent.Player
NAMES = ("neck pitch", "head pitch", "head yaw", "head roll", "beak")


def post(url, body=b"{}"):
    return urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=10).read()


def present():
    """The five positions in ticks, from one sync read (one round trip to the duck)."""
    body = [0xFE, len(P.IDS) + 4, 0x82, 56, 2, *P.IDS]
    r = post(f"{BUS}/bus/{8 * len(P.IDS)}/30", bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]))
    got = {r[i + 2]: r[i + 5] | r[i + 6] << 8 for i in range(0, len(r) - 7, 8) if r[i:i + 2] == b"\xff\xff"}
    return [got.get(sid) for sid in P.IDS]


if __name__ == "__main__":
    clips = {c["name"]: c for c in anim_pack.pack()[1]}
    cal = {j["id"]: j for j in json.loads(urllib.request.urlopen(BUS + "/calibration", timeout=5).read())["joints"]}
    deg = lambda ticks: [None if t is None else (t - cal[i]["zero_raw"]) * 360 / 4096 * cal[i]["direction"] for i, t in zip(P.IDS, ticks)]
    results, traces = {}, {}
    for name in sys.argv[1:] or ("yes", "hello", "curious"):
        c = clips[name]
        k = np.array(c["k"], dtype=float)
        head = np.clip(k[:, 1:5] / 1000, P.LO, P.HI)
        cmd = np.degrees(np.column_stack([head, k[:, 5:6] / 100 * P.BEAK]) + np.array(P.DEFAULT))      # commanded joint angles
        t0, rows = time.time(), []
        post(URL + "/api/anim_play", json.dumps(dict(name=name)).encode())
        asked = time.time() - t0                         # how long the request itself took; the clip starts within it
        while time.time() - t0 < c["T"] + 1.4:
            t = time.time() - t0
            d = deg(present())
            if None not in d:
                rows.append([t, *d])
        m = np.array(rows)
        want = lambda lag: np.column_stack([np.interp(m[:, 0] - 0.5 - lag, k[:, 0] / 1000, cmd[:, j]) for j in range(5)])
        play = (m[:, 0] > 0.5) & (m[:, 0] < 0.5 + c["T"])
        lags = np.arange(-0.1, 0.61, 0.01)
        err = [np.sqrt(np.mean((m[play, 1:] - want(l)[play]) ** 2)) for l in lags]
        lag = float(lags[int(np.argmin(err))])
        e = m[play, 1:] - want(lag)[play]
        rest = np.abs(m[-1, 1:] - np.degrees(P.DEFAULT))
        results[name] = dict(duration_s=c["T"], samples=int(play.sum()), rate_hz=round(float(play.sum() / c["T"]), 1), delay_s=round(lag, 2), request_s=round(asked, 2),
                             rms_deg=round(float(np.sqrt(np.mean(e ** 2))), 2), worst_deg=round(float(np.abs(e).max()), 2),
                             swing_deg=round(float((cmd.max(0) - cmd.min(0)).max()), 1), end_off_rest_deg=round(float(rest.max()), 2),
                             rms_by_joint={n: round(float(np.sqrt(np.mean(e[:, j] ** 2))), 2) for j, n in enumerate(NAMES)})
        traces[name] = (m, want(lag), c["T"])
        print(name, results[name], flush=True)
        time.sleep(1.0)
    post(URL + "/api/anim_stop")
    out = HERE.parent / "docs/anim_results.json"
    out.write_text(json.dumps(dict(when=time.strftime("%Y-%m-%d %H:%M"), results=results), indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, len(traces), figsize=(3.2 * len(traces), 7), sharex="col", squeeze=False)
    for col, (name, (m, w, T)) in enumerate(traces.items()):
        for j in range(5):
            ax = axes[j][col]
            ax.plot(m[:, 0], w[:, j], color="#9aa0a6", lw=1.6, label="commanded")
            ax.plot(m[:, 0], m[:, j + 1], color="#1a73e8", lw=1.0, label="measured")
            ax.axvspan(0.5, 0.5 + T, color="#000", alpha=0.04, lw=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(NAMES[j] + " (deg)", fontsize=8)
        axes[0][col].set_title(f"{name}: RMS {results[name]['rms_deg']} deg, delay {results[name]['delay_s'] * 1000:.0f} ms", fontsize=8)
        axes[4][col].set_xlabel("time since the request (s)", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE.parent / "docs/figures/anim_tracking.pdf")
    print(out, HERE.parent / "docs/figures/anim_tracking.pdf", sep="\n")
