"""How fast does the real duck come fully online after each kind of interruption? Run with the duck powered:

    ~/.venvs/microduck_live/bin/python firmware/test_bringup.py [--reps 3] [--no-reboot]

"Online" is strict: the studio is linked to the duck, the camera, ToF, IMU, microphone, speaker and servo port all
report ok, all 15 servos answer, a battery reading exists, AND each sensor actually delivers data (a camera frame
that is not flat, a ToF frame, an IMU reading, a live microphone). Each case is timed from the moment of the
interruption, REPS times. Writes docs/bringup_results.json and docs/figures/bringup_times.pdf, prints a table.

Cases: a new studio.py started (it takes over from the old one); the SSH link cut; the agent killed on the duck;
Run then Setup; the servo port disconnected and connected; the duck rebooted.
"""
import argparse, json, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8770"
SENSORS = ("camera", "tof", "imu", "mic", "speaker", "servos")


def get(path, raw=False, timeout=6):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        body = r.read()
    return body if raw else json.loads(body)


def post(name, body=None):
    req = urllib.request.Request(f"{URL}/api/{name}", data=json.dumps(body or {}).encode())
    try:
        return json.load(urllib.request.urlopen(req, timeout=90))
    except urllib.error.HTTPError as e:
        return json.load(e)


def missing():
    """What is not online yet, as a list of words; empty = everything is."""
    try:
        s = get("/api/state")
    except Exception:
        return ["studio"]
    if not s["brain"]["on"]:
        return ["link"]
    sen = s["brain"]["sensors"]
    out = [k for k in SENSORS if not sen.get(k, {}).get("ok")]
    if len(s["ids"]) != 15:
        out.append(f"servos {len(s['ids'])}/15")
    if not s["battery"]:
        out.append("battery")
    if out:
        return out
    try:
        frame = get("/sense/camera", raw=True)
        if len(frame) < 1000 or len(set(frame[:20000:97])) < 8:
            out.append("camera frame")
        if not get("/sense/tof"):
            out.append("tof frame")
        if not get("/sense/imu"):
            out.append("imu reading")
        if not (get("/sense/mic") or {}).get("live"):
            out.append("mic signal")
    except Exception:
        out.append("data")
    return out


def online(limit=240):
    """Seconds until everything is online (two checks in a row), or None after `limit`."""
    t0, good = time.time(), 0
    while time.time() - t0 < limit:
        m = missing()
        good = good + 1 if not m else 0
        if good >= 2:
            return round(time.time() - t0, 1), []
        time.sleep(0.5)
    return None, m


def ssh(cmd, timeout=20):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "microduck", cmd],
                          capture_output=True, text=True, timeout=timeout)


def studio_restart():
    subprocess.Popen([sys.executable, str(HERE / "studio.py"), "--no-open"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


CASES = {
    "studio restarted": studio_restart,
    "SSH link cut": lambda: subprocess.run(["pkill", "-f", "8772:127.0.0.1"]),
    "agent killed on the duck": lambda: ssh("pkill -f 'firmware/duck_agent\\.py$'"),
    "Run, then Setup": lambda: (post("mode", dict(to="run")), time.sleep(3), post("mode", dict(to="setup")),
                                post("connect", dict(port="duck"))),
    "servo port reconnected": lambda: (post("disconnect"), post("connect", dict(port="duck"))),
    "duck rebooted": lambda: ssh("systemctl reboot"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-reboot", action="store_true")
    a = ap.parse_args()
    t, m = online(120)
    if t is None:
        sys.exit(f"not online to begin with: {m}")
    print(f"online to begin with ({t} s)")
    results = {}
    for name, cut in CASES.items():
        if name == "duck rebooted" and a.no_reboot:
            continue
        results[name] = []
        for i in range(2 if name == "duck rebooted" else a.reps):
            t0 = time.time()
            cut()
            if name in ("SSH link cut", "agent killed on the duck", "duck rebooted", "studio restarted"):
                for _ in range(60):                    # wait for the interruption to be visible, then time recovery
                    if missing():
                        break
                    time.sleep(0.25)
            t, m = online()
            total = None if t is None else round(time.time() - t0, 1)
            results[name].append(dict(seconds=total, missing=m))
            print(f"  {name:<26} run {i + 1}: {'FAILED, missing ' + ', '.join(m) if total is None else f'{total} s'}", flush=True)
            time.sleep(3)
    out = HERE.parent / "docs/bringup_results.json"
    out.write_text(json.dumps(dict(when=time.strftime("%Y-%m-%d %H:%M"), reps=a.reps, results=results), indent=1))

    print(f"\n{'case':<28}{'n':>3}{'ok':>4}{'mean s':>9}{'min':>7}{'max':>7}")
    rows = []
    for name, runs in results.items():
        ok = [r["seconds"] for r in runs if r["seconds"] is not None]
        rows.append((name, ok, len(runs)))
        print(f"{name:<28}{len(runs):>3}{len(ok):>4}" + (f"{statistics.mean(ok):>9.1f}{min(ok):>7.1f}{max(ok):>7.1f}" if ok else "        -"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for y, (name, ok, n) in enumerate(rows):
        if ok:
            ax.barh(y, statistics.mean(ok), color="#3b82f6", height=0.55)
            ax.plot(ok, [y] * len(ok), "k.", ms=5)
            ax.text(max(ok) + 1, y, f"{statistics.mean(ok):.0f} s" + ("" if len(ok) == n else f"  ({n - len(ok)} failed)"), va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time until every sensor and all 15 servos are online and delivering data (s)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE.parent / "docs/figures/bringup_times.pdf")
    print(f"\n{out}\n{HERE.parent / 'docs/figures/bringup_times.pdf'}")
