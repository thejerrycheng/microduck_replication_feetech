"""Does the duck's one hardware bridge bring every sensor back by itself? Run on the Mac with the duck powered:

    ~/.venvs/microduck_live/bin/python firmware/test_bridge.py [--reps 5] [--reboots 3] [--only reboot] [--host microduck]

Everything is measured through the duck's OWN server (http://<duck>:8770, the one the phone app uses), with no Mac
studio involved, so it shows what the duck does alone. "Streaming" is strict: the camera, the ToF and the
microphone each deliver data that changes between two readings 0.3 s apart. Each case is one interruption, timed
from the interruption until all three stream again. After each recovery the processes on the duck are listed, and
the run counts how often anything other than the one bridge and its readers held a device.

Cases: the bridge killed; the camera, ToF and microphone readers killed; the duck rebooted (clean reboot).
Servos and IMU are not judged here (they need servo power): test_bringup.py covers them.
Writes docs/bridge_results.json and docs/figures/bridge_recovery.pdf, prints a table.
"""
import argparse, json, statistics, subprocess, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KILL = {"bridge killed": "pkill -f 'firmware/duck_agent\\.py$'", "camera reader killed": "pkill -x v4l2-ctl",
        "ToF reader killed": "pkill -x tof_vl53l5cx; pkill -x tof_vl53l8cx", "microphone reader killed": "pkill -x arecord"}
OWNERS = "pgrep -af 'duck_agent|v4l2-ctl|arecord|tof_vl53|robotd|tofd|mediad' | grep -v pgrep"


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=4) as r:
        return r.read()


def missing(url):
    """What is not streaming, as a list; empty = camera, ToF and microphone all deliver changing data."""
    try:
        a = [get(url, "/sense/" + k) for k in ("camera", "tof")]
        time.sleep(0.3)
        b = [get(url, "/sense/" + k) for k in ("camera", "tof")]
        mic = json.loads(get(url, "/sense/mic")) or {}
    except Exception:
        return ["server"]
    out = [k for k, x, y in zip(("camera", "tof"), a, b) if len(y) < 100 or x == y]
    return out + ([] if mic.get("live") else ["mic"])


def streaming(url, limit=180):
    """Seconds until everything streams (two checks in a row), or None after `limit`."""
    t0, good, m = time.time(), 0, []
    while time.time() - t0 < limit:
        m = missing(url)
        good = good + 1 if not m else 0
        if good >= 2:
            return round(time.time() - t0, 1), []
    return None, m


def ssh(host, cmd):
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", host, cmd],
                              capture_output=True, text=True, timeout=20).stdout
    except subprocess.TimeoutExpired:
        return ""


def strays(host):
    """Processes holding a device that are not the one bridge or a reader it started."""
    lines = [l.split(None, 1) for l in ssh(host, OWNERS).splitlines() if l.strip()]
    agents = [pid for pid, cmd in lines if "duck_agent.py" in cmd]
    kids = set(ssh(host, f"pgrep -P {agents[0]}").split()) if len(agents) == 1 else set()
    return [cmd for pid, cmd in lines if pid not in kids and pid not in agents[:1]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--reboots", type=int, default=3)
    ap.add_argument("--host", default="microduck")
    ap.add_argument("--only", help="run only the cases whose name contains this, keep the others' saved results")
    a = ap.parse_args()
    url = f"http://{ssh(a.host, 'hostname').strip()}.local:8770"
    t, m = streaming(url, 60)
    if t is None:
        raise SystemExit(f"not streaming to begin with: {m}")
    print(f"{url}: streaming to begin with; strays: {strays(a.host) or 'none'}")
    cases = {**KILL, "duck rebooted": "sudo -n /usr/bin/systemctl reboot"}
    out = HERE.parent / "docs/bridge_results.json"
    results = json.loads(out.read_text())["results"] if a.only and out.exists() else {}      # --only: keep the rest
    for name, cmd in cases.items():
        if a.only and a.only not in name:
            continue
        results[name] = []
        for i in range(a.reboots if name == "duck rebooted" else a.reps):
            t0 = time.time()
            ssh(a.host, cmd)
            if name == "duck rebooted":
                while "server" not in missing(url) and time.time() - t0 < 90:      # wait until it is really down
                    pass
                if "server" not in missing(url):
                    print("  the duck did not reboot: the login may not (mode_install.sh grants it); skipped")
                    break
            t, m = streaming(url, 300 if name == "duck rebooted" else 120)
            total = None if t is None else round(time.time() - t0, 1)
            results[name].append(dict(seconds=total, missing=m, strays=strays(a.host)))
            print(f"  {name:<26} run {i + 1}: {'FAILED, missing ' + ', '.join(m) if total is None else f'{total} s'}"
                  f"  strays: {results[name][-1]['strays'] or 'none'}", flush=True)
            time.sleep(4)
    results = {k: v for k, v in results.items() if v}
    out.write_text(json.dumps(dict(when=time.strftime("%Y-%m-%d %H:%M"), url=url, results=results), indent=1))

    print(f"\n{'case':<28}{'n':>3}{'ok':>4}{'mean s':>9}{'min':>7}{'max':>7}{'strays':>8}")
    rows = []
    for name, runs in results.items():
        ok = [r["seconds"] for r in runs if r["seconds"] is not None]
        rows.append((name, ok, len(runs)))
        print(f"{name:<28}{len(runs):>3}{len(ok):>4}" + (f"{statistics.mean(ok):>9.1f}{min(ok):>7.1f}{max(ok):>7.1f}" if ok else f"{'-':>23}")
              + f"{sum(bool(r['strays']) for r in runs):>8}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for y, (name, ok, n) in enumerate(rows):
        if ok:
            ax.barh(y, statistics.mean(ok), color="#3b82f6", height=0.55)
            ax.plot(ok, [y] * len(ok), "k.", ms=5)
            ax.text(max(ok) * 1.15, y, f"{statistics.mean(ok):.1f} s" + ("" if len(ok) == n else f"  ({n - len(ok)} failed)"), va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [f"{r[0]} (n={r[2]})" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time until camera, ToF and microphone all stream again (s, log scale)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE.parent / "docs/figures/bridge_recovery.pdf")
    print(f"\n{out}\n{HERE.parent / 'docs/figures/bridge_recovery.pdf'}")
