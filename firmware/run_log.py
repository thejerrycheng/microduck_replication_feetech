"""The duck's flight recorder: every policy run (live or dry) is written to the duck's card, tick by tick.

On the duck (`duck_agent.Drive` uses `Recorder`; numpy and the standard library only):
    /data/runs/run_0007_20261122-175300.f32     one float32 row per 20 ms control tick, appended once a second
    /data/runs/run_0007_20261122-175300.json    what the columns are, the calibration, the gains, how the run ended

On the Mac:
    ~/.venvs/microduck_live/bin/python firmware/run_log.py pull              # copy new runs into logs/duck_runs/
    ~/.venvs/microduck_live/bin/python firmware/run_log.py list              # the runs here, one line each
    ~/.venvs/microduck_live/bin/python firmware/run_log.py show [run]        # numbers of one run (default: the newest)
    ~/.venvs/microduck_live/bin/python firmware/run_log.py plot [run]        # docs/figures/run_<name>.pdf
    ~/.venvs/microduck_live/bin/python firmware/run_log.py csv [run]         # logs/duck_runs/<name>.csv
    ~/.venvs/microduck_live/bin/python firmware/run_log.py replay [run]      # the logged obs through the pack: same actions?

In Python: `run = run_log.load(path)`; `run["obs"]` is (ticks, 61), `run["load"]` (ticks, 15), `run.meta` the header.

One row holds (see LAYOUT; all in the policies' joint order, duck_pilot.JOINTS, unless said):
  the policy's side  obs (the 61 numbers given to the network, exactly), action (its 14 outputs), which policy (role)
                     and pilot state, the command, target (what was asked of the joints after the head mix and the
                     neck limit, rad) and goal (the ticks written to the servos; in a dry run, what would have been)
  the joints         pos (ticks) and vel (counts) as read for the policy; then, read after the goals are written, per
                     servo and with the beak as the 15th: load (-1000..1000, the servo's torque measure, 0.1 % of
                     stall), current (counts), temperature (deg C), supply voltage (V) and the servo's status byte
  the IMU            the calibrated body quaternion, the chip's raw quaternion, the gyro in the sensor's and in the
                     body's axes (deg/s)
  the battery        `volt` is the supply at each servo; `battery()` turns its median into percent of charge
  the loop           time, how long the tick took, late / missed flags, the driver's stick and keys, a playing clip's
                     weight, the computer's temperature and load

A missed reading keeps the previous values in obs (as the policy saw them) and sets `missed`; servo extras not read
that tick are NaN. The recorder never stops a run: if the card fills or fails it stops recording and says so.
The oldest runs are deleted to keep the folder under CAP bytes.
"""
import json, os, re, sys, threading, time
from pathlib import Path

import numpy as np

CAP = 500e6                         # bytes of runs kept on the duck (about 3 h of driving); the oldest go first
STATES = ["stand", "walk", "run", "sprint", "stopping", "sit_down", "seated", "stand_up", "jump", "flip", "kick", "recover"]
ROLES = ["walk", "run", "sprint", "sitstand", "jump", "roulade", "recovery", "kick"]
KEYS = ["sit", "jump", "flip", "kick", "faster", "slower"]            # `keys` is a bit mask, bit i = KEYS[i]
LAYOUT = [("t", 1),                 # s since the run began (monotonic clock), at the start of the tick
          ("ms", 1),                # how long the tick took: read, policy, write (not the extras read after it)
          ("late", 1), ("missed", 1),   # 1 if the tick before overran its 20 ms / if this tick's reading failed
          ("state", 1), ("role", 1),    # indices into STATES and ROLES (-1: unknown)
          ("level", 1), ("tilt", 1), ("falls", 1),      # speed level, trunk tilt (deg), falls counted so far
          ("blend", 1),             # 0..1: how far the start-up blend from the first pose has come (1 = the policy alone)
          ("stick", 4), ("keys", 1),    # the driver: x, y, w, run; keys pressed this tick
          ("cmd", 3),               # the pilot's slewed walk command vx, vy, yaw
          ("clip", 1),              # weight of a playing clip on the head's targets, 0 if none
          ("quat", 4), ("quat_raw", 4), ("gyro", 3), ("rate", 3),
          ("obs", 61), ("action", 14), ("target", 14), ("goal", 14),
          ("pos", 14), ("vel", 14),
          ("load", 15), ("current", 15), ("temp", 15), ("volt", 15), ("status", 15),
          ("cpu_temp", 1), ("cpu_load", 1)]
WIDTH = sum(n for _, n in LAYOUT)
OBS = [("gyro", 3), ("gravity", 3), ("joint_pos", 14), ("joint_vel", 14), ("last_action", 14), ("command", 13)]


def slices(layout=LAYOUT):
    out, i = {}, 0
    for name, n in layout:
        out[name], i = slice(i, i + n), i + n
    return out


def prune(root, cap=CAP):
    """Delete the oldest runs until the folder holds less than `cap` bytes."""
    runs = {}
    for f in os.listdir(root):
        m = re.match(r"(run_\d+_[\d-]+)\.(f32|json)$", f)
        if m:
            runs.setdefault(m.group(1), []).append(os.path.join(root, f))
    total = sum(os.path.getsize(p) for ps in runs.values() for p in ps)
    for name in sorted(runs):
        if total < cap:
            break
        for p in runs[name]:
            total -= os.path.getsize(p)
            os.remove(p)


class Recorder:
    """Rows are filled by the control loop (`put`, then `commit`) and written by a thread of its own, so the card
    never holds up a tick."""

    def __init__(self, root, meta, cap=CAP):
        os.makedirs(root, exist_ok=True)
        prune(root, cap)
        last = [int(m.group(1)) for m in (re.match(r"run_(\d+)_", f) for f in os.listdir(root)) if m]
        self.name = f"run_{max(last, default=0) + 1:04d}_{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
        self.path = os.path.join(root, self.name)
        self.meta = dict(meta, version=1, name=self.name, started=time.time(), layout=LAYOUT, obs=OBS, states=STATES,
                         roles=ROLES, keys=KEYS, dtype="float32", width=WIDTH)
        self._meta()
        self.at, self.row, self.rows, self.error = slices(), np.full(WIDTH, np.nan, np.float32), [], None
        self.file, self.done = open(self.path + ".f32", "ab"), threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _meta(self):
        with open(self.path + ".json.part", "w") as f:
            json.dump(self.meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self.path + ".json.part", self.path + ".json")

    def put(self, name, value):
        self.row[self.at[name]] = value

    def commit(self):
        if self.error is None:
            self.rows.append(self.row.tobytes())
        self.row[:] = np.nan

    def _flush(self):
        rows, self.rows = self.rows, []
        if rows and self.error is None:
            try:
                self.file.write(b"".join(rows))
                self.file.flush()
            except OSError as e:
                self.error = str(e)

    def _write(self):
        while not self.done.wait(1.0):
            self._flush()

    def close(self, **end):
        """End the run: the last rows go to the card and the header gets how it ended."""
        self.done.set()
        self.thread.join(3)
        self._flush()
        try:
            os.fsync(self.file.fileno())
            self.file.close()
            self.meta.update(end, ended=time.time(), log_error=self.error)
            self._meta()
        except OSError as e:
            self.error = str(e)


# ---- on the Mac -------------------------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "logs/duck_runs"
REMOTE = "/data/runs"


class Run(dict):
    """{column: array (ticks, width) or (ticks,)} with `.meta` (the header) and `.name`."""


def load(path):
    path = Path(str(path)[:-4] if str(path).endswith((".f32", "json")) else str(path))
    path = Path(str(path).rstrip("."))
    meta = json.loads(Path(str(path) + ".json").read_text())
    raw = np.fromfile(str(path) + ".f32", dtype=np.float32)
    rows = raw[:len(raw) // meta["width"] * meta["width"]].reshape(-1, meta["width"])      # a cut-off last row is dropped
    run = Run({k: rows[:, s].squeeze(1) if s.stop - s.start == 1 else rows[:, s] for k, s in slices(meta["layout"]).items()})
    run.meta, run.name = meta, meta["name"]
    return run


def battery(run):
    """Percent of charge per tick from the median servo supply voltage (feetech.battery's curve; low under load)."""
    import feetech
    v = np.nanmedian(run["volt"], axis=1)
    return np.array([np.nan if np.isnan(x) else feetech.battery(float(x)) for x in v]), v


def pick(name=None):
    runs = sorted(p.with_suffix("") for p in LOCAL.glob("run_*.json"))
    if not runs:
        sys.exit(f"no runs in {LOCAL}: run `run_log.py pull` first")
    if name is None:
        return runs[-1]
    hit = [p for p in runs if name in p.name]
    if not hit:
        sys.exit(f"no run matching {name!r}")
    return hit[-1]


def pull(host="microduck"):
    import subprocess
    LOCAL.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host]
    out = subprocess.run([*ssh, f"cd {REMOTE} 2>/dev/null && stat -c '%s %n' run_* 2>/dev/null"], capture_output=True, text=True, timeout=30)
    if out.returncode == 255:
        sys.exit(f"cannot reach {host} over SSH")
    want = [(int(s), n) for s, n in (line.split() for line in out.stdout.splitlines()) if not n.endswith(".part")]
    todo = [n for s, n in want if not (LOCAL / n).exists() or (LOCAL / n).stat().st_size != s or n.endswith(".json")]
    if todo:
        tar = subprocess.run([*ssh, f"tar cf - -C {REMOTE} " + " ".join(todo)], capture_output=True, timeout=600)
        subprocess.run(["tar", "xf", "-", "-C", str(LOCAL)], input=tar.stdout, check=True)
    print(f"{len(want) // 2} runs on {host}, {len([n for n in todo if n.endswith('.f32')])} copied to {LOCAL}")


def summary(run):
    m, n = run.meta, len(run["t"])
    pct, v = battery(run)
    err = np.degrees(run["target"] - run["obs"][:, 6:20] - np.array(m["default_pose"]))        # asked - reached, deg
    roles = [m["roles"][int(r)] for r in run["role"] if r >= 0]
    out = dict(name=run.name, live=m["live"], ticks=n, seconds=round(float(run["t"][-1]), 1) if n else 0.0,
               ended=m.get("error") or ("stopped" if "ended" in m else "cut off (no end record: power lost?)"))
    if not n:
        return out
    dt = np.diff(run["t"])
    out.update(
        hz=round(float(1 / dt.mean()), 1) if n > 1 else None, tick_ms_mean=round(float(np.nanmean(run["ms"])), 2),
        tick_ms_max=round(float(np.nanmax(run["ms"])), 1), late_pct=round(100 * float(np.nanmean(run["late"])), 1),
        missed=int(np.nansum(run["missed"])), falls=int(np.nanmax(run["falls"])),
        role_seconds={r: round(roles.count(r) * 0.02, 1) for r in dict.fromkeys(roles)},
        tilt_deg=dict(mean=round(float(np.nanmean(run["tilt"])), 1), max=round(float(np.nanmax(run["tilt"])), 1)),
        tracking_err_deg={j: dict(rms=round(float(np.sqrt(np.nanmean(err[:, i] ** 2))), 2), max=round(float(np.nanmax(np.abs(err[:, i]))), 1))
                          for i, j in enumerate(m["joints"])},
        load_max_pct={j: round(float(np.nanmax(np.abs(run["load"][:, i]))) / 10, 1) for i, j in enumerate(m["servos"])}
        if np.isfinite(run["load"]).any() else None,
        temp_max_c=None if not np.isfinite(run["temp"]).any() else float(np.nanmax(run["temp"])),
        volts=None if not np.isfinite(v).any() else dict(first=round(float(v[np.isfinite(v)][:50].mean()), 2),
                                                         last=round(float(v[np.isfinite(v)][-50:].mean()), 2), min=round(float(np.nanmin(run["volt"])), 1)),
        battery_pct=None if not np.isfinite(pct).any() else dict(first=float(np.nanmedian(pct[np.isfinite(pct)][:50])),
                                                                 last=float(np.nanmedian(pct[np.isfinite(pct)][-50:]))),
        extras_read_pct=round(100 * float(np.isfinite(run["load"][:, 0]).mean()), 1),
        cpu_temp_max_c=round(float(np.nanmax(run["cpu_temp"])), 1) if np.isfinite(run["cpu_temp"]).any() else None)
    return out


def replay(run, pack=None):
    """Feed every logged observation to the pack's network of the logged role: the largest difference from the logged
    action (rad). 0 means the log holds exactly what the policy saw and did."""
    import policy_net
    nets = policy_net.load(str(pack or HERE / "policies.npz"))
    worst = {}
    for obs, act, r in zip(run["obs"], run["action"], run["role"]):
        role = run.meta["roles"][int(r)]
        worst[role] = max(worst.get(role, 0.0), float(np.abs(nets[role](obs) - act).max()))
    return worst


def plot(run, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m, t = run.meta, run["t"]
    J, S = m["joints"], m["servos"]
    legs = [i for i, j in enumerate(J) if "hip" in j or "knee" in j or "ankle" in j]
    pct, v = battery(run)
    err = np.degrees(run["target"] - run["obs"][:, 6:20] - np.array(m["default_pose"]))
    fig, ax = plt.subplots(8, 1, figsize=(9, 17), sharex=True)
    ax[0].plot(t, run["tilt"], "k", lw=0.8, label="tilt")
    ax[0].set_ylabel("trunk tilt [deg]")
    edges = [0, *(np.flatnonzero(np.diff(run["role"])) + 1), len(t)]
    for a, b in zip(edges, edges[1:]):
        r = int(run["role"][a])
        ax[0].axvspan(t[a], t[b - 1], color=f"C{r}", alpha=0.18, lw=0)
    for r in sorted(set(int(x) for x in run["role"])):
        ax[0].plot([], [], color=f"C{r}", lw=6, alpha=0.4, label=m["roles"][r])
    ax[0].legend(ncol=6, fontsize=7, loc="upper right")
    ax[1].plot(t, run["rate"], lw=0.6)
    ax[1].legend(["front", "left", "up"], ncol=3, fontsize=7)
    ax[1].set_ylabel("body turn rate [deg/s]")
    for i in legs:
        ax[2].plot(t, err[:, i], lw=0.6, label=J[i])
        ax[3].plot(t, np.degrees(run["action"][:, i]), lw=0.6)
    ax[2].set_ylabel("asked - reached [deg]")
    ax[2].legend(ncol=5, fontsize=6)
    ax[3].set_ylabel("policy action, legs [deg]")
    for i in [S.index(J[k]) for k in legs]:
        ax[4].plot(t, run["load"][:, i] / 10, lw=0.5)
        ax[5].plot(t, run["current"][:, i], lw=0.5)
    ax[4].set_ylabel("servo load, legs [% of stall]")
    ax[5].set_ylabel("servo current, legs [counts]")
    ax[6].plot(t, np.nanmax(run["temp"], axis=1), "C3", lw=0.8, label="hottest servo")
    ax[6].plot(t, run["cpu_temp"], "C1", lw=0.8, label="computer")
    ax[6].set_ylabel("temperature [deg C]")
    ax[6].legend(fontsize=7, loc="upper left")
    b = ax[6].twinx()
    b.plot(t, v, "C0", lw=0.6)
    b.set_ylabel("supply at the servos [V]", color="C0")
    ax[7].plot(t, run["ms"], "k", lw=0.4)
    ax[7].axhline(20, color="C3", lw=0.6)
    ax[7].set_ylabel("tick [ms]")
    ax[7].set_xlabel("time [s]")
    ax[0].set_title(f"{run.name}: {'live' if m['live'] else 'dry run'}, {len(t)} ticks, battery "
                    f"{np.nanmedian(pct[:50]):.0f} -> {np.nanmedian(pct[-50:]):.0f} %" if np.isfinite(pct).any() else run.name, fontsize=9)
    fig.tight_layout()
    out = Path(out or HERE.parent / f"docs/figures/{run.name}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    what, arg = (sys.argv[1] if len(sys.argv) > 1 else "list"), (sys.argv[2] if len(sys.argv) > 2 else None)
    if what == "pull":
        pull(arg or "microduck")
    elif what == "list":
        for p in sorted(LOCAL.glob("run_*.json")):
            s = summary(load(p.with_suffix("")))
            print(f"{s['name']}  {'live' if s['live'] else 'dry '}  {s['seconds']:7.1f} s  falls {s.get('falls', 0):2d}  {s['ended']}")
    elif what == "show":
        print(json.dumps(summary(load(pick(arg))), indent=1))
    elif what == "plot":
        print(plot(load(pick(arg))))
    elif what == "replay":
        for role, d in replay(load(pick(arg))).items():
            print(f"{role:<10} worst |logged action - pack(logged obs)| = {d:.2e} rad")
    elif what == "csv":
        run = load(pick(arg))
        names = [n if w == 1 else f"{n}_{i}" for n, w in run.meta["layout"] for i in range(w)]
        out = LOCAL / f"{run.name}.csv"
        np.savetxt(out, np.column_stack([run[n] for n, _ in run.meta["layout"]]), delimiter=",", header=",".join(names), comments="", fmt="%.6g")
        print(out)
    else:
        sys.exit(__doc__)
