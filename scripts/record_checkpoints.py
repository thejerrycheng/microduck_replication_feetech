#!/usr/bin/env python
"""Roll out every checkpoint of a run and keep one video per checkpoint.

For each ``model_<iteration>.pt`` in the run directory (optionally every N-th), runs ``scripts/play.py``
with video recording on, then moves the clip to ``rollouts/<experiment>/<run>/model_<iteration>.mp4``
and writes an index (``rollouts/<experiment>/<run>/index.json``) with the checkpoint's iteration,
SHA-256 and the clip path, so a clip can always be traced to the weights it shows.

Example::

    uv run --project external/IsaacLab python scripts/record_checkpoints.py \
        --task Microduck-Sit-Flat-HD1910 --run external/IsaacLab/logs/rsl_rl/microduck_sit_flat_hd1910/<run> \
        --every 4 --video_length 600 presets=backlash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAY = REPO / "scripts" / "play.py"
_STEP_RE = re.compile(r"model_(\d+)\.pt$")


def checkpoints(run: Path) -> list[tuple[int, Path]]:
    found = []
    for path in run.glob("model_*.pt"):
        match = _STEP_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run", required=True, type=Path, help="Run directory holding model_*.pt")
    parser.add_argument("--every", type=int, default=1, help="Record every N-th checkpoint (the last one always)")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--video_length", type=int, default=600, help="Clip length in control steps (50 Hz)")
    parser.add_argument("--out", type=Path, default=REPO / "rollouts")
    parser.add_argument("--only", type=int, nargs="*", help="Record only these iterations")
    args, extra = parser.parse_known_args()

    run = args.run.resolve()
    ckpts = checkpoints(run)
    if not ckpts:
        print(f"no model_*.pt in {run}", file=sys.stderr)
        return 1
    if args.only:
        ckpts = [c for c in ckpts if c[0] in set(args.only)]
    else:
        ckpts = [c for i, c in enumerate(ckpts) if i % args.every == 0 or i == len(ckpts) - 1]
    out_dir = args.out / run.parent.name / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    for iteration, ckpt in ckpts:
        key = f"model_{iteration}"
        clip = out_dir / f"{key}.mp4"
        if clip.exists() and index.get(key, {}).get("sha256") == sha256(ckpt):
            print(f"[skip] {clip} already recorded for this checkpoint")
            continue
        started = time.time()
        cmd = [
            sys.executable, str(PLAY), "--rl_library", "rsl_rl", "--task", args.task,
            "--checkpoint", str(ckpt), "--num_envs", str(args.num_envs),
            "--video", "--video_length", str(args.video_length), "--video_interval", str(10**9),
            *extra,
        ]
        print("[record]", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=REPO / "external" / "IsaacLab")
        if result.returncode != 0:
            print(f"[fail] play exited {result.returncode} for {ckpt}", file=sys.stderr)
            continue
        new_clips = sorted(
            (p for p in run.rglob("*.mp4") if p.stat().st_mtime >= started - 1), key=lambda p: p.stat().st_mtime
        )
        if not new_clips:
            print(f"[fail] no video produced for {ckpt}", file=sys.stderr)
            continue
        shutil.move(str(new_clips[-1]), clip)
        index[key] = {
            "iteration": iteration,
            "checkpoint": str(ckpt),
            "sha256": sha256(ckpt),
            "clip": str(clip),
            "task": args.task,
            "num_envs": args.num_envs,
            "video_length_steps": args.video_length,
            "extra_args": extra,
        }
        index_path.write_text(json.dumps(index, indent=2))
        print(f"[done] {clip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
