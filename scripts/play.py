#!/usr/bin/env python
"""Play / record / export a Microduck HD1910 checkpoint (Isaac Lab ``play`` with this package's tasks).

Example::

    uv run --project external/IsaacLab python scripts/play.py --rl_library rsl_rl \
        --task Microduck-Sit-Flat-HD1910 --checkpoint latest --num_envs 4 presets=backlash \
        --video --video_length 600

Playback writes ``exported/policy.pt`` and ``exported/policy.onnx`` next to the checkpoint.
"""

import warp as wp

wp.config.enable_backward = False

import microduck_rl  # noqa: E402, F401
from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
