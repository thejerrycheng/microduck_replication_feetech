#!/usr/bin/env python
"""Train a Microduck HD1910 task: the Isaac Lab ``train`` entry point with this package's tasks registered.

Example::

    uv run --project external/IsaacLab python scripts/train.py --rl_library rsl_rl \
        --task Microduck-Velocity-Flat-HD1910 --num_envs 4096 presets=backlash \
        --video --video_length 500 --video_interval 6000
"""

# Warp captures ``enable_backward`` at module creation; set it before any kernel module is imported.
import warp as wp

wp.config.enable_backward = False

import microduck_rl  # noqa: E402, F401  (registers the tasks)
from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_train_cli())
