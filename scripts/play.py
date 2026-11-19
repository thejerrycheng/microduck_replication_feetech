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
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

from microduck_rl.realism import coerce_override_ranges  # noqa: E402

# Hydra overrides such as ``vin_range=[6.4,6.4]`` arrive as lists; the BAM actuator hashes them. The evaluators fix
# this in each task's ``play_mode``, but ``run_play_cli`` never calls it, so coerce here, just before the env is built.
_init = ManagerBasedRLEnv.__init__


def _init_coerced(self, cfg, *args, **kwargs):
    coerce_override_ranges(cfg)
    _init(self, cfg, *args, **kwargs)


ManagerBasedRLEnv.__init__ = _init_coerced

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
