"""Video-clip camera for training runs.

Isaac Lab's video recorder uses the scene visualizer: a fixed camera near the world origin. The env grid
is centred on the origin, so with thousands of environments the robots nearest the camera are the ones
in the middle of the grid. ``max_visible_envs`` alone samples visible robots at random across the whole
grid and the clip shows an empty floor with the ducks on the horizon; this module picks the central
environments explicitly.
"""

from __future__ import annotations

import math

from isaaclab.visualizers import VisualizerCfg

CLIP_EYE = (1.4, 1.4, 0.7)
CLIP_LOOKAT = (0.0, 0.0, 0.08)


def central_env_indices(num_envs: int, count: int = 16) -> list[int]:
    """Indices of the ``count`` environments nearest the centre of Isaac Lab's row-major square env grid."""
    per_row = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / per_row)
    centre = ((rows - 1) / 2.0, (per_row - 1) / 2.0)
    indices = sorted(
        range(num_envs),
        key=lambda i: (divmod(i, per_row)[0] - centre[0]) ** 2 + (divmod(i, per_row)[1] - centre[1]) ** 2,
    )
    return indices[:count]


def clip_visualizer_cfg(num_envs: int, count: int = 16) -> VisualizerCfg:
    """A visualizer that frames the ``count`` central environments from a low camera near the origin."""
    return VisualizerCfg(eye=CLIP_EYE, lookat=CLIP_LOOKAT, visible_env_indices=central_env_indices(num_envs, count))


PLAY_EYE = (0.9, 0.9, 0.45)
PLAY_LOOKAT = (0.0, 0.0, 0.06)


def play_visualizer_cfg() -> VisualizerCfg:
    """Playback renders every environment (there are only a few): a closer camera, no debug markers."""
    return VisualizerCfg(eye=PLAY_EYE, lookat=PLAY_LOOKAT, enable_markers=False)
