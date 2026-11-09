"""PPO + fallen-gated behaviour cloning from a frozen stand-up expert, upright frames anchored to the walk.

Port of Pollen's ``PpoWithExpertBc`` (``mjlab_microduck/tasks/distill.py``), the mechanism behind their deployed
VelStand policy (one network that walks, falls and gets up). Their measured lessons, which this port keeps:

* A warm-started walk NEVER discovers getting up on its own (0 recoveries in ~1000 iterations in two runs):
  its action std is ~0.2, flailing from lying never yields a partial rise, so the recovery rewards have no
  gradient. The skill exists in the stand-up expert; distil it.
* DAgger-flavoured: after every PPO update, regress the student's deterministic action onto the expert's on the
  FALLEN frames the student itself just visited (tilt above ``gate_tilt_deg``, read off the projected-gravity
  block of the actor observation). The expert keeps its OWN observation normaliser and gets a zeroed twist
  slot (it never saw a velocity command).
* Without an anchor the first BC pass rewrote the shared weights and killed the walk inside 12 iterations.
  The warm-start WALK checkpoint is therefore a second frozen teacher on the UPRIGHT frames (tilt below
  ``anchor_tilt_deg``). PPO alone owns the band in between and everything the reward adds on top.
* Gentle passes: lr 3e-4, a minimum mini-batch so a first pass on a handful of frames cannot take 20 steps.
* Addition here: upright frames with a ZERO velocity command are cloned from the stand-up expert as well, which
  holds a robust 0.115 m stand -- the walk crouches to 0.073 m at zero command, and two fine-tunes with height
  costs up to -50 left that untouched (`docs/squat_walk_v4_19496.json`). Frames with a command keep the walk anchor.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

__all__ = ["PpoWithExpertBc", "default_bc_cfg", "tilt_deg_from_obs"]


def default_bc_cfg() -> dict:
    return {
        "expert_checkpoint": "",  # rsl_rl checkpoint of the stand-up expert (actor_state_dict, its own normaliser)
        "anchor_checkpoint": "",  # the warm-start walk checkpoint: anchors the upright frames
        "gravity_slice": (3, 6),  # projected gravity inside the 61-D actor observation
        "twist_slice": (48, 51),  # velocity command slot (zeroed for the expert)
        "gate_tilt_deg": 35.0,  # expert teaches above this tilt
        "anchor_tilt_deg": 25.0,  # walk anchors below this tilt; PPO alone in between
        "lr": 3.0e-4,
        "min_batch": 512,
        "max_steps": 20,
        "min_frames": 64,
        "expert_weight": 1.0,
        "anchor_weight": 1.0,
        "max_grad_norm": 1.0,
        # Zero-command upright frames go to the EXPERT too (it holds a robust 0.115 m stand; the walk crouches to
        # 0.073 m at zero command and kept doing so under a -50 height cost). Frames with a velocity command stay
        # anchored to the walk.
        "expert_on_zero_twist": True,
        "zero_twist_eps": 1.0e-3,
    }


def _flat_obs(model, obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[g] for g in model.obs_groups], dim=-1)


def _det_action(model, flat: torch.Tensor) -> torch.Tensor:
    out = model.mlp(model.obs_normalizer(flat))
    return model.distribution.deterministic_output(out) if model.distribution is not None else out


def tilt_deg_from_obs(flat: torch.Tensor, gravity_slice=(3, 6)) -> torch.Tensor:
    g = flat[:, gravity_slice[0] : gravity_slice[1]]
    return torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))


class PpoWithExpertBc(PPO):
    """PPO whose update is followed by a fallen-gated BC pass toward a frozen expert and an upright anchor."""

    def __init__(self, actor, critic, storage, *args, bc_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.bc_cfg = {**default_bc_cfg(), **(bc_cfg or {})}
        self.expert = self._load_teacher(self.bc_cfg["expert_checkpoint"])
        self.anchor = self._load_teacher(self.bc_cfg["anchor_checkpoint"])
        self.bc_optimizer = torch.optim.Adam(self._raw_actor.parameters(), lr=float(self.bc_cfg["lr"]))
        print(
            f"[PpoWithExpertBc] expert={'on' if self.expert is not None else 'OFF'} "
            f"({self.bc_cfg['expert_checkpoint']}), anchor={'on' if self.anchor is not None else 'OFF'} "
            f"({self.bc_cfg['anchor_checkpoint']}), gates {self.bc_cfg['anchor_tilt_deg']}/{self.bc_cfg['gate_tilt_deg']} deg"
        )

    def _load_teacher(self, path: str):
        if not path:
            return None
        teacher = copy.deepcopy(self._raw_actor)
        state = torch.load(path, map_location=self.device, weights_only=False)
        teacher.load_state_dict(state["actor_state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def update(self) -> dict[str, float]:
        # the raw (un-normalised) actor observations of the rollout, taken before PPO clears the storage
        flat = _flat_obs(self._raw_actor, self.storage.observations.flatten(0, 1)).detach().clone()
        loss_dict = super().update()
        loss_dict.update(self._bc_update(flat))
        return loss_dict

    def _bc_update(self, flat: torch.Tensor) -> dict[str, float]:
        cfg = self.bc_cfg
        stats = {"bc_expert": 0.0, "bc_anchor": 0.0, "bc_fallen_frac": 0.0, "bc_standing_frac": 0.0}
        if self.expert is None and self.anchor is None:
            return stats
        tilt = tilt_deg_from_obs(flat, tuple(cfg["gravity_slice"]))
        fallen = tilt > float(cfg["gate_tilt_deg"])
        upright = tilt < float(cfg["anchor_tilt_deg"])
        t0, t1 = cfg["twist_slice"]
        zero_cmd = flat[:, t0:t1].abs().amax(dim=1) < float(cfg["zero_twist_eps"])
        if cfg.get("expert_on_zero_twist", True):
            expert_frames = fallen | (upright & zero_cmd)
            upright = upright & ~zero_cmd
        else:
            expert_frames = fallen
        stats["bc_fallen_frac"] = float(fallen.float().mean())
        stats["bc_standing_frac"] = float((expert_frames & ~fallen).float().mean())
        jobs = []
        if self.expert is not None and int(expert_frames.sum()) >= int(cfg["min_frames"]):
            x = flat[expert_frames]
            xe = x.clone()
            xe[:, t0:t1] = 0.0
            with torch.no_grad():
                y = _det_action(self.expert, xe)
            jobs.append(("bc_expert", x, y, float(cfg["expert_weight"])))
        if self.anchor is not None and int(upright.sum()) >= int(cfg["min_frames"]):
            x = flat[upright]
            with torch.no_grad():
                y = _det_action(self.anchor, x)
            jobs.append(("bc_anchor", x, y, float(cfg["anchor_weight"])))
        params = list(self._raw_actor.parameters())
        for name, x, y, w in jobs:
            n = x.shape[0]
            steps = int(max(1, min(int(cfg["max_steps"]), n // int(cfg["min_batch"]))))
            bs = int(min(n, max(int(cfg["min_batch"]), n // steps)))
            total = 0.0
            for _ in range(steps):
                idx = torch.randint(0, n, (bs,), device=x.device)
                pred = _det_action(self._raw_actor, x[idx])
                loss = w * torch.nn.functional.mse_loss(pred, y[idx])
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
                self.bc_optimizer.step()
                total += float(loss)
            stats[name] = total / steps
        return stats
