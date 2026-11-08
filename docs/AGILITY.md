# Front flip (roulade) and jump: what was ported, from where, and why

Both tasks were built from existing open-source MicroDuck work, read before a line was written. Reference
clones live outside the repo (scratchpad); the designs are summarised here so the provenance survives.

## Why the "front flip" is the roulade

* **Lulzx/microduck-backflip** (mjlab, ~90 logged experiments on the stock Dynamixel XL330 duck): an aerial
  standing flip was never achieved without assistance. Unassisted the best standing rotation was 55 deg;
  with a 16 N / 1.5 N.m virtual spotter and a landing damper the best was 1 of 64. Measured plant limits:
  0.77 m/s takeoff, 0.15 s flight, 9 rad/s. The repo is explicit that it is a curriculum bridge, not a result.
* **nocrab/microduck-backflip** (Hugging Face, sim only): "pure standing backflip is out of reach"; its flip
  uses the head as a spring and is a different manoeuvre.
* **HD1910 vs XL330**: our BAM fit gives a no-load speed of ~12 rad/s against the XL330's 20.2, and the
  community's hop ceiling on the XL330 is a SPEED wall (19.6 of 20.2 rad/s at 2.6 cm). Open-loop
  crouch-and-extend on our plant never leaves the ground (`docs/jump_ceiling*.json`; a lower bound only).
* **Pollen's official `Mjlab-Roulade-Flat-MicroDuck`** is a forward roll over the flat top of the head that
  lands back on the feet, deployed on real ducks (hero video). That is the forward flip this robot performs,
  so `Microduck-Roulade-Flat-HD1910` is a term-for-term port of its run-5 recipe (`microduck_rl/mdp/roulade.py`,
  `microduck_rl/tasks/roulade_env_cfg.py`).

Roulade design points kept from upstream: one dense progress reward on the max-so-far forward rotation,
integrated only while supported and sagittal (support gate, flatness gate), paid-rate cap 5 rad/s; landing
annuity gated on >=260 deg AND a flat-top-of-head ground latch; stand tax after completion; reverse curriculum
with mid-roll tucked spawns 50-340 deg and 0-3 rad/s; motion blockers near zero during discovery, ramped by
curriculum; no fall termination. Changes: entropy 0.001 (the sit stage showed a noise-initiated behaviour is
not one the mean action performs), the replica's wider mass/CoM/friction/servo randomisation, Isaac Lab
contact sensors filtered to the ground SHAPE (`/World/ground/terrain/GroundPlane/CollisionPlane`).

## Jump

`Microduck-Jump-Flat-HD1910` ports **ThomasBurgess2000/microduck-max-height-jump v6** (one-shot launch flag in
the twist slot, paid-once frontiers for launch speed / airborne rise / air time, touchdown quality, settle
progress, stable-landing pulse, durable-landing jackpot only at the horizon, failed-recovery termination with
an explicit cost, no success termination) with the measured lessons of **chenp9527/microduck-jump**: per-episode
standing baseline (a fixed 0.115 m baseline paid 46 % of the height reward for unloading the feet), height and
air-time rewards saturating together, one-shot landing payout (a latch or a window was farmed by minimal
hops), smoothness introduced only after the skill exists, checkpoint selection by landing rate not recency.
Upstream's performance-gated curricula are iteration schedules here. The 12 mm height scale is a normaliser,
not a cap; the HD1910's real ceiling is whatever the deterministic evaluator (`scripts/eval_jump.py`) measures.

## Evaluate the mean action

Every claim about these policies comes from `scripts/eval_roulade.py` / `scripts/eval_jump.py` run on
standing starts with the mean action, never from training-rollout clips or the training log.
