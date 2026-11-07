# microduck_replication_feetech

RL policies for a **Microduck** built from the [kissqy/Microduck-kissqy](https://github.com/kissqy/Microduck-kissqy)
replication files with **Feetech HD1910** servos instead of Dynamixel XL330s, trained on
**Isaac Lab 3.0 + Newton (MuJoCo-Warp)** and deployed zero-shot through the stock Microduck runtime
(`robotctl policy load`).

Three policies, one observation/action contract (61 obs / 14 actions, the official layout):

| Policy | Task id | Recipe |
|---|---|---|
| walk / run (velocity tracking) | `Microduck-Velocity-Flat-HD1910` (+ `-Rough-`, `Microduck-Run-Flat-HD1910`) | official Isaac Lab `IsaacContrib-Velocity-*-MicroDuck` |
| sit (stand → sit) | `Microduck-Sit-Flat-HD1910` | `Mjlab-SitStand-Flat-MicroDuck` ported from `pollen-robotics/microduck_rl`, SIT always commanded |
| stand (sit → stand) | `Microduck-Stand-Flat-HD1910` | same port, STAND always commanded, seated starts |
| sit ↔ stand, one policy | `Microduck-SitStand-Flat-HD1910` | the upstream task as is |

Jump and forward roll come later (`Mjlab-Roulade-*` is the next port).

## What is different from the official robot, and how it is handled

| Difference | Source | Handling |
|---|---|---|
| Feetech HD1910 servos (320:1, 12-bit, 23 g) | kissqy `hardware.json`, `servo_references.json` | Rhoban BAM m6 fit of the HD1910 from LuwuDynamics/xgoduck_rl (`1910_m6.json`, Apache-2.0) with the **Feetech** firmware law: `duty = err · kp_fw(5) · 0.166 · error_gain_ratio`, no current limiter, 0.97 duty ceiling. Supply 7.4–8.0 V with sag, 3–6 physics-step command delay — the values every HD1910 training repo uses (Ley-WL, kissqy). `microduck_rl/assets/microduck_hd1910.py` |
| +5 g per servo (×15), different compute/HAT/IMU/battery placement | your build | Not modelled: **domain-randomised**. Every body's mass ×0.90–1.25 (official: trunk ×0.95–1.05), trunk CoM up to ±25 mm, head CoM up to ±15 mm (official 15/10 mm), sole friction 0.5–1.3. `microduck_rl/dr.py`, applied to all tasks. |
| Geometry | official Pollen USD (CC BY-NC-SA meshes, shipped with Isaac Lab) | unchanged; the kissqy prints are the official skeleton |

Everything else — observation noise, IMU misalignment and latency, encoder bias, joint friction and armature randomisation, pushes, backlash hinges (`presets=backlash`), curricula, PPO hyper-parameters — is the official recipe, untouched.

## Setup

```bash
git clone git@github.com:thejerrycheng/microduck_replication_feetech.git && cd microduck_replication_feetech
make setup      # clones Isaac Lab main into external/, `uv sync`, installs this package into that env
make list-envs
make smoke      # 64 envs × 3 iterations of walk/sit/stand — a few minutes, catches config errors
make measure    # standing / seated rest heights of the Newton model -> docs/rest_heights.json
```

Requirements: Linux, an NVIDIA GPU, `uv`. Isaac Sim is **not** needed (Kit-less Newton).

## Train

```bash
make train-all         # walk -> sit -> stand, one trainer at a time, in a tmux session (tmux attach -t microduck)
make train-walk        # Microduck-Velocity-Flat-HD1910, 4096 envs, presets=backlash
make train-sit
make train-stand
make train-run         # faster forward command range (experimental)
# any extra CLI args: make train-walk ARGS="--seed 1 --max_iterations 20000"
```

Every run records a rollout clip of the live policy every 250 iterations (`--video`, one clip per
checkpoint interval) under `external/IsaacLab/logs/rsl_rl/<experiment>/<run>/videos/train/`.
Checkpoints are `model_<iteration>.pt` in the run directory, every 250 iterations.

The default budget is the official one (50k iterations walking, 15k sit/stand); `--max_iterations`
overrides it. Training logs go to TensorBoard (`--logger wandb` for W&B).

## Roll out checkpoints

```bash
# every checkpoint of a run -> rollouts/<experiment>/<run>/model_<iter>.mp4 (+ index.json with checkpoint SHA)
make record TASK=Microduck-Sit-Flat-HD1910 RUN=external/IsaacLab/logs/rsl_rl/microduck_sit_flat_hd1910_backlash/<run> EVERY=4
# one checkpoint, 4 robots, also exports exported/policy.onnx next to it
make play TASK=Microduck-Velocity-Flat-HD1910 CKPT=latest
```

## Evaluate (before trusting a checkpoint)

```bash
# deterministic rollouts; a policy that only works with exploration noise is not deployable
uv run --project external/IsaacLab python scripts/eval_posture.py --task Microduck-Sit-Flat-HD1910 --run <run> --checkpoint model_N.pt presets=backlash
uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --run <run> --checkpoint model_N.pt presets=backlash   # L/R air, stance, stride CV
scripts/record_speeds.sh <run> model_N.pt Microduck-Run-Flat-HD1910                       # labelled fixed-command clips
```

## Deploy

`make play` writes `exported/policy.onnx` (61 → 14, normaliser baked in). On the robot:

```bash
scp exported/policy.onnx microduck@ROBOT:/home/microduck/walk_hd1910.onnx
ssh microduck@ROBOT sudo robotctl policy load walk /home/microduck/walk_hd1910.onnx
```

The sit/stand policies are driven by writing 0/1 into the `vx` command slot (the upstream
`SitStand` contract); the kissqy runtime fork (`microduck-hd1910/firmware`) is what talks to the
HD1910 bus.

## Layout

```
microduck_rl/
  assets/microduck_hd1910.py   HD1910 BAM fit, Feetech firmware law, robot configs
  dr.py                        replica domain randomisation (mass / CoM / friction)
  mdp/posture.py               sit/stand command, rewards, reset + hardening events (port of mjlab mdp.py)
  tasks/velocity_env_cfg.py    walk / run / rough on the official recipe
  tasks/sitstand_env_cfg.py    sit / stand / sit-stand (port of microduck_sitstand_env_cfg.py v1.5)
  tasks/agents.py              PPO runners
scripts/
  train.py, play.py            Isaac Lab entry points with these tasks registered
  record_checkpoints.py        one clip per checkpoint
  measure_rest_heights.py      verify STAND_Z / SIT_Z on the Newton model
docs/DESIGN.md                 why each decision, with the sources
external/IsaacLab              (git-ignored) Isaac Lab main
```

## References

- [Train in Simulation, Deploy on a Real Robot: Microduck with Isaac Lab 3.0](https://huggingface.co/blog/nvidia/microduck-isaac-lab-sim-to-real) — NVIDIA/HF
- [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) `source/isaaclab_tasks/isaaclab_tasks/contrib/microduck` — the velocity recipe
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) — the mjlab tasks (sit/stand source)
- [kissqy/Microduck-kissqy](https://github.com/kissqy/Microduck-kissqy) — the HD1910 replica this robot is built from
- [Ley-WL/microduck-rl](https://github.com/Ley-WL/microduck-rl), [LuwuDynamics/xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl) — HD1910 BAM parameters
- [Rhoban/bam](https://github.com/Rhoban/bam) — the actuator model
- [joeynyc/awesome-microduck](https://github.com/joeynyc/awesome-microduck) — index of the ecosystem

Code: Apache-2.0. The Microduck meshes in the official USDs are CC BY-NC-SA 4.0 (Pollen Robotics).
