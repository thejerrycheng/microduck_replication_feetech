# microduck_replication_feetech

Reinforcement-learning policies, a desktop/phone driving app and the on-robot runtime for a **MicroDuck replica
on Feetech HD1910 servos** (instead of the original Dynamixel XL330s). The hardware is built from
**[JoyandAI/OpenMicroDuck](https://github.com/JoyandAI/OpenMicroDuck)** — CAD, BOM, electronics and the Radxa
system image — and the policies are trained on **Isaac Lab 3.0 + Newton (MuJoCo-Warp)** against a model of the
slower servo, then run on the robot through the same 61-observation / 14-action contract as the official duck.

**Status (Nov 2026).** Seven policies train and are measured in simulation (walk, run, sprint, sit/stand,
jump, front flip, get-up). The first hardware build — *iteration one* — stands, sits down and stands back up,
and flips on the real HD1910s; what was measured on it and how it differs from the training model is in
[`docs/SIM2REAL.md`](docs/SIM2REAL.md). Project page with videos:
[thejerrycheng.github.io/microduck.html](https://thejerrycheng.github.io/microduck.html).

## Hardware: build it from OpenMicroDuck

Everything physical comes from **[JoyandAI/OpenMicroDuck](https://github.com/JoyandAI/OpenMicroDuck)**, the
open, HD1910-based MicroDuck (25 cm, < 1 kg, 15 DOF). Use it for:

| What | Where in OpenMicroDuck | Notes |
|---|---|---|
| **CAD** — full printable duck | [`cad/openmicroduck.3mf`](https://github.com/JoyandAI/OpenMicroDuck/tree/main/cad), [`cad/openmicroduck-assembly-opt.stl`](https://github.com/JoyandAI/OpenMicroDuck/tree/main/cad), per-part files in [`cad/parts/`](https://github.com/JoyandAI/OpenMicroDuck/tree/main/cad/parts) | PLA FDM; TPU (or a silicone pad) for the feet if you can |
| **Servo model** | [`cad/HD-1910-C001-20260902.stp`](https://github.com/JoyandAI/OpenMicroDuck/tree/main/cad), [`hardware_spec/servo/`](https://github.com/JoyandAI/OpenMicroDuck/tree/main/hardware_spec) | Feetech HD-1910-C001 ×15 |
| **BOM** | [`docs/bom.md`](https://github.com/JoyandAI/OpenMicroDuck/blob/main/docs/bom.md) and the table in its README | servos, Radxa ZERO 3W (2 GB), Zero Robot HAT, IMU-to-servo board (LSM6DSV16X), IMX219 camera, 6700K ×3 + ET2216 ×11 bearings, M2 self-tappers, 2S 7.4 V 3500 mAh pack, mics + speaker |
| **Electronics / architecture** | [`docs/architecture.md`](https://github.com/JoyandAI/OpenMicroDuck/blob/main/docs/architecture.md), [`docs/main_controller.md`](https://github.com/JoyandAI/OpenMicroDuck/blob/main/docs/main_controller.md), [`docs/servo.md`](https://github.com/JoyandAI/OpenMicroDuck/blob/main/docs/servo.md), [`docs/structure.md`](https://github.com/JoyandAI/OpenMicroDuck/blob/main/docs/structure.md) | servo wiring, IDs, control loop at 50 Hz |
| **System image** (Radxa ZERO 3W) | linked from the OpenMicroDuck README ("预部署镜像") | microduck runtime preinstalled; login `joyandai`; hotspot Wi-Fi setup. The image is not in this repo (`openmicroduck_image/` is git-ignored) |
| **Reference policies** | [`policies/`](https://github.com/JoyandAI/OpenMicroDuck/tree/main/policies) (`OpenMD_*.onnx`) | same 61 → 14 contract; a useful baseline next to the policies here |

OpenMicroDuck licences: hardware drawings, PCB and BOM CERN-OHL-S; software Apache-2.0; docs CC BY-SA (see its
`LICENSE*` files). The servo ID map and bus registers used by `firmware/` follow
[kissqy/Microduck-kissqy](https://github.com/kissqy/Microduck-kissqy) (IDs 10–14 right leg, 20–24 left leg,
30–34 neck/head/mouth; Feetech STS/SCS TTL at 1 Mbit).

## The seven policies

The defaults the app and the robot load (`app/engine.py` `DEFAULTS`); ONNX files in [`exports/`](exports/),
resumable checkpoints in [`checkpoints/`](checkpoints/) (see `checkpoints/INDEX.json`).

| Skill | Export | Task id |
|---|---|---|
| walk / stand / turn | `velstand_allcoll_hd1910_iter21495.onnx` | `Microduck-VelStand-Flat-HD1910-AllColl` |
| run | `run_v3_allcoll_hd1910_iter5999.onnx` | `Microduck-RunV3-Flat-HD1910-AllColl` |
| sprint | `sprint_max_allcoll_hd1910_iter7998.onnx` | `Microduck-SprintMax-Flat-HD1910-AllColl` |
| sit ↔ stand | `sitstand_fast_hd1910_iter5499.onnx` | `Microduck-SitStandFast-Flat-HD1910` |
| jump | `jump_hd1910_iter1500.onnx` | `Microduck-Jump-Flat-HD1910` |
| front flip (roulade) | `roulade_head_hd1910_iter7000.onnx` | `Microduck-RouladeHead-Flat-HD1910` |
| get up after a fall | `standup_hd1910_iter9999.onnx` | `Microduck-StandUp-Flat-HD1910` |

Every task keeps the official observation/action layout (61 obs / 14 actions). `make list-envs` prints all
registered tasks, including the earlier walk/run/sit/stand variants. Where the flip and jump recipes came
from, and why the "flip" is a roulade: [`docs/AGILITY.md`](docs/AGILITY.md).

## Drive it

| Tool | What it does | Start | Docs |
|---|---|---|---|
| **Duck Live** (macOS) | the duck in MuJoCo on your Mac, all seven policies on the keyboard, 27 emotions | `app/setup.sh` once, then `app/ducklive.sh` | [`app/README.md`](app/README.md), `docs/duck_live_app.pdf`, `docs/duck_live_animation.pdf` |
| **Duck Servo Studio** | find, set IDs, test and live-monitor the 15 servos from a browser; hot-plug | `python firmware/studio.py` (`--mock` without hardware) | [`firmware/README.md`](firmware/README.md) |
| **On-robot runtime** | policies on the Radxa ZERO 3W at 50 Hz, the pilot that switches skills | `firmware/duck_agent.py`, `firmware/duck_pilot.py` | [`docs/SIM2REAL.md`](docs/SIM2REAL.md) |
| **Duck** (iPhone) | studio + Duck Live on a phone, or straight to the duck with no Mac | `cd ios && xcodegen generate` | [`ios/README.md`](ios/README.md), `docs/duck_ios_app.pdf` |

## What is different from the official robot, and how it is handled

| Difference | Source | Handling |
|---|---|---|
| Feetech HD1910 servos (320:1, 12-bit, 23 g) | OpenMicroDuck / kissqy servo specs | Rhoban BAM m6 fit of the HD1910 from LuwuDynamics/xgoduck_rl (`1910_m6.json`, Apache-2.0) with the **Feetech** firmware law: `duty = err · kp_fw(5) · 0.166 · error_gain_ratio`, no current limiter, 0.97 duty ceiling. Supply 7.4–8.0 V with sag, 3–6 physics-step command delay. `microduck_rl/assets/microduck_hd1910.py` |
| +5 g per servo (×15), different compute/HAT/IMU/battery placement | the OpenMicroDuck build | Not modelled: **domain-randomised**. Every body's mass ×0.90–1.25 (official: trunk ×0.95–1.05), trunk CoM up to ±25 mm, head CoM up to ±15 mm (official 15/10 mm), sole friction 0.5–1.3. `microduck_rl/dr.py`, applied to all tasks. |
| Geometry | official Pollen USD (CC BY-NC-SA meshes, shipped with Isaac Lab) | training geometry unchanged; the printed parts follow the MicroDuck skeleton and their differences fall inside the randomisation above. Measured real-vs-sim gaps: [`docs/SIM2REAL.md`](docs/SIM2REAL.md) |

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

That is the stock-runtime path. On the OpenMicroDuck/Radxa build the policies here run through
`firmware/duck_agent.py` (one sync read of the 14 joint servos and the IMU board per 20 ms tick) and
`firmware/duck_pilot.py` (which policy, which command); install with the `firmware/*_install.sh` scripts, see
[`firmware/README.md`](firmware/README.md) and [`docs/SIM2REAL.md`](docs/SIM2REAL.md). The sit/stand policies are
driven by writing 0/1 into the `vx` command slot (the upstream `SitStand` contract).

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
exports/, checkpoints/        the selected ONNX policies and their resumable checkpoints
app/                           Duck Live (macOS): MuJoCo + onnxruntime on the CPU, keyboard driving, emotions
firmware/                      servo studio, on-robot runtime (duck_agent / duck_pilot), install scripts
ios/                           the Duck iPhone app
docs/DESIGN.md                 why each decision, with the sources
docs/SIM2REAL.md               measured on the real duck vs. the training model
docs/AGILITY.md                jump + front-flip provenance
docs/*.pdf                     technical reports (Duck Live app, animation, iOS app)
external/IsaacLab              (git-ignored) Isaac Lab main
```

## References

- [Train in Simulation, Deploy on a Real Robot: Microduck with Isaac Lab 3.0](https://huggingface.co/blog/nvidia/microduck-isaac-lab-sim-to-real) — NVIDIA/HF
- [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) `source/isaaclab_tasks/isaaclab_tasks/contrib/microduck` — the velocity recipe
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) — the mjlab tasks (sit/stand source)
- [JoyandAI/OpenMicroDuck](https://github.com/JoyandAI/OpenMicroDuck) — CAD, BOM, electronics and system image this robot is built from
- [kissqy/Microduck-kissqy](https://github.com/kissqy/Microduck-kissqy) — the earlier HD1910 replica; servo map and bus registers
- [Ley-WL/microduck-rl](https://github.com/Ley-WL/microduck-rl), [LuwuDynamics/xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl) — HD1910 BAM parameters
- [Rhoban/bam](https://github.com/Rhoban/bam) — the actuator model
- [joeynyc/awesome-microduck](https://github.com/joeynyc/awesome-microduck) — index of the ecosystem

Code: Apache-2.0. The Microduck meshes in the official USDs are CC BY-NC-SA 4.0 (Pollen Robotics).
OpenMicroDuck hardware files keep their own licences (CERN-OHL-S hardware, Apache-2.0 software, CC BY-SA docs);
nothing from it is copied into this repo — it is linked.
