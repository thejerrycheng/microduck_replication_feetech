# Duck Live (Mac app)

The HD1910 Microduck driven from the keyboard, as a macOS app. **The physics and the policies run on this Mac**
(C MuJoCo + onnxruntime on the CPU, ~0.6 ms per 50 Hz control tick); the 5080 desktop is only where the policies
are trained. Same conventions as WUJI Live (`~/Desktop/WUJI_Tools/rl/tools/mac_app`) and M2 Live. Design,
equations and measured numbers: `docs/duck_live_app.pdf`.

```bash
app/setup.sh            # once: ~/.venvs/microduck_live (MuJoCo, onnxruntime, onnx, Rhoban BAM, websockets)
app/ducklive.sh         # open the app (builds /Applications/Duck Live.app on first use)
app/ducklive.sh --browser | --serve | --install-app | --sync
app/sync_policies.sh --local   # git pull + repack exports whose weights sit in a policy.onnx.data sidecar
app/bench_mac.py        # the doc's experiments -> docs/figures/duck_app_*.pdf, docs/bench_results.json
app/bench_anim.py       # the animation doc's experiments -> docs/figures/duck_anim_*.pdf, docs/bench_anim_results.json
app/tools/make_animations.py   # rebuild app/animations/ (Pollen's emotions + the Duck Live originals)
```

## Controls (keep them obvious)

| key | does |
|---|---|
| **▲ ▼** | walk forward / backward (VelStand walk policy) |
| **◀ ▶** | turn in place, or turn while walking (hold with ▲) |
| **⌘ + ▲** | run (run policy, straight only; release ⌘ to turn). At speed level 5 it is the **sprint** policy. Releasing the arrow = stop |
| **⌥ + ◀ ▶** | sidestep |
| **A / D** | speed level down / up — one knob for the walking and the running top speed, live while moving; level 5 = sprint |
| **S** | sit; S again stands up (fast sit/stand policy, ~0.3 s). Pressed while moving: stop first, then sit. ▲ while seated stands up, then walks |
| **space** | jump (jump policy: launch flag for 0.75 s, ~3 cm hop, lands, hands back to the walk) |
| **R** | front flip (roulade policy, the straight-head `roulade_head` export: a roll over the head, lands on the feet with the head back home, hands back to the walk) |
| **1 – 0**, **Q W E T Y U I O P**, **F G H J K L Z X** | emotions (see below; the key is printed on each tile of the left panel). **M** mutes the voice |
| **⏎** | reset to standing. A fall is **never undone by a reset**: under the VelStand walk the duck gets up by itself; under run / sprint / sit / jump / flip the stand-up policy takes over at zero twist and hands back once upright |
| **⌘ + drag the duck** | pull it with a spring, MuJoCo-viewer style (force = mass · 120 /s² · offset, ≤ 12 N) |
| mouse | orbit / zoom (the camera follows the duck; switch off in ⚙︎) |

Multiple keys combine (▲+◀ = walk a curve). Everything is **held-state**: the duck does exactly what is held
right now, and releasing everything always ends in the stand (= the walk policy at zero twist; VelStand balances
there at 0.116 m without the squat of the earlier walk exports).

Speed levels (measured on this plant, `docs/duck_live_app.pdf`): walk 0.20 / 0.30 / 0.40 / 0.50 / 0.60 m/s,
run 0.30 / 0.50 / 0.70 / 0.90 m/s then sprint (~1.2 m/s, the policy's command is fixed), turn 0.8–1.0 rad/s.

## Emotions, idle life, reactions

The duck is a character, not only a plant. The left panel plays **27 emotion clips**; the legs stay on the RL policy
the whole time (design, equations and measurements: `docs/duck_live_animation.pdf`).

- **Pollen Robotics' 14 open-source emotions** ([pollen-robotics/microduck_emotions](https://github.com/pollen-robotics/microduck_emotions),
  Apache-2.0, copied unchanged with their sounds): yes, no, mmh?, curious, laugh, excited, angry, sad, mock, defiant,
  impatient, yep, devastated (sits down), play dead (policies off, lies on its back; any key and the stand-up policy
  gets it up).
- **13 Duck Live originals**, voiced with Pollen's own voice synth (same duck) and lip-synced: hello (turns round to
  face you first), happy (wiggles on the walk policy, then the jump policy), dance (side-steps to a tune), sing, love,
  proud, show off (front-flip policy, then the fanfare), peck, yawn, sneeze, startled (steps back), shake off, nap
  (sit-stand policy).
- **alive** (chip, on by default): glances, re-fixations, breathing, looks at you (the page reports the camera), leads
  turns with its head while walking, looks at the point you pull it by, dozes off when left seated.
- **reacts** (chip, on by default): shakes off a fall once it is back up, "yep" after a landed jump, proud after a
  landed flip, "mmh?" when you let go after a pull, yawns when bored, says hello when the app opens.

An arrow key always interrupts. A clip asked for while walking waits for the stop; one that needs legs stands a seated
duck up first. The head is driven on the head servos while VelStand balances (it does not track its head command on
this checkpoint) and through the sit-stand policy's head command when seated; the beak is drawn (the MJCF has no jaw
joint). `server.py --still --no-react` starts the bare policies.

## Files

| | |
|---|---|
| `engine.py` | the duck: MuJoCo plant (Pollen `scene_backlash.xml`, 0.005 s), BAM HD1910 servos (Feetech law, sag, 3–6-step delay), the seven ONNX roles hot-swapped at 50 Hz, the key → command state machine (incl. the jump's launch flag, the flip's hand-back, fall handling), the ⌘-drag spring |
| `server.py` | 127.0.0.1 HTTP (page, scene, meshes) + WebSocket (held keys in, 50 Hz snapshots out); real-time pacing thread |
| `web/` | the page (three.js, meshes from the compiled MuJoCo model with crease-smoothed normals, HUD, key legend) |
| `macos/DuckLive.swift` | the window (WKWebView) — reads keys natively so ⌘+arrow key-ups are never lost, starts/stops the server |
| `animation.py` | the animation layer: clips, idle life, reactions; outputs head offsets, beak, a clip twist and skill requests that `engine.Duck.step` applies |
| `animations/` | the clip library (`manifest.json`, `keyframes/`, `sounds/`); generated, see `animations/README.md` |
| `tools/make_animations.py`, `tools/quack.py` | builds the library; Pollen's voice synth (Apache-2.0) |
| `bench_anim.py` | head tracking through the policy vs the servos, every clip over jittered starts, idle life, driving and pushes with it on, reactions, cost |
| `bench_mac.py` | step time, speed / turn tracking, sit–stand and stop transitions, jump, flip, pushes + get-up, the keyboard scenario |
| `sync_policies.sh` | `git pull`, repack sidecar exports, newer ONNX exports from the 5080 |
| `assets/microduck/` | Pollen Robotics' MJCF (Apache-2.0) |

Logs: `logs/app/` (`server.log`, `app_stdout.log`). In the browser (`--browser`) WebKit/Chrome may drop the arrow
key-up while ⌘ is held; the page clears the held keys when ⌘ is released, the app window does not have the problem.

**Exports with a sidecar.** Isaac Lab's exporter writes `policy.onnx` + `policy.onnx.data`; the `exports/` copies
of the 2026-11-10 batch (jump, roulade, run v3, sprint, sitstand fast, standup, velstand) referenced a sidecar that
was not next to them and could not load anywhere. `sync_policies.sh --local` repacks them from
`checkpoints/<run>/policy.onnx(+.data)` into single self-contained files (bit-identical outputs).

---

# What the engine needs from the training side (contract)

The app itself is built on the Mac. This folder holds the pieces it needs from the training side.

## Assets
`assets/microduck/` is Pollen Robotics' MJCF (Apache-2.0, licence included): `scene_backlash.xml` includes
`robot_groundcontact_backlash.xml` (the model our sit/stand, roll and jump policies train on; the walk/run
exports named `*_allcoll_*` are fine-tuned on Isaac Lab's equivalent). `scene.xml` is the non-backlash
variant. Keyframe `INIT` stands the robot at z 0.12. Pollen's reference CPU runner is
`microduck_rl/scripts/infer_policy.py` (BAM servo model wired through `bam.mujoco.MujocoController`).

## Policies (`../exports/*.onnx`, 61 -> 14, observation normaliser baked in, 50 Hz)
| file | role | command slot use |
|---|---|---|
| **`velstand_allcoll_hd1910_iter21495.onnx`** | **the walk policy: walking + balance at zero speed + fall recovery, one network** (Pollen's VelStand recipe). Idle = this policy at zero twist (stands at 0.116 m, no squat). If it is knocked over it gets up by itself (100 % from face down / face up / side / random / real fall poses, 0.3-0.6 s) and keeps following the command. | twist = (vx [-0.4,0.6], vy [-0.3,0.3], yaw [-1,1]); zero = balanced stand |
| `sitstand_fast_hd1910_iter5499.onnx` | sit <-> stand, transition 0.3 s (0.26 s to standing, 0.32 s to seated) | twist = (flag, 0, 0): 0 stand, 1 sit |
| `standup_hd1910_iter9999.onnx` | standalone get-up + standing hold; fallback if a non-recovering policy (run, sit) ends on the floor: run it at zero twist until upright, then hand back | twist = (0, 0, 0) |
| `sprint_max_allcoll_hd1910_iter7998.onnx` | sprint (straight only): 1.1-1.2 m/s regardless of the command magnitude (servos at their speed limit, flight-phase gait); 0 falls at the 0.7-1.2 commands. Use it as the 'sprint' button; no fall recovery -> on a fall switch to velstand or standup | twist = (1.2, 0, 0) |
| `run_v3_allcoll_hd1910_iter5999.onnx` | running (straight only), peak 0.63 m/s at the 0.9 command (servo speed limit); no fall recovery of its own -> on a fall switch to velstand or standup | twist = (vx [0.2,1.1], 0, 0) |
| `run_allcoll_hd1910_iter4498.onnx` | previous runner (0.49 m/s peak; superseded) | twist = (vx [0.2,1.2], 0, 0) |
| `walk_allcoll_hd1910_iter16498.onnx` | previous walking policy (squats to 0.073 m at zero command; superseded by velstand) | as velstand |
| `sitstand_hd1910_iter4000.onnx` | previous sit <-> stand (2 s transitions; superseded) | twist = (flag, 0, 0) |
| `walk_sym_hd1910_iter15000.onnx`, `run_v2_hd1910_iter2999.onnx` | the reduced-collision originals | same |
| `sit_hd1910_iter12000.onnx` | sit-only (standing start) | twist = (1, 0, 0) |
| `jump_hd1910_iter1500.onnx` | vertical hop (~3 cm, the servo ceiling). Training command (`microduck_rl/mdp/jump.py::JumpOneShotCommand`): launch flag = 1 once both feet have been on the ground for 4 control steps, for 0.75 s of support time, until the first touchdown after a flight; the app reproduces exactly that, keeps the policy running until the landing is quiet for 1 s (at most 2.5 s), then hands back to velstand. 83 % take-off, 94 % stable landing (96 trials in Newton) | twist = (launch flag, 0, 0) |
| `jump_kp58_hd1910_iter2250.onnx` | the jump for servos whose P register is 5-8 (fine-tuned with P drawn per servo in [5, 8]): at P 8 it lands 67 % of jumps where `jump_hd1910_iter1500` lands 19 %; at P 5 the two match. Provisional. |
| `roulade_head_kp58_hd1910_iter7999.onnx` | the front flip for servos whose P register is 5-8: at P 8 it completes the roll and stands in 72 % of episodes where `roulade_head_hd1910_iter7000` manages 51 %; at P 5 the two match (87 % vs 84 %). Provisional. |
| `roulade_head_hd1910_iter7000.onnx` | front flip (forward roll over the head, lands on the feet, head back to straight within 4 deg): run it from a stand at zero twist; one roll takes ~1.5 s, then hand back to velstand. 90.6 % complete the roll and stand, 100 % end standing (96 trials) | twist = (0, 0, 0) |
| `roulade_hd1910_iter6000.onnx` | previous flip (96.9 % land standing but the head stays 58 deg off home; superseded) | twist = (0, 0, 0) |

The app's roles: walk = `velstand_*` (else `walk_*`), run = `run_*`, sprint = `sprint_*`, sitstand = `sitstand_*`,
jump = `jump_*`, roulade = `roulade_*`, recovery = `standup_*` (else `recovery_*` / `getup_*`); the contract's
file above is the default for each, else the newest iteration. `exports/` must hold self-contained ONNX files
(see "Exports with a sidecar" above).

## Observation (61 floats, this order, raw units; the ONNX normalises)
1. `imu` (6): base angular velocity in the trunk frame [rad/s] (3), then projected gravity in the trunk frame (3)
   = R(q)^T (0, 0, -1).
2. `joint_pos` (14): encoder angle minus DEFAULT_POSE, in the order below. The encoder sees the servo joint
   **plus its paired backlash hinge** (Isaac Lab `joint_pos_rel_biased(..., backlash_cfg)`).
3. `joint_vel` (14): servo + backlash-hinge velocities [rad/s], same order.
4. `actions` (14): the previous policy output (zeros at start).
5. `velocity_commands` (3), `head_pose_commands` (4, zeros), `body_pose_commands` (6, zeros).

Joint order (actuators and observations): left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee,
right_ankle. DEFAULT_POSE = [0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579,
0.0049, -0.4530] rad.

## Action
servo target = DEFAULT_POSE + action (scale 1.0). **Not clipped to the joint range** (Isaac Lab's joint-position
action does not clip; the stand-up from the learned seated pose drives the knee against its limit with a target
beyond it — clipped, the knee stalls 17° short and the duck never rises). Control at 50 Hz, physics 0.005 s
(decimation 4). Keep ONE shared last-action buffer across the continuous policies (walk / run / sprint / sit-stand)
when hot-swapping; the episodic ones (jump, flip, stand-up) start the way their training episodes start, with a
zeroed buffer and a zero twist.

## Servo (Feetech HD1910, BAM m6 fit used in training)
kt 0.6238 N.m/A, R 4.910 ohm, kp_fw 5, error gain 0.166 x 1.3217 duty/rad per unit, max PWM 0.97, no current
limit, supply 7.4-8.0 V (sag 0-0.2 V/A, floor 7.0 V), command delay 3-6 physics steps, backlash +-1 deg
hinges. Full dict: `microduck_rl/assets/microduck_hd1910.py::HD1910_BAM_FIT` (the engine reads it from there).

## Operator logic that worked in simulation (`scripts/demo_sequence.py`, `scripts/push_test.py`)
Default policy is **velstand**: idle = velstand at zero twist (balanced stand, no squat), arrow keys set its twist.
Running = run policy with vx in [0.2, 1.1] (it cannot turn); sprint = the sprint policy. Stop = velstand at zero
twist. Sit/stand toggles the fast sit/stand flag (0.3 s transitions); start sitting from velstand's stand, return to
velstand once standing. Fall detection (tilt > 60 deg for 0.2 s): under velstand nothing to do, it gets up on its
own; under run / sprint / sit / jump / flip, switch to the stand-up policy (or velstand) at zero twist until
tilt < 20 deg and trunk z > 0.095 m for 0.5 s, then hand back. R (front flip) = roulade policy from a stand, then
back to velstand. Space (jump) = jump policy with its launch flag, then back to velstand.
