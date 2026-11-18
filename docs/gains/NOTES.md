# P 5–8 gain fine-tune — per-policy verdicts

Chain: `scripts/finetune_gains.sh` (unit `microduck-finetune-gains`). Each deployed policy is graded at fixed
firmware P 5 (`kp_scale 1.0`) and P 8 (`kp_scale 1.6`), fine-tuned 1000 iterations with per-servo P ∈ [5, 8],
and graded again. `RUNS.txt` lists the fine-tune run dirs. The squat column was missing for velstand until
`eval_squat.py` stopped assuming the sit-stand event name (fixed 2026-11-17, re-run by hand).

## Baselines: which deployed policies degrade at P 8 (32 envs, `*_base_p5_*` vs `*_base_p8_*`)

| policy | metric | P 5 | P 8 |
|---|---|---|---|
| jump 1500 | stable landing / durable | 0.94 / 0.59 | 0.31 / 0.22 |
| roulade_head 7000 | complete+standing | 0.875 | 0.438 |
| sprint_max 7998 | vx at cmd 1.2 / falls | 1.17 m/s / 0.06 | 0.91 m/s / 0.12 |
| run_v3 5999 | vx p10 at cmd 0.9 / falls | 0.62 / 0.03 | 0.41 / 0.09 |
| standup 9999 | pushed standing, upright at end | 1.00 | 0.81 |
| sitstand_fast 5499 | posture / transition | no change | no change |
| velstand 21495 | walk 0.3 falls (128 envs) / get-up | 0.02 / fine | 0.04 / fine |

The dynamic policies (jump, flip, sprint) are the ones a stiffer servo breaks: they were trained at P 4.25-5.75
and time their take-offs and landings against that response. The fine-tune is for them first; the chain was
restarted on 2026-11-17 22:24 in the order jump, roulade_head, sprint_max, run_v3, standup, sitstand_fast
(POLICIES env override, script unchanged; it had been waiting at the GPU gate since 19:06 behind other
sessions' trainers).

## velstand (base 21495 → ft 22494, run 2026-11-17_18-13-49) — HOLD, mixed

Walking, vx 0.3, 128 envs (`velstand_ckpts/`):

| ckpt | P | vx mean | vx p10 | fallen |
|---|---|---|---|---|
| base 21495 | 5 | 0.258 | 0.109 | 0.020 |
| ft 22494   | 5 | 0.283 | 0.207 | 0.080 |
| base 21495 | 8 | 0.310 | 0.180 | 0.040 |
| ft 22494   | 8 | 0.306 | 0.175 | 0.080 |

Every intermediate fine-tune checkpoint (21500…22494) falls more than base at P5 (0.06–0.18); the drift is not
monotone, so it is the fine-tune's stochastic drift rather than a trend toward P 8.

Get-up battery, P5, 96 envs/family (`fell_after_standing`): prone 0.052→0.010, supine 0.073→0.010,
side 0.073→0.010, bank 0.052→0.021, random 0.094→0.042, sitting 0.115→0.125, pushed-standing end 0.938→0.917.
Squat (16 envs): no difference beyond noise at either P.

Reading: the base already walks and gets up at P 8 as well as at P 5, so the gain range is not what the
fine-tune fixed; the extra 1000 iterations traded walking falls (2–4 % → 8 %) for re-falls after standing
(5–7 % → 1 %). Neither checkpoint dominates. Options: (a) ship base as `_kp58` (it is robust to P 8 already),
(b) a second arm at reduced learning rate to keep the gait while taking the get-up gain. Decide after the
other six policies report.

## jump (base 1500 → ft run 2026-11-17_22-34-52, 1000 its) — PROVISIONAL PICK ft 2250

96 envs per cell (`jump_ckpts/`); landing = stable landing rate, durable = still standing later, early = terminated early:

| ckpt | P | take-off | rise med mm | landing | durable | early |
|---|---|---|---|---|---|---|
| base 1500 | 5 | 0.865 | 29.9 | 0.969 | 0.583 | 0.292 |
| ft 2000   | 5 | 0.812 | 27.0 | 0.938 | 0.573 | 0.281 |
| ft 2250   | 5 | 0.854 | 24.8 | 0.958 | 0.667 | 0.250 |
| ft 2499   | 5 | 0.729 | 24.6 | 0.885 | 0.531 | 0.312 |
| base 1500 | 8 | 0.500 | 27.8 | 0.188 | 0.073 | 0.896 |
| ft 2000   | 8 | 0.656 | 27.9 | 0.688 | 0.344 | 0.542 |
| ft 2250   | 8 | 0.635 | 26.2 | 0.667 | 0.344 | 0.521 |
| ft 2499   | 8 | 0.635 | 24.2 | 0.656 | 0.323 | 0.604 |

At P 8 the base lands 19 % of its jumps; every fine-tune checkpoint lands 59–69 %. ft 2250 keeps the P 5
numbers (landing 0.958, durable 0.667 vs base 0.969/0.583) and is the pick; the last checkpoint (2499) lost
take-offs at P 5 and is not. Rise shrinks 5 mm across the fine-tune at both gains (the policy jumps a little
lower to land). P 8 is still far from P 5 parity (0.67 vs 0.96 landing), so a longer continuation from 2250
with the same P ∈ [5, 8] range is the next single-change run for this policy once the chain has finished.

## roulade_head (base 7000 → ft run 2026-11-18_06-33-23, 1000 its) — PROVISIONAL PICK ft 7999

Fine-tune with P drawn per servo in [5, 8] (kp_scale [1.0, 1.6]), entropy 0.0 / lr as the chain sets
them. Every saved checkpoint graded at 96 envs, standing start, mean action, FIXED gains
(`docs/gains/roulade_head_ckpts/*_n96.json`, `scripts/grade_gains_ckpts.sh roulade_head`):

| ckpt | P | complete | complete+standing | standing at end | head err (deg, median) |
|---|---|---|---|---|---|
| base 7000 | 5 | 0.844 | 0.844 | 1.000 | 3.8 |
| base 7000 | 8 | 0.719 | 0.510 | 0.729 | 5.7 |
| ft 7250 | 5 | 0.917 | 0.833 | 0.906 | 2.7 |
| ft 7250 | 8 | 0.677 | 0.667 | 0.990 | 2.6 |
| ft 7500 | 5 | 0.896 | 0.854 | 0.958 | 4.3 |
| ft 7500 | 8 | 0.583 | 0.583 | 1.000 | 1.6 |
| ft 7750 | 5 | 0.917 | 0.865 | 0.948 | 3.5 |
| ft 7750 | 8 | 0.698 | 0.698 | 1.000 | 0.9 |
| **ft 7999** | 5 | 0.896 | 0.865 | 0.958 | 2.4 |
| **ft 7999** | 8 | 0.740 | 0.719 | 0.979 | 1.0 |

Verdict: ft 7999. At P 8 complete+standing 0.51 → 0.72 and standing at the end 0.73 → 0.98; at P 5 it
matches the base (0.865 vs 0.844; standing at the end 0.958 vs 1.000 is 4 episodes of 96). The run was
still improving at P 8 across its last three checkpoints (0.58, 0.70, 0.72), so a longer continuation
from 7999 with the same KP range is the next single-change run. Exported as
`checkpoints/roulade_head_kp58_7999`, `exports/roulade_head_kp58_hd1910_iter7999.onnx` (self-contained);
`roulade_head_7000` stays the P 5 pick.
