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
