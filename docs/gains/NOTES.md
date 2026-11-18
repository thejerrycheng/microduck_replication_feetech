# P 5–8 gain fine-tune — per-policy verdicts

Chain: `scripts/finetune_gains.sh` (unit `microduck-finetune-gains`). Each deployed policy is graded at fixed
firmware P 5 (`kp_scale 1.0`) and P 8 (`kp_scale 1.6`), fine-tuned 1000 iterations with per-servo P ∈ [5, 8],
and graded again. `RUNS.txt` lists the fine-tune run dirs. The squat column was missing for velstand until
`eval_squat.py` stopped assuming the sit-stand event name (fixed 2026-11-17, re-run by hand).

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
