# Real-duck parameters in training (2026-11-19)

Source: the 2026-11-18 measurements on the HD1910 replica (user document "Matching the training simulation to the real
duck"). Each row is a training change picked up by EVERY task (realism.py / dr.py / the sit-stand DR stack), unless noted.

| Measured on the duck | Training before | Training now | Where |
|---|---|---|---|
| supply 7.6-7.8 V at rest, 6.5-7.2 V under load | vin 7.4-8.0, sag floor 7.0 | vin 6.4-8.2 per env, sag floor 6.2 | `assets/microduck_hd1910.py` |
| control period 20.2-21.7 ms cool/hot, 30-38 ms throttled; user: 50 Hz down to 25 Hz | fixed 20 ms | per-env period 20-40 ms drawn at reset; the policy's output is taken on TICKS and held between them (`DropoutJointPositionAction`), the held action fed back to `last_action` | `realism.py`, `mdp/realism.py` |
| compute+bus 10-14 ms, late ticks | BAM command delay 3-6 physics steps | 3-8 physics steps (15-40 ms) | `assets/microduck_hd1910.py` |
| servo P 5 (replicas 5-8) | kp_scale 0.85-1.15 | kp_scale 1.0-1.6 = P 5-8 | `realism.py` |
| stalled get-up (legs do not move under a 40 deg error) | servo friction x0.8-1.2 | x0.8-1.5, with the low voltage | `realism.py` |
| joint speed in whole counts, 0.077 rad/s each | continuous +-0.25 rad/s noise | rounded to 0.077 rad/s, +-0.04 noise | `realism.py` (`dropout_obs quantization`) |
| 0 missed reads in thousands of ticks | 5 % dropped reads | 0.5 % | `realism.py` |
| battery/board heavier than modelled (not weighed) | trunk in the all-body x0.90-1.25 | trunk x0.90-1.60 (`battery_mass`), other bodies x0.90-1.25 | `dr.py`, `sitstand_env_cfg.py` |
| soles slip on the real floor | sole 0.5-1.3 but the contact never below 1.0 (docs/friction) | soles own the contact (priority 1), 0.25-1.2 per sole per reset | `dr.py`, `sitstand_env_cfg.py`, `mdp/slip.py` |
| -- | always-on slip cost | slip charged only above 0.2 m/s (walk, run, sprint) | `velocity_env_cfg.py` |
| the jump flag is a 0.36 s clock on the duck | flag until touchdown | flag on for 0.36 s | `mdp/jump.py` |
| stand must survive disturbance | VelStand pushes +-0.4 every 3-5 s, topples to +-1.2 | +-0.6 every 2-4 s, topples to +-1.5 | `velstand_env_cfg.py` |

Not changed, and why: the servo D term (the duck runs D 20, training has none; no step response at D 20 exists to fit --
set D 0 on the duck or measure first), the backlash observation (encoder location unknown), the 2 s start blend
(next wave), dropped writes 3 % (not measured), IMU misalignment (6 deg covers the calibrated 3 deg).

Smoke, current walker `velstand_21495`, vx 0.3, 32 envs under the full STRESS condition (P 8, 6.4 V, friction 0.3,
25 Hz): fallen 0.97, vx 0.04 (`docs/real/smoke_velstand_base_stress32.json`). Single-factor smokes are in
`docs/real/smoke_velstand_base_*32.json`.

Chain `scripts/train_real.sh` (unit `microduck-real`): velstand, standup, run_v3, sitstand_fast, jump, roulade_head,
sprint_max, 2000 iterations each from the current picks, entropy 0.001; graded at 96 envs under NOMINAL (P 5,
7.6-7.8 V, friction 1.0, 50 Hz) and STRESS before and after, `docs/real/<name>_{base,ft}_{nominal,stress}_*.json`.

## Baselines: the current picks under the new env (96 envs, 2026-11-19 01:30-01:50)

NOMINAL = P 5, 7.6-7.8 V, sole friction 1.0, 50 Hz; STRESS = P 8, 6.4 V, sole friction 0.3, 25 Hz. Everything else
(delays 15-40 ms, trunk mass x0.9-1.6, speed quantisation, servo friction x0.8-1.5) is randomised in both.
`docs/real/<policy>_base_{nominal,stress}_*.json`.

| policy (pick) | metric | nominal | stress |
|---|---|---|---|
| velstand_21495, walk 0.3 m/s | vx achieved / fallen | 0.262 / 0.04 | 0.049 / 0.98 |
| velstand_21495, get-up (prone, supine, side, random, bank, sitting) | standing at end | 0.97-1.00 | 0.65-0.82 |
| velstand_21495, get-up | re-fell after standing | 0.07-0.21 | 0.69-0.82 |
| velstand_21495, pushed stand (1 m/s every 1.5 s) | standing at end | 0.94 | 0.52 |
| standup_9999, get-up (six families) | standing at end / re-fell | 1.00 / 0.00-0.03 | 0.75-0.91 / 0.48-0.69 |
| standup_9999, pushed stand | standing at end | 0.92 | 0.48 |
| run_v3_5999, 0.5 / 0.9 m/s | vx achieved | 0.43 / 0.64 (no falls) | 0.10 / 0.08 (falls 0.44 / 0.12) |
| sprint_max_7998, 1.2 m/s | vx achieved / fallen | 1.03 / 0.02 | 0.35 / 0.02 |
| sitstand_fast_5499 | standing / seated-upright at the end | 0.58 / 0.40 (head up 0.98) | 0.14 / 0.50 (head up 0.64, trunk 0.063 m) |
| jump_kp58_2250 | valid take-off / stable landing / durable | 0.74 / 0.55 / 0.42 | 0.00 / 0.00 / 0.00 |
| roulade_head_kp58_7999 | complete+standing / standing at end | 0.75 / 0.80 | 0.26 / 0.33 (tilt 60 deg) |

Reading: under stress every policy still starts the motion (get-ups succeed, the roll completes half the time) and
then cannot hold the result: re-falls 0.5-0.8, pushed stands halve, the sit-stand parks the trunk at 0.063 m, the
jump never leaves the ground. That is the "lying still / falling again" behaviour the duck showed. The nominal
numbers are themselves below the old fixed-condition grades (jump landing 0.55 vs 0.96 at P 5) because the delay,
trunk-mass and speed-quantisation ranges now apply at nominal too.
