# Design notes

Dated 2026-11-06. Every number below has a source; "measured" means measured in this repo.

## Stack

**Isaac Lab `main` (3.0, Newton + MuJoCo-Warp), Kit-less.** The official Microduck training repo
(`pollen-robotics/microduck_rl`) runs on mjlab; NVIDIA/HF ported the velocity task to Isaac Lab
contrib with the same solver, timing (200 Hz physics / 50 Hz control), BAM servo model, backlash
preset and the full DR list, and deployed it on the real robot through the stock runtime. We build
on that port and add what it lacks (sit/stand) by porting the mjlab tasks, which are already
written against an Isaac-Lab-shaped API.

The community Isaac Lab port (`kabilankb/isaaclab-microduck`) reports locomotion "not working"
(policies converge to standing) and lacks the BAM actuators and observation delays; it is not used.

## Actuator: Feetech HD1910 in Isaac Lab's BAM drive

The Newton BAM kernel computes `duty = (target - measured) * kp_fw * error_gain`, clamps it to
`±max_pwm`, and (if `max_current > 0`) to a current window. Rhoban BAM's Feetech model
(`bam/feetech/actuator.py`, `STS3215Actuator`) is `duty = err * kp * 0.166 * error_gain_ratio`,
clamped to ±0.97, no current limiter, plus a target-position rate limiter. The HD1910 fit
(`1910_m6.json`, LuwuDynamics, m6 model) has `error_gain_ratio = 1.3217` and `max_velocity = 100`
(upstream's way of disabling the rate limiter, which the Isaac Lab drive does not have anyway). So:

| BamMotorCfg | value | from |
|---|---|---|
| `kt` | 0.6238 V·s/rad | fit |
| `resistance` | 4.910 Ω | fit |
| `error_gain` | 0.166 × 1.3217 = 0.2194 /rad per P unit | BAM Feetech law × fit |
| `max_pwm` | 0.97 | BAM Feetech |
| `max_current` | 0 (disabled) | BAM Feetech has none |
| friction terms | fit | fit |
| `kp_fw` | 5 | LuwuDynamics / Ley-WL / kissqy |
| `vin_range`, sag, `vin_min` | 7.4–8.0 V, 0–0.2 V/N·m, 7.0 V | kissqy task configs |
| delay | 3–6 physics steps | kissqy task configs |
| `armature` | 0.00178 kg·m² | fit |

Stall torque at 8.0 V = 8.0·kt/R = **1.02 N·m** (the drive's effort limit). `q_offset` in the fit is
an identification bias and is not used (Ley-WL's README says the same).

The fit is Apache-2.0 and redistributed unchanged; it was identified by LuwuDynamics on an STS3215
test bench with a 1910, not by us. Nothing here is a measurement of *your* servos.

## Mass and CoM: domain randomisation, not a model

kissqy's `hardware.json` records 23 g per HD1910 (datasheet 21±2 g) against 18 g for the XL330:
+75 g over 15 servos, ≈ +7 % of the robot. The compute (Radxa ZERO 3W), HAT, IMU board and
battery holder are different parts in different places. The official USD cannot be edited per
build, so the replica is covered by ranges:

| | official | here |
|---|---|---|
| mass scale | trunk 0.95–1.05 | **every body 0.90–1.25** |
| trunk CoM | ±3 → ±15 mm (curriculum) | ±3 → **±25 mm** |
| head CoM | ±3 → ±10 mm | ±3 → **±15 mm** |
| sole friction | 0.7–1.3 | **0.5–1.3** |

Applied identically to every task (`microduck_rl/dr.py`). If a policy fails to transfer, the first
check is whether the real robot sits inside these ranges (weigh it; the trunk and head CoM can be
estimated by balancing on an edge).

## Sit / stand

`Mjlab-SitStand-Flat-MicroDuck` v1.5 is ported term for term (`mdp/posture.py`,
`tasks/sitstand_env_cfg.py`); kissqy's own SitStand training config was checked against upstream and
is identical apart from the actuator. Three policies come from one recipe by fixing the start
distribution and the command: *Sit* (standing start, SIT), *Stand* (seated start, STAND) and the full
*SitStand*. The observation keeps the 61-wide walking layout (posture flag in the `vx` slot).

Deviations from upstream, all forced by the backend:

- self-collision is measured between the left and right leg chains (a filtered contact sensor) rather
  than the whole trunk subtree against itself;
- `STAND_Z = 0.115`, `SIT_Z = 0.060` are upstream's MuJoCo rest heights. `scripts/measure_rest_heights.py`
  re-measures them on the Newton model; update the constants if they disagree by more than a few mm;
- `joint_torque_rate_l2` / `neck_strain_leaned` read `applied_torque`; the same script reports whether
  the Newton BAM drive populates it.

## Rollout videos

Training records a clip of the live policy every 250 iterations (`--video --video_interval 6000`,
24 steps per iteration); `scripts/record_checkpoints.py` re-renders every saved checkpoint after the
fact and writes an index with each checkpoint's SHA-256, so a clip can always be traced to its weights
(a live `latest.pt` is not one policy).
