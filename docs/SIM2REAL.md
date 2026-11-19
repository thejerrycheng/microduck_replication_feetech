# Matching the training simulation to the real duck

What was measured on the real HD1910 Microduck replica on 2026-11-18, how it differs from what the policies were
trained on, and what to change in training (`microduck_rl/`) so that the next policies see in simulation what they
will meet on the duck. Every number below is from a run on the duck unless it says "not measured".

The policies on the duck are the seven Duck Live defaults (`app/engine.py` `DEFAULTS`): walk/stand
`velstand_allcoll_hd1910_iter21495`, run `run_v3_allcoll_hd1910_iter5999`, sprint `sprint_max_allcoll_hd1910_iter7998`,
sit-stand `sitstand_fast_hd1910_iter5499`, jump `jump_hd1910_iter1500`, flip `roulade_head_hd1910_iter7000`,
get-up `standup_hd1910_iter9999`.

## 1. What the real duck does each control tick

`firmware/duck_agent.py` `Drive` (the runner) and `firmware/duck_pilot.py` (which policy, what command), on the Radxa
Zero 3W, every 20 ms:

1. One sync read of registers 56-59 from the 14 joint servos: position (ticks) and speed (counts).
2. One read of the IMU board (bus ID 200, register 124): gyro (int16, 17.5 mdps per count) and the chip's own
   orientation quaternion.
3. Observation, 61 values, the policies' own layout:
   `gyro (3) | gravity in the trunk frame (3) | joint angle - default pose (14) | joint speed (14) | last action (14) | command (13)`.
4. The policy (numpy, 16-bit weights; `firmware/policy_net.py`).
5. One sync write of 14 goal positions: `default pose + action`, not clipped to the joint range.

Conversions, all from the calibration file (`/data/calibration.json`):

| quantity | real duck |
|---|---|
| joint angle [rad] | `(ticks - zero_raw) * 2 pi / 4096 * direction` |
| joint speed [rad/s] | `counts * 50 * 2 pi / 4096 * direction` |
| gyro [rad/s], trunk frame | chip gyro rotated by the calibrated mounting (`mount^-1 g mount`) |
| gravity, trunk frame | `(0, 0, -1)` rotated by the inverse of `chip quaternion x mount` |
| target [ticks] | `zero_raw + (default + action) / (2 pi / 4096 * direction)` |

Joint order is `duck_pilot.JOINTS` (left leg, neck/head, right leg), the same as `app/engine.py`.

## 2. Measured differences

| # | Quantity | Training (file, value) | Real duck, measured | How it was measured |
|---|---|---|---|---|
| 1 | Control period | fixed 20 ms (`sim.dt` 0.005, decimation 4) | 20.2-20.6 ms with the board cool (48.5-49.9 Hz, about 7 % of ticks late); 21.7 ms at 83 C (46 Hz); 30-38 ms once the board throttles (33.0 and 26.5 Hz) | the runner's own tick counter, 5 live runs and 4 dry runs |
| 2 | Compute + bus time inside a tick | not modelled separately; BAM command delay 3-6 physics steps (15-30 ms) | 9.7-10.3 ms cool, 13-14 ms at 83 C, 29-41 ms throttled | same counter |
| 3 | Supply voltage | `HD1910_VIN_RANGE` (7.4, 8.0) V, `HD1910_VIN_MIN` 7.0 V | 7.6-7.8 V at rest; 6.5-7.2 V while the legs push (6.5 V was the lowest reading) | servo register 62 during live runs |
| 4 | Servo gains | P 5 (`HD1910_KP_FW`), no D term in the drive model | P 5, D 20 in the servos (set 2026-11-18; they were at the factory P 32, D 40) | registers 21, 22, 50, 51 |
| 5 | Joint speed reading | continuous, noise +-0.25 rad/s, lag 1-3 steps (`realism.py`) | quantised: 1 count = 50 ticks/s = 0.077 rad/s (fitted 42.7-49.6 ticks/s per count on servos 30, 31, 33, correlation 0.96-0.98) | position differences against register 58 during a head animation |
| 6 | Joint position reading | +-0.002 rad noise, lag 0-2 steps, 5 % dropped reads | 12-bit, 0.0015 rad per tick; **0 missed readings** in every run (thousands of ticks) | sync-read checksums |
| 7 | Dropped writes | 3 % (`WRITE_DROPOUT_PROB`) | not measured (a sync write is not acknowledged) | |
| 8 | IMU frame | trunk frame, misalignment up to 6 deg | after the two-pose calibration: front = chip +z, left = +y, up = -x, each within about 3 deg of square; poses 91.7 deg apart | `/data/calibration.json`, 11:51 |
| 9 | Backlash in the joint observation | `servo angle + backlash hinge` (`app/engine.py` `observe`) | the servo's own encoder only; backlash not measured | |
| 10 | Foot contacts, trunk height, trunk velocity | used by Duck Live's operator logic (jump launch flag until touchdown, hand-backs) | none of the three exists on the duck: `duck_pilot.py` uses tilt and the clock | |
| 11 | Start of an episode | reset near the default pose (or the task's own reset) | any limp pose; torque comes on where the joints are and the target is blended in over 2 s | `Drive.BLEND` |
| 12 | Joint range | model limits, e.g. hip pitch +-90 deg | the tucked sitting duck read hip pitch -167.6 and +133.1 deg, knee -98.0 deg: the real joints go past the model's limits | calibrated angles, sitting |
| 13 | Board temperature | none | 51 C at boot, 69 C after 6 min, 83-87 C after about 15 min, then throttled to 408 MHz; three shutdowns in one day | `/sys/class/thermal` |

Not measured at all, and worth measuring before trusting any conclusion: end-to-end command delay (write to first
motion), floor friction, the duck's real mass and centre of mass with battery and electronics, backlash per joint,
and the delay of the chip's fused orientation.

## 3. What the real duck did

All with the walk/stand policy at zero command (`firmware/live_try.py`, logs in `docs/live_*.json`).

| Run | Condition | Result |
|---|---|---|
| 11:05 | half torque, tucked sit, IMU front axis wrong | tipped at 2.0 s |
| 11:08 (a) | 80 % torque, IMU front wrong, gyro rotated the wrong way (software bug) | got up from 119 deg in about 2 s; fell and recovered about 3 times in 20 s |
| 11:08 (b) | full torque, same faults | upright in 112 of 173 samples; one steady stretch of 12 s |
| 11:14 | board throttled, loop at 26.5 Hz | never upright (0 of 22 samples) |
| 11:27 | gyro fixed, IMU front still wrong | upright in 440 of 607 samples over 2 min; 13 falls, all recovered; last 30 s upright |
| 11:30 | plus hand-over to the stand-up policy after 2 s down | 14 falls; after the last the stand-up policy held still at 95 deg for 8 s |
| 11:56 | two-pose IMU calibration | up in under 1 s; 2.2-4.2 deg of tilt for 16 s, then longer; later lay still at 94 deg with the same target for minutes (battery 6.8-7.2 V under load, 46 Hz) |
| 12:00 | plus alternating walk / stand-up policy when down | up at about 6 s; 1-6 deg of tilt for 27 s; fell at 34 s; run stopped from outside before the recovery could be seen |

Three things follow from these and nothing else has been shown:

- **With the IMU frame right, a cool board and a fresh battery, the stand holds for tens of seconds.** Every earlier
  failure had a wrong IMU frame in it, so those runs say little about the policy.
- **Loop rate matters a great deal.** At 26.5 Hz the same policy never got up.
- **Lying still is a real failure mode that the simulator does not have.** In the simulator the walk policy is up
  1.0-1.2 s after any of 20 lying starts. On the duck it can ask for a pose the legs do not reach, nothing moves, the
  observation does not change, and it asks again. Low supply voltage is the most likely reason the legs do not reach
  it (6.8 V against a trained minimum of 7.0), but that is an inference, not a measurement.

Walking, running, sitting, jumping and the flip have not been run on the real duck.

## 4. What to change in training, in order of expected effect

1. **Supply voltage.** `microduck_rl/assets/microduck_hd1910.py`: widen `HD1910_VIN_RANGE` from (7.4, 8.0) to about
   (6.4, 8.2) and lower `HD1910_VIN_MIN` from 7.0 to about 6.2, so that a sagging 2-cell pack is inside the
   distribution. Check the sag term (`HD1910_VIN_DROP_GAIN_RANGE` 0.0-0.2) against a logged voltage-against-load
   trace before changing it.
2. **Control period and delay.** Randomise the control period per episode over about 20-24 ms (decimation jitter or
   an extra whole-step action delay), and keep the BAM command delay. If the board cannot be cooled, a second range
   of 30-40 ms is what throttling produces, but a policy trained for both will be worse at each: cooling the board
   is the better fix.
3. **Get-up from a stall.** Add resets lying on the side, front and back **combined with** the low end of the
   voltage range and the high end of friction (`FRICTION_SCALE_RANGE` 0.8-1.2: try up to 1.5), so that "the first
   push does not move the duck" happens in training. A small penalty on a repeated identical action while tilted
   past 60 deg would also discourage waiting.
4. **Joint speed observation.** In `realism.py`, replace the +-0.25 rad/s uniform noise on `joint_vel` by
   quantisation to 0.077 rad/s (round to the nearest multiple) plus a smaller noise. This is how the real reading
   differs, and it removes all speeds below about 0.04 rad/s from what the policy sees.
5. **Dropped reads.** `READ_DROPOUT_PROB` 0.05 is far above what the bus does (0 in thousands of ticks). Lower it
   to about 0.005 so the policy stops paying for robustness it does not need. Leave `WRITE_DROPOUT_PROB` until it
   is measured.
6. **D gain.** Either add the servo's D term to the drive model (the real servos run D 20), or set D to 0 on the
   real servos and compare. Which is closer has not been tested; kissqy's replica walked with P 5, D 20.
7. **Backlash observation.** The real observation is the encoder alone. If the encoder sits on the output shaft it
   already includes backlash; if it sits before the gears it does not. Find out which, and make the simulated
   observation the same (`servo + backlash` or `servo` only).
8. **Start pose.** Add episodes that start with the joints anywhere inside (and a little outside) the model's
   range, with a 2 s linear blend from that pose to the policy's target, since that is how every real run starts.
9. **Jump without contacts.** The jump's launch flag is ended by touchdown in training and by the clock on the duck
   (0.36 s, the simulator's touchdown: launch on at 0.08 s, feet off at 0.24-0.26 s, down at 0.34-0.36 s). Train
   the jump with a flag that is on for a fixed time, so both are the same.
10. **IMU.** The trained misalignment (up to 6 deg) already covers the calibrated mounting (about 3 deg). No change
    needed; do not deploy without the two-pose calibration (`firmware/studio.py` `imu_cal`).

## 5. Measurements still needed, and the tools for them

| Measurement | Tool | State |
|---|---|---|
| Per-tick observation and action log from the duck, to replay through the simulator | add a ring buffer to `Drive._run` and a route to fetch it | not written yet |
| Servo step response (delay, rise, overshoot) at P 5, D 20 and several voltages | `firmware/step_response.py` | exists; not rerun at these gains |
| Voltage against load, to fit the sag term | log register 62 and the load register during a live run | `live_watch.py` logs voltage only |
| Head tracking of commanded motion | `firmware/test_anim.py` | done: 0.8-2.2 deg RMS, 0.12-0.14 s from request to motion |
| Loop timing | the runner's status (`hz`, `ms`, `late`) | done, table above |
| Fall recovery time on the real duck | `firmware/live_try.py --down` | no clean run yet with the IMU fixed |

## 6. Checking a new policy before it goes on the duck

```
app/sync_policies.sh --local                                   # new exports into exports/
~/.venvs/microduck_live/bin/python firmware/policy_pack.py     # numpy pack + check against ONNX
~/.venvs/microduck_live/bin/python firmware/test_pilot.py      # the duck's pilot in the simulator, 10 trials
```

Then start the Mac studio (it sends the pack to the duck when it differs), run **Dry run** in the Drive panel with
the duck standing (loop rate, tilt, and the largest step the policy would ask for), and only then `live_try.py`
with its tilt and time limits. Numbers to beat, from the current policies: pack against ONNX 0.3-0.5 mrad worst
action difference (walk 5.7 mrad); pilot in the simulator 8 of 10 trials without a fall, 10 of 10 flips upright;
on the duck, 27 s of standing at 1-6 deg.
