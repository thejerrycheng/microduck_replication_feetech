# firmware — Duck Servo Studio

Find, rename, test and live-monitor the Microduck's 15 Feetech HD1910 servos from a browser.

```bash
~/.venvs/microduck_live/bin/python firmware/studio.py          # real bus (needs pyserial)
~/.venvs/microduck_live/bin/python firmware/studio.py --mock   # no hardware
```

**Hot-plug:** twice a second the studio drops servos that have been silent for 1.2 s and pings the factory and
joint IDs for new ones, so swapping servos needs no Scan. A lone new servo opens by itself; a pulled adapter
reconnects when it returns. If the duck's computer goes away while the bus runs through it, the port menu falls
back to the USB adapter; Connect refuses a port that is no longer there. A servo answers only with servo power
on: the USB adapter powers itself, not the bus.

**Set an ID:** plug in one new servo → Connect → tap it in the found row (orange dot = new) → tap the joint it belongs to → Set.

| IDs | Joints |
|---|---|
| 10–14 | right hip yaw, hip roll, hip pitch, knee, ankle |
| 20–24 | left hip yaw, hip roll, hip pitch, knee, ankle |
| 30–34 | neck pitch, head pitch, head yaw, head roll, mouth |

Bus: Feetech STS/SCS TTL, 1 Mbit. Map and registers follow
[kissqy/Microduck-kissqy](https://github.com/kissqy/Microduck-kissqy) (`hardware.json`, `duck-control/src/feetech.rs`),
which keeps the official Pollen numbering. The servo protocol is the one used for the MABEL ORCA hand.

**Select a servo** from the found row under the top bar, by its tag, or by clicking it on the duck. Green = answering, red = found earlier but silent now,
see-through = not on the bus, blue = selected. The card that opens plots its position, velocity, load and temperature
live (sampled at 40 Hz, one Sync Read of registers 40–70) and exports the last two minutes as CSV in raw units.
Velocity uses kissqy's HD1910 scale (1 count = 50 ticks/s), not yet checked against a real servo.

**P / D:** the position-loop gains. The servo runs on RAM registers 50 / 51, loaded at power-up from EEPROM
21 / 22; Set writes both and reads both back. A factory HD1910 reads P 32, D 40. **Duck** fills the replica's
gains: P 5, D 20, beak P 10 (kissqy console note v1.0.131; P 5 is also what the policies were trained against).
The official Pollen duck runs P 200, I 0, D 0 on Dynamixel XL330s, a different servo and scale.

**Zero:** makes the present pose read 2048 (Feetech offset calibration, instruction 0x0B). A saved calibration follows the servo: its zero shifts by the same amount, so it still means the same pose. A servo standing outside its joint range is the exception: its saved zero cannot be right, so the present pose becomes the joint zero.

**Joint range:** the slider and Test never leave the joint's travel in the model (`app/assets/microduck/robot_groundcontact.xml`,
read at start-up; the mouth, absent from the model, uses kissqy's −5° … 30°). Travel is taken about 2048, so pose the
joint at its zero pose and press Zero first. A servo that reads more than 5° outside its range is refused, not swung
back. Encoder direction per joint is kissqy's calibrated `hardware.json` (`FLIP` in `feetech.py`); it only changes the
result for the lopsided joints (hip yaw, neck pitch, mouth) and has not been checked on this duck. A servo without a
joint ID keeps ±45° about the pose where torque came on.

| ID | joint | ticks | servo frame, about 2048 |
|---|---|---|---|
| 10 | right_hip_yaw | 1707–2332 | -30° … 25° |
| 11 | right_hip_roll | 1798–2298 | -22° … 22° |
| 12 | right_hip_pitch | 1024–3072 | -90° … 90° |
| 13 | right_knee | 1024–3072 | -90° … 90° |
| 14 | right_ankle | 1024–3072 | -90° … 90° |
| 20 | left_hip_yaw | 1764–2389 | -25° … 30° |
| 21 | left_hip_roll | 1798–2298 | -22° … 22° |
| 22 | left_hip_pitch | 1024–3072 | -90° … 90° |
| 23 | left_knee | 1024–3072 | -90° … 90° |
| 24 | left_ankle | 1024–3072 | -90° … 90° |
| 30 | neck_pitch | 1024–2731 | -90° … 60° |
| 31 | head_pitch | 1024–3072 | -90° … 90° |
| 32 | head_yaw | 114–3982 | -170° … 170° |
| 33 | head_roll | 1764–2332 | -25° … 25° |
| 34 | mouth | 1707–2105 | -30° … 5° |

| File | |
|---|---|
| `feetech.py` | serial bus, ID map, mock bus |
| `studio.py` | local server and routes |
| `web/index.html` | the page |
| `web/duck.bin` | 3-D duck, baked by `bake_duck.py` from `app/assets/microduck` |
| `duck_agent.py` | runs on the duck's computer; serves its sensors (and a mock of them) |
| `run_log.py`, `test_run_log.py` | the record of every policy run, kept on the duck; pull, summary, plot, CSV and replay on the Mac; its test |
| `duck_voice.py` | the duck's voice: Pollen's vocalisation synth, ported; renders the sound bank |
| `tof/` | ST's VL53L5CX / L8CX driver (BSD-3, via Pollen) and `main.c`, built on the duck |
| `sd_flash.sh`, `sd_wifi.sh` | write an image to the microSD card; put Wi-Fi (and USB networking) on it |
| `mode_install.sh` | on the duck, once: the Setup / Run permission, and `mediad` off |
| `net_install.sh`, `duck_net.py` | on the duck, once: Wi-Fi retried forever, fallback hotspot, USB networking (not yet run on a duck) |

Safety: an ID is refused if another servo already answers on it; the Test wiggle ramps at 44°/s, the slider commands the servo directly (full speed) within the joint range
above, and Disconnect releases every servo.

## Battery

The top bar shows the charge left as a percentage, with the pack voltage in the tooltip; it turns orange under 40 %
and red under 15 %, where a warning also pops up. The voltage is the servos' own supply reading (the median of the
connected servos, or one servo read by the agent when only the duck is connected), smoothed. Percent comes from a
typical lithium-ion discharge curve for 2 cells in series (`feetech.CELL`): 8.4 V = 100 %, 7.4 V = 42 %,
7.0 V = 14 %, 6.0 V = 0 %. It is an estimate: voltage sags under load, the readings step by 0.1 V (about 8 %), and
the curve is not this pack's own. On a wired supply it shows whatever that voltage maps to.

## Board temperature

Beside the battery the top bar shows the temperature of the duck's processor (the RK3566's own sensor, read by the
agent each time it reports status), with the load average in the tooltip. It turns orange from 75 °C and red from
85 °C, where a warning also pops up: those are the board's own two throttling points (`soc-thermal` passive trips;
it switches itself off at 115 °C). `HOT` in `web/index.html`. It shows only while the duck's computer is connected.

## One hardware bridge

`duck_agent.py` is the only program on the duck that opens a device: camera, microphone, speaker, ToF ranger and
the servo bus (servos and IMU). It runs as the systemd user service `duck-agent`, enabled at boot and restarted
2 s after it dies. Every client asks it over HTTP on the duck's loopback: the Mac's studio through an SSH tunnel,
and the duck's own studio (`duck-studio`, for the phone). Neither starts an agent or opens a device. A Mac studio
that connects only checks that the duck has its version of the agent and installs or updates the service if not.

Each device is opened when first asked for and opened again after it fails: a reader that exits is started again
at the next request, one that stays silent for 30 s is ended and started again, the ToF reader resets its sensor
after 3 s without a frame, and the serial port is reopened after an error. The agent starts with any device
absent. At the first start after a boot it stops the vendor runtime (`robotd`, `tofd`, enabled in the image), so
the duck comes up in Setup with the bridge owning everything; Run hands the servo bus and the ToF to that runtime
until Setup is chosen again.

This follows Pollen's runtime (pollen-robotics/microduck, `docs/design/architecture.md`): systemd-supervised
daemons, one owner per device, clients on a socket, daemons that keep running and retry when hardware is missing.
Pollen splits it into `robotd` (servos and IMU), `tofd` and `mediad`; here it is one process.

Measured with `test_bridge.py` on 2026-11-18, through the duck's own server with no Mac studio running; time from
the interruption until camera, ToF and microphone all deliver changing data again (`docs/bridge_results.json`,
`docs/figures/bridge_recovery.pdf`):

| interruption | n | recovered | mean (s) | range (s) |
|---|---|---|---|---|
| bridge killed | 5 | 5 | 5.2 | 4.8 to 5.7 |
| camera reader killed | 5 | 5 | 2.3 | 1.9 to 2.6 |
| ToF reader killed | 5 | 5 | 2.4 | 2.0 to 2.7 |
| microphone reader killed | 5 | 5 | 2.5 | 2.3 to 2.7 |

After all 20 recoveries no process other than the bridge and its own readers held a device. Not yet measured:
start at boot (the login could not reboot the duck; `mode_install.sh` now grants that), a cable physically
unplugged and replugged, and the servos and IMU (servo power was off).

## Driving: the policies on the duck

The seven policies Duck Live drives with (walk and stand, run, sprint, sit and stand, jump, front flip, get up) run on
the duck itself. The page has two halves, switched in the top bar as on the phone app:

- **Setup**: the joints, calibration, gains and sensors.
- **Control**: the phone app's pad at the bottom of the page. A row of animations to scroll through, power and the
  speed slider (1 to 5), the stick (forward / back, sidestep), sit, run, jump, flip, and the turn slider; the chip under
  the bar shows what the pilot is doing, the speed it commands and the loop rate. Keys: arrows, Shift = run, S sit,
  Space jump, R flip, A / D speed. `http://127.0.0.1:8770/#control` opens straight into it.

The switch only changes what is on screen. Power starts the policies (pressed twice: the legs move) and, pressed twice
again, sits the duck down and lets every joint go limp; Shift-click is a dry run (it reads and computes, nothing
moves). Going back to Setup with the policies on does the same sit-and-let-go first.

| piece | file | what it is |
|---|---|---|
| pack | `policy_pack.py` -> `policies.npz`, kept on the duck in `/data` | the seven actors as 16-bit weights, 2.58 MB (the ONNX exports: 5.6 MB) |
| network | `policy_net.py` | the actor in numpy: no ONNX runtime on the duck |
| pilot | `duck_pilot.py` | which policy, what command: Duck Live's operator logic with only an IMU and joints |
| runner | `duck_agent.Drive` | 50 Hz on the bridge: read IMU and 14 joints, pilot, write 14 targets |

**Dry run** reads and computes and writes nothing. **Start** (pressed twice) drives the servos: torque on where each
joint is, one second of blending to the policy's target. **Stop** lets every joint go limp. Sticks not refreshed for
0.5 s count as released; ten ticks in a row without a full reading stop a live run.

Measured (`docs/policy_pack_results.json`, `docs/pilot_results.json`, `docs/figures/pilot_sim.pdf`), 2026-11-18:

- Pack against ONNX over a 32 s simulated scenario (1600 ticks): worst action difference 0.3 to 0.5 mrad for run,
  sprint, sit-stand, jump and flip, 5.7 mrad for walk (RMS 0.84 mrad). Closed loop with the pack: 0 falls, as ONNX;
  the duck ends 1.66 m from where ONNX took it, because a jump and a flip amplify any difference.
- Pilot in the simulator, 10 trials with random start and servo delay (walk, turn, run, sprint, sit, stand, back,
  sidestep, jump, flip): 8 without a fall; 2 falls, both stopping from a sprint (Duck Live's own benchmark: 4 falls in
  8 trials), each recovered by the walk policy. Walk 0.42 m/s for 0.40 asked, run 0.62 for 0.70, sprint 1.33; jump
  rise 24 to 28 mm; 10 of 10 flips end upright.
- On the duck, dry run: 49.5 Hz, 10.3 ms of each 20 ms tick used, about 7 % of ticks late, no missed reading.

Not yet done on the real duck: a live run. Open before it: the jump is timed by the clock (launch flag until 0.36 s,
the simulator's touchdown) because the duck has no foot contacts; the IMU's axes against the model's trunk frame and
the servo gains (the policies were trained at P 5) are not verified; the stand-up policy was not reached in the tests.

## Run log

Every policy run, live or dry, is recorded on the duck tick by tick (`run_log.py`, written by `duck_agent.Drive`):
`/data/runs/run_NNNN_<UTC time>.f32` (one row of 241 float32 per 20 ms tick, appended once a second by a
thread of its own) and `.json` (the columns, the calibration, the servos' P and D, the pack's SHA-256, how the run
ended). A run cut by a power loss keeps its rows up to the last second and has no end record.

| in a row | columns |
|---|---|
| the policy | `obs` (the 61 numbers given to the network), `action` (its 14 outputs), `role` and `state`, `cmd`, `target` (rad, after the head mix and neck limit), `goal` (ticks written; in a dry run, what would have been) |
| the joints | `pos` (ticks), `vel` (counts) as read for the policy; per servo with the beak as the 15th: `load` (-1000..1000, 0.1 % of stall: the servo's torque measure), `current` (counts), `temp` (deg C), `volt` (V), `status` |
| the IMU | `quat` (body, calibrated), `quat_raw` (the chip's), `gyro` (sensor axes) and `rate` (body axes), deg/s |
| the battery | `volt`; `run_log.battery` gives percent of charge from its median (low under load, as in Battery above) |
| the loop | `t`, `ms`, `late`, `missed`, `blend`, `stick`, `keys`, `clip`, `level`, `tilt`, `falls`, `cpu_temp`, `cpu_load` |

Load, current, temperature and voltage need a second bus read (registers 60-70 of the 15 servos, 255 reply bytes,
about 2.6 ms at 1 Mbit). It is made after the goals are written, so the policy's reading and its write are no later
than without the log, and it is skipped (NaN in the row) on a tick already past 14 ms. The log never stops a run: on
any error it stops recording and the status says why (`log_error`). The folder is kept under 500 MB
(173.5 MB per hour of driving), the oldest runs deleted first.

```bash
~/.venvs/microduck_live/bin/python firmware/run_log.py pull      # new runs -> logs/duck_runs/ (not in git)
~/.venvs/microduck_live/bin/python firmware/run_log.py list
~/.venvs/microduck_live/bin/python firmware/run_log.py show      # the newest run: rate, falls, tracking error, load, temperature, battery
~/.venvs/microduck_live/bin/python firmware/run_log.py plot      # docs/figures/run_<name>.pdf
~/.venvs/microduck_live/bin/python firmware/run_log.py csv
~/.venvs/microduck_live/bin/python firmware/run_log.py replay    # logged obs through the pack: the logged actions?
```

`run_log.load(path)` gives the columns as arrays (`run["obs"]`, `run["load"]`, ...) and the header as `run.meta`.

Measured with `test_run_log.py` on 2026-11-22 (`docs/run_log_results.json`, `docs/figures/run_log_test.pdf`): the real
drive loop, pilot and pack against a simulated servo bus on the Mac, 6 s each, dry, live, and live with the
beak's servo silent. In all three: one row per tick (315, 313, 314 rows), the logged
observation through the pack gives the logged action exactly (worst difference 0e+00 rad), the joint angles
in `obs` are the logged ticks through the calibration to 1e-07 rad, the goals are the logged targets to
0.50 tick (rounding), load, current and temperature are the bus's own, the silent servo's columns are
NaN, and the key press and the walk are in the log. With a log folder that cannot be made the run went on
(50 ticks in 1 s) and reported why. Filling a row costs 18 us on the Mac.

Not yet measured: anything on the real duck. The cost of the second bus read on its loop (expected about +2.6 ms
on a 10.7 ms tick, outside the read-to-write path) and how often it is skipped are the first things to read from
the first real run (`show`: `tick_ms_mean`, `late_pct`, `extras_read_pct`). The simulated bus has no physics, so
the figure's joint traces mean nothing about the duck. The pack holds no export names, so a run identifies its
policies by the pack's SHA-256 only. The current's scale is not published by Feetech (counts, as elsewhere here).

## Animations

The 27 emotion clips of Duck Live (`app/animations`: Pollen's 14 and the 13 originals) play on the real duck from the
row of animations at the top of the **Control** pad. The duck plays them by itself: the bridge (`duck_agent.Player`) runs the clip on
a 50 Hz thread, writing the head's four joints and the beak (servos 30-34) and playing the sound, so it is the same
from the Mac, from the phone's server, or with nobody connected. The legs are never written: they stay limp or held.
Seven clips also walk, jump, flip or sit in the simulator (orange dot); here they play their head, beak and sound
only. Pressing the clip that is playing stops it and lets the head go limp. A clip needs a calibrated duck.

A clip plays at once whether or not the policies are driving (Control). In Setup the bridge moves the head itself.
Under the policies the clip jumps in on top, as in Duck Live: the legs stay with the policy, which balances against
the moving head; the head's four joints are mixed into the runner's own targets (eased in over 0.5 s and out over
0.4 s, offsets kept to Duck Live's standing limits) and the beak and sound are played beside it.

The clips live on the duck, on its always-writable partition (`/data/animations`), so they survive power cuts and
the read-only lock. `anim_pack.py` builds the pack and a Mac studio sends it when it connects, only if the duck's
differs (compared by SHA-256):

| | in the repo | on the duck |
|---|---|---|
| motion, 27 clips, 98.3 s | 257 kB (keyframe JSON, 50 ms steps) | 31 kB (head and beak only, redundant keyframes dropped) |
| sound | 8.23 MB (16-bit 48 kHz WAV) | 2.11 MB (8-bit mu-law, 24 kHz) |
| sent over the link | | 0.74 MB (zip) |

Measured on the real duck with `test_anim.py` on 2026-11-18 (`docs/anim_results.json`,
`docs/figures/anim_tracking.pdf`), servos 30-34 read 18 times a second while the clip plays:

| clip | length (s) | largest commanded swing (deg) | tracking error, RMS (deg) | request to motion (s) | off rest at the end (deg) |
|---|---|---|---|---|---|
| yes | 1.4 | 40.1 | 1.59 | 0.12 | 0.65 |
| hello | 2.6 | 34.4 | 0.83 | 0.14 | 0.14 |
| curious | 2.9 | 45.8 | 2.22 | 0.12 | 0.75 |

The worst single samples (6 to 23 deg) fall on the fastest moves, where a 50 ms sampling gap is itself a large
angle. Only these three clips were measured; the other 24 were not played by the test. Not checked: whether the
beak opens the right way on this duck, and head yaw (none of the three clips turns the head).

## Sensors

The studio connects to the duck by itself: every four seconds it looks for the duck on the Wi-Fi and, once it
answers, starts the agent there. If the duck drops off it reconnects when it is back. **Duck** in the top bar switches
this off (and on again). The link is to the duck's own computer over SSH (host alias `microduck` in `~/.ssh/config`,
key login). The studio copies `duck_agent.py` to `/tmp` there, runs it, and reads it through an SSH tunnel; the agent
listens on the duck's loopback only, needs no root and installs nothing. Closing the connection stops it.

The dock on the left has one button per sensor, and each sensor is also a part of the 3-D duck that lights up like a
servo housing (the lens, the speaker, the HAT board for the microphone; a small chip for the ToF and IMU, which have
no part in the model). Click either to turn
it on: its window opens in the dock and it draws into the scene. Colours match the servos: green = present,
red = missing, grey = duck not connected, blue = on.

| Sensor | Window | In the 3-D scene | On a real duck |
|---|---|---|---|
| Camera (IMX219) | live picture, 320 × 240 | the picture on the camera's view frustum (62° wide) | `v4l2-ctl` raw NV12 frames, converted in the page |
| ToF (VL53L5CX / L8CX) | 8 × 8 distance grid, nearest distance | 64 tiles at the measured distances (45° field), warm = near | ST's driver, built on the duck from `tof/` (`tof/build.sh`); the sensor goes on the HAT socket marked **IO2/3** (I2C3) |
| IMU | horizon; roll, pitch, yaw | the whole duck turns to the IMU's orientation | `imu_to_servo` board on the servo bus, ID 200 (gyro + quaternion at register 124) |
| Mic | waveform, level in dBFS | the HAT board glows with level | HAT codec TLV320AIC3104 on I2C3, set up from user space (the image has no driver); MIC3R → right channel |
| Speaker | Quack, Greet, Coo, Peck, Ask, Alarm: a random variant each | – | same codec's line outputs; plays the duck's voice bank (`duck_voice.py`, 64 sounds in `~/microduck/sounds`) |

Sensor positions come from the model (Pollen's `microduck_rl` MJCF): the `head_camera`, `tof` and `imu` sites, the
`speaker` part, and for the microphone the centre of the robot HAT board (`elec_rpi_robot_hat_pcb`), which carries
the MEMS microphone. Where on that board the microphone sits is not in the model. Both lookers are drawn facing +x from the standing pose; they do not follow the head joints yet.

**Servos through the duck.** When the servos are wired to the duck's own computer (the HAT, UART `/dev/ttyS2`) and
not to a USB adapter on the Mac, the port list gains a `duck` entry: packets then travel through the agent, and Scan,
Set ID, the live plots and the rest work as with an adapter. The duck's user must be allowed to open the port:
`sudo usermod -aG dialout radxa` on the duck, then reboot. Checked only against a fake servo on a pseudo-terminal
(with and without the transmit echo a half-duplex adapter produces), not on the duck.

### Setting up a freshly flashed duck

For the JoyandAI first-version image on a Radxa Zero 3W with the Zero Robot HAT (logins `radxa`/`radxa`,
`rock`/`rock`). Everything the sensors need, in order; each step is run once and survives reboots.

| Step | Command | Why |
|---|---|---|
| Wi-Fi (card in the Mac) | `sudo bash firmware/sd_wifi.sh` | writes the network to the card's `config` partition; the duck joins it on first boot |
| SSH key | `ssh-copy-id rock@radxa-zero3.local`, and a `Host microduck` entry in `~/.ssh/config` | the studio logs in without a password |
| Servo port, runtime sockets | `sudo usermod -aG robot,dialout rock` | `/dev/ttyS2` is `root:dialout`; the runtime's sockets are `root:robot` |
| I2C3 on header pins 3/5 | `sudo cp /boot/dtbo/rk3568-i2c3-m0.dtbo.disabled /boot/dtbo/zz-i2c3-m0.dtbo && sudo u-boot-update` | the HAT's audio codec (0x18) and ToF socket (0x29) are on I2C3; the image's `rk3568-i2s3-m0` overlay switches it off, so the copy must sort after it |
| Setup / Run switch, camera | `sudo bash /tmp/mode_install.sh` (copy `firmware/mode_install.sh` over first), then reboot | lets the studio pause the runtime; switches `mediad` off, which otherwise breaks the camera at every boot |

Everything else is done by the studio when it connects: it copies the agent and the voice synth, builds the ToF
reader (`tof/build.sh`), renders the voice bank, and programs the audio codec.

**Setup / Run.** The duck's own runtime (`robotd`, `tofd`) holds the servo bus (exclusively) and the ToF ranger.
The switch in the top bar pauses it (**Setup**: the studio has the servos, IMU and ToF) or starts it again
(**Run**). Connecting the `duck` servo port switches to Setup by itself. The runtime starts at every boot, so a
freshly booted duck is in Run. Pausing the runtime makes a standing duck go limp.

**What broke, and the fix for each** (all found on the real duck):

| Symptom | Cause | Fix |
|---|---|---|
| Speaker silent, microphone constant | no kernel driver for the TLV320AIC3104; only a "dummy" I2S card | `duck_agent.Sensors.codec` programs the codec over I2C (clock from BCLK through the PLL, DACs to line out, MIC3R to the right ADC); both streams 32-bit |
| Codec and ToF absent from every bus | I2C3 disabled by the I2S overlay | the `zz-i2c3-m0` overlay above |
| IMU absent | it is not on I2C: it is the `imu_to_servo` board, ID 200 on the servo bus | read register 124 (gyro + quaternion) through the bus |
| Servos unreachable | port permission; then the runtime holding the port | `dialout` group; Setup mode |
| ToF never produces a frame after the runtime has used it | a sensor that already has its firmware rejects a second upload for 60 s | `tof/main.c` ranges first and uploads only if no frame arrives |
| Camera: green noise at 0.1 fps, or bright bands | `mediad` leaves the sensor in a mode the ISP mishandles; stopping it does not undo that | `mediad` disabled; the camera then boots in its default mode (measured: 11 frames/s to the studio, clean) |
| Camera picture sideways | the sensor is mounted on its side | the page turns it a quarter turn clockwise |
| Board drops off the network | under-powered: a Mac USB port or an empty pack cannot hold it up | 5 V / 2 A or more on the `OTG` port; a steady (not blinking) LED means it has not booted |
| Studio connects, then loses the duck | the agent announced itself before it was listening; two studio copies killing each other's tunnel | announce after bind; a new `studio.py` takes over from an old one |

Measured on this duck in Setup mode: 15 servos answering at 16 readings/s over Wi-Fi, IMU live, ToF 13 frames/s
(VL53L5CX), camera 11 frames/s, a quack raising the microphone from -56 to -11 dBFS, pack at 8.0 V (83 %).
Not yet done on the real duck: joint calibration, the USB-cable link (`radxa-ncm` is enabled but fails to start),
and the three-way network setup in `net_install.sh`.

## Gains vs. responsiveness (measured)

`step_response.py` steps one servo ±10° under a grid of gains (RAM only, restored afterwards) and writes
`docs/figures/servo_step_response.pdf` + `docs/servo_step_response.json`. One unloaded HD1910 (ID 22), 7.6 V, one
step per setting, ~1400 Hz sampling:

| P / D | time to 90 % | overshoot | settled within 5 % |
|---|---|---|---|
| 5 / 20 (replica) | 261 ms | none | 345 ms |
| 10 / 20 | 124 ms | none | 170 ms |
| 20 / 20 | 51 ms | none | 56 ms |
| 32 / 40 (factory) | 35 ms | 4 % | 37 ms |
| 50 / 40 | 31 ms | 18 % | 63 ms |
| 80 / 40 | 30 ms | 27 % | 85 ms |

Above P ≈ 32 the servo is speed-limited: no faster, only more overshoot. The policies were trained with P 5.

## Calibration

**Calibrate** (top bar, needs all 15 servos) records where the duck's home pose is, so every joint angle and limit
is measured from the real mechanism instead of from an assumed 2048.

1. Press **Calibrate**: every servo goes limp. Lay the duck flat on its back and pose it by hand at home: every
   joint at the model's zero pose (legs straight, head straight, beak closed), which is the pose the 3-D duck shows.
2. Switch **Hold** on: every joint holds where it is. Click a joint and trim it with the slider or the ‹ › buttons
   (0.2° a step, at most ±22° from where it was caught). A joint's own torque switch releases just that joint to
   move it by hand again.
   If a joint cannot be trimmed, its home pose sits at the end of the servo's encoder (within 400 ticks of 0 or
   4095), where the servo cannot move both ways. The studio names such joints; select one and press **Zero**
   (Feetech offset calibration: the present position will read 2048), then switch its torque back on.
3. **Save**. Each joint's present servo position becomes its zero, the IMU's mounting rotation is taken from the
   same pose (a duck on its back has its nose pointing up), and the torque is released. **Cancel** changes nothing.

**Live pose.** Once a calibration is saved, the 3-D duck takes the real joints' angles: every state update (8-16
a second over Wi-Fi) each calibrated joint's angle is applied to the model, and the ID tags, sensor parts and
overlays move with it. `bake_duck.py` stores, for every part, the chain of joints that carries it; the page poses
a chain as the product, trunk first, of each joint's rotation about its baked zero-pose axis. While calibrating,
the model stays at home, as the target to match. The beak (34) has no joint in Pollen's model, so `bake_duck.py`
adds one the way the Duck Live app draws it: the two lower-beak meshes turn about the beak servo's axis, positive =
open. The iOS app does the same from the `jaw` entry in `duck.bin`.
Joint directions are still kissqy's, unmeasured: a joint that turns the wrong way on screen has the wrong sign.

The IMU can also be re-levelled alone: stand the duck upright on a level surface and press **Level** in the IMU
window. Hold the duck still for either; a reading with more than 3°/s of rotation is refused.

The result is one file, `~/microduck/calibration.json` on the duck, with a copy at `firmware/calibration.json`:
per joint `{name, id, zero_raw, direction, min_rad, max_rad, calibrated}` (the kissqy `hardware.json` fields) and
`imu.mount`, a quaternion. Joint angle = (ticks - zero_raw) x 360/4096 x direction. Only Save and Level write it;
the studio reads it when it connects to the duck and never changes it otherwise. After calibration the slider's
limits are the model's joint range about the calibrated zero and the card shows the joint angle.

IMU maths: the chip reports q_raw (sensor to its own gravity-aligned world). With the body at a known orientation
q_pose, mount = q_raw^-1 x q_pose, and from then on body orientation = q_raw x mount. Roll and pitch are absolute;
yaw has no reference and drifts.

Directions (`direction`) start as kissqy's and are not measured; with a zero-pose home they do not affect the
zeros, only which way positive angle runs. **Flip**, in a calibrated joint's card, reverses one: use it when the
joint on screen turns the opposite way to the real one. The choice is stored in the calibration and kept when
the duck is calibrated again. Checked against the mock only; no real duck has been calibrated yet.

