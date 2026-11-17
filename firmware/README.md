# firmware — Duck Servo Studio

Find, rename, test and live-monitor the Microduck's 15 Feetech HD1910 servos from a browser.

```bash
~/.venvs/microduck_live/bin/python firmware/studio.py          # real bus (needs pyserial)
~/.venvs/microduck_live/bin/python firmware/studio.py --mock   # no hardware
```

**Hot-plug:** twice a second the studio drops servos that have been silent for 1.2 s and pings the factory and
joint IDs for new ones, so swapping servos needs no Scan. A lone new servo opens by itself; a pulled adapter
reconnects when it returns.

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

**Zero:** makes the present pose read 2048 (Feetech offset calibration, instruction 0x0B).

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
3. **Save**. Each joint's present servo position becomes its zero, the IMU's mounting rotation is taken from the
   same pose (a duck on its back has its nose pointing up), and the torque is released. **Cancel** changes nothing.

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

Directions (`direction`) are not measured: they are kissqy's, and with a zero-pose home they do not affect the
zeros, only which way positive angle runs. Checked against the mock only; no real duck has been calibrated yet.

