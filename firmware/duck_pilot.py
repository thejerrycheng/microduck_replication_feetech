"""The duck's pilot: which policy runs, with what command, from the driver's inputs. numpy only; runs on the duck.

It is the operator logic of Duck Live's simulator (app/engine.py `Duck.operate`) reduced to what a real duck can
measure. The simulator decides with the trunk's height, its velocity and the feet's contacts; the duck has an IMU
and its joints. So here "upright" is tilt alone, and what the simulator ends on a height or a contact ends on a
time (marked * below). Everything else is the same: roles, commands, speed levels, slew rates, thresholds.

    pilot = Pilot(policy_net.load("policies.npz"))
    pilot.drive(x, y, w, run)      # sticks, each -1..1: x sidestep right, y forward, w turn clockwise
    pilot.press("sit" | "jump" | "flip" | "faster" | "slower")
    target = pilot.tick(gyro, quat, jpos, jvel)      # every 20 ms -> 14 joint angles [rad] to send

`tick` takes the trunk's turn rate [rad/s, trunk frame], its orientation (w, x, y, z; z up), and the 14 joint
angles and speeds [rad, rad/s] in JOINTS order. The observation is the policies' own: gyro, gravity in the trunk
frame, joint angles minus the default pose, joint speeds, the previous action, the 13-wide command.

States: stand, walk, run, sprint, stopping, sit_down, seated, stand_up, jump, flip, recover.
  * sit_down -> seated and stand_up -> stand after SIT_S / STAND_S (simulator: trunk height)
  * stopping -> stand after STOP_S (simulator: or once the trunk is slow)
  * jump: launch flag off for 4 ticks, then on until 0.36 s, where the simulator's duck touches down (simulator:
    until the feet's contact says so); ends once upright and turning slowly for 0.1 s after that, or at
    JUMP_MAX_S (simulator: a quiet landing by contact, height and speed)
  * a fall is tilt past 60 deg for 0.2 s (simulator: or collapsed below a height)
A fall under the walk policy is left to it for GETUP_S (VelStand gets up by itself in the simulator); if the duck is
still down then, or fell under anything else, the stand-up policy takes over until upright for 0.5 s. A jump or flip that does not end upright goes to the stand-up policy too.
"""
import math

import numpy as np

JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
DEFAULT_POSE = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                         0.0, 0.0873, 0.4579, 0.0049, -0.4530])
DT = 0.02
LEVELS = [dict(walk=0.20, run=0.30, yaw=0.50), dict(walk=0.30, run=0.50, yaw=0.60), dict(walk=0.40, run=0.70, yaw=0.70),
          dict(walk=0.50, run=0.80, yaw=0.80), dict(walk=0.60, run="sprint", yaw=0.80)]
WALK_VX_BACK, WALK_VY_MAX, RUN_VX, SPRINT_VX = 0.4, 0.3, (0.2, 1.1), 1.2
VX_SLEW, YAW_SLEW, DEAD = 1.5, 4.0, 0.12
FALL_DEG, FALL_S, UPRIGHT_DEG, RECOVERED_S = 60.0, 0.2, 20.0, 0.5
STOP_S, SIT_S, STAND_S = 0.8, 1.0, 1.0
# The jump by the clock. In the simulator (10 starts) the launch flag goes on at 0.08 s, the feet leave at 0.24-0.26 s,
# touch down at 0.34-0.36 s (flag off), and the walk policy has the duck back at 0.44-0.56 s. Left running longer, the
# jump policy tips over. The duck has no foot contacts, so it uses those times.
JUMP_WAIT, JUMP_LAUNCH_S, JUMP_MIN_S, JUMP_MAX_S, QUIET_W, QUIET_S = 4, 0.28, 0.36, 1.0, 1.5, 0.1
FLIP_MIN_S, FLIP_MAX_S = 1.0, 3.0
# Getting up, on the real duck. In the simulator the walk policy is back on its feet a second after any fall. The
# real one sometimes lies still instead: the policy asks for a pose the legs cannot reach against the floor, nothing
# moves, so nothing changes what it sees and it asks for the same again (2026-11-18: tilt 94 deg, the same target for
# minutes). So the duck does not wait: down GETUP_S under the walk policy, the stand-up policy tries for GETUP_TRY_S,
# then the walk policy again, and so on. Two different policies push in different ways, which is what unsticks it.
GETUP_S, GETUP_TRY_S = 3.0, 5.0


class Pilot:
    def __init__(self, nets, level=3):
        self.nets, self.level = nets, level
        self.state, self.t, self.t0 = "stand", 0.0, 0.0
        self.stick, self.run, self.sit_pending = (0.0, 0.0, 0.0), False, False
        self.cmd, self.last = np.zeros(13), np.zeros(14)
        self.v = [0.0, 0.0, 0.0]
        self.down_t = self.up_t = self.quiet_t = None
        self.role, self.tilt, self.falls, self.note, self.down = "walk", 0.0, 0, "", False

    # ---- the driver ----
    def drive(self, x=0.0, y=0.0, w=0.0, run=False):
        self.stick = tuple(0.0 if abs(a) < DEAD else float(min(1.0, max(-1.0, a))) for a in (x, y, w))
        self.run = bool(run) and any(self.stick)

    def press(self, key):
        st = self.state
        if key in ("faster", "slower"):
            self.level = min(len(LEVELS), max(1, self.level + (1 if key == "faster" else -1)))
        elif key == "sit":
            if st in ("seated", "sit_down"):
                self._to("stand_up")
            elif st in ("stand", "stopping", "stand_up"):
                self._to("sit_down")
            elif st in ("walk", "run", "sprint"):
                self.sit_pending = True
        elif key in ("jump", "flip"):
            role = "jump" if key == "jump" else "roulade"
            if role in self.nets and st in ("stand", "stopping") and self.tilt < UPRIGHT_DEG:
                self._to(key)
                self.last[:] = 0                 # an episodic policy starts as its training episode does
                self.v = [0.0, 0.0, 0.0]
                self.up_t = self.quiet_t = None
            else:
                self.note = f"{key}: only from a stand"

    def _to(self, s):
        if s != self.state:
            self.state, self.t0 = s, self.t

    def _slew(self, vx, vy, yaw, lo=None):
        self.v[0] += min(VX_SLEW * DT, max(-VX_SLEW * DT, vx - self.v[0]))
        if lo is not None:
            self.v[0] = max(lo, self.v[0])
        self.v[1] += min(VX_SLEW * DT, max(-VX_SLEW * DT, vy - self.v[1]))
        self.v[2] += min(YAW_SLEW * DT, max(-YAW_SLEW * DT, yaw - self.v[2]))

    # ---- one control tick ----
    def tick(self, gyro, quat, jpos, jvel):
        self.t += DT
        w, x, y, z = quat
        grav = np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])      # (0, 0, -1) in the trunk frame
        self.tilt = math.degrees(math.acos(max(-1.0, min(1.0, -grav[2]))))
        self._operate(float(np.linalg.norm(gyro)))
        obs = np.concatenate([gyro, grav, np.asarray(jpos) - DEFAULT_POSE, jvel, self.last, self.cmd]).astype(np.float32)
        self.last = np.asarray(self.nets[self.role](obs), dtype=np.float64)
        return DEFAULT_POSE + self.last

    def _operate(self, turn):
        st, age = self.state, self.t - self.t0
        x, y, w = self.stick
        moving, lv = any(self.stick), LEVELS[self.level - 1]
        run = self.run and y > 0 and x == 0 and w == 0
        gait = ("sprint" if lv["run"] == "sprint" and "sprint" in self.nets else "run") if run else "walk"
        self.down_t = (self.down_t if self.down_t is not None else self.t) if self.tilt > FALL_DEG else None
        self.up_t = (self.up_t if self.up_t is not None else self.t) if self.tilt < UPRIGHT_DEG else None
        up_for = self.t - self.up_t if self.up_t is not None else 0.0
        self.cmd[:] = 0
        if self.down and self.tilt < UPRIGHT_DEG:
            self.down = False
        if self.down_t is not None and self.t - self.down_t >= FALL_S and not self.down and st not in ("jump", "flip", "recover"):
            self.falls += 1
            self.down = True                     # one fall until it is upright again
            self.sit_pending = False
            if st in ("stand", "walk", "stopping") and not GETUP_S:
                self.note = "fell: the walk policy gets up by itself"      # VelStand keeps its command and recovers
            elif st in ("stand", "walk", "stopping"):
                self.note = "fell: the walk policy first"                   # ... and the stand-up policy if it has not
            else:
                self._to("recover")
                self.last[:] = 0
                st, age = "recover", 0.0
        if (GETUP_S and self.down and st in ("stand", "walk", "stopping") and "recovery" in self.nets
                and self.down_t is not None and self.t - self.down_t >= GETUP_S):
            self._to("recover")                  # still down: the policy trained for exactly this takes over
            self.last[:] = 0
            st, age = "recover", 0.0
        if st == "recover":
            self.role = "recovery" if "recovery" in self.nets else "walk"
            self.v = [0.0, 0.0, 0.0]
            if up_for >= RECOVERED_S and age > RECOVERED_S:
                self._to("stand")
            elif GETUP_S and age > GETUP_TRY_S and self.role == "recovery":
                self._to("stand")                # not up yet: the walk policy's turn again
                self.down_t = self.t
            return
        if st == "jump":
            self.role = "jump"
            self.cmd[0] = 1.0 if JUMP_WAIT * DT <= age <= JUMP_WAIT * DT + JUMP_LAUNCH_S else 0.0
            quiet = age > JUMP_MIN_S and self.tilt < UPRIGHT_DEG and turn < QUIET_W
            self.quiet_t = (self.quiet_t if self.quiet_t is not None else self.t) if quiet else None
            if (self.quiet_t is not None and self.t - self.quiet_t >= QUIET_S) or age >= JUMP_MAX_S:
                self._end()
            return
        if st == "flip":
            self.role = "roulade"
            if (age >= FLIP_MIN_S and up_for >= RECOVERED_S) or age >= FLIP_MAX_S:
                self._end()
            return
        if st in ("seated", "sit_down"):
            if moving:
                self._to("stand_up")
            elif st == "sit_down" and age > SIT_S:
                self._to("seated")
        elif st == "stand_up" and age > STAND_S:
            self._to("stand")
        elif st == "stand" and moving:
            self._to(gait)
        elif st in ("walk", "run", "sprint"):
            self._to(gait if moving else "stopping")
        elif st == "stopping":
            if moving:
                self._to(gait)
            elif age > STOP_S:
                self._to("sit_down" if self.sit_pending else "stand")
                self.sit_pending = False
        st = self.state
        if st in ("walk", "stopping", "stand"):
            self.role = "walk"
            if st == "walk":                     # +vy and +yaw are to the duck's left
                self._slew(y * (lv["walk"] if y > 0 else min(WALK_VX_BACK, lv["walk"])), -x * min(WALK_VY_MAX, 0.7 * lv["walk"]), -w * lv["yaw"])
            else:
                self._slew(0.0, 0.0, 0.0)
            self.cmd[0:3] = self.v
        elif st in ("run", "sprint"):
            self.role = st
            if st == "sprint":
                self.v = [SPRINT_VX, 0.0, 0.0]
            else:
                self._slew(RUN_VX[1] if lv["run"] == "sprint" else lv["run"], 0.0, 0.0, lo=RUN_VX[0])
            self.cmd[0] = self.v[0]
        else:
            self.role = "sitstand"
            self.v = [0.0, 0.0, 0.0]
            self.cmd[0] = 1.0 if st in ("sit_down", "seated") else 0.0

    def _end(self):
        if self.tilt < UPRIGHT_DEG:
            self._to("stand")
        else:
            self.falls += 1
            self.down = True
            self._to("recover")
            self.last[:] = 0

    def status(self):
        return dict(state=self.state, role=self.role, level=self.level, tilt=round(self.tilt, 1), falls=self.falls,
                    cmd=[round(v, 2) for v in self.v], note=self.note)
