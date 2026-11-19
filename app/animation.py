"""Duck Live animation layer: emotions, idle life and reactions on top of the policies.

The duck stays under its RL policies the whole time. This module only decides, 50 times a second,

* ``head``  -- the four neck/head joint offsets from the default pose [rad] (neck_pitch, head_pitch, head_yaw, head_roll;
  Pollen's convention: head_pitch > 0 = beak down, head_yaw > 0 = to the duck's left),
* ``mouth`` -- the beak opening, 0..1,
* ``twist`` -- a (vx, vy, wz) command for the walk policy while a clip asks for one (wiggle, side-step, step back),
* skills    -- "sit" / "jump" / "flip": the sit-stand, jump and front-flip policies, fired from a clip's timeline,

and ``engine.Duck`` applies them (``Duck.step``): the legs always come from the active policy, which sees the moving
head in its joint observations and balances against it. How the head offsets reach the servos depends on the policy
(``HEAD_MODE``; measured in ``bench_anim.py``, docs/duck_live_app.pdf): the VelStand walk does not track its head
command on this checkpoint, so the head servos are driven directly; the sit-stand policy tracks it, so seated the
offsets go through the policy's own head command.

Three sources of motion, blended:

1. **clips** (``animations/``, built by ``tools/make_animations.py``): Pollen Robotics' 14 open-source emotions and the
   Duck Live originals, with their sounds;
2. **idle life** (:class:`IdleLife`, after the ideas of Pollen's ``robotd/src/idle_head.rs``): glances, re-fixations,
   breathing, a drift that never quite stops, looking at the viewer (the page sends the camera position), leading
   turns with the head while walking, calming down and dozing off;
3. **reactions**: a clip chosen by what just happened (got up from a fall -> shakes it off, landed a jump -> "yep",
   landed a flip -> proud, let go after a pull -> "mmh?", bored -> yawns).

Deterministic for a given seed and input sequence (its own ``numpy`` generator, simulation time only).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANIM_DIR = os.path.join(HERE, "animations")

# head offset limits [rad] (neck_pitch, head_pitch, head_yaw, head_roll). Standing: beyond about -0.45 / +0.30 of neck
# the head's mass moves the centre of mass far enough that VelStand steps to stay up (measured; Pollen notes the same
# on the real duck). Seated the sit-stand policy holds more.
HEAD_LO_STAND = np.array([-0.45, -0.75, -1.20, -0.42])
HEAD_HI_STAND = np.array([0.30, 1.00, 1.20, 0.42])
HEAD_LO_SEATED = np.array([-0.80, -0.75, -1.50, -0.42])
HEAD_HI_SEATED = np.array([0.45, 1.00, 1.50, 0.42])
VIEWER_YAW_MAX = 2.0        # the duck looks back over its shoulder at a viewer up to this far round [rad]; further, it does not try
LOOK_YAW_MAX = 1.05         # how far the head itself turns for that
FACE_TURN_START = 0.7       # a clip that faces the viewer first turns the body (walk policy, yaw twist) when the viewer is further round ...
FACE_TURN_STOP = 0.3        # ... until it is within this
FACE_TURN_MAX_S = 4.5
HEAD_RATE = 12.0            # rad/s slew on the commanded head offsets (the clips peak near 7 rad/s)
MOUTH_RATE = 14.0           # 1/s: full open in ~70 ms
CLIP_FADE_S = 0.18          # cross-fade between idle life and a clip
W_FADE_S = 0.20             # fade of the head override itself (in/out of the states where it applies)
BODY_PITCH_TO_HEAD = 0.6    # Pollen's body_pitch (a bow) is shown as extra head pitch: the body-pose command is not usable here
# how the head offsets reach the servos, per policy role (None: the policy keeps its head)
HEAD_MODE = dict(walk="direct", sitstand="cmd")
TILT_RELEASE_DEG = 30.0     # past this tilt the policy gets its head back (it may need it to catch itself)
QUEUE_MAX_S = 6.0           # an emotion asked for while the duck is busy waits this long, then is dropped
SKILL_MAX_S = 6.0           # a jump / flip fired by a clip is given this long to finish


def ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6 * u - 15) + 10)


class Clip:
    """One emotion: keyframes (Pollen's format, linear between samples), sounds, skills, metadata."""

    def __init__(self, meta: dict, root: str = ANIM_DIR):
        self.meta = meta
        self.name = meta["name"]
        with open(os.path.join(root, meta["keyframes"])) as f:
            keys = json.load(f)["keyframes"]
        self.t = np.array([k["t"] for k in keys], dtype=np.float64)
        if len(self.t) == 0 or np.any(np.diff(self.t) <= 0):
            raise ValueError(self.name + ": keyframes empty or out of order")
        g = lambda k, n: float(k.get(n) or 0.0)   # noqa: E731
        self.head = np.array([[g(k, "neck"), g(k, "head_pitch") + BODY_PITCH_TO_HEAD * g(k, "body_pitch"), g(k, "head_yaw"), g(k, "head_roll")] for k in keys])
        self.mouth = np.clip(np.array([g(k, "mouth") for k in keys]), 0.0, 1.0)
        self.twist = np.array([(k.get("twist") or [0.0, 0.0, 0.0]) for k in keys], dtype=np.float64)
        self.has_twist = bool(np.any(self.twist != 0.0))
        self.skills = []
        for k in keys:                                   # a skill repeated on consecutive keyframes is one request
            s = k.get("skill")
            if s and (not self.skills or self.skills[-1][1] != s):
                self.skills.append((float(k["t"]), s))
        self.T = float(self.t[-1])
        self.sounds = list(meta.get("sounds", []))
        self.sound_at = float(meta.get("sound_at_s", 0.0))
        self.seated_ok = bool(meta.get("seated", False))
        self.face = bool(meta.get("face", False))
        self.stages = (meta.get("pose_joints") or {}).get("stages", [])
        self.needs = set(s for _, s in self.skills)

    def at(self, t: float):
        t = min(max(t, 0.0), self.T)
        i = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2)) if len(self.t) > 1 else 0
        j = min(i + 1, len(self.t) - 1)
        w = (t - self.t[i]) / (self.t[j] - self.t[i]) if j > i else 0.0
        return (self.head[i] + (self.head[j] - self.head[i]) * w, float(self.mouth[i] + (self.mouth[j] - self.mouth[i]) * w),
                self.twist[i] + (self.twist[j] - self.twist[i]) * w)


def load_library(root: str = ANIM_DIR) -> dict:
    """{name: Clip} in the manifest's order; an unreadable clip is skipped (and reported), not fatal."""
    out = {}
    path = os.path.join(root, "manifest.json")
    if not os.path.exists(path):
        return out
    with open(path) as f:
        man = json.load(f)
    for m in man["clips"]:
        try:
            out[m["name"]] = Clip(m, root)
        except Exception as e:  # noqa: BLE001
            print("[animation] skipped %s: %r" % (m.get("name"), e))
    return out


class IdleLife:
    """What the head does when nobody is animating it: never a sinusoid, never frozen.

    Channels: (neck, head_pitch, head_yaw, head_roll) + beak. A *glance* picks somewhere to look (mostly left / right,
    sometimes the middle, sometimes the viewer), moves there on a minimum-jerk curve and holds; while it holds the gaze
    *re-fixates* a couple of degrees around the spot; a slow *drift* and the *breath* ride on top. ``energy`` falls
    from 1 to 0.35 over the first minute of stillness (glances get rarer and smaller); seated and still it *dozes*.
    Walking, the glances shrink and the head leads the turn.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.reset()

    def reset(self):
        self.t = 0.0
        self.idle_t = 0.0
        self.g_from = np.zeros(5)
        self.g_to = np.zeros(5)
        self.g_t0 = 0.0
        self.g_travel = 0.3
        self.g_until = 0.6
        self.g_viewer = False
        self.refix = np.zeros(2)
        self.refix_to = np.zeros(2)
        self.refix_next = 1.0
        self.phase = self.rng.uniform(0, 2 * math.pi, 6)
        self.doze = 0.0
        self.turn = 0.0
        self.look = np.zeros(2)          # filtered gaze towards a point of interest (grab), yaw / pitch
        self.look_w = 0.0

    def step(self, dt, state, yaw_cmd, viewer, grab, mood):
        """-> (head[4], beak). ``viewer`` / ``grab``: (yaw, pitch) of that point seen from the head, or None."""
        r = self.rng
        self.t += dt
        t = self.t
        still = state in ("stand", "seated")
        self.idle_t = self.idle_t + dt if still else 0.0
        energy = (0.35 + 0.65 * math.exp(-self.idle_t / 45.0)) * (1.0 + 0.35 * mood)
        walking = state in ("walk", "stopping")
        seated = state == "seated"
        # dozing: seated and still for a while
        doze_to = 1.0 if (seated and self.idle_t > 25.0) else 0.0
        self.doze += (doze_to - self.doze) * min(1.0, dt / (4.0 if doze_to else 0.4))
        reach = energy * (0.45 if walking else 1.0) * (1.0 - 0.85 * self.doze)

        if t >= self.g_until:                               # a new glance
            cur = self.pose5(t)
            u = r.random()
            self.g_viewer = False
            if viewer is not None and u < 0.30 and abs(viewer[0]) < VIEWER_YAW_MAX and not walking:
                yaw, pitch = float(np.clip(viewer[0], -LOOK_YAW_MAX, LOOK_YAW_MAX)), float(np.clip(viewer[1], -0.45, 0.35))
                self.g_viewer = True
            elif u < 0.46:
                yaw, pitch = r.uniform(-0.06, 0.06), r.uniform(-0.05, 0.05)
            else:
                side = -np.sign(cur[2]) if (abs(cur[2]) > 0.12 and r.random() < 0.65) else r.choice([-1.0, 1.0])
                yaw = side * r.uniform(0.18, 0.62) * reach
                pitch = r.uniform(-0.22, 0.14) * reach
            roll = r.choice([-1.0, 1.0]) * r.uniform(0.10, 0.24) * min(1.0, reach + 0.3) if r.random() < 0.28 else 0.0
            neck = r.uniform(-0.09, 0.06) * reach
            beak = r.uniform(0.04, 0.16) if r.random() < 0.3 else 0.0
            to = np.array([neck, pitch, yaw, roll, beak])
            self.g_from, self.g_to, self.g_t0 = cur, to, t
            self.g_travel = r.uniform(0.22, 0.42) + 0.30 * abs(to[2] - cur[2])
            self.g_until = t + self.g_travel + r.uniform(0.7, 2.8) / (0.45 + 0.55 * min(1.0, energy)) * (1.0 + 2.0 * self.doze)
            self.refix_to[:] = 0.0
        if t >= self.refix_next:                            # the gaze shifts a little while it holds
            self.refix_next = t + r.uniform(0.35, 0.95)
            if t > self.g_t0 + self.g_travel:
                self.refix_to = np.array([r.uniform(-0.035, 0.035), r.uniform(-0.02, 0.02)]) * (0.4 + 0.6 * reach)
        self.refix += (self.refix_to - self.refix) * min(1.0, dt / 0.06)

        p = self.pose5(t)
        if self.g_viewer and viewer is not None and t > self.g_t0 + self.g_travel:    # keep the viewer in sight as either moves
            p[2] += 0.15 * (float(np.clip(viewer[0], -LOOK_YAW_MAX, LOOK_YAW_MAX)) - p[2])
            self.g_to[2] = p[2]
        head = p[:4].copy()
        beak = float(p[4])
        head[2] += self.refix[0]
        head[1] += self.refix[1]
        ph = self.phase                                       # drift: incommensurate slow sines, a degree or so
        head[2] += 0.020 * math.sin(0.37 * t + ph[0]) + 0.012 * math.sin(0.91 * t + ph[1])
        head[1] += 0.012 * math.sin(0.53 * t + ph[2])
        head[3] += 0.010 * math.sin(0.29 * t + ph[3])
        breath_T = 3.4 + 1.6 * self.doze - 0.5 * max(0.0, mood)
        b = math.sin(2 * math.pi * t / breath_T + ph[4])
        head[0] += (0.016 + 0.030 * self.doze) * b            # the chest rises: the neck follows ...
        head[1] -= (0.010 + 0.012 * self.doze) * b            # ... and the head stays level
        head[1] += 0.10 * max(0.0, -mood) + 0.50 * self.doze - 0.04 * max(0.0, mood)   # sad: head low; happy: chin up
        head[3] += 0.12 * self.doze
        beak += 0.06 * self.doze * max(0.0, -b)
        # walking: the head leads the turn
        self.turn += ((yaw_cmd if walking else 0.0) - self.turn) * min(1.0, dt / 0.18)
        head[2] += 0.38 * self.turn
        head[3] += 0.07 * self.turn
        # something has hold of the duck: look at it
        want = 1.0 if grab is not None else 0.0
        self.look_w += (want - self.look_w) * min(1.0, dt / 0.12)
        if grab is not None:
            self.look += (np.array([np.clip(grab[0], -LOOK_YAW_MAX, LOOK_YAW_MAX), np.clip(grab[1], -0.5, 0.6)]) - self.look) * min(1.0, dt / 0.10)
        if self.look_w > 1e-3:
            head[2] += self.look_w * (self.look[0] - head[2])
            head[1] += self.look_w * (self.look[1] - head[1])
            beak = max(beak, 0.3 * self.look_w)
        return head, beak

    def pose5(self, t):
        return self.g_from + (self.g_to - self.g_from) * ease((t - self.g_t0) / max(self.g_travel, 1e-3))


class Animator:
    """Owns the clip being played, the idle life and the reactions; ticked once per control step by the duck."""

    def __init__(self, duck, seed: int = 0, lively: bool = False, react: bool = False, library: dict | None = None):
        self.duck = duck
        self.lib = library if library is not None else load_library()
        self.seed = int(seed)
        self.lively = bool(lively)       # idle life + locomotion head
        self.react = bool(react)         # automatic reactions
        self.sfx_id = 0
        self.sfx = None                  # the last sound started: dict(id, file, clip)
        self.sfx_cut = 0                 # bumped when a clip is cut short (the page fades its sound)
        self.viewer_pos = None           # the page's camera, world frame
        self.pin = None                  # bench_anim.py: a fixed head offset held as if a clip asked for it
        self.reset()

    # ---- state ------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.idle = IdleLife(self.rng)
        self.clip = None
        self.clip_t = 0.0
        self.clip_fired = 0              # skills fired so far
        self.clip_sound = False
        self.wait = None                 # the skill the clip is waiting on ("jump" / "flip") and since when
        self.stage = -1                  # play-dead: index of the pose stage reached
        self.queue = None                # (name, since, auto)
        self.facing = False              # turning the body towards the viewer before a clip that faces them
        self.cw = 0.0                    # clip weight in the head blend
        self.clip_head = np.zeros(4)
        self.clip_mouth = 0.0
        self.head = np.zeros(4)          # the output, slewed and limited
        self.mouth = 0.0
        self.w = 0.0                     # override weight
        self.twist = None
        self.mood = 0.0
        self.last = None                 # dict(name, t) of the last finished clip
        self.next_yawn = None
        self.grab_t = 0.0
        self.cool = 0.0
        self.played = []                 # (sim time, name, auto) log, for the bench

    def names(self):
        return list(self.lib.keys())

    def manifest(self):
        """What the page needs to draw the palette."""
        out = []
        for c in self.lib.values():
            m = c.meta
            ok = all(dict(jump="jump", flip="roulade", sit="sitstand").get(s, "walk") in self.duck.policies for s in c.needs)
            out.append(dict(name=c.name, label=m.get("label", c.name), emoji=m.get("emoji", ""), fx=m.get("fx", ""), key=m.get("key", ""),
                            group=m.get("group", ""), T=round(c.T, 2), description=m.get("description", ""), uses=m.get("uses", []),
                            sounds=c.sounds, available=ok))
        return out

    # ---- requests ---------------------------------------------------------------------------------------------------------
    def play(self, name: str, auto: bool = False) -> bool:
        """Ask for an emotion. It starts as soon as the duck can (standing still, or seated for the clips that allow
        it); a duck that is walking stops first, a seated one stands up if the clip needs its legs."""
        if name not in self.lib:
            return False
        d = self.duck
        if d.state == "dead":
            d.wake()
            return False
        if self.clip is not None and not auto:
            self.cancel(cut=True)
        elif self.clip is not None:
            return False
        self.queue = (name, d.data.time, auto)
        return True

    def cancel(self, cut: bool = False):
        if self.clip is not None:
            if cut and self.clip_t < self.clip.T - 0.15:
                self.sfx_cut += 1
            self.finish(completed=False)
        self.queue = None

    def finish(self, completed: bool):
        c = self.clip
        if c is None:
            return
        if completed:
            self.mood = float(c.meta.get("mood", self.mood * 0.5))
        self.last = dict(name=c.name, t=self.duck.data.time, completed=completed)
        self.clip = None
        self.wait = None
        self.twist = None
        self.idle.g_until = self.idle.t + 0.25        # pick a fresh glance right after

    # ---- one tick (before the duck's operator logic) -----------------------------------------------------------------------
    def update(self):
        d = self.duck
        dt = d.control_dt
        now = d.data.time
        st = d.state
        self.cool = max(0.0, self.cool - dt)
        self.mood *= math.exp(-dt / 25.0)
        for ev in d.take_events():
            self.on_event(ev)
        moving = bool(d.held)
        # ---- the user is driving: a clip gives way (play-dead is woken instead)
        if moving and self.clip is not None and st != "dead":
            self.cancel(cut=True)
        if moving and self.queue is not None and not self.queue[2]:
            self.queue = None
        # ---- start what is queued, when the duck can
        pre_twist = None
        if self.queue is None:
            self.facing = False
        if self.queue is not None and self.clip is None:
            name, since, auto = self.queue
            c = self.lib[name]
            if now - since > QUEUE_MAX_S or (auto and moving):
                self.queue = None
            elif st == "stand" and not d.down and d.upright and not moving:
                v = self.direction_to(self.viewer_pos) if c.face else None
                if v is not None and now - since < FACE_TURN_MAX_S and abs(v[0]) > (FACE_TURN_STOP if self.facing else FACE_TURN_START):
                    self.facing = True               # turn round to face the viewer first: the walk policy, turning on the spot
                    pre_twist = np.array([0.0, 0.0, math.copysign(min(1.0, 0.45 + 0.6 * abs(v[0])), v[0])])
                else:
                    self.facing = False
                    self.start(c, auto)
            elif st == "seated" and c.seated_ok and not c.has_twist and not c.needs:
                self.start(c, auto)
            elif st == "seated" and not auto and not d.sit_toggled_by_anim:
                d.sit_toggled_by_anim = True
                d.toggle_sit()                          # stand up first
            elif st in ("walk", "run", "sprint") and auto:
                self.queue = None
        # ---- advance the clip
        self.twist = None
        if self.clip is not None:
            c = self.clip
            if st in ("recover", "fallen") or (d.down and st != "dead"):
                self.cancel(cut=True)                   # it fell: the policies deal with that first
            elif self.wait is not None:
                kind, since = self.wait
                if st in ("stand", "stopping") and now - since > 0.3:
                    self.wait = None                    # landed: the timeline goes on
                elif now - since > SKILL_MAX_S:
                    self.cancel()
            else:
                self.clip_t += dt
                if not self.clip_sound and self.clip_t >= c.sound_at and c.sounds:
                    self.clip_sound = True
                    self.say(c.sounds[int(self.rng.integers(len(c.sounds)))], c.name)
                while self.clip is not None and self.clip_fired < len(c.skills) and self.clip_t >= c.skills[self.clip_fired][0]:
                    self.fire(c.skills[self.clip_fired][1])
                    self.clip_fired += 1
                if self.clip is not None:
                    for i, sg in enumerate(c.stages):
                        if i > self.stage and self.clip_t >= sg["at_s"]:
                            self.stage = i
                            d.pose_stage(sg)
                    if self.clip_t >= c.T:
                        self.finish(completed=True)
                    else:
                        self.clip_head, self.clip_mouth, tw = c.at(self.clip_t)
                        if c.has_twist and st in ("stand", "stopping"):
                            self.twist = tw
        if self.clip is None and pre_twist is not None:
            self.twist = pre_twist
        # ---- idle life / reactions
        idle_head, idle_beak = np.zeros(4), 0.0
        if self.lively:
            viewer = self.direction_to(self.viewer_pos)
            grab = self.direction_to(d.perturb["target"]) if d.perturb is not None else (viewer if self.facing else None)
            idle_head, idle_beak = self.idle.step(dt, st, d.yaw_cmd, viewer, grab, self.mood)
        if self.react:
            self.reactions(dt, st, moving)
        # ---- blend, face the viewer, slew, limit
        playing = (self.clip is not None and self.wait is None) or self.pin is not None
        self.cw += ((1.0 if playing else 0.0) - self.cw) * min(1.0, dt / CLIP_FADE_S * 2.0)
        clip_head = self.clip_head.copy() if self.pin is None else np.asarray(self.pin, dtype=np.float64).copy()
        if self.clip is not None and self.clip.face and self.viewer_pos is not None:
            v = self.direction_to(self.viewer_pos)
            if v is not None:
                clip_head[2] += float(np.clip(v[0], -1.2, 1.2))
                clip_head[1] += float(np.clip(v[1], -0.4, 0.3))
        want = self.cw * clip_head + (1.0 - self.cw) * idle_head
        beak = self.cw * self.clip_mouth + (1.0 - self.cw) * idle_beak
        seated = st in ("seated", "sit_down", "dead")
        lo, hi = (HEAD_LO_SEATED, HEAD_HI_SEATED) if seated else (HEAD_LO_STAND, HEAD_HI_STAND)
        want = np.clip(want, lo, hi)
        self.head += np.clip(want - self.head, -HEAD_RATE * dt, HEAD_RATE * dt)
        self.mouth += float(np.clip(beak - self.mouth, -MOUTH_RATE * dt, MOUTH_RATE * dt))
        # ---- where the override applies
        active = (self.clip is not None or self.lively or self.pin is not None) and st in ("stand", "stopping", "walk", "seated", "sit_down") \
            and not d.down and d.tilt_deg < TILT_RELEASE_DEG and HEAD_MODE.get(d.policy_role) is not None
        if st == "sit_down" and self.clip is None and self.pin is None:
            active = False
        self.w += ((1.0 if active else 0.0) - self.w) * min(1.0, dt / W_FADE_S * 2.0)
        if self.w < 1e-3:
            self.w = 0.0

    def start(self, c: Clip, auto: bool):
        d = self.duck
        self.queue = None
        d.sit_toggled_by_anim = False
        self.clip, self.clip_t, self.clip_fired, self.clip_sound, self.wait, self.stage = c, 0.0, 0, False, None, -1
        self.clip_head, self.clip_mouth, _ = c.at(0.0)
        self.played.append((round(d.data.time - d.sim_t0, 2), c.name, auto))

    def fire(self, skill: str):
        """A clip's skill: the same thing the S / space / R keys do, when the duck is in a state to do it."""
        d = self.duck
        if skill == "sit":
            if d.state in ("stand", "stopping"):
                d.state_to("sit_down")
        elif skill in ("jump", "flip"):
            role = "jump" if skill == "jump" else "roulade"
            if role in d.policies and d.state in ("stand", "stopping") and d.upright:
                d.start_behavior(role)
                d.behavior_by_anim = True
                self.wait = (skill, d.data.time)
        # anything else: ignored, like Pollen's player

    def say(self, file: str, clip: str = ""):
        self.sfx_id += 1
        self.sfx = dict(id=self.sfx_id, file=file, clip=clip)

    # ---- reactions --------------------------------------------------------------------------------------------------------
    def on_event(self, ev: str):
        if ev == "reset":
            return
        if not self.react or self.clip is not None or self.cool > 0:
            return
        pick = dict(back_up="dizzy", jump_ok="yes_fast", flip_ok="proud", jump_fail="mmh").get(ev)
        if pick and pick in self.lib:
            self.queue = (pick, self.duck.data.time, True)
            self.cool = 3.0

    def reactions(self, dt, st, moving):
        d = self.duck
        now = d.data.time
        # let go after a pull: "mmh?"
        if d.perturb is not None:
            self.grab_t += dt
        else:
            if self.grab_t > 1.0 and self.clip is None and self.cool <= 0 and st == "stand" and "mmh" in self.lib:
                self.queue = ("mmh", now, True)
                self.cool = 6.0
            self.grab_t = 0.0
        # bored: a yawn now and then
        if st == "stand" and not moving and self.clip is None and self.queue is None and d.perturb is None:
            if self.next_yawn is None:
                self.next_yawn = self.idle.idle_t + float(self.rng.uniform(40.0, 70.0))
            elif self.idle.idle_t >= self.next_yawn and "yawn" in self.lib:
                self.queue = ("yawn", now, True)
                self.next_yawn = self.idle.idle_t + float(self.rng.uniform(50.0, 90.0))
        elif st != "stand" or moving:
            self.next_yawn = None

    # ---- geometry ---------------------------------------------------------------------------------------------------------
    def direction_to(self, point):
        """(yaw, pitch) of a world point seen from the head, as head-joint offsets: yaw > 0 = to the duck's left,
        pitch > 0 = below the horizon (beak down). None when there is no point or it is on top of the duck."""
        if point is None:
            return None
        d = self.duck
        hp = d.data.xpos[d.head_body]
        v = np.asarray(point, dtype=np.float64) - hp
        h = math.hypot(v[0], v[1])
        if h < 0.05:
            return None
        q = d.data.qpos[d.free_qadr + 3:d.free_qadr + 7]
        trunk_yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
        yaw = (math.atan2(v[1], v[0]) - trunk_yaw + math.pi) % (2 * math.pi) - math.pi
        return yaw, -math.atan2(v[2], h)

    def snapshot(self):
        c = self.clip
        q = self.queue
        return dict(
            mouth=round(self.mouth, 3),
            anim=(dict(name=c.name, label=c.meta.get("label", c.name), emoji=c.meta.get("emoji", ""), fx=c.meta.get("fx", ""),
                       t=round(self.clip_t, 2), T=round(c.T, 2), wait=self.wait[0] if self.wait else None) if c is not None else None),
            queued=q[0] if q else None, sfx=self.sfx, sfx_cut=self.sfx_cut, mood=round(self.mood, 2), doze=round(self.idle.doze, 2),
            lively=self.lively, react=self.react, head=[round(float(x), 3) for x in self.head], head_w=round(self.w, 2))
