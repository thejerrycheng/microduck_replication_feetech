#!/usr/bin/env python3
"""Build app/animations/: the emotion clips Duck Live plays (keyframes + sounds + manifest.json).

  ~/.venvs/microduck_live/bin/python app/tools/make_animations.py [--pollen /path/to/microduck_emotions]

Two sources, one format (Pollen's: one sample per step, head deltas in rad, mouth 0..1, see animations/README.md):

* **Pollen Robotics' 14 emotions** (github.com/pollen-robotics/microduck_emotions, Apache-2.0): copied as they are
  (keyframes and wavs; the repo is cloned to a temp dir when --pollen is not given). Nothing is edited: the runtime
  (animation.py) applies this plant's head limits when it plays them.
* **Duck Live originals** (this file): motion written as control points + oscillations, voice synthesised with
  Pollen's own voice synth (tools/quack.py, the robot's personality seed), so they sound like the same duck. The
  beak track is lip-synced to the generated sound (RMS envelope), plus whatever the clip opens by hand. Unlike
  Pollen's set these also use the **policies**: a `twist` per keyframe (the walk policy's command: wiggles,
  side-steps, hop back) and `skill` events ("jump", "flip", "sit": the jump / front-flip / sit-stand policies).

Deterministic: fixed seeds, no wall clock. Needs numpy + scipy (scipy only here, not at run time).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "animations")
sys.path.insert(0, HERE)
import quack as Q  # noqa: E402

POLLEN_URL = "https://github.com/pollen-robotics/microduck_emotions"
POLLEN_COMMIT = "431c73df8a9d3cb6b6067d09ced6c169fdfa59fd"
STEP = 0.05                     # originals: one keyframe every 50 ms (Pollen's files: 100 ms)
P = Q.Personality(Q.ROBOT_SEED)
PC = P.pitch_center_hz          # 238 Hz


# ---- voice helpers ----------------------------------------------------------------------------------------------------
def st(n: float) -> float:
    """n semitones above the duck's pitch centre, in Hz."""
    return Q.semitones(PC, n)


def quack(dur, contour_st, env=("expdecay", None, None), variant=0, tag="live", gain=1.0, **kw):
    """One vocalisation; ``contour_st`` = [(fraction of dur, semitones from the pitch centre), ...]."""
    contour = [(f * dur, st(s)) for f, s in contour_st]
    sig = Q.quack(P, dur, contour, env, rng=P.variant_rng(tag, variant), **kw)
    return gain * Q.normalise(sig, -3.0)


def chirp(s0, s1, dur=0.13, variant=0, gain=0.9):
    return quack(dur, [(0, s0), (0.25, s1), (1, s1 - 1.5)], ("expdecay", 0.008, dur * 0.45), variant, "chirp", gain, am_scale=0.5)


def wak(s=0.0, dur=0.2, variant=0, gain=1.0):
    return quack(dur, [(0, s + 2), (0.08, s + 4), (1, s - 3)], ("expdecay", 0.006, dur * 0.4), variant, "wak", gain, click_gain=0.15)


def coo(s0, s1, dur=0.5, variant=0, gain=0.7):
    return quack(dur, [(0, s0), (0.6, s1), (1, s1 - 0.5)], ("bell", dur * 0.3, dur * 0.4), variant, "coo", gain,
                 mods=dict(quackiness=0.1, brightness=0.15), am_scale=0.2, breath_scale=1.5)


def tone(hz, dur, variant=0, gain=0.9):
    """A sung note at an absolute pitch (the duck sings two octaves around its centre)."""
    sig = Q.quack(P, dur, [(0, hz * 0.97), (0.05, hz), (dur, hz)], ("expdecay", 0.012, dur * 0.7), rng=P.variant_rng("sing", variant),
                  mods=dict(quackiness=0.25), am_scale=0.35)
    return gain * Q.normalise(sig, -3.0)


def noise(dur, env_decay, seed, lo=0.0, gain=1.0):
    """A breath / sneeze burst: pink noise under an exponential decay."""
    n = np.random.default_rng(seed)
    t = Q.t_axis(dur)
    x = Q.pink_noise(len(t), n) * (1 - lo) + lo * n.standard_normal(len(t)) * 0.4
    return gain * Q.normalise(x * Q.expdecay(t, 0.01, env_decay), -3.0)


# ---- motion builder -----------------------------------------------------------------------------------------------------
CH = ("neck", "head_pitch", "head_yaw", "head_roll", "mouth", "vx", "vy", "wz")


def ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)       # minimum-jerk


class Clip:
    """Control points per channel (minimum-jerk between them) + additive oscillation bursts + sounds + skills."""

    def __init__(self, name, T, **meta):
        self.name, self.T, self.meta = name, float(T), meta
        self.pts = dict((c, [(0.0, 0.0)]) for c in CH)
        self.oscs = []
        self.sounds = []
        self.skills = []

    def key(self, ch, *tv):
        """key("head_yaw", t0, v0, t1, v1, ...): control points."""
        for i in range(0, len(tv), 2):
            self.pts[ch].append((float(tv[i]), float(tv[i + 1])))
        return self

    def hold(self, ch, t0, t1, v, ramp=0.15):
        return self.key(ch, t0, v, t1, v, t1 + ramp, 0.0) if t0 <= ramp else self.key(ch, t0 - ramp, self.value(ch, t0 - ramp), t0, v, t1, v, t1 + ramp, 0.0)

    def value(self, ch, t):
        return float(self._base(ch, np.array([t]))[0])

    def osc(self, ch, t0, t1, amp, hz, phase=0.0, decay=0.0, square=0.0):
        """amp * sin(2 pi hz (t - t0) + phase) inside [t0, t1], Hann-faded at both ends; ``decay`` [1/s] damps it,
        ``square`` in 0..1 flattens the sine towards a square wave (snappier swings)."""
        self.oscs.append((ch, t0, t1, amp, hz, phase, decay, square))
        return self

    def say(self, t, sig):
        self.sounds.append((float(t), np.asarray(sig, dtype=np.float64)))
        return self

    def skill(self, t, name):
        self.skills.append((float(t), name))
        return self

    def _base(self, ch, t):
        pts = sorted(self.pts[ch])
        ts, vs = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2) if len(ts) > 1 else np.zeros(len(t), dtype=int)
        if len(ts) == 1:
            return np.full(len(t), vs[0])
        u = (t - ts[i]) / np.maximum(ts[i + 1] - ts[i], 1e-9)
        return vs[i] + (vs[i + 1] - vs[i]) * ease(u)

    def sample(self):
        t = np.round(np.arange(0.0, self.T + 1e-9, STEP), 3)
        out = dict((c, self._base(c, t)) for c in CH)
        for ch, t0, t1, amp, hz, phase, decay, square in self.oscs:
            w = np.where((t >= t0) & (t <= t1), 1.0, 0.0)
            fade = min(0.12, 0.5 / hz, (t1 - t0) / 4)
            w *= np.clip((t - t0) / fade, 0, 1) * np.clip((t1 - t) / fade, 0, 1)
            s = np.sin(2 * math.pi * hz * (t - t0) + phase)
            if square > 0:
                s = np.tanh(s * (1 + 6 * square)) / math.tanh(1 + 6 * square)
            out[ch] = out[ch] + w * amp * s * np.exp(-decay * np.maximum(t - t0, 0))
        return t, out

    def audio(self):
        if not self.sounds:
            return None
        sig = Q.mix(self.sounds, total=self.T)
        return Q.normalise(sig, -3.0) * self.meta.get("gain", 1.0)

    def build(self):
        t, out = self.sample()
        sig = self.audio()
        if sig is not None:                                   # lip-sync: the beak follows the sound's loudness
            win = int(0.03 * Q.SR)
            env = np.sqrt(np.convolve(sig ** 2, np.ones(win) / win, mode="same"))
            env = env / (env.max() + 1e-9)
            lead = 0.03                                       # the servo needs a moment: open slightly early
            lip = np.interp(t + lead, np.arange(len(env)) / Q.SR, env, right=0.0) ** 0.6
            out["mouth"] = np.maximum(out["mouth"], self.meta.get("lip", 0.85) * lip)
        out["mouth"] = np.clip(out["mouth"], 0.0, 1.0)
        skills = dict((round(round(ts / STEP) * STEP, 3), n) for ts, n in self.skills)
        keys = []
        for i, ti in enumerate(t):
            keys.append(dict(t=float(ti), neck=round(float(out["neck"][i]), 4), head_pitch=round(float(out["head_pitch"][i]), 4),
                             head_yaw=round(float(out["head_yaw"][i]), 4), head_roll=round(float(out["head_roll"][i]), 4),
                             body_pitch=0.0, body_z=0.0,
                             twist=[round(float(out["vx"][i]), 3), round(float(out["vy"][i]), 3), round(float(out["wz"][i]), 3)],
                             skill=skills.get(float(ti)), mouth=round(float(out["mouth"][i]), 3)))
        return keys, sig


# ---- the originals --------------------------------------------------------------------------------------------------------
def originals():
    clips = []

    # hello: looks at you, two cheerful quacks with a head tilt each way, then a nod
    c = Clip("hello", 2.6, label="Hello", emoji="👋", fx="👋", face=True, group="live", seated=True,
             description="Looks at you, tilts its head one way then the other on two cheerful quacks, and nods.")
    c.key("head_pitch", 0.3, -0.22, 1.35, -0.22, 1.6, 0.38, 1.85, -0.08, 2.3, 0.0)
    c.key("head_roll", 0.3, 0.0, 0.5, 0.26, 0.85, 0.26, 1.05, -0.26, 1.35, -0.26, 1.6, 0.0)
    c.key("neck", 0.3, -0.08, 1.4, -0.08, 2.2, 0.0)
    c.say(0.45, quack(0.22, [(0, 1), (0.3, 5), (1, 3)], ("expdecay", 0.008, 0.11), 1)).say(1.0, quack(0.24, [(0, 3), (0.3, 8), (1, 6)], ("expdecay", 0.008, 0.12), 2))
    c.say(1.6, chirp(6, 9, 0.12, 3, 0.6))
    clips.append(c)

    # happy: a tail-wiggle dance on the walk policy (alternating yaw twist), chirps climbing, then a jump
    c = Clip("happy", 3.3, label="Happy", emoji="😄", fx="✨", group="live", mood=0.8,
             description="Wiggles on the spot (the walk policy turning left-right), chirps climbing in pitch, then jumps for joy (the jump policy).")
    c.key("head_pitch", 0.25, -0.3, 2.3, -0.3, 2.5, 0.0, 2.75, 0.3, 3.0, -0.1, 3.3, 0.0)
    c.osc("wz", 0.3, 2.3, 0.8, 1.6, square=0.5).osc("head_roll", 0.3, 2.3, 0.24, 1.6, phase=math.pi, square=0.3)
    c.osc("head_yaw", 0.3, 2.3, 0.2, 1.6, phase=math.pi / 2)
    for i in range(7):
        c.say(0.3 + i * 0.3125, chirp(2 + i * 1.3, 5 + i * 1.3, 0.12, i))
    c.skill(2.5, "jump").say(2.5, quack(0.45, [(0, 4), (0.5, 12), (1, 14)], ("bell", 0.05, 0.2), 9, "whee"))
    c.say(2.75, chirp(9, 12, 0.14, 11, 0.7))
    clips.append(c)

    # dance: side-steps to a beat (walk policy, lateral twist), head bobbing, a little tune
    beat = 0.6
    c = Clip("dance", 8 * beat + 1.2, label="Dance", emoji="💃", fx="🎵", group="live", mood=0.6,
             description="Side-steps left and right on a beat (the walk policy's lateral command), bobs its head and sings an eight-note tune.")
    tune = ["C4", "E4", "G4", "E4", "C4", "E4", "G4", "C5"]
    for i in range(8):
        tb = 0.3 + i * beat
        side = 1.0 if (i // 2) % 2 == 0 else -1.0
        c.key("vy", tb, 0.2 * side, tb + beat - 0.1, 0.2 * side)
        c.key("head_yaw", tb + 0.05, 0.32 * side, tb + beat - 0.1, 0.32 * side)
        c.key("head_pitch", tb, 0.26, tb + 0.22, -0.12, tb + beat - 0.06, -0.12)
        c.key("head_roll", tb + 0.1, 0.12 * (1 if i % 2 else -1))
        c.say(tb, tone(Q.note(tune[i]), 0.34, i))
    te = 0.3 + 8 * beat
    c.key("vy", te, 0.0).key("head_yaw", te + 0.1, 0.0).key("head_roll", te + 0.1, 0.0)
    c.key("head_pitch", te + 0.15, -0.5, te + 0.7, -0.5, te + 0.9, 0.0)
    c.say(te + 0.1, tone(Q.note("G4"), 0.2, 20)).say(te + 0.3, tone(Q.note("C5"), 0.5, 21))
    clips.append(c)

    # sing: a short phrase, head swaying
    c = Clip("sing", 5.0, label="Sing", emoji="🎤", fx="🎶", group="live", seated=True, mood=0.5,
             description="Beak up, sways and sings a ten-note phrase; the beak follows every note.")
    phrase = [("E4", .3), ("G4", .3), ("A4", .3), ("G4", .6), ("E4", .3), ("G4", .3), ("A4", .3), ("C5", .6), ("A4", .3), ("G4", .9)]
    tcur = 0.4
    for i, (nm, d) in enumerate(phrase):
        c.say(tcur, tone(Q.note(nm), d * 0.95, 30 + i))
        tcur += d
    c.key("head_pitch", 0.35, -0.35, 4.4, -0.35, 4.8, 0.0).osc("head_roll", 0.4, 4.5, 0.2, 0.55).osc("head_yaw", 0.4, 4.5, 0.25, 0.275, phase=1.0)
    c.osc("head_pitch", 0.4, 4.5, 0.07, 1.67)
    clips.append(c)

    # love: shy
    c = Clip("love", 3.6, label="Love", emoji="🥰", fx="❤️", face=True, group="live", seated=True, mood=0.7,
             description="Looks at you, then goes shy: head tilted and turned away, a slow sway on two soft coos, and a peek back.")
    c.key("head_roll", 0.5, 0.0, 0.9, 0.3, 2.4, 0.3, 2.8, 0.1, 3.3, 0.0)
    c.key("head_pitch", 0.4, -0.15, 0.9, 0.35, 2.4, 0.35, 2.8, -0.15, 3.3, 0.0)
    c.key("head_yaw", 0.5, 0.0, 0.95, 0.45, 2.4, 0.45, 2.8, 0.0)
    c.osc("head_roll", 1.0, 2.4, 0.08, 0.9).osc("neck", 1.0, 2.4, 0.05, 0.9, phase=1.5)
    c.say(1.05, coo(-3, 1, 0.5, 0)).say(1.75, coo(-1, 4, 0.55, 1)).say(2.85, chirp(7, 10, 0.12, 5, 0.6))
    clips.append(c)

    # proud: ta-daa
    c = Clip("proud", 2.4, label="Proud", emoji="😎", fx="⭐", group="live", seated=True, mood=0.6,
             description="Beak high on a two-note fanfare, then looks slowly over its audience.")
    c.key("head_pitch", 0.3, -0.55, 1.9, -0.5, 2.3, 0.0).key("neck", 0.3, 0.12, 1.9, 0.12, 2.3, 0.0)
    c.key("head_yaw", 0.9, 0.0, 1.3, 0.35, 1.8, -0.35, 2.2, 0.0)
    c.say(0.3, tone(Q.note("G4"), 0.18, 40)).say(0.52, tone(Q.note("C5"), 0.55, 41))
    clips.append(c)

    # showoff: wiggle, front flip, ta-daa
    c = Clip("showoff", 3.4, label="Show off", emoji="🤸", fx="🌟", group="live", mood=0.8,
             description="'Watch this': two quacks, a front flip (the front-flip policy), then the proud pose and its fanfare.")
    c.key("head_roll", 0.15, 0.22, 0.4, -0.22, 0.65, 0.0).key("head_pitch", 0.15, -0.2, 0.7, 0.0)
    c.say(0.15, wak(2, 0.16, 1)).say(0.42, wak(5, 0.16, 2))
    c.skill(0.9, "flip")
    c.key("head_pitch", 1.0, 0.0, 1.35, -0.55, 2.9, -0.5, 3.3, 0.0).key("neck", 1.0, 0.0, 1.35, 0.12, 2.9, 0.12, 3.3, 0.0)
    c.key("head_yaw", 1.9, 0.0, 2.3, 0.35, 2.8, -0.35, 3.2, 0.0)
    c.say(1.3, tone(Q.note("G4"), 0.18, 40)).say(1.52, tone(Q.note("C5"), 0.55, 41))
    clips.append(c)

    # peck: forage
    c = Clip("peck", 3.4, label="Peck", emoji="🌾", fx="🌾", group="live",
             description="Head down to the floor, three pecks with a beak click each, then beak up to swallow with a content hum.")
    c.key("neck", 0.5, -0.35, 1.8, -0.35, 2.2, 0.05, 3.0, 0.05, 3.3, 0.0)
    c.key("head_pitch", 0.5, 0.7, 0.78, 0.7, 0.86, 0.98, 0.98, 0.7, 1.13, 0.7, 1.21, 0.98, 1.33, 0.7, 1.48, 0.7, 1.56, 0.98, 1.68, 0.7,
          1.8, 0.7, 2.25, -0.4, 2.95, -0.4, 3.3, 0.0)
    c.key("head_yaw", 0.9, 0.0, 1.05, 0.12, 1.4, -0.12, 1.75, 0.0)
    for i, tp in enumerate((0.84, 1.19, 1.54)):
        c.say(tp, Q.peck(P, i) * 0.8).key("mouth", tp - 0.1, 0.45, tp, 0.0)
    c.key("mouth", 2.3, 0.0, 2.4, 0.3, 2.5, 0.0, 2.6, 0.3, 2.7, 0.0)
    c.say(2.75, coo(-4, -2, 0.4, 3, 0.5))
    clips.append(c)

    # yawn
    c = Clip("yawn", 4.2, label="Yawn", emoji="🥱", fx="💤", group="live", seated=True, mood=-0.4, lip=0.3,
             description="A long yawn: the beak opens wide as the head tips back, then it droops, drowsy, and shakes itself awake.")
    c.key("mouth", 0.3, 0.0, 1.3, 1.0, 1.8, 1.0, 2.3, 0.0)
    c.key("head_pitch", 0.3, 0.0, 1.3, -0.45, 1.8, -0.45, 2.4, 0.1, 2.9, 0.38, 3.3, 0.38, 3.5, 0.0)
    c.key("neck", 0.3, 0.0, 1.3, 0.1, 1.8, 0.1, 2.4, 0.0).key("head_roll", 0.6, 0.0, 1.3, 0.12, 2.2, 0.0)
    c.osc("head_yaw", 3.4, 3.95, 0.22, 5.0, decay=2.0)
    c.say(0.5, quack(1.5, [(0, 2), (0.35, 6), (1, -8)], ("bell", 0.5, 0.7), 0, "yawn", 0.6, mods=dict(quackiness=0.05, brightness=0.1), breath_scale=4.0, am_scale=0.1))
    c.say(3.45, chirp(3, 5, 0.1, 8, 0.4))
    clips.append(c)

    # sneeze
    c = Clip("sneeze", 2.5, label="Sneeze", emoji="🤧", fx="💨", group="live", seated=True,
             description="'Ah... ah...': the head tips back in two steps, then snaps down on the sneeze and shakes it off.")
    c.key("head_pitch", 0.15, 0.0, 0.35, -0.22, 0.55, -0.22, 0.75, -0.48, 1.05, -0.5, 1.15, 0.65, 1.35, 0.3, 1.6, 0.0)
    c.key("neck", 0.3, 0.05, 1.05, 0.12, 1.15, -0.2, 1.6, 0.0)
    c.key("mouth", 0.2, 0.0, 0.35, 0.35, 0.5, 0.15, 0.75, 0.6, 1.02, 0.7, 1.12, 1.0, 1.3, 0.0)
    c.osc("head_yaw", 1.5, 2.1, 0.25, 4.5, decay=2.5)
    c.say(0.2, noise(0.2, 0.08, 1, gain=0.25)).say(0.6, noise(0.3, 0.12, 2, gain=0.35))
    c.say(1.08, Q.mix([(0, noise(0.35, 0.07, 3, lo=0.6)), (0, quack(0.2, [(0, 14), (1, 2)], ("expdecay", 0.004, 0.06), 4, "choo", 0.7))]))
    c.say(1.75, coo(-2, 0, 0.3, 6, 0.4))
    clips.append(c)

    # startled: jerk back, alarm call, hop back, look around
    c = Clip("startled", 2.7, label="Startled", emoji="😱", fx="❗", group="live",
             description="Head snaps back with an alarm call, a quick step backwards (the walk policy), looks left and right, then creeps forward.")
    c.key("head_pitch", 0.08, -0.5, 0.6, -0.3, 0.9, -0.1, 2.0, -0.1, 2.3, 0.1, 2.6, 0.0)
    c.key("neck", 0.08, 0.22, 0.7, 0.1, 1.9, 0.0, 2.2, -0.2, 2.6, 0.0)
    c.key("vx", 0.1, -0.35, 0.65, -0.35, 0.8, 0.0)
    c.key("head_yaw", 0.8, 0.0, 0.95, 0.7, 1.3, 0.7, 1.45, -0.7, 1.8, -0.7, 2.0, 0.0)
    c.key("mouth", 0.0, 0.0, 0.08, 1.0, 0.35, 1.0, 0.5, 0.0)
    c.say(0.02, Q.alarm(P, 0)).say(0.3, Q.alarm(P, 1) * 0.8).say(2.15, quack(0.25, [(0, -2), (1, 3)], ("bell", 0.06, 0.1), 3, "mm", 0.45, mods=dict(quackiness=0.1)))
    clips.append(c)

    # dizzy: shake it off (played after a get-up)
    c = Clip("dizzy", 2.0, label="Shake off", emoji="😵‍💫", fx="💫", group="live", seated=True,
             description="A fast decaying head shake with a 'brrr', then a short 'ha'. Plays by itself after the duck has got up from a fall.")
    c.osc("head_yaw", 0.1, 1.3, 0.55, 5.0, decay=1.8).osc("head_roll", 0.1, 1.3, 0.12, 5.0, phase=1.2, decay=1.8)
    c.key("head_pitch", 0.1, 0.12, 1.2, 0.12, 1.45, -0.25, 1.75, 0.0)
    c.say(0.12, quack(1.0, [(0, 3), (1, -4)], ("bell", 0.05, 0.5), 0, "brr", 0.7, mods=dict(am_rate_hz=14.0, am_depth=1.0, quackiness=1.0), am_scale=1.5))
    c.say(1.45, wak(4, 0.16, 5, 0.7))
    clips.append(c)

    # nap: sit down and doze (the sit-stand policy); the idle life keeps it dozing afterwards
    c = Clip("nap", 6.0, label="Nap", emoji="😴", fx="💤", group="live", ends="seated", mood=-0.8, lip=0.25,
             description="Sits down (the sit-stand policy), the head sinks and it snores twice. It stays seated; S or an arrow wakes it.")
    c.skill(0.3, "sit")
    c.key("head_pitch", 0.9, 0.0, 2.0, 0.55, 6.0, 0.55).key("head_roll", 1.2, 0.0, 2.2, 0.14, 6.0, 0.14)
    c.osc("neck", 2.0, 6.0, 0.05, 0.42)
    for i, ts in enumerate((2.2, 4.1)):
        c.say(ts, noise(0.8, 0.5, 10 + i, gain=0.25)).say(ts + 0.95, quack(0.6, [(0, 12), (1, 9)], ("bell", 0.2, 0.3), 20 + i, "snore", 0.25, mods=dict(quackiness=0.0, brightness=0.05), breath_scale=3.0))
    clips.append(c)
    return clips


# keyboard: number row = Pollen's quick expressions, Q..P and F..L = the originals and the long ones
POLLEN_META = dict(
    yes=dict(label="Yes", emoji="👍", fx="✔️", key="1"), no=dict(label="No", emoji="🙅", fx="✖️", key="2"),
    mmh=dict(label="Mmh?", emoji="🤨", fx="❔", key="3"), curious=dict(label="Curious", emoji="🤔", fx="❓", key="4"),
    laugh=dict(label="Laugh", emoji="😂", fx="😂", key="5", mood=0.7), excited=dict(label="Excited", emoji="🤩", fx="✨", key="6", mood=0.8),
    angry=dict(label="Angry", emoji="😠", fx="💢", key="7", mood=-0.3), sad=dict(label="Sad", emoji="😢", fx="💧", key="8", mood=-0.7),
    mock=dict(label="Mock", emoji="😜", fx="🎵", key="9"), defiant=dict(label="Defiant", emoji="😤", fx="💢", key="0"),
    impatient=dict(label="Impatient", emoji="😒", fx="⏱️", key="k"), yes_fast=dict(label="Yep", emoji="👌", fx="✔️", key="l"),
    devastated=dict(label="Devastated", emoji="😭", fx="💧", key="z", mood=-0.9, ends="seated"),
    play_dead=dict(label="Play dead", emoji="💀", fx="👻", key="x", ends="down"),
)
LIVE_KEYS = dict(hello="q", happy="w", dance="e", sing="t", love="y", proud="u", showoff="i", peck="o", yawn="p",
                 sneeze="f", startled="g", dizzy="h", nap="j")
# clips that may also be played seated (head and beak only; the sit-stand policy holds the seat)
POLLEN_SEATED = ("yes", "no", "mmh", "curious", "laugh", "excited", "angry", "sad", "mock", "defiant", "impatient", "yes_fast")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pollen", help="a clone of pollen-robotics/microduck_emotions (default: clone it to a temp dir)")
    args = ap.parse_args()
    src = args.pollen
    tmp = None
    if not src:
        tmp = tempfile.mkdtemp(prefix="microduck_emotions_")
        subprocess.check_call(["git", "clone", "-q", POLLEN_URL, tmp])
        subprocess.check_call(["git", "-C", tmp, "checkout", "-q", POLLEN_COMMIT])
        src = tmp
    for d in ("keyframes", "sounds"):
        shutil.rmtree(os.path.join(OUT, d), ignore_errors=True)
        os.makedirs(os.path.join(OUT, d))
    shutil.copy(os.path.join(src, "LICENSE"), os.path.join(OUT, "LICENSE-pollen-microduck_emotions-apache-2.0"))
    manifest = []

    for c in originals():
        keys, sig = c.build()
        with open(os.path.join(OUT, "keyframes", c.name + ".json"), "w") as f:
            json.dump(dict(step_s=STEP, keyframes=keys), f, separators=(",", ":"))
        sounds = []
        if sig is not None:
            Q.write_wav(os.path.join(OUT, "sounds", c.name + ".wav"), sig, peak_dbfs=None)
            sounds = ["sounds/%s.wav" % c.name]
        m = dict(name=c.name, source="duck-live", duration_s=c.T, keyframes="keyframes/%s.json" % c.name, sounds=sounds, sound_at_s=0.0,
                 key=LIVE_KEYS.get(c.name, ""), uses=sorted(set(["head", "beak"] + (["walk policy"] if any(any(k["twist"]) for k in keys) else [])
                                                               + ["%s policy" % dict(jump="jump", flip="front-flip", sit="sit-stand")[s] for _, s in c.skills])))
        m.update((k, v) for k, v in c.meta.items() if k not in ("gain", "lip"))
        manifest.append(m)

    with open(os.path.join(src, "emotions.json")) as f:
        pol = json.load(f)["emotions"]
    order = list(POLLEN_META)                           # palette order = keyboard order
    for e in sorted(pol, key=lambda e: order.index(e["name"])):
        n = e["name"]
        shutil.copy(os.path.join(src, e["keyframes"]), os.path.join(OUT, "keyframes", n + ".json"))
        sounds = []
        for s in e["sounds"]:
            shutil.copy(os.path.join(src, s), os.path.join(OUT, "sounds", os.path.basename(s)))
            sounds.append("sounds/" + os.path.basename(s))
        with open(os.path.join(src, e["keyframes"])) as f:
            kf = json.load(f)["keyframes"]
        m = dict(name=n, source="pollen-robotics/microduck_emotions", duration_s=kf[-1]["t"], keyframes="keyframes/%s.json" % n, sounds=sounds,
                 sound_at_s=e["sound_at_s"], description=e["description"], group="pollen", seated=n in POLLEN_SEATED,
                 uses=["head", "beak"] + (["sit-stand policy"] if any(k.get("skill") == "sit" for k in kf) else []))
        if "pose_joints" in e:
            m["pose_joints"] = e["pose_joints"]
            m["uses"].append("stand-up policy")
        m.update(POLLEN_META[n])
        manifest.append(m)

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(dict(about="Duck Live emotion clips. Built by app/tools/make_animations.py; do not edit by hand.",
                       pollen=dict(url=POLLEN_URL, commit=POLLEN_COMMIT, licence="Apache-2.0"), clips=manifest), f, indent=1, ensure_ascii=False)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print("%-11s %-10s %5s  %3s  %s" % ("clip", "source", "T [s]", "key", "uses"))
    for m in manifest:
        print("%-11s %-10s %5.1f  %3s  %s" % (m["name"], m["group"], m["duration_s"], m["key"], ", ".join(m["uses"])))
    print("wrote", OUT, "(%d clips)" % len(manifest))


if __name__ == "__main__":
    main()
