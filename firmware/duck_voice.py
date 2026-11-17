"""The Microduck's voice: Pollen Robotics' seedable pet-vocalisation synth, in plain Python.

    python3 duck_voice.py ~/microduck/sounds          # render this duck's whole bank (about a minute on the duck)
    python3 duck_voice.py out/ --seed 100             # ... or any duck's

A port of `sounds/src/{rng,personality,synth,voices}.rs` from pollen-robotics/microduck (Apache-2.0), itself a port
of apirrone/microduck_sounds. One integer seed fixes a `Personality` (pitch, timbre, quackiness, tempo); each tag
has several variants so the duck does not sound like a stuck recording. On a robot the seed is
sha256(SoC serial)[:8], so every duck has its own stable voice.

Tags: alarm, greet, inquire, peck, chirp (the quack), coo. `wheee` (the looped roller-blade ride) is not ported.
The random stream (xoshiro256++ / splitmix64, Box-Muller) follows the Rust exactly; samples are computed in
double precision where the Rust uses float32, so a bank matches Pollen's in every draw but not bit for bit. It has
not been compared against a bank rendered by the Rust.

Bank layout, as the runtime expects: <dir>/<tag>/<tag>_<letter>.wav, 48 kHz mono 16-bit.
"""
import argparse, copy, hashlib, math, os, wave, zlib
from array import array

SR, TUNED_SR, TAU, M64 = 48000, 22050.0, 2 * math.pi, (1 << 64) - 1
VARIANTS = dict(greet=12, chirp=12)          # the most-heard tags get the most variants; the rest get 10


class Rng:
    def __init__(self, seed):
        x, self.s, self.spare = seed & 0xFFFFFFFF, [], None
        for _ in range(4):                   # splitmix64 spreads the seed over the four xoshiro words
            x = (x + 0x9E3779B97F4A7C15) & M64
            z = x
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
            self.s.append(z ^ (z >> 31))

    def random(self):
        """Uniform in [0, 1): the top 53 bits of xoshiro256++."""
        s, rot = self.s, lambda v, k: ((v << k) | (v >> (64 - k))) & M64
        out = (rot((s[0] + s[3]) & M64, 23) + s[0]) & M64
        t = (s[1] << 17) & M64
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t
        s[3] = rot(s[3], 45)
        return (out >> 11) / (1 << 53)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.random()

    def integers(self, lo, hi):
        return lo + math.floor(self.random() * (hi - lo))

    def choice(self, items):
        return items[self.integers(0, len(items))]

    def normal(self):
        if self.spare is not None:
            z, self.spare = self.spare, None
            return z
        r, theta = 1.0 - self.random(), TAU * self.random()
        mag = math.sqrt(-2.0 * math.log(r))
        self.spare = mag * math.sin(theta)
        return mag * math.cos(theta)


class Personality:
    def __init__(self, seed):
        r = Rng(seed)
        self.seed = seed & 0xFFFFFFFF
        self.register = r.choice([-1.0, 0.0, 0.0, 1.0]) + r.uniform(-0.4, 0.4)
        self.pitch_center_hz = min(620.0, max(110.0, r.uniform(160.0, 380.0) * 2.0 ** (self.register * 0.45)))
        self.pitch_spread, self.glide_bias = r.uniform(0.4, 1.2), r.uniform(-1.0, 1.0)
        self.brightness, self.tilt, self.nasal = r.uniform(0.05, 0.55), r.uniform(1.4, 2.8), r.uniform(0.1, 1.0)
        self.harmonic_skew, self.formant_n, self.formant_gain = r.uniform(-1.0, 1.0), r.integers(1, 6), r.uniform(0.0, 1.4)
        self.vibrato_rate_hz, self.vibrato_depth = r.uniform(3.5, 9.5), r.uniform(0.0, 0.7)
        self.jitter_depth, self.breath, self.quackiness = r.uniform(0.03, 0.35), r.uniform(0.0, 0.30), r.uniform(0.2, 1.0)
        self.am_rate_hz, self.am_depth = r.uniform(18.0, 55.0), r.uniform(0.15, 0.70)
        self.warble_hz, self.warble_depth = r.uniform(7.0, 18.0), r.uniform(0.0, 1.4)
        self.attack_sharpness, self.speed = r.uniform(0.0, 1.0), r.uniform(0.82, 1.22)

    def but(self, **changes):
        q = copy.copy(self)
        q.__dict__.update(changes)
        return q

    def variant_rng(self, tag, variant):
        return Rng((self.seed * 1_000_003) ^ zlib.crc32(tag.encode()) ^ (variant * 2_654_435_761))

    def harmonics(self):
        """Weights of the first seven harmonics: rolloff, a lifted high end, nasality, odd/even skew, one formant."""
        w = []
        for n in range(1, 8):
            base = 1.0 / n ** self.tilt
            skew = self.harmonic_skew * (0.4 if n % 2 == 0 else -0.2) if self.harmonic_skew >= 0 else \
                -self.harmonic_skew * (-0.3 if n % 2 == 0 else 0.4)
            w.append(max(0.0, base + self.brightness * (n / 7) ** 1.5 + self.nasal * (0.6 if n in (2, 3) else 0.0)
                         + skew + (self.formant_gain if n == self.formant_n else 0.0) * base * 1.5))
        w[0] = max(w[0], 0.7)                # the fundamental always carries the pitch
        return w


# ---- building blocks ---------------------------------------------------------------------------------------------
def t_axis(dur):
    return [i / SR for i in range(max(1, round(dur * SR)))]


def lerp(t, pts):
    out, k = [], 0
    for x in t:
        while k < len(pts) - 2 and x > pts[k + 1][0]:
            k += 1
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        out.append(y0 if x <= x0 else y1 if x >= x1 or x1 <= x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return out


def expdecay(t, attack, decay):
    attack, decay = max(attack, 1e-4), max(decay, 1e-4)
    return [min(1.0, max(0.0, x / attack)) * math.exp(-max(0.0, x - attack) / decay) for x in t]


def bell(t, attack, release):
    start, a, r = max(0.0, t[-1] - release), max(attack, 1e-4), max(release, 1e-4)
    return [min(1.0, max(0.0, max(0.0, 1.0 - (x - start) / r) if x > start else x / a if x < attack else 1.0)) for x in t]


def vibrato(t, rate, depth, phase):
    if rate <= 0 or depth <= 0:
        return [1.0] * len(t)
    return [2.0 ** (depth * math.sin(TAU * rate * x + phase) / 12.0) for x in t]


def jitter(n, depth, rng):
    """Slow random pitch wobble: white noise through a moving average ("same" convolution, as numpy did it)."""
    if depth <= 0:
        return [1.0] * n
    raw, k = [rng.normal() for _ in range(n)], max(1, round(64.0 * SR / TUNED_SR))
    acc = [0.0]
    for v in raw:
        acc.append(acc[-1] + v)
    off = (k - 1) // 2
    return [2.0 ** (depth * (acc[min(n, i + off + 1)] - acc[max(0, i + off + 1 - k)]) / k / 12.0) for i in range(n)]


def pink_noise(n, rng):
    a, leak, out = 0.985 ** (TUNED_SR / SR), 0.0, []
    for _ in range(n):
        leak = a * leak + rng.normal()
        out.append(leak)
    peak = max(map(abs, out)) + 1e-9
    return [v / peak for v in out]


def normalise(x, dbfs):
    gain = 10.0 ** (dbfs / 20.0) / (max(map(abs, x)) + 1e-9)
    return [v * gain for v in x]


def attack(p, dur, snappy):
    soft, sharp = 0.04 * dur, 0.003 * dur
    return max(0.001, soft + (sharp - soft) * p.attack_sharpness * snappy)


def voice(p, t, freq, rng, am_scale, breath_scale):
    """The shared core: harmonic oscillator + vibrato + jitter + the quack's AM buzz + breath."""
    vib = vibrato(t, p.vibrato_rate_hz, p.vibrato_depth, rng.uniform(0.0, TAU))
    jit, w = jitter(len(t), p.jitter_depth, rng), p.harmonics()
    acc, body = 0.0, []
    for f, v, j in zip(freq, vib, jit):
        acc += f * v * j
        ph = TAU * acc / SR
        body.append(sum(wn * math.sin((n + 1) * ph) for n, wn in enumerate(w) if wn))
    am = p.am_depth * am_scale * p.quackiness            # quackiness gates the buzz: pure-tone ducks have none
    if am > 0.01:
        body = [b * (1.0 - am * (0.5 + 0.5 * math.sin(TAU * p.am_rate_hz * x))) for b, x in zip(body, t)]
    breath = p.breath * breath_scale
    if breath > 0:
        body = [b + breath * nz for b, nz in zip(body, pink_noise(len(t), rng))]
    return body


def shaped(body, env):
    return [b * e for b, e in zip(body, env)]


# ---- the recipes -------------------------------------------------------------------------------------------------
def alarm(p, variant):
    rng = p.variant_rng("alarm", variant)
    dur = (0.20 + 0.12 * rng.random()) / p.speed
    t = t_axis(dur)
    f0 = p.pitch_center_hz * (1.25 + 0.35 * p.pitch_spread) * (0.94 + 0.12 * rng.random())
    peak = 1.15 + 0.25 * p.pitch_spread + 0.10 * rng.random()
    freq = lerp(t, [(0.0, f0), (0.05 * dur, f0 * peak), (dur, f0 * (0.75 + 0.20 * (1.0 - p.pitch_spread)))])
    env = expdecay(t, attack(p, dur, 1.0), dur * (0.40 + 0.20 * rng.random()))
    sig, crackle = shaped(voice(p, t, freq, rng, 0.5, 1.0), env), 0.04 + 0.10 * p.brightness
    return normalise([v + crackle * rng.normal() * e for v, e in zip(sig, env)], -3.0)      # bright ducks rasp


def greet_syllable(p, rng, dur_scale, f0_scale):
    dur = (0.32 + 0.25 * rng.random()) * dur_scale / p.speed
    t = t_axis(dur)
    f0 = p.pitch_center_hz * (0.9 + 0.15 * rng.random()) * f0_scale
    bias, bend = p.glide_bias, 0.10 + 0.15 * p.pitch_spread         # positive ducks bend up, negative down
    end = f0 * (1.0 - bias * bend * 0.3) * (0.92 + 0.08 * rng.random())
    freq = lerp(t, [(0.0, f0 * (1.0 - bias * bend * 0.5)), (0.18 * dur, f0 * (1.0 + bias * bend)), (dur, end)])
    return shaped(voice(p, t, freq, rng, 1.0, 1.0), expdecay(t, attack(p, dur, 0.5), dur * 0.7))


def greet(p, variant):
    rng = p.variant_rng("greet", variant)
    sig = greet_syllable(p, rng, 1.0, 1.0)
    if rng.random() < 0.4:                               # some greets are double: "wak-wak"
        sig += [0.0] * int((0.05 + 0.06 * rng.random()) / p.speed * SR)
        sig += greet_syllable(p, rng, 0.8, 0.95 + 0.06 * rng.random())
    return normalise(sig, -3.0)


def inquire(p, variant):
    rng = p.variant_rng("inquire", variant)
    dur = (0.42 + 0.25 * rng.random()) / p.speed
    t = t_axis(dur)
    f0 = p.pitch_center_hz * (0.88 + 0.10 * rng.random())
    rise = 1.15 + 0.50 * p.pitch_spread + 0.20 * max(0.0, p.glide_bias) + 0.10 * rng.random()      # a question rises
    freq = lerp(t, [(0.0, f0 * 0.92), (0.30 * dur, f0 * 0.95), (dur, f0 * rise)])
    env = bell(t, dur * (0.06 + 0.10 * (1.0 - p.attack_sharpness)), dur * 0.32)
    return normalise(shaped(voice(p, t, freq, rng, 0.6, 1.0), env), -3.0)


def peck(p, variant):
    rng = p.variant_rng("peck", variant)
    dur = (0.16 + 0.12 * rng.random()) / p.speed
    t = t_axis(dur)
    f0 = p.pitch_center_hz * (0.45 + 0.20 * rng.random())
    freq = lerp(t, [(0.0, f0 * 1.5), (0.04 * dur, f0), (dur, f0 * 0.80)])
    body = shaped(voice(p, t, freq, rng, 0.3, 0.5), expdecay(t, attack(p, dur, 1.0), dur * 0.35))
    n = min(len(t), int((0.003 + 0.006 * p.attack_sharpness) * SR))                # the "tock": a faded noise click
    gain = 0.4 + 0.4 * p.attack_sharpness
    for i in range(n):
        body[i] += gain * rng.uniform(-1.0, 1.0) * (1.0 - i / n) ** 2
    return normalise(body, -3.0)


def chirp_syllable(p, rng, f0, dur, contour):
    t = t_axis(dur)
    warble = vibrato(t, p.warble_hz, p.warble_depth, rng.uniform(0.0, TAU))
    freq = [f0 * c * w for c, w in zip(lerp(t, contour), warble)]
    soft = p.but(quackiness=p.quackiness * 0.2, brightness=p.brightness * 0.5, formant_gain=p.formant_gain * 0.5)
    return shaped(voice(soft, t, freq, rng, 1.0, 0.4), expdecay(t, attack(p, dur, 0.7), dur * 0.55))


def chirp(p, variant):
    """The mouth-trigger sound, the duck's quack. Variants cycle four shapes: rise, fall, trill, double."""
    rng = p.variant_rng("chirp", variant)
    f0 = p.pitch_center_hz * (0.95 + 0.75 * rng.random())
    shape = variant % 4
    if shape == 0:
        dur = (0.10 + 0.10 * rng.random()) / p.speed
        peak = 1.12 + 0.10 * rng.random()
        sig = chirp_syllable(p, rng, f0, dur, [(0.0, 0.88), (0.5 * dur, peak), (dur, 1.05)])
    elif shape == 1:
        dur = (0.10 + 0.10 * rng.random()) / p.speed
        start = 1.12 + 0.10 * rng.random()
        sig = chirp_syllable(p, rng, f0, dur, [(0.0, start), (0.3 * dur, 1.0), (dur, 0.78)])
    elif shape == 2:
        dur = (0.22 + 0.14 * rng.random()) / p.speed
        trill = p.but(warble_depth=max(p.warble_depth, 0.7) * 1.6)
        sig = chirp_syllable(trill, rng, f0, dur, [(0.0, 1.0), (0.5 * dur, 1.06), (dur, 0.94)])
    else:                                                # "wek-wek"
        dur = (0.08 + 0.05 * rng.random()) / p.speed
        sig = chirp_syllable(p, rng, f0, dur, [(0.0, 0.95), (0.4 * dur, 1.10), (dur, 0.88)])
        sig += [0.0] * int((0.03 + 0.03 * rng.random()) / p.speed * SR)
        dur_b, f0_b = dur * 0.9, f0 * (0.92 + 0.10 * rng.random())
        sig += chirp_syllable(p, rng, f0_b, dur_b, [(0.0, 0.95), (0.4 * dur_b, 1.10), (dur_b, 0.88)])
    return normalise(sig, -6.0)


def coo(p, variant):
    rng = p.variant_rng("coo", variant)
    dur = (0.85 + 0.55 * rng.random()) / p.speed
    t = t_axis(dur)
    f0 = p.pitch_center_hz * (0.42 + 0.15 * (1.0 - p.attack_sharpness)) * (0.94 + 0.12 * rng.random())
    drift = 1.0 + 0.05 * rng.random() + 0.04 * p.glide_bias
    freq = lerp(t, [(0.0, f0 * 0.94), (dur * 0.5, f0 * drift), (dur, f0 * 0.90)])
    env = bell(t, dur * (0.18 + 0.10 * (1.0 - p.attack_sharpness)), dur * 0.30)
    soft = p.but(breath=max(p.breath, 0.12) + 0.10, quackiness=p.quackiness * 0.25, am_rate_hz=p.am_rate_hz * 0.30,
                 vibrato_rate_hz=p.vibrato_rate_hz * 0.45, vibrato_depth=p.vibrato_depth * 0.7)
    return normalise(shaped(voice(soft, t, freq, rng, 1.0, 1.0), env), -5.0)


TAGS = dict(chirp=chirp, greet=greet, coo=coo, peck=peck, inquire=inquire, alarm=alarm)


# ---- the bank ----------------------------------------------------------------------------------------------------
def hardware_seed():
    """sha256(SoC serial)[:8] as an integer: what gives each duck its own voice. /etc/machine-id if there is none."""
    for path in ("/proc/device-tree/serial-number", "/etc/machine-id"):
        if os.path.exists(path):
            ident = open(path, "rb").read().split(b"\0")[0].decode().strip()
            if ident:
                return int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16)
    raise OSError("no SoC serial and no machine-id to derive a voice from")


def path(bank, tag, variant):
    return os.path.join(bank, tag, f"{tag}_{chr(ord('a') + variant)}.wav")


def render_all(bank, seed=None, log=lambda *_: None):
    """Write every missing sound of the bank. A `.seed` marker makes a bank for another seed re-render."""
    seed = hardware_seed() if seed is None else seed
    p, mark = Personality(seed), os.path.join(bank, ".seed")
    fresh = os.path.exists(mark) and open(mark).read().strip() == str(seed)
    for tag, recipe in TAGS.items():
        os.makedirs(os.path.join(bank, tag), exist_ok=True)
        for v in range(VARIANTS.get(tag, 10)):
            if fresh and os.path.exists(path(bank, tag, v)):
                continue
            pcm = array("h", (max(-32768, min(32767, int(s * 32767.0))) for s in recipe(p, v)))
            with wave.open(path(bank, tag, v) + ".part", "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
                f.writeframes(pcm.tobytes())
            os.replace(path(bank, tag, v) + ".part", path(bank, tag, v))
            log(tag, v, len(pcm) / SR)
    open(mark, "w").write(str(seed))
    return seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bank")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    seed = render_all(a.bank, a.seed, lambda tag, v, secs: print(f"  {tag}_{chr(97 + v)}  {secs:.2f} s", flush=True))
    p = Personality(seed)
    print(f"seed {seed}: pitch {p.pitch_center_hz:.0f} Hz, quackiness {p.quackiness:.2f}, speed {p.speed:.2f}")
