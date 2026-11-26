#!/usr/bin/env python3
"""PPO velocity-command flight policy for the four fans, trained in MuJoCo on a CPU (interim: the Isaac Lab chain needs
the GPU). Command: forward, sideways and VERTICAL velocity plus yaw rate, in the heading frame, like the walk policies'
velocity command with a z component. Take-off is a positive vertical-velocity command from the ground.

    FLIGHT_LEGS=1 python scripts/tools/flight_rl_mujoco.py train     # residual policy: fans + ten leg joints (default mode)
    FLIGHT_RESIDUAL=0 python scripts/tools/flight_rl_mujoco.py train # from scratch (did not learn to fly in 3 M steps)
    python scripts/tools/flight_rl_mujoco.py eval                    # grid of plant errors, figure, JSON
    FLIGHT_LEGS=1 python scripts/tools/flight_rl_mujoco.py train     # stage 2: + ten leg joints in the air, warm-started

The policy outputs collective, roll, pitch and yaw commands at 50 Hz, mixed to the 4 throttles; the legs and head hold the stand pose with the model's position servos
(stiffened x4). Ground episodes start with the rotors idling at half the hover speed, because the stand with the pack
is not passively stable: the fans hold attitude from the first tick.
Plant: scene_flight.xml + the fan/battery model of microduck_rl/flight/fan_model.py, redrawn every episode
(thrust +-20 %, per fan +-7 %, torque coefficient +-15 %, rotor inertia x0.7-1.8, cell resistance x0.6-1.6,
state of charge 0.5-0.98, command delay 5-20 ms, all-up centre of mass +-8 mm in x and y).
Observation (23): gravity in the body frame, body rates, body velocity, height, the 4 commands, 4 rotor speeds,
bus voltage, last throttles, with sensor noise. The body velocity presumes an estimator (optical flow or similar).
Seeds are fixed. Outputs: checkpoints/flight_hover_mujoco/policy.npz, docs/flight_rl_results.json,
docs/figures/flight_rl_*.pdf.
"""
import argparse, copy, json, math, multiprocessing as mp, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
XML = os.path.join(ROOT, "app", "assets", "microduck", "scene_flight.xml")
LEGS = os.environ.get("FLIGHT_LEGS") == "1"      # the policy also moves the ten leg joints in the air
RESIDUAL = os.environ.get("FLIGHT_RESIDUAL", "1") == "1"   # the policy adds to the model-based controller (default)
CK_FANS = os.path.join(ROOT, "checkpoints", "flight_hover_mujoco")
CK = os.path.join(ROOT, "checkpoints", ("flight_res_legs_mujoco" if LEGS else "flight_res_fans_mujoco") if RESIDUAL else ("flight_legs_mujoco" if LEGS else "flight_hover_mujoco"))
TAG = ("flight_rl_res_legs" if LEGS else "flight_rl_res_fans") if RESIDUAL else ("flight_rl_legs" if LEGS else "flight_rl")
RES_C, RES_D = 0.10, 0.10     # residual authority on collective and on each differential channel [throttle]
I_DIAG = json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))["layout"]["I_diag"]


class _Belief:
    """What the base controller knows about the fans: the nominal model and the measured bus voltage, not the drawn plant."""

    def __init__(self, plant, p_nominal):
        self._plant, self.p = plant, p_nominal

    @property
    def aux(self):
        return self._plant.aux
DT, DECIM, EP_S = 0.005, 4, 6.0
N_LEG = 10
OBS, ACT = 23 + (2 * N_LEG if LEGS else 0), 4 + (N_LEG if LEGS else 0)
LEG_SCALE, LEG_MAX, LEG_RATE, HIP_ROLL_MAX = 0.5, 0.6, 0.06, 0.08   # rad per unit action, rad, rad per control step, rad
CMD_RANGE = dict(vx=(-0.5, 0.5), vy=(-0.3, 0.3), vz=(-0.4, 0.6), wz=(-1.0, 1.0))
CMD_RESAMPLE_S = 2.0
Z_FLOOR, Z_CEIL = 0.25, 1.0     # outside this band the vertical command is overridden to come back
U0, U_SCALE, U_DIFF = 0.55, 0.25, 0.12
# fan order FL, RL, FR, RR (flight_pack.json)
MIX_ROLL, MIX_PITCH, MIX_YAW = np.array([1.0, 1.0, -1.0, -1.0]), np.array([1.0, -1.0, 1.0, -1.0]), np.array([1.0, -1.0, -1.0, 1.0])


class Env:
    def __init__(self, seed):
        import mujoco
        import flight_sim as S
        import fan_model as F
        self.mj, self.S, self.F = mujoco, S, F
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.m.opt.timestep = DT
        # the legs only hold a pose here: stiffen the model's position servos (kp x4, joint damping) so they do
        self.m.actuator_gainprm[:, 0] *= 4; self.m.actuator_biasprm[:, 1] *= 4; self.m.dof_damping[6:] += 0.02
        an = [self.m.actuator(i).name for i in range(self.m.nu)]
        self.ankles = (an.index("left_ankle"), an.index("right_ankle"))
        self.leg_act = [i for i, n in enumerate(an) if "neck" not in n and "head" not in n]
        self.leg_qadr = [int(self.m.jnt_qposadr[self.m.actuator_trnid[i, 0]]) for i in self.leg_act]
        self.hip_roll = [k for k, i in enumerate(self.leg_act) if "hip_roll" in an[i]]
        assert len(self.leg_act) == N_LEG
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.trunk = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.pack = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "flight_pack")
        self.ipos0 = self.m.body_ipos[self.pack].copy()
        self.mass = float(self.m.body_subtreemass[self.trunk])
        self.p0, self.b0 = S.prior_params()
        self.w_ref = math.sqrt(4.0 / self.p0.kf)
        self.dr = dict(thrust=(0.8, 1.2), fan=(0.93, 1.07), kq=(0.85, 1.15), J=(0.7, 1.8), r=(0.6, 1.6), soc=(0.5, 0.98), delay=(1, 4), com=0.012 if LEGS else 0.008)
        self.fixed = None      # evaluation: dict(thrust=..., com_x=...) overrides the draw
        self.p_air = 0.5
        self.use_policy = True   # False: actions ignored (the base controller alone, for comparison)
        self.cmd_scale = 1.0   # curriculum: scales the non-take-off commands
        self.dr_scale = 1.0    # curriculum: scales the spread of the plant draw

    def reset(self):
        mj, m, d, rng, F = self.mj, self.m, self.d, self.rng, self.F
        mj.mj_resetDataKeyframe(m, d, self.key)
        d.ctrl[:] = m.key_ctrl[self.key]
        d.ctrl[self.ankles[0]] -= 0.08; d.ctrl[self.ankles[1]] += 0.08   # lean that roughly balances the pack on the back
        def U(k):                                    # draw, with the spread scaled by the curriculum
            lo, hi = self.dr[k]; mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * self.dr_scale
            return rng.uniform(mid - half, mid + half)
        fx = self.fixed or {}
        scale = fx.get("thrust", U("thrust"))
        p = F.replace(self.p0, eta_inst=self.p0.eta_inst * scale, kq=self.p0.kq * (1.0 if fx else U("kq")), J=self.p0.J * (1.0 if fx else U("J")))
        bat = F.replace(self.b0, r_cell=self.b0.r_cell * (1.0 if fx else U("r")))
        self.plant = self.S.FanPlant(m, d, p, bat, cmd_delay_steps=2 if fx else int(rng.integers(self.dr["delay"][0], self.dr["delay"][1] + 1)),
                                     soc0=0.8 if fx else U("soc"))
        self.plant.fan_gain = np.ones(4) if fx else 1.0 + self.dr_scale * (rng.uniform(*self.dr["fan"], 4) - 1.0)
        cm = self.dr["com"] * self.dr_scale
        com = np.array([fx.get("com_x", rng.uniform(-cm, cm)), 0.0 if fx else rng.uniform(-cm, cm), 0.0])
        m.body_ipos[self.pack] = self.ipos0 + com * self.mass / m.body_mass[self.pack]     # shifts the all-up CoM by `com`
        self.air = (not fx) and rng.random() < self.p_air
        self.schedule = fx.get("schedule")          # evaluation: list of (t_start, [vx, vy, vz, wz])
        if self.air:
            d.qpos[2] = rng.uniform(0.30, 0.65)
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = math.radians(rng.uniform(0, 15))
            d.qpos[3:7] = [math.cos(ang / 2), *(math.sin(ang / 2) * ax)]
            d.qvel[0:3] = rng.uniform(-0.3, 0.3, 3); d.qvel[3:6] = rng.uniform(-1.0, 1.0, 3)
            self.plant.w[:] = math.sqrt(self.mass * 9.81 / 4 / self.p0.kf) * rng.uniform(0.8, 1.1)
        else:
            # ground start with the rotors idling at half the hover speed: the stand with the pack is not passively
            # stable (it tips over in ~0.5 s with the fans off), so the fans must hold attitude from the first tick
            self.plant.w[:] = 0.5 * math.sqrt(self.mass * 9.81 / 4 / self.p0.kf)
        mj.mj_forward(m, d)
        self.t = 0
        self.u = np.zeros(4)
        self.leg = np.zeros(N_LEG)
        if RESIDUAL:
            self.ctrl = self.S.FlightController(m, d, self.plant, I_DIAG)
            self.ctrl.dt = DT * DECIM
            self.ctrl.plant = _Belief(self.plant, self.p0)       # nominal fan model
            self.ctrl.com_bias = -com                            # nominal centre of mass
            self.z_ref = float(d.xpos[self.trunk][2])
            self.t_full = self.F.steady_state(1.0, 4, 0.8, self.p0, self.b0)["thrust"]
        self.hold = d.ctrl.copy()
        self.cmd = np.zeros(4)
        self._draw_cmd(first=True)
        return self._obs()

    def _draw_cmd(self, first=False):
        rng = self.rng
        if self.schedule is not None:
            return
        if first and not self.air:                  # from the ground the first command is a take-off
            self.cmd = np.array([0.0, 0.0, rng.uniform(0.3, 0.6), 0.0]); return
        if rng.random() < 0.25:
            self.cmd = np.zeros(4); return          # hover
        self.cmd = np.array([rng.uniform(*CMD_RANGE[k]) * self.cmd_scale for k in ("vx", "vy", "vz", "wz")])

    def command(self):
        """Commanded (vx, vy, vz, wz); the vertical part is overridden near the floor and the ceiling."""
        if self.schedule is not None:
            t = self.t * DT * DECIM
            c = np.array([c_ for t0, c_ in self.schedule if t >= t0][-1], dtype=float)
        else:
            c = self.cmd.copy()
        z = float(self.d.xpos[self.trunk][2])
        if z > Z_CEIL:
            c[2] = min(c[2], -0.3)
        elif z < Z_FLOOR and (self.air or self.t * DT * DECIM > 2.0):
            c[2] = max(c[2], 0.2)
        return c

    def _state(self):
        d = self.d
        R = d.xmat[self.trunk].reshape(3, 3)
        v = np.zeros(6)
        # world-frame velocity of the body frame (XBODY), rotated into it. Not mjOBJ_BODY with the local flag: that is
        # the body's INERTIAL frame, whose axes are the trunk's principal axes, not forward/left/up.
        self.mj.mj_objectVelocity(self.m, d, self.mj.mjtObj.mjOBJ_XBODY, self.trunk, v, 0)
        return R, R.T @ v[:3], R.T @ v[3:], float(d.xpos[self.trunk][2])

    def _obs(self):
        R, om, vb, z = self._state()
        n = self.rng.normal
        return np.concatenate([-R[2] + 0.01 * n(size=3), 0.25 * (om + 0.05 * n(size=3)), vb + 0.05 * n(size=3), [z + 0.01 * n()], self.command(),
                               self.plant.w[0] / self.w_ref * (1 + 0.005 * n(size=4)), [float(self.plant.aux["v_bus"][0]) / 16.8], self.u]
                              + ([self.leg, self.d.qpos[self.leg_qadr] - self.hold[self.leg_act] + 0.005 * n(size=N_LEG)] if LEGS else [])).astype(np.float32)

    def step(self, a):
        mj, m, d = self.mj, self.m, self.d
        # the policy acts in mixer coordinates (collective, roll, pitch, yaw), mapped to the four throttles by the fan
        # geometry: per-fan exploration noise on this unstable plant tumbled it before anything could be learned
        a = np.clip(a, -3, 3) if self.use_policy else np.zeros(ACT)
        if RESIDUAL:
            # base: the cascaded controller of flight_sim.py tracking the velocity command (height reference integrates
            # the vertical command; a climb from the floor starts with a 0.1 m step so the feet clear at once)
            c = self.command(); z = float(d.xpos[self.trunk][2]); dtc = DT * DECIM
            R0 = d.xmat[self.trunk].reshape(3, 3); yaw = math.atan2(R0[1, 0], R0[0, 0])
            if self.ctrl.yaw_ref is None:
                self.ctrl.yaw_ref = yaw
            self.ctrl.yaw_ref += c[3] * dtc
            self.z_ref = float(np.clip(self.z_ref + c[2] * dtc, 0.10, 1.3))
            if z < 0.16 and c[2] > 0.1:
                self.z_ref = max(self.z_ref, z + 0.10)
            self.z_ref = float(np.clip(self.z_ref, z - 0.25, z + 0.25))
            v_ref = np.array([math.cos(yaw) * c[0] - math.sin(yaw) * c[1], math.sin(yaw) * c[0] + math.cos(yaw) * c[1]])
            u_base, _ = self.ctrl.update(self.z_ref, c[2], v_ref, self.t_full)
            u = np.clip(u_base + RES_C * a[0] + RES_D * (MIX_ROLL * a[1] + MIX_PITCH * a[2] + MIX_YAW * a[3]), 0.0, 1.0)
        else:
            u = np.clip(U0 + U_SCALE * a[0] + U_DIFF * (MIX_ROLL * a[1] + MIX_PITCH * a[2] + MIX_YAW * a[3]), 0.0, 1.0)
        du = u - self.u
        self.u = u
        dleg = np.zeros(N_LEG)
        if LEGS:
            # leg joint offsets from the hold pose, rate-limited, hip roll kept small (feet out of the jets), and only
            # once the duck is clear of the floor (on the ground the legs hold the stand)
            tgt = np.clip(LEG_SCALE * a[4:], -LEG_MAX, LEG_MAX)
            tgt[self.hip_roll] = np.clip(tgt[self.hip_roll], -HIP_ROLL_MAX, HIP_ROLL_MAX)
            tgt *= float(np.clip((d.xpos[self.trunk][2] - 0.16) / 0.08, 0.0, 1.0))
            dleg = np.clip(tgt - self.leg, -LEG_RATE, LEG_RATE)
            self.leg = self.leg + dleg
            d.ctrl[self.leg_act] = self.hold[self.leg_act] + self.leg
        for _ in range(DECIM):
            d.qfrc_applied[:] = 0
            self.plant.command(u)
            self.plant.substep()
            mj.mj_step(m, d)
        self.t += 1
        if self.schedule is None and self.t % int(CMD_RESAMPLE_S / (DT * DECIM)) == 0:
            self._draw_cmd()
        R, om, vb, z = self._state()
        c = self.command()
        yaw = math.atan2(R[1, 0], R[0, 0])
        vw = R @ vb                                                    # world-frame velocity
        vh = np.array([math.cos(yaw) * vw[0] + math.sin(yaw) * vw[1], -math.sin(yaw) * vw[0] + math.cos(yaw) * vw[1], vw[2]])   # heading frame
        wz = float((R @ om)[2])
        e_xy = float(np.sum((vh[:2] - c[:2]) ** 2)); e_z = float((vh[2] - c[2]) ** 2); e_w = float((wz - c[3]) ** 2)
        tilt2 = R[2, 0] ** 2 + R[2, 1] ** 2
        # each tracking term = a sharp kernel (precision) + a linear ramp (a gradient from far away: with the kernel alone,
        # standing still under a 0.5 m/s climb command scored exp(-5) and nothing pulled the duck off the floor)
        lin = lambda e2, span: max(1.0 - math.sqrt(e2) / span, 0.0)
        r = (0.8 * math.exp(-e_xy / 0.10) + 0.7 * lin(e_xy, 1.0) + 0.8 * math.exp(-e_z / 0.05) + 1.2 * lin(e_z, 0.8) + 0.5 * math.exp(-e_w / 0.5) + 0.5 * math.exp(-tilt2 / 0.10)
             - 0.02 * float(om[0] ** 2 + om[1] ** 2) - 0.05 * float(du @ du) + 0.2)
        if z < 0.17 and c[2] > 0.1:
            r -= 1.5                                # on the floor while a climb is commanded: staying put was the safe local optimum
        diff = self.u - self.u.mean()
        if LEGS:   # differential thrust costs: a steady push or an off-centre mass is cheaper to answer with the legs
            r -= 4.0 * float(diff @ diff) + 0.3 * float(dleg @ dleg)
        bad = (not np.isfinite(z)) or R[2, 2] < math.cos(math.radians(70)) or z > 1.6
        timeout = self.t >= int(EP_S / (DT * DECIM))
        if bad:
            r -= 5.0
        info = dict(tilt=math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))), z=z, v=vh, wz=wz, cmd=c, diff=float(np.sqrt(diff @ diff)), leg=self.leg.copy())
        return self._obs(), r, bool(bad), bool(timeout), info


def worker(conn, seeds):
    envs = [Env(s) for s in seeds]
    conn.send(np.stack([e.reset() for e in envs]))
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            ob, rw, term, tout, fin = [], [], [], [], []
            for e, a in zip(envs, data):
                o, r, bad, to, _ = e.step(a)
                fin.append(o)
                if bad or to:
                    o = e.reset()
                ob.append(o); rw.append(r); term.append(bad); tout.append(to and not bad)
            conn.send((np.stack(ob), np.array(rw, np.float32), np.array(term), np.array(tout), np.stack(fin)))
        elif cmd == "set":
            for e in envs:
                e.dr_scale, e.cmd_scale = data
        elif cmd == "close":
            conn.close(); return


def policy_np(P, obs):
    x = (obs - P["mu"]) / P["sd"]
    h = np.tanh(x @ P["w0"] + P["b0"]); h = np.tanh(h @ P["w1"] + P["b1"])
    return h @ P["w2"] + P["b2"]


def train(iters, n_proc=9, per_proc=6, T=96, seed=0, init=None, cur0=0.25):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(2)
    N = n_proc * per_proc
    conns, procs = [], []
    for i in range(n_proc):
        a, b = mp.Pipe()
        pr = mp.Process(target=worker, args=(b, [seed * 1000 + i * per_proc + j for j in range(per_proc)]), daemon=True)
        pr.start(); conns.append(a); procs.append(pr)
    obs = np.concatenate([c.recv() for c in conns])
    mlp = lambda o: nn.Sequential(nn.Linear(OBS, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, o))
    actor, critic = mlp(ACT), mlp(1)
    log_std = nn.Parameter(torch.full((ACT,), -1.6 if RESIDUAL else -1.2))
    if RESIDUAL:
        with torch.no_grad():                        # start as the base controller: near-zero residual
            last = [l for l in actor if isinstance(l, nn.Linear)][-1]; last.weight.mul_(0.01); last.bias.zero_()
    mu, var, cnt = np.zeros(OBS), np.ones(OBS), 1e-4
    warm = 0
    if LEGS and not RESIDUAL and os.path.exists(os.path.join(CK_FANS, "policy.npz")):
        # stage 2 starts as the fans-only policy: its weights fill the fan rows / the old observation columns, the new
        # leg outputs and inputs start at zero, and the actor is frozen while the fresh critic catches up
        P0 = dict(np.load(os.path.join(CK_FANS, "policy.npz")))
        L = [l for l in actor if isinstance(l, nn.Linear)]
        with torch.no_grad():
            L[0].weight.zero_(); L[0].weight[:, :23] = torch.as_tensor(P0["w0"].T, dtype=torch.float32); L[0].bias.copy_(torch.as_tensor(P0["b0"], dtype=torch.float32))
            L[1].weight.copy_(torch.as_tensor(P0["w1"].T, dtype=torch.float32)); L[1].bias.copy_(torch.as_tensor(P0["b1"], dtype=torch.float32))
            L[2].weight.zero_(); L[2].bias.zero_()
            L[2].weight[:4] = torch.as_tensor(P0["w2"].T, dtype=torch.float32); L[2].bias[:4] = torch.as_tensor(P0["b2"], dtype=torch.float32)
            log_std.fill_(-1.8)
        mu[:23], var[:23], cnt = P0["mu"], P0["sd"] ** 2, 2.0e5
        warm = 20
        print("warm start from", os.path.relpath(CK_FANS, ROOT), flush=True)
    if init and not LEGS:
        P0 = dict(np.load(init))
        L = [l for l in actor if isinstance(l, nn.Linear)]
        with torch.no_grad():
            for i, l in enumerate(L):
                l.weight.copy_(torch.as_tensor(P0[f"w{i}"].T, dtype=torch.float32)); l.bias.copy_(torch.as_tensor(P0[f"b{i}"], dtype=torch.float32))
            log_std.fill_(-1.4)
        mu, var, cnt = P0["mu"], P0["sd"] ** 2, 2.0e5
        warm = 15
        print("resumed actor from", os.path.relpath(init, ROOT), flush=True)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=3e-4)
    gamma, lam, clip = 0.99, 0.95, 0.2
    hist = []
    ep_ret, ep_len, done_ret, done_len, done_bad = np.zeros(N), np.zeros(N), [], [], []
    t0 = time.time()
    for it in range(iters):
        cur = min(1.0, cur0 + (1.0 - cur0) * it / (0.4 * iters))          # plant spread and command size ramp over the first 40 %
        for c in conns:
            c.send(("set", (cur, cur)))
        B = dict(o=np.zeros((T, N, OBS), np.float32), a=np.zeros((T, N, ACT), np.float32), lp=np.zeros((T, N), np.float32), r=np.zeros((T, N), np.float32),
                 v=np.zeros((T, N), np.float32), d=np.zeros((T, N), np.float32))
        for t in range(T):
            b_mu, b_var, b_n = obs.mean(0), obs.var(0), len(obs)             # running normaliser
            tot = cnt + b_n; delta = b_mu - mu
            mu = mu + delta * b_n / tot; var = (var * cnt + b_var * b_n + delta ** 2 * cnt * b_n / tot) / tot; cnt = tot
            sd = np.sqrt(var) + 1e-6
            with torch.no_grad():
                x = torch.as_tensor((obs - mu) / sd, dtype=torch.float32)
                m_ = actor(x); dist = torch.distributions.Normal(m_, log_std.exp())
                a = dist.sample(); lp = dist.log_prob(a).sum(-1); v = critic(x).squeeze(-1)
            for c, chunk in zip(conns, np.split(a.numpy(), n_proc)):
                c.send(("step", chunk))
            res = [c.recv() for c in conns]
            nobs = np.concatenate([r_[0] for r_ in res]); rew = np.concatenate([r_[1] for r_ in res])
            term = np.concatenate([r_[2] for r_ in res]); tout = np.concatenate([r_[3] for r_ in res]); fin = np.concatenate([r_[4] for r_ in res])
            if tout.any():                                                    # time limit is not a failure: bootstrap
                with torch.no_grad():
                    vf = critic(torch.as_tensor((fin[tout] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
                rew[tout] += gamma * vf
            B["o"][t], B["a"][t], B["lp"][t], B["r"][t], B["v"][t], B["d"][t] = (obs - mu) / sd, a.numpy(), lp.numpy(), rew, v.numpy(), (term | tout)
            ep_ret += rew; ep_len += 1
            for i in np.where(term | tout)[0]:
                done_ret.append(ep_ret[i]); done_len.append(ep_len[i]); done_bad.append(bool(term[i])); ep_ret[i] = 0; ep_len[i] = 0
            obs = nobs
        with torch.no_grad():
            last_v = critic(torch.as_tensor((obs - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
        adv = np.zeros((T, N), np.float32); g = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else B["v"][t + 1]
            delta = B["r"][t] + gamma * nv * (1 - B["d"][t]) - B["v"][t]
            g = delta + gamma * lam * (1 - B["d"][t]) * g
            adv[t] = g
        ret = adv + B["v"]
        fl = lambda k: torch.as_tensor(B[k].reshape(T * N, *B[k].shape[2:]))
        o_, a_, lp_, adv_, ret_ = fl("o"), fl("a"), fl("lp"), torch.as_tensor(adv.reshape(-1)), torch.as_tensor(ret.reshape(-1))
        adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)
        for g_ in opt.param_groups:
            g_["lr"] = 3e-4 * (1 - 0.7 * it / iters)
        for _ in range(5):
            for idx in torch.randperm(T * N).split(1296):
                dist = torch.distributions.Normal(actor(o_[idx]), log_std.exp())
                ratio = (dist.log_prob(a_[idx]).sum(-1) - lp_[idx]).exp()
                loss = ((0.0 if it < warm else 1.0) * -torch.min(ratio * adv_[idx], ratio.clamp(1 - clip, 1 + clip) * adv_[idx]).mean()
                        + 0.5 * (critic(o_[idx]).squeeze(-1) - ret_[idx]).pow(2).mean() - (0.0 if it < warm else 0.003) * dist.entropy().sum(-1).mean())
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0); opt.step()
        k = 200
        hist.append(dict(it=it, steps=(it + 1) * T * N, ret=float(np.mean(done_ret[-k:])) if done_ret else 0.0, len=float(np.mean(done_len[-k:])) if done_len else 0.0,
                         crash=float(np.mean(done_bad[-k:])) if done_bad else 1.0, std=float(log_std.exp().mean()), t=time.time() - t0))
        if it % 10 == 0 or it == iters - 1:
            h = hist[-1]
            print(f"it {it:4d} steps {h['steps']/1e6:.2f}M  return {h['ret']:7.1f}  len {h['len']:5.0f}  crash rate {h['crash']:.2f}  std {h['std']:.2f}  {h['t']:.0f}s", flush=True)
            os.makedirs(CK, exist_ok=True)
            L = [l for l in actor if isinstance(l, nn.Linear)]
            np.savez(os.path.join(CK, "policy.npz"), mu=mu, sd=np.sqrt(var) + 1e-6, **{f"w{i}": l.weight.detach().numpy().T for i, l in enumerate(L)},
                     **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(L)})
            json.dump(dict(history=hist, seed=seed, envs=N, horizon=T, obs=OBS, u0=U0, u_scale=U_SCALE), open(os.path.join(CK, "train_log.json"), "w"))
    for c in conns:
        c.send(("close", None))


def evaluate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = dict(np.load(os.path.join(CK, "policy.npz")))
    log = json.load(open(os.path.join(CK, "train_log.json")))["history"]
    env = Env(123)
    # scripted command profile: take off, hover, fly forward, climb, sidestep, yaw, descend
    SCHED = [(0.0, [0, 0, 0.5, 0]), (1.2, [0, 0, 0, 0]), (3.0, [0.4, 0, 0, 0]), (5.0, [0, 0, 0.4, 0]), (6.0, [0, 0.25, 0, 0]), (8.0, [0, 0, 0, 0.8]), (10.0, [0, 0, -0.3, 0]), (11.0, [0, 0, 0, 0])]
    T_END = 12.0

    def fly(thrust, com_x):
        env.fixed = dict(thrust=thrust, com_x=com_x, schedule=SCHED)
        o = env.reset(); L = dict(t=[], v=[], cmd=[], z=[], tilt=[], wz=[], u=[], vbus=[], diff=[], leg=[]); crashed = False
        for k in range(int(T_END / (DT * DECIM))):
            o, r, bad, to, info = env.step(policy_np(P, o))
            for key, val in (("t", k * DT * DECIM), ("v", info["v"]), ("cmd", info["cmd"]), ("z", info["z"]), ("tilt", info["tilt"]), ("wz", info["wz"]),
                             ("u", env.u.copy()), ("vbus", float(env.plant.aux["v_bus"][0])), ("diff", info["diff"]), ("leg", info["leg"])):
                L[key].append(val)
            if bad:
                crashed = True; break
        L = {k_: np.array(v_) for k_, v_ in L.items()}
        air = L["t"] > 1.5
        err = L["v"] - L["cmd"][:, :3]
        out = dict(crashed=crashed, max_tilt=float(L["tilt"].max()), z_peak=float(L["z"].max()), z_end=float(L["z"][-1]))
        if not crashed:
            out.update(rms_vxy=float(np.sqrt(np.mean(np.sum(err[air, :2] ** 2, axis=1)))), rms_vz=float(np.sqrt(np.mean(err[air, 2] ** 2))),
                       rms_wz=float(np.sqrt(np.mean((L["wz"][air] - L["cmd"][air, 3]) ** 2))), took_off=bool(L["z"].max() > 0.3),
                       diff_throttle=float(L["diff"][air].mean()), leg_offset_rms=float(np.sqrt(np.mean(L["leg"][air] ** 2))), leg_offset_max=float(np.abs(L["leg"][air]).max()))
        out["success"] = bool((not crashed) and out["max_tilt"] < 45 and out.get("took_off", False) and out["rms_vxy"] < 0.25 and out["rms_vz"] < 0.2)
        return out, L

    grid_s, grid_c = (0.80, 0.90, 1.00, 1.10, 1.20), (-0.015, -0.0075, 0.0, 0.0075, 0.015)
    base = {}
    if RESIDUAL:                                     # the base controller alone on the same grid
        env.use_policy = False
        for s_ in grid_s:
            for c in grid_c:
                base[f"thrust{s_:.2f}_com{c*1e3:+.1f}mm"] = fly(s_, c)[0]
        env.use_policy = True
        print(f"  base controller alone: {sum(v['success'] for v in base.values())}/25")
    res, ok, vz_rms = {}, np.zeros((5, 5)), np.full((5, 5), np.nan)
    for i, s_ in enumerate(grid_s):
        for j, c in enumerate(grid_c):
            o, L = fly(s_, c)
            res[f"thrust{s_:.2f}_com{c*1e3:+.1f}mm"] = o; ok[i, j] = o["success"]; vz_rms[i, j] = o.get("rms_vz", np.nan)
            if s_ == 1.0 and c == 0.0:
                trace = L
        print(f"  thrust x{s_:.2f}: " + "  ".join(f"com {c*1e3:+5.1f} {'ok ' if ok[i,j] else 'FAIL'} tilt {res[f'thrust{s_:.2f}_com{c*1e3:+.1f}mm']['max_tilt']:4.1f}" for j, c in enumerate(grid_c)))
    # randomised plants from the training distribution (unseen seed), random commands, 6 s each
    env.fixed = None; env.p_air = 0.0; env.rng = np.random.default_rng(999)
    n, good = 100, 0
    for _ in range(n):
        o = env.reset(); bad = False
        for k in range(int(EP_S / (DT * DECIM))):
            o, r, bad, to, info = env.step(policy_np(P, o))
            if bad:
                break
        good += (not bad)
    nom = res["thrust1.00_com+0.0mm"]
    out = dict(command="vx, vy, vz, yaw rate (heading frame)", schedule=SCHED, grid=res, grid_success=int(ok.sum()), grid_n=int(ok.size), random_no_crash=int(good), random_n=n,
               train_steps=log[-1]["steps"], train_time_s=log[-1]["t"], final_return=log[-1]["ret"], final_crash_rate=log[-1]["crash"], nominal=nom)
    out["legs"] = LEGS; out["residual"] = RESIDUAL
    if base:
        both = [k for k in res if res[k]["success"] and base[k]["success"]]
        mean = lambda d_, key: float(np.mean([d_[k][key] for k in both])) if both else None
        out["base"] = dict(grid=base, grid_success=sum(v["success"] for v in base.values()),
                           paired=dict(n=len(both), **{f"{key}_{nm}": mean(d_, key) for key in ("rms_vxy", "rms_vz", "diff_throttle", "max_tilt") for nm, d_ in (("base", base), ("policy", res))}))
        print("  paired cases (both succeed):", out["base"]["paired"])
    json.dump(out, open(os.path.join(ROOT, "docs", TAG + "_results.json"), "w"), indent=1)
    print(f"grid {int(ok.sum())}/{ok.size}; randomised plants without a crash {good}/{n}; nominal: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in nom.items()))
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.0))
    ax = axs[0, 0]
    ax.plot([h["steps"] / 1e6 for h in log], [h["ret"] for h in log], color="#2a6f97"); ax.set_xlabel("environment steps [M]"); ax.set_ylabel("episode return", color="#2a6f97")
    a2 = ax.twinx(); a2.plot([h["steps"] / 1e6 for h in log], [h["crash"] for h in log], color="#d1603d"); a2.set_ylabel("crash rate", color="#d1603d"); ax.set_title("PPO training", fontsize=9)
    L = trace; ax = axs[0, 1]
    for k_, (lab, col) in enumerate((("forward", "#2a6f97"), ("sideways", "#3a7d44"), ("vertical", "#d1603d"))):
        ax.plot(L["t"], L["cmd"][:, k_], color=col, ls=":", lw=1); ax.plot(L["t"], L["v"][:, k_], color=col, label=lab)
    ax.set_xlabel("time [s]"); ax.set_ylabel("velocity [m/s]"); ax.legend(fontsize=7); ax.set_title("Nominal plant: commanded (dotted) and flown velocity", fontsize=9)
    ax = axs[1, 0]
    ax.plot(L["t"], L["z"], color="#2a6f97", label="height [m]"); ax.plot(L["t"], L["tilt"] / 50, color="#d1603d", label="tilt [deg] / 50")
    ax.plot(L["t"], L["cmd"][:, 3], color="#7a5195", ls=":", lw=1); ax.plot(L["t"], L["wz"], color="#7a5195", label="yaw rate [rad/s]"); ax.plot(L["t"], L["u"].mean(1), color="#3a7d44", label="mean throttle")
    ax.set_xlabel("time [s]"); ax.legend(fontsize=7); ax.set_title("Height, tilt, yaw rate, throttle", fontsize=9)
    ax = axs[1, 1]
    im = ax.imshow(vz_rms, origin="lower", cmap="viridis", aspect="auto")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, ("%.2f" % vz_rms[i, j] if np.isfinite(vz_rms[i, j]) else "crash") + ("" if ok[i, j] else "\nfail"), ha="center", va="center", color="w", fontsize=7)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{c*1e3:+.1f}" for c in grid_c]); ax.set_xlabel("centre-of-mass offset, x [mm]")
    ax.set_yticks(range(5)); ax.set_yticklabels([f"{s_:.2f}" for s_ in grid_s]); ax.set_ylabel("true thrust / nominal")
    plt.colorbar(im, ax=ax, label="vertical-velocity rms error [m/s]"); ax.set_title(f"Plant error: {int(ok.sum())}/25 pass", fontsize=9)
    for a_ in (axs[0, 0], axs[0, 1], axs[1, 0]):
        a_.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", TAG + ".pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "eval"]); ap.add_argument("--iters", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init"); ap.add_argument("--cur0", type=float, default=0.25)
    a = ap.parse_args()
    if a.mode == "train":
        mp.set_start_method("spawn"); train(a.iters, seed=a.seed, init=a.init, cur0=a.cur0)
    else:
        evaluate()
