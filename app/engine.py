"""Duck Live engine: the HD1910 Microduck in C MuJoCo on this Mac, driven by the ONNX policies trained on the 5080.

One object (:class:`Duck`) owns the physics (Pollen's ``scene_backlash.xml``, 0.005 s steps), the servo model (Rhoban
BAM m6 fit of the Feetech HD1910 with the Feetech firmware law, battery sag and a 3--6-step command delay, the same
plant the policies trained on), the policies (velstand walk / run / sprint / sit-stand / jump / front flip / stand-up,
hot-swapped at 50 Hz) and the operator logic that turns held keys into policy commands (``app/README.md``, "Operator
logic").

    duck = Duck()                      # loads the contract's default exports
    duck.reset()
    duck.set_keys({"up", "left"}, meta=False, alt=False)
    for _ in range(50): duck.step()    # one second of control at 50 Hz
    duck.snapshot()                    # what the page draws: body poses + telemetry

No rendering, no network: ``server.py`` wraps it for the app, ``bench_mac.py`` measures it for the doc.
"""
from __future__ import annotations

import ast
import glob
import math
import os
import re
import time
from collections import deque

import mujoco
import numpy as np
import onnxruntime as ort

import animation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCENE_XML = os.path.join(HERE, "assets", "microduck", "scene_backlash.xml")
EXPORTS = os.path.join(REPO, "exports")

# ---- the training contract (app/README.md) ------------------------------------------------------------------------------
JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
DEFAULT_POSE = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                         0.0, 0.0873, 0.4579, 0.0049, -0.4530], dtype=np.float32)
PHYSICS_DT = 0.005          # training: sim.dt 0.005, decimation 4 -> 50 Hz control
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION
N_OBS, N_ACT = 61, 14

# Servo: HD1910 BAM m6 fit + Feetech firmware law, as in microduck_rl/assets/microduck_hd1910.py (read from there).
KP_FW = 5.0
ERROR_GAIN_PER_P = 0.166
MAX_PWM = 0.97
VIN_RANGE = (7.4, 8.0)
VIN_DROP_GAIN_RANGE = (0.0, 0.2)
VIN_MIN = 7.0
DELAY_RANGE = (3, 6)        # physics steps

# Command ranges the policies were trained on (velocity_env_cfg.py / velstand_env_cfg.py)
WALK_VX_RANGE = (-0.4, 0.6)
WALK_VY_MAX = 0.3
WALK_YAW_MAX = 1.0
RUN_VX_RANGE = (0.2, 1.1)   # run v3
SPRINT_VX = 1.2             # sprint-max: 1.1-1.2 m/s whatever the command (servos at their speed limit)

# Speed levels (A slower / D faster): one knob sets the walking AND the running top speed. ``run`` = "sprint" means the
# sprint policy (its command is fixed); it falls back to the run policy at RUN_VX_RANGE[1] when no sprint export exists.
# Measured on this plant (app/bench_mac.py, docs/duck_live_app.pdf).
LEVELS = [
    dict(walk=0.20, run=0.30, yaw=0.50, label="1 · stroll"),
    dict(walk=0.30, run=0.50, yaw=0.60, label="2 · easy"),
    dict(walk=0.40, run=0.70, yaw=0.70, label="3 · brisk"),
    dict(walk=0.50, run=0.80, yaw=0.80, label="4 · quick"),
    dict(walk=0.60, run="sprint", yaw=0.80, label="5 · sprint"),
]
# ⌘-drag perturbation, MuJoCo-viewer style: F = m (k_s (target - grab) - k_d v_grab), clamped
PERTURB_STIFFNESS = 120.0   # 1/s^2 (MuJoCo's default map.stiffness is 100)
PERTURB_DAMPING = 8.0       # 1/s
PERTURB_MAX_N = 12.0        # ~1.7 x the duck's weight
STICK_DEAD = 0.12           # the phone's sticks: deflection (of 1) below which an axis reads zero
DEFAULT_LEVEL = 3

STOP_HOLD_S = 0.8           # "stopping" = the walk policy at zero twist; it is the stand once the trunk is slow or after this
STOP_V = 0.05               # m/s
STAND_UP_MIN_S = 0.3        # sit->stand (fast sit/stand policy: 0.26 s): z > STAND_UP_Z and at least this long
STAND_UP_Z = 0.100
STAND_UP_MAX_S = 6.0        # still seated after this: say so (⏎ resets); never walk off a seated pose
SIT_SETTLED_Z = 0.075
SIT_MIN_S = 0.5
FALL_TILT_DEG = 60.0        # contract: tilt past 60 deg for FALL_HOLD_S (or collapsed below FALL_Z) is a fall
FALL_HOLD_S = 0.2
FALL_Z = 0.045              # below the seated height (0.060): collapsed on the belly even if the trunk is level
FALL_RESET_S = 2.0          # only with auto_reset=True and no policy that can get up (a fall is recovered, not undone)
UPRIGHT_TILT_DEG = 20.0     # contract: upright = tilt < 20 deg and trunk z > 0.095 m ...
UPRIGHT_Z = 0.095
RECOVERED_S = 0.5           # ... for 0.5 s, then the get-up policy hands back
ON_FEET_Z = 0.080           # a jump / flip ends "on its feet" above this (the landing crouch is lower than the stand;
                            # the jump task's own stable-landing test uses 0.085); the walk policy then stands it up
RECOVERY_MAX_S = 15.0       # still down after this: say so (⏎ resets), keep trying
VX_SLEW = 1.5               # m/s^2 on the commanded forward speed (steps are in-distribution; this just feels nicer)
YAW_SLEW = 4.0              # rad/s^2
# jump: the training command (microduck_rl/mdp/jump.py JumpOneShotCommand) -- launch flag 1 while "supported" (both
# feet on the ground for JUMP_SUPPORT_STEPS control steps), inside the launch window and not yet landed
JUMP_SUPPORT_STEPS = 4
JUMP_LAUNCH_WINDOW_S = 0.75
JUMP_TAKEOFF_VZ = 0.05
JUMP_TAKEOFF_TILT_DEG = 30.0
JUMP_STABLE_S = 0.1         # landed + quiet for this long -> hand back to the walk policy. Measured on this plant: the jump
                            # policy holds its landing only ~0.3-0.5 s, then drifts forward and tips over (10/12 trials with a
                            # 1 s wait, 1/12 with 0.3 s), so the walk policy takes the stand over as soon as the landing is quiet
JUMP_MAX_S = 2.5            # hand back at the latest here (the task's episode is 3.5 s; recovery times out at 1.4 s)
QUIET_TILT_DEG = 20.0       # "quiet landing": a foot down, tilt below this, z >= ON_FEET_Z + 5 mm, slow (the task's test uses 18 deg)
ROULADE_MIN_S = 1.0         # a front flip takes ~1.5 s; hand back once upright for RECOVERED_S after this ...
ROULADE_MAX_S = 3.0         # ... or here at the latest (not upright -> the get-up policy)
QUIET_V = 0.30              # m/s, QUIET_W rad/s: "quiet" after a landing (the jump task's stable-landing test: 0.2 / 1.5)
QUIET_W = 1.5
# the beak: Pollen's MJCF has no jaw joint (the jaw is welded into the head body), so the opening is drawn, not simulated:
# the page turns the two lower-beak meshes about the beak servo's axis (head-body frame; the XL330 horn at x 3.2 mm,
# z -18 mm, axis +y; + opens). The jaw's mass stays where the policies trained with it.
JAW_MESHES = ("jaw", "jaw_soft")
JAW_PIVOT = (0.0032, 0.0, -0.018)
JAW_AXIS = (0.0, 1.0, 0.0)
JAW_OPEN_RAD = 0.42
HEAD_BODY = "jaw_soft"      # the head link (camera, beak)


def read_bam_fit() -> dict:
    """``HD1910_BAM_FIT`` from microduck_rl/assets/microduck_hd1910.py (no Isaac Lab import on the Mac)."""
    src = os.path.join(REPO, "microduck_rl", "assets", "microduck_hd1910.py")
    with open(src) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(tgt, ast.Name) and tgt.id == "HD1910_BAM_FIT":
                return ast.literal_eval(node.value)
    raise RuntimeError("HD1910_BAM_FIT not found in " + src)


def quat_rotate_inverse(q, v):
    """Rotate ``v`` by the inverse of the unit quaternion ``q`` = (w, x, y, z)."""
    w, xyz = q[0], q[1:4]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---- policies -----------------------------------------------------------------------------------------------------------
# role -> export name pattern(s), newest iteration first. walk: VelStand (walk + balance + get-up, the contract's default)
# or a plain walk export; recovery: the standalone stand-up policy (used after a fall under run / sprint / sit / jump /
# flip; the VelStand walk gets up by itself).
ROLES = dict(walk=("velstand_*.onnx", "walk_*.onnx"), run="run_*.onnx", sprint="sprint_*.onnx", sitstand="sitstand_*.onnx",
             jump="jump_*.onnx", roulade="roulade_*.onnx", recovery=("standup_*.onnx", "recovery_*.onnx", "getup_*.onnx"))
DEFAULTS = dict(walk="velstand_allcoll_hd1910_iter21495.onnx", run="run_v3_allcoll_hd1910_iter5999.onnx",
                sprint="sprint_max_allcoll_hd1910_iter7998.onnx", sitstand="sitstand_fast_hd1910_iter5499.onnx",
                jump="jump_hd1910_iter1500.onnx", roulade="roulade_head_hd1910_iter7000.onnx", recovery="standup_hd1910_iter9999.onnx")
REQUIRED = ("walk", "run", "sitstand")
SELF_RECOVERING = ("velstand_",)    # walk exports that get up on their own (VelStand recipe: no fall termination in training)


def list_exports() -> dict:
    """{role: [file names]} of the deployable ONNX exports, newest iteration first."""
    def it(f):
        m = re.search(r"iter(\d+)", f)
        return int(m.group(1)) if m else -1
    out = {}
    for role, pat in ROLES.items():
        pats = (pat,) if isinstance(pat, str) else pat
        fs = sorted(set(os.path.basename(f) for p in pats for f in glob.glob(os.path.join(EXPORTS, p))), key=lambda f: -it(f))
        out[role] = fs
    return out


class Policy:
    """One ONNX actor: 61 raw observations in (normaliser baked in), 14 actions out."""

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        i, o = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
        if list(i.shape) != [1, N_OBS] or list(o.shape) != [1, N_ACT]:
            raise ValueError(f"{self.name}: expected [1,{N_OBS}] -> [1,{N_ACT}], got {i.shape} -> {o.shape}")
        self.iname, self.oname = i.name, o.name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.sess.run([self.oname], {self.iname: obs.reshape(1, -1).astype(np.float32)})[0][0]


# ---- servos: BAM HD1910 on C MuJoCo -------------------------------------------------------------------------------------
class Servos:
    """The 14 HD1910 servos as the training drive: Feetech P law on a rate-unlimited target, BAM m6 motor + friction,
    battery sag ``V = max(V_min, V_0 - g * sum|tau|)``, a fixed command delay of ``lag`` physics steps, armature on
    the dofs. Position actuators become torque motors (``forcerange`` = +-V_max kt / R); joint damping and friction
    loss are zeroed because BAM rewrites them every step, with the stiff friction constraint training used.
    """

    def __init__(self, xml: str, vin: float, sag_gain: float, lag: int, timestep: float = PHYSICS_DT):
        from bam.model import load_model_from_dict
        from bam.mujoco import MujocoController

        fit = read_bam_fit()
        self.fit = fit
        bam = load_model_from_dict(fit)
        act = bam.actuator
        act.kp = KP_FW
        act.vin = vin
        act.error_gain = ERROR_GAIN_PER_P
        act.max_pwm = MAX_PWM
        act.max_current = None
        self.bam = bam
        self.kt, self.R = float(bam.kt.value), float(bam.R.value)
        force_limit = VIN_RANGE[1] * self.kt / self.R

        spec = mujoco.MjSpec.from_file(xml)
        names = []
        for a in spec.actuators:
            tgt = a.target
            tgt = tgt.name if hasattr(tgt, "name") else str(tgt)
            if tgt.startswith("passive_"):
                continue
            a.set_to_motor()
            a.forcelimited = True
            a.forcerange = (-force_limit, force_limit)
            a.ctrllimited = False
            a.gear = [1.0, 0, 0, 0, 0, 0]
            names.append(a.name)
            for j in spec.joints:
                if j.name == tgt:
                    j.damping = np.zeros((3, 1))     # MjsJoint takes a (3,1) array
                    j.frictionloss = 0.0
                    j.solref_friction = (-5.0e4, -2.0e2)
                    j.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)
                    break
        if names != JOINTS:
            raise RuntimeError(f"actuator order {names} != contract {JOINTS}")
        self.model = spec.compile()
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.ctrl = MujocoController(bam, names, self.model, self.data)
        self.names = names
        self.vin0, self.sag_gain, self.vin_min = float(vin), float(sag_gain), VIN_MIN
        self.lag = int(lag)
        self.force_limit = force_limit
        self.qpos_idx = np.array(self.ctrl.qpos_indexes)
        self.dof_idx = np.array(self.ctrl.dof_indexes)
        jr = np.array([self.model.jnt_range[self.model.actuator_trnid[i, 0]] for i in range(self.model.nu)])
        self.q_lo, self.q_hi = jr[:, 0], jr[:, 1]
        self.queue: deque = deque(maxlen=max(1, self.lag + 1))
        self.vin_eff = self.vin0
        self.reset(DEFAULT_POSE)

    def set_lag(self, lag: int):
        """Change the command delay [physics steps]; takes effect at the next reset."""
        self.lag = int(lag)
        self.queue = deque(maxlen=max(1, self.lag + 1))

    def reset(self, target: np.ndarray):
        self.ctrl.reset()
        self.queue.clear()
        for _ in range(self.queue.maxlen):
            self.queue.append(np.array(target, dtype=np.float64))
        self.ctrl.q_target[:] = target

    def command(self, target: np.ndarray):
        """New servo targets from the policy [rad]; applied after ``lag`` physics steps (push once per physics step).

        Not clipped to the joint range: Isaac Lab's joint-position action does not clip either, and the policies use
        targets beyond a limit to drive a joint against it with more torque (the stand-up from the learned seated
        pose relies on it: with the targets clipped the knee stalls 17 deg short and the duck never rises).
        """
        self.queue.append(np.asarray(target, dtype=np.float64))

    def substep(self):
        """One physics step: delayed target -> firmware law -> motor torque -> frictions -> mj_step."""
        self.ctrl.q_target[:] = self.queue[0]
        # battery sag from the previous step's torques (the Isaac Lab BAM drive's model)
        tau = np.abs(self.data.qfrc_actuator[self.dof_idx]).sum()
        self.vin_eff = max(self.vin_min, self.vin0 - self.sag_gain * tau)
        self.bam.actuator.vin = self.vin_eff
        self.ctrl.update()
        mujoco.mj_step(self.model, self.data)


# ---- the duck -------------------------------------------------------------------------------------------------------------
class Duck:
    """Physics + servos + policies + operator logic. All public methods are called from one thread."""

    STATES = ("stand", "walk", "run", "sprint", "stopping", "sit_down", "seated", "stand_up", "jump", "roulade", "recover", "fallen", "dead")

    def __init__(self, policies: dict | None = None, vin: float = sum(VIN_RANGE) / 2, sag_gain: float = 0.1,
                 lag: int = 4, level: int = DEFAULT_LEVEL, auto_reset: bool = False, walk_self_recovery: bool = True,
                 timestep: float = PHYSICS_DT, decimation: int = DECIMATION, lively: bool = False, react: bool = False,
                 anim_seed: int = 0):
        # lively: idle life (glances, breathing, the head leading turns ...); react: automatic reactions (animation.py).
        # Both off by default, so the plain engine is exactly the policies (bench_mac.py); the app turns them on.
        # walk_self_recovery: a fall under a self-recovering walk (VelStand) is left to that policy, which keeps tracking
        # the command (the contract); False hands every fall to the stand-up policy instead (bench_mac.py compares them)
        self.walk_self_recovery = bool(walk_self_recovery)
        self.servos = Servos(SCENE_XML, vin=vin, sag_gain=sag_gain, lag=lag, timestep=timestep)
        self.model, self.data = self.servos.model, self.servos.data
        self.decimation = int(decimation)
        self.control_dt = self.model.opt.timestep * self.decimation
        m = self.model
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")]
        self.free_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
        self.free_dadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
        # the encoder sees servo + gearbox play (Isaac Lab joint_pos_rel_biased / joint_vel_rel_backlash with backlash_cfg)
        bl = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "passive_%s_backlash" % j) for j in JOINTS]
        self.bl_qidx = np.array([m.jnt_qposadr[j] for j in bl])
        self.bl_didx = np.array([m.jnt_dofadr[j] for j in bl])
        self.key_stand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.key_sit = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "SIT")
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"),
                     mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"))

        self.policies: dict[str, Policy] = {}
        self.load_policies(policies or {})
        self.level = int(level)
        self.auto_reset = auto_reset
        self.held: set[str] = set()
        self.meta = False
        self.alt = False
        self.stick = None           # (x right, y forward, w clockwise) in -1..1 from the phone's sticks; None = the keys rule
        self.sit_pending = False
        self.notice = ""
        self.notice_t = 0.0
        self.falls = 0
        self.ticks = 0
        self.step_ms = deque(maxlen=250)
        self.perturb = None          # dict(body, local, target) while the user ⌘-drags the duck
        self.perturb_force = np.zeros(3)
        self.perturb_point = np.zeros(3)
        self.total_mass = float(self.model.body_subtreemass[self.trunk])
        self.head_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HEAD_BODY)
        self.events: list[str] = []
        self.pose = None             # play-dead: the scripted joint pose while the policies are off
        self.sit_toggled_by_anim = False
        self.behavior_by_anim = False
        self.anim = None
        self.anim = animation.Animator(self, seed=anim_seed, lively=lively, react=react)
        self.reset()

    # ---- policies --------------------------------------------------------------------------------------------------------
    def load_policies(self, chosen: dict):
        """``chosen`` = {role: file name under exports/ ("" = off)}; unnamed roles take DEFAULTS, else the newest export;
        a missing optional role (sprint / jump / roulade / recovery) is simply absent."""
        avail = list_exports()
        for role in ROLES:
            if role in chosen and not chosen[role] and role not in REQUIRED:
                self.policies.pop(role, None)
                continue
            name = chosen.get(role) or DEFAULTS.get(role) or (avail[role][0] if avail[role] else None)
            if name is not None and not os.path.exists(name if os.path.isabs(name) else os.path.join(EXPORTS, name)):
                name = avail[role][0] if avail[role] else None       # the default is not in this checkout: newest export
            if name is None:
                if role in REQUIRED:
                    raise FileNotFoundError(f"no export for the {role} role in {EXPORTS}")
                self.policies.pop(role, None)
                continue
            path = name if os.path.isabs(name) else os.path.join(EXPORTS, name)
            if role not in self.policies or self.policies[role].path != path:
                self.policies[role] = Policy(path)
        self.walk_recovers = any(self.policies["walk"].name.startswith(p) for p in SELF_RECOVERING)

    def recover_role(self):
        """The policy that gets the duck up after a fall: the stand-up export, else a self-recovering walk, else none."""
        if "recovery" in self.policies:
            return "recovery"
        return "walk" if self.walk_recovers else None

    # ---- state -----------------------------------------------------------------------------------------------------------
    def reset(self, seated: bool = False, yaw: float = 0.0, xy=(0.0, 0.0)):
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.key_sit if seated else self.key_stand)
        d.qpos[self.free_qadr:self.free_qadr + 2] = xy
        d.qpos[self.free_qadr + 3:self.free_qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[:] = 0
        d.ctrl[:] = 0
        target = d.qpos[self.servos.qpos_idx].copy() if seated else DEFAULT_POSE.astype(np.float64)
        self.servos.reset(target)
        mujoco.mj_forward(m, d)
        self.last_action = np.zeros(N_ACT, dtype=np.float32)
        self.cmd = np.zeros(13, dtype=np.float32)
        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.yaw_cmd = 0.0
        self.state = "seated" if seated else "stand"
        self.state_t0 = d.time
        self.sit_pending = False
        self.down_since = None       # the trunk has been past FALL_TILT_DEG / below FALL_Z since
        self.down = False            # a fall is latched until the trunk is upright again
        self.upright_since = None
        self.quiet_since = None
        self.jump = None             # the jump task's per-episode state while jumping (see operate)
        self.roll = None             # the front flip's progress while rolling
        self.policy_role = "sitstand" if seated else "walk"
        self.v_body = np.zeros(3)
        self.wz = 0.0
        self.v_filt = np.zeros(3)
        self.wz_filt = 0.0
        self.sim_t0 = d.time
        self.p0 = d.qpos[self.free_qadr:self.free_qadr + 2].copy()
        self.odometer = 0.0
        self.p_prev = self.p0.copy()
        self.events = ["reset"]
        self.pose = None
        self.sit_toggled_by_anim = False
        self.behavior_by_anim = False
        if self.anim is not None:
            self.anim.reset()

    def emit(self, ev: str):
        """Something the animation layer may react to: back_up, jump_ok, jump_fail, flip_ok, reset."""
        self.events.append(ev)

    def take_events(self):
        ev, self.events = self.events, []
        return ev

    def emote(self, name: str) -> bool:
        """Play an emotion clip (animations/manifest.json) as soon as the duck can."""
        ok = self.anim.play(name)
        if not ok and name not in self.anim.lib:
            self.say("no such emotion: " + name)
        return ok

    def set_keys(self, held, meta: bool = False, alt: bool = False):
        """The movement keys currently held ({"up","down","left","right"}) and the modifier state."""
        self.held = set(k for k in held if k in ("up", "down", "left", "right"))
        self.meta, self.alt = bool(meta), bool(alt)
        self.stick = None

    def set_stick(self, x: float, y: float, w: float, run: bool = False):
        """Proportional command (the phone app's two sticks): x = sidestep right, y = forward, w = turn clockwise, each
        -1..1 of the current speed level's walk, sidestep and turn speeds. All three inside the dead zone = released.
        `run` asks for the run / sprint policy; it only acts on a pure forward push (those policies cannot turn)."""
        v = tuple(0.0 if abs(a) < STICK_DEAD else float(np.clip(a, -1.0, 1.0)) for a in (x, y, w))
        self.held, self.alt = set(), False
        self.stick = v if any(v) else None
        self.meta = bool(run) and self.stick is not None

    def press(self, key: str):
        """Momentary keys: s (sit / stand up), space (jump), r (front flip), a / d (speed), enter (reset)."""
        if self.state == "dead" and key != "enter":
            self.wake()
            return
        if key in ("s", "space", "r"):
            self.anim.cancel(cut=True)          # the user's own command comes first
        if key == "a":
            self.level = max(1, self.level - 1)
            self.say("speed %s" % LEVELS[self.level - 1]["label"])
        elif key == "d":
            self.level = min(len(LEVELS), self.level + 1)
            self.say("speed %s" % LEVELS[self.level - 1]["label"])
        elif key == "enter":
            self.reset()
        elif key == "s":
            self.toggle_sit()
        elif key in ("space", "r"):
            role = "jump" if key == "space" else "roulade"
            what = "jump" if role == "jump" else "front flip"
            if role not in self.policies:
                self.say(what + ": no policy in exports/")
            elif self.state in ("stand", "stopping") and not self.down and self.tilt_deg < UPRIGHT_TILT_DEG and self.z > UPRIGHT_Z:
                self.start_behavior(role)
            else:
                self.say(what + ": stand still first")

    def start_behavior(self, role: str):
        """Hand over to an episodic policy (jump / front flip / stand-up) the way its training episode starts: fresh
        last-action buffer, zero command; the walk / run / sit-stand handovers keep the shared buffer instead."""
        self.state_to(role if role != "recovery" else "recover")
        self.last_action[:] = 0
        self.vx_cmd = self.vy_cmd = self.yaw_cmd = 0.0
        self.upright_since = None
        self.quiet_since = None
        if role == "jump":
            self.jump = dict(grounded=0, support=False, support_s=0.0, airborne=False, landed=False, air_steps=0,
                             z0=self.z, rise=0.0, launch=False)
        elif role == "roulade":
            self.roll = dict(angle=0.0, peak=0.0)

    def toggle_sit(self):
        if self.state in ("seated", "sit_down"):
            self.state_to("stand_up")
        elif self.state == "stand_up":
            self.state_to("sit_down")
        elif self.state in ("stand", "stopping"):
            self.state_to("sit_down")
        elif self.state in ("walk", "run", "sprint"):
            self.sit_pending = True
            self.say("stopping, then sitting")

    # ---- play dead: the one clip that switches the policies off (Pollen's pose_joints stages) --------------------------------
    def pose_stage(self, sg: dict):
        """Enter / advance the scripted pose: the joints in ``targets`` ramp linearly from where they are to their
        absolute angles over ``ramp_s``, the ``torque_off`` joints hang free (their target follows the encoder), the
        rest hold. No policy runs until :meth:`wake`."""
        q = self.data.qpos[self.servos.qpos_idx].copy()
        start = self.pose_target() if self.pose is not None else q
        goal = start.copy()
        for name, val in sg.get("targets", {}).items():
            goal[JOINTS.index(name)] = float(val)
        free = [JOINTS.index(n) for n in sg.get("torque_off", []) if n in JOINTS]
        self.pose = dict(start=start, goal=goal, free=free, t0=self.data.time, ramp=max(1e-3, float(sg.get("ramp_s", 1.0))))
        self.state_to("dead")
        self.last_action[:] = 0

    def pose_target(self) -> np.ndarray:
        p = self.pose
        u = min(1.0, (self.data.time - p["t0"]) / p["ramp"])
        target = p["start"] + (p["goal"] - p["start"]) * u
        if p["free"]:
            target[p["free"]] = self.data.qpos[self.servos.qpos_idx][p["free"]]
        return target

    def wake(self):
        """End of playing dead: the get-up policy takes over (it is a fall as far as the policies are concerned)."""
        if self.state != "dead":
            return
        self.pose = None
        self.anim.cancel(cut=True)
        self.down = True
        if self.recover_role() is not None:
            self.start_behavior("recovery")
            self.say("alive after all · getting up")
        else:
            self.reset()

    def set_perturb(self, body: int, local=None, target=None):
        """Start / move / end the ⌘-drag spring: ``body`` < 0 ends it; ``local`` is the grab point in the body frame,
        ``target`` the cursor's point in the world (on a plane parallel to the camera through the grab point)."""
        if body is None or body < 0 or local is None or target is None:
            self.perturb = None
            self.perturb_force[:] = 0
            return
        self.perturb = dict(body=int(body), local=np.asarray(local, dtype=np.float64), target=np.asarray(target, dtype=np.float64))

    def apply_perturb(self):
        """The spring force on the grabbed body for this physics step (into qfrc_applied, cleared first)."""
        d = self.data
        d.qfrc_applied[:] = 0
        if self.perturb is None:
            return
        b = self.perturb["body"]
        R = d.xmat[b].reshape(3, 3)
        p = d.xpos[b] + R @ self.perturb["local"]
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)   # world frame: (rot, lin)
        lin = v[3:6] + np.cross(v[0:3], p - d.xpos[b])
        F = self.total_mass * (PERTURB_STIFFNESS * (self.perturb["target"] - p) - PERTURB_DAMPING * lin)
        n = np.linalg.norm(F)
        if n > PERTURB_MAX_N:
            F *= PERTURB_MAX_N / n
        mujoco.mj_applyFT(self.model, d, F, np.zeros(3), p, b, d.qfrc_applied)
        self.perturb_force[:] = F
        self.perturb_point[:] = p

    def push(self, vx: float, vy: float):
        """A velocity impulse on the trunk (training's ``push_by_setting_velocity``, scripts/push_test.py): adds
        (vx, vy) [m/s] to the whole robot's horizontal velocity."""
        d = self.data
        d.qvel[self.free_dadr] += vx
        d.qvel[self.free_dadr + 1] += vy
        mujoco.mj_forward(self.model, d)

    def say(self, text: str):
        self.notice, self.notice_t = text, self.data.time

    def state_to(self, s: str):
        if s != self.state:
            self.state = s
            self.state_t0 = self.data.time

    @property
    def state_time(self):
        return self.data.time - self.state_t0

    # ---- one control tick --------------------------------------------------------------------------------------------------
    def step(self):
        t0 = time.perf_counter()
        a = self.anim
        a.update()                                        # emotions / idle life: head offsets, beak, clip twist, skills
        self.operate()
        role = self.policy_role
        if self.state == "fallen":
            target = DEFAULT_POSE.astype(np.float64)      # nothing can get it up: hold the default pose where it fell
        elif self.state == "dead":
            target = self.pose_target()                   # playing dead: scripted pose, no policy
        else:
            mode = animation.HEAD_MODE.get(role) if a.w > 0.0 else None
            if mode == "cmd":                             # the policy tracks its head command (sit-stand)
                self.cmd[3:7] = a.w * a.head
            obs = self.observe()
            action = self.policies[role](obs)
            self.last_action = action.astype(np.float32)
            target = DEFAULT_POSE + action
            if mode == "direct":                          # the policy does not (VelStand): drive the head servos, it balances
                target[5:9] = (1.0 - a.w) * target[5:9] + a.w * (DEFAULT_POSE[5:9] + a.head)
        for _ in range(self.decimation):
            self.servos.command(target)
            self.apply_perturb()
            self.servos.substep()
        self.measure()
        self.ticks += 1
        self.step_ms.append((time.perf_counter() - t0) * 1e3)

    def observe(self) -> np.ndarray:
        d, s = self.data, self.servos
        q = d.qpos[self.free_qadr + 3:self.free_qadr + 7]
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        grav = quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))
        jpos = d.qpos[s.qpos_idx] + d.qpos[self.bl_qidx] - DEFAULT_POSE
        jvel = d.qvel[s.dof_idx] + d.qvel[self.bl_didx]
        return np.concatenate([gyro, grav, jpos, jvel, self.last_action, self.cmd]).astype(np.float32)

    def measure(self):
        d = self.data
        q = d.qpos[self.free_qadr + 3:self.free_qadr + 7]
        v_world = d.qvel[self.free_dadr:self.free_dadr + 3]
        self.v_body = quat_rotate_inverse(q, v_world)
        self.wz = float(d.qvel[self.free_dadr + 5])
        a = 0.08  # ~0.25 s time constant at 50 Hz
        self.v_filt = (1 - a) * self.v_filt + a * self.v_body
        self.wz_filt = (1 - a) * self.wz_filt + a * self.wz
        p = d.qpos[self.free_qadr:self.free_qadr + 2]
        self.odometer += float(np.linalg.norm(p - self.p_prev))
        self.p_prev = p.copy()

    @property
    def z(self):
        return float(self.data.qpos[self.free_qadr + 2])

    @property
    def tilt_deg(self):
        q = self.data.qpos[self.free_qadr + 3:self.free_qadr + 7]
        g = quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))
        return math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))

    @property
    def upright(self):
        return self.tilt_deg < UPRIGHT_TILT_DEG and self.z > UPRIGHT_Z

    def feet_contact(self):
        """(left, right): foot collision geom touching the floor."""
        d = self.data
        on = [False, False]
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.floor in (g1, g2):
                other = g2 if g1 == self.floor else g1
                for k in range(2):
                    if other == self.feet[k]:
                        on[k] = True
        return on[0], on[1]

    # ---- operator logic ----------------------------------------------------------------------------------------------------
    def intent(self):
        if self.stick is not None:
            x, y, w = self.stick
            turning = x != 0.0 or w != 0.0
            return y > 0, y < 0, x < 0 or w < 0, x > 0 or w > 0, True, self.meta and y > 0 and not turning
        h = self.held
        fwd = "up" in h and "down" not in h
        back = "down" in h and "up" not in h
        left = "left" in h and "right" not in h
        right = "right" in h and "left" not in h
        moving = fwd or back or left or right
        run = self.meta and fwd
        return fwd, back, left, right, moving, run

    def gait_for(self, run: bool):
        """walk / run / sprint for the held keys at the current speed level."""
        if not run:
            return "walk"
        lv = LEVELS[self.level - 1]
        return "sprint" if lv["run"] == "sprint" and "sprint" in self.policies else "run"

    def operate(self):
        """Held keys + momentary presses -> state, active policy and the 13-wide command."""
        fwd, back, left, right, moving, run = self.intent()
        lv = LEVELS[self.level - 1]
        t = self.data.time
        st = self.state
        if st == "dead":                                  # playing dead: no policy, no fall logic; any key wakes it
            self.policy_role = "walk"
            self.cmd[:] = 0
            self.vx_cmd = self.vy_cmd = self.yaw_cmd = 0.0
            self.down_since = None
            if moving:
                self.wake()
            return
        # ---- falls (contract: tilt > 60 deg for 0.2 s, or collapsed below 0.045 m). Not judged mid-jump / mid-flip
        # (a flip is upside down by design); those hand over to the get-up policy themselves if they end on the floor.
        down_now = self.tilt_deg > FALL_TILT_DEG or (self.z < FALL_Z and self.state_time > 0.5)
        if down_now:
            self.down_since = self.down_since if self.down_since is not None else t
        else:
            self.down_since = None
        fell = self.down_since is not None and t - self.down_since >= FALL_HOLD_S
        if fell and not self.down and st not in ("jump", "roulade", "recover", "fallen"):
            self.down = True
            self.falls += 1
            self.sit_pending = False
            rr = self.recover_role()
            if self.walk_recovers and self.policy_role == "walk" and self.walk_self_recovery:
                self.say("fell · the walk policy gets up by itself")   # VelStand: keep its command, it recovers while tracking it
            elif rr is not None:
                self.start_behavior("recovery")
                self.say("fell · stand-up policy" if rr == "recovery" else "fell · walk policy at zero twist gets up")
            else:
                self.state_to("fallen")
                self.say("fell · resetting" if self.auto_reset else "fell · no get-up policy in exports/ · ⏎ resets")
            st = self.state
        if self.down and self.upright and st not in ("recover", "fallen", "jump", "roulade"):
            self.down = False
            self.say("back up")
            self.emit("back_up")
        # upright-for-RECOVERED_S timer (hand-back rule of the get-up policy and the flip)
        if self.upright:
            self.upright_since = self.upright_since if self.upright_since is not None else t
        else:
            self.upright_since = None
        upright_for = (t - self.upright_since) if self.upright_since is not None else 0.0

        if st == "recover":
            rr = self.recover_role()
            if rr is None:
                self.state_to("fallen")
                st = "fallen"
            else:
                self.policy_role = rr
                self.cmd[:] = 0
                self.vx_cmd = self.vy_cmd = self.yaw_cmd = 0.0
                if upright_for >= RECOVERED_S and self.state_time > RECOVERED_S:
                    self.down = False
                    self.state_to("stand")
                    self.say("back up")
                    self.emit("back_up")
                elif self.state_time > RECOVERY_MAX_S and int(self.state_time * 50) % 150 == 0:
                    self.say("still trying to get up · ⏎ resets")
                return
        if st == "fallen":
            if self.auto_reset and self.state_time > FALL_RESET_S:
                yaw = yaw_of(self.data.qpos[self.free_qadr + 3:self.free_qadr + 7])
                self.reset(yaw=yaw, xy=self.data.qpos[self.free_qadr:self.free_qadr + 2].copy())
            self.policy_role = "walk"
            self.cmd[:] = 0
            return
        if st == "jump":
            self.operate_jump(upright_for)
            return
        if st == "roulade":
            self.operate_roulade(upright_for)
            return
        if st in ("seated", "sit_down"):
            if moving:
                self.state_to("stand_up")
                st = "stand_up"
            elif st == "sit_down" and self.z < SIT_SETTLED_Z and self.state_time > SIT_MIN_S:
                self.state_to("seated")
                st = "seated"
        if st == "stand_up":
            if self.z > STAND_UP_Z and self.state_time > STAND_UP_MIN_S:
                self.state_to("stand")
                st = "stand"
            elif self.state_time > STAND_UP_MAX_S and int(self.state_time * 50) % 100 == 0:
                self.say("still getting up · ⏎ resets")
        if st == "stand":
            if moving:
                self.state_to(self.gait_for(run))
                st = self.state
        elif st in ("walk", "run", "sprint"):
            if not moving:
                self.state_to("stopping")
                st = "stopping"
            elif self.gait_for(run) != st:
                self.state_to(self.gait_for(run))
                st = self.state
        if st == "stopping":
            if moving:
                self.state_to(self.gait_for(run))
                st = self.state
            elif self.state_time > STOP_HOLD_S or (self.state_time > 0.3 and float(np.linalg.norm(self.v_filt[:2])) < STOP_V):
                if self.sit_pending:
                    self.sit_pending = False
                    self.state_to("sit_down")
                    st = "sit_down"
                else:
                    self.state_to("stand")
                    st = "stand"
        # ---- command + policy for the resolved state
        self.cmd[:] = 0
        if st in ("walk", "stopping", "stand"):
            # the walk policy at zero twist is the stand (VelStand: balanced stand at 0.116 m; a plain walk export squats)
            self.policy_role = "walk"
            if st == "walk":
                vx = lv["walk"] if fwd else (-min(-WALK_VX_RANGE[0], lv["walk"]) if back else 0.0)
                if self.stick is not None:               # proportional: fractions of this level's speeds (+vy and +yaw are to the left)
                    x, y, w = self.stick
                    vx = y * (lv["walk"] if y > 0 else min(-WALK_VX_RANGE[0], lv["walk"]))
                    vy = -x * min(WALK_VY_MAX, 0.7 * lv["walk"])
                    yaw = -w * lv["yaw"]
                elif self.alt:
                    vy = (min(WALK_VY_MAX, 0.7 * lv["walk"]) if left else 0.0) - (min(WALK_VY_MAX, 0.7 * lv["walk"]) if right else 0.0)
                    yaw = 0.0
                else:
                    vy = 0.0
                    yaw = (lv["yaw"] if left else 0.0) - (lv["yaw"] if right else 0.0)
            else:
                vx = vy = yaw = 0.0
            tw = self.anim.twist
            if tw is not None and st != "walk":           # a clip's own twist (wiggle, side-step): already smooth, not slewed
                self.vx_cmd = float(np.clip(tw[0], WALK_VX_RANGE[0], WALK_VX_RANGE[1]))
                self.vy_cmd = float(np.clip(tw[1], -WALK_VY_MAX, WALK_VY_MAX))
                self.yaw_cmd = float(np.clip(tw[2], -WALK_YAW_MAX, WALK_YAW_MAX))
            else:
                self.slew(vx, vy, yaw)
            self.cmd[0:3] = (self.vx_cmd, self.vy_cmd, self.yaw_cmd)
        elif st in ("run", "sprint"):
            self.policy_role = st
            if left or right:
                self.say("running goes straight · release ⌘ to turn")
            if st == "sprint":
                self.vx_cmd, self.vy_cmd, self.yaw_cmd = SPRINT_VX, 0.0, 0.0     # the sprint policy's command is fixed
            else:
                vx = RUN_VX_RANGE[1] if lv["run"] == "sprint" else lv["run"]
                self.slew(vx, 0.0, 0.0, vx_lo=RUN_VX_RANGE[0])
            self.cmd[0:3] = (self.vx_cmd, 0.0, 0.0)
        else:
            self.policy_role = "sitstand"
            self.vx_cmd = self.vy_cmd = self.yaw_cmd = 0.0
            self.cmd[0] = 1.0 if st in ("sit_down", "seated") else 0.0

    def operate_jump(self, upright_for: float):
        """Space: the jump policy with the training command (``JumpOneShotCommand``): launch flag 1 once both feet have
        been down for 4 control steps, for 0.75 s of support time, until the first touchdown after a flight. Hands back
        to the walk policy once the landing is quiet for 1 s (or at JUMP_MAX_S); on the floor -> the get-up policy."""
        j = self.jump
        dt = self.control_dt
        lf, rf = self.feet_contact()
        both, any_ = lf and rf, lf or rf
        vz = float(self.data.qvel[self.free_dadr + 2])
        tilt = self.tilt_deg
        if not j["airborne"]:
            j["grounded"] = j["grounded"] + 1 if both else 0
        j["support"] = j["support"] or j["grounded"] >= JUMP_SUPPORT_STEPS
        if j["support"]:
            j["support_s"] += dt
        if j["support"] and not j["airborne"] and not any_ and vz >= JUMP_TAKEOFF_VZ and tilt <= JUMP_TAKEOFF_TILT_DEG:
            j["airborne"] = True
        if j["airborne"] and not j["landed"]:
            if not any_:
                j["air_steps"] += 1
                j["rise"] = max(j["rise"], self.z - j["z0"])
            elif j["air_steps"] >= 1:
                j["landed"] = True
        j["launch"] = j["support"] and j["support_s"] <= JUMP_LAUNCH_WINDOW_S and not j["landed"]
        self.policy_role = "jump"
        self.cmd[:] = 0
        self.cmd[0] = 1.0 if j["launch"] else 0.0
        quiet = (any_ and tilt < QUIET_TILT_DEG and self.z >= ON_FEET_Z + 0.005 and float(np.linalg.norm(self.v_body)) < QUIET_V
                 and float(np.linalg.norm(self.data.qvel[self.free_dadr + 3:self.free_dadr + 6])) < QUIET_W)
        if quiet and j["landed"]:
            self.quiet_since = self.quiet_since if self.quiet_since is not None else self.data.time
        else:
            self.quiet_since = None
        quiet_for = (self.data.time - self.quiet_since) if self.quiet_since is not None else 0.0
        if quiet_for >= JUMP_STABLE_S - 1e-9 or self.state_time >= JUMP_MAX_S - 1e-9:
            self.end_behavior("jump · rose %.0f mm" % (j["rise"] * 1000) if j["airborne"] else "jump · no take-off")

    def operate_roulade(self, upright_for: float):
        """R: the front flip policy at zero twist (training: near-zero twist noise). Rotation about the trunk's pitch
        axis is integrated from the gyro for the HUD / bench. Hands back to the walk once upright for RECOVERED_S after
        ROULADE_MIN_S, or at ROULADE_MAX_S (on the floor -> the get-up policy)."""
        r = self.roll
        wy = float(self.data.sensordata[self.gyro_adr + 1])         # trunk-frame pitch rate; nose-down positive
        r["angle"] += wy * self.control_dt
        r["peak"] = max(r["peak"], r["angle"])
        self.policy_role = "roulade"
        self.cmd[:] = 0
        if (self.state_time >= ROULADE_MIN_S and upright_for >= RECOVERED_S - 1e-9) or self.state_time >= ROULADE_MAX_S - 1e-9:
            self.end_behavior("front flip · rolled %.0f°" % math.degrees(r["peak"]))

    def end_behavior(self, text: str):
        """A jump / flip is over: back to the walk policy if on its feet, else the get-up policy (the fall is counted)."""
        by_anim, self.behavior_by_anim = self.behavior_by_anim, False
        if self.tilt_deg < UPRIGHT_TILT_DEG and self.z > ON_FEET_Z:
            self.state_to("stand")
            self.say(text)
            if not by_anim:                               # a trick the user asked for: the duck has an opinion about it
                self.emit("flip_ok" if self.roll is not None else ("jump_ok" if self.jump and self.jump["airborne"] else "jump_fail"))
        else:
            self.falls += 1
            self.down = True
            rr = self.recover_role()
            if rr is None:
                self.state_to("fallen")
                self.say(text + " · fell · ⏎ resets")
            else:
                self.start_behavior("recovery")
                self.say(text + " · fell · getting up")
        self.jump = None
        self.roll = None

    def slew(self, vx, vy, yaw, vx_lo=None):
        dt = self.control_dt
        self.vx_cmd += float(np.clip(vx - self.vx_cmd, -VX_SLEW * dt, VX_SLEW * dt))
        self.vy_cmd += float(np.clip(vy - self.vy_cmd, -VX_SLEW * dt, VX_SLEW * dt))
        self.yaw_cmd += float(np.clip(yaw - self.yaw_cmd, -YAW_SLEW * dt, YAW_SLEW * dt))
        if vx_lo is not None and self.vx_cmd < vx_lo:
            self.vx_cmd = vx_lo   # the run policy never saw less than 0.2 m/s

    # ---- what the page draws -------------------------------------------------------------------------------------------------
    def scene(self) -> dict:
        """Static scene: bodies, visual mesh geoms (body, pos, quat, mesh id, rgba) -- meshes come from :meth:`mesh`."""
        m = self.model
        geoms = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            geoms.append(dict(body=int(m.geom_bodyid[g]), pos=[round(float(x), 6) for x in m.geom_pos[g]],
                              quat=[round(float(x), 6) for x in m.geom_quat[g]], mesh=int(m.geom_dataid[g]),
                              rgba=[round(float(x), 3) for x in rgba]))
        jaw = [i for i, e in enumerate(geoms) if e["body"] == self.head_body and m.mesh(e["mesh"]).name in JAW_MESHES]
        return dict(bodies=[m.body(i).name for i in range(m.nbody)], geoms=geoms, trunk=self.trunk, head=self.head_body,
                    jaw=dict(geoms=jaw, pivot=JAW_PIVOT, axis=JAW_AXIS, open_rad=JAW_OPEN_RAD), anims=self.anim.manifest(),
                    joints=JOINTS, default_pose=DEFAULT_POSE.tolist(), control_dt=self.control_dt,
                    physics_dt=float(m.opt.timestep), lag=self.servos.lag, vin=self.servos.vin0,
                    sag_gain=self.servos.sag_gain, levels=LEVELS)

    def mesh(self, i: int, crease_deg: float = 40.0) -> bytes:
        """Mesh ``i`` as [uint32 nv, uint32 ni][float32 xyz * nv][float32 normal * nv][uint32 idx * ni], with
        crease-smoothed vertex normals: STL triangles share no vertices, so per-face normals render every facet
        ("mesh lines"); here coincident vertices are merged, each corner's normal is the area-weighted mean of
        the incident faces within ``crease_deg`` of its own face, and the result is re-indexed by (position,
        normal) so smooth regions share vertices and real edges stay sharp. Cached per mesh."""
        if not hasattr(self, "_mesh_cache"):
            self._mesh_cache, self._mesh_lock = {}, __import__("threading").Lock()
        with self._mesh_lock:
            if i in self._mesh_cache:
                return self._mesh_cache[i]
            return self._build_mesh(i, crease_deg)

    def _build_mesh(self, i: int, crease_deg: float) -> bytes:
        m = self.model
        va, vn = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
        fa, fn = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
        V = m.mesh_vert[va:va + vn].astype(np.float64)
        F = m.mesh_face[fa:fa + fn].astype(np.int64)
        Vr = np.round(V, 6)
        uniq, inv = np.unique(Vr, axis=0, return_inverse=True)
        F2 = inv.reshape(-1)[F]                                   # faces on merged vertices
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        area = np.linalg.norm(n, axis=1)
        N = n / np.maximum(area, 1e-18)[:, None]
        cv, cf = F2.reshape(-1), np.repeat(np.arange(fn), 3)     # corners: vertex, face
        order = np.argsort(cv, kind="stable")
        vs, fs = cv[order], cf[order]
        nu = len(uniq)
        counts = np.bincount(vs, minlength=nu)
        starts = np.cumsum(counts) - counts
        inc = np.full((nu, int(counts.max())), -1, dtype=np.int64)
        inc[vs, np.arange(len(vs)) - starts[vs]] = fs            # incident faces per merged vertex
        nb = inc[cv]                                              # (ncorner, dmax)
        valid = nb >= 0
        nbN = N[np.clip(nb, 0, None)]
        cosang = np.einsum("cdk,ck->cd", nbN, N[cf])
        w = (valid & (cosang > math.cos(math.radians(crease_deg)))) * area[np.clip(nb, 0, None)]
        nrm = (nbN * w[..., None]).sum(1)
        bad = np.linalg.norm(nrm, axis=1) < 1e-12
        nrm[bad] = N[cf][bad]
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        key = np.concatenate([uniq[cv], np.round(nrm, 3)], axis=1)      # cv indexes the MERGED vertices
        ukey, idx = np.unique(key, axis=0, return_inverse=True)
        pos = ukey[:, :3].astype(np.float32)
        nor = ukey[:, 3:].astype(np.float32)
        nor /= np.maximum(np.linalg.norm(nor, axis=1), 1e-9)[:, None]
        idx = idx.reshape(-1).astype(np.uint32)
        out = np.array([len(pos), len(idx)], dtype=np.uint32).tobytes() + pos.tobytes() + nor.tobytes() + idx.tobytes()
        self._mesh_cache[i] = out
        return out

    def snapshot(self) -> dict:
        d = self.data
        n = self.model.nbody
        sm = self.step_ms
        trick = None
        if self.state == "jump" and self.jump is not None:
            j = self.jump
            trick = dict(kind="jump", launch=bool(j["launch"]), airborne=bool(j["airborne"]), landed=bool(j["landed"]), rise_mm=round(j["rise"] * 1000, 1))
        elif self.state == "roulade" and self.roll is not None:
            trick = dict(kind="roulade", deg=round(math.degrees(self.roll["angle"]), 0), peak_deg=round(math.degrees(self.roll["peak"]), 0))
        return dict(
            **self.anim.snapshot(),
            t=round(float(d.time - self.sim_t0), 3),
            pos=np.round(d.xpos[1:n], 5).ravel().tolist(),
            quat=np.round(d.xquat[1:n], 5).ravel().tolist(),
            state=self.state, policy=self.policies[self.policy_role].name if self.policy_role in self.policies else "",
            role=self.policy_role, level=self.level, level_label=LEVELS[self.level - 1]["label"],
            cmd=[round(float(x), 3) for x in self.cmd[0:3]],
            v=[round(float(x), 3) for x in self.v_filt], wz=round(float(self.wz_filt), 3),
            z=round(self.z, 4), tilt=round(self.tilt_deg, 1), odometer=round(self.odometer, 3),
            vin=round(self.servos.vin_eff, 2), falls=self.falls, down=self.down, sit_pending=self.sit_pending,
            step_ms=round(float(np.median(sm)), 2) if sm else 0.0,
            notice=self.notice if d.time - self.notice_t < 2.5 else "",
            held=sorted(self.held), meta=self.meta, alt=self.alt,
            has=dict((r, r in self.policies) for r in ROLES), walk_recovers=self.walk_recovers, trick=trick,
            perturb=(dict(p=np.round(self.perturb_point, 4).tolist(), target=np.round(self.perturb["target"], 4).tolist(),
                          F=round(float(np.linalg.norm(self.perturb_force)), 2)) if self.perturb is not None else None),
        )
