# Robots that walk and fly, and RL for them: survey notes

Compiled 2026-11-25 by web search; every row was opened at its link unless marked UNVERIFIED. "n/s" = not stated on
the page read.

**Bottom line.** One paper trains a single RL policy over legs and thrusters for walking and flying (L'Erario et al.,
iRonCub), and it is simulation only with no public code. No sim-to-real RL result was found for a legged robot with
thrusters. Every legged-aerial hardware demonstration found uses model-based control. The closest hardware-validated RL
is on wheeled or morphing air-ground robots (ATMO, DoubleBee).

## Legged + aerial robots

| Work | Robot | Control | Shown on hardware | Link |
|---|---|---|---|---|
| LEONARDO, Kim ... Chung, Science Robotics 2021 | 0.76 m biped, 4 propellers | model-based | walking, flying, slackline, skateboard | https://aerospacerobotics.caltech.edu/leonardo |
| Learning to Walk and Fly with Adversarial Motion Priors, L'Erario ... Scaramuzza, arXiv 2309.12784 | iRonCub, 44 kg, 4 jet turbines | **RL (PPO + AMP), one policy** | no (simulation) | https://arxiv.org/html/2309.12784v4 |
| iRonCub 3, Gorbani ... Pucci, arXiv 2506.01125 (2025) | 4 jet turbines | MPC on centroidal momentum | tethered lift-off, about 3 s airborne | https://arxiv.org/html/2506.01125 |
| Small humanoid with flight unit and wheels, Sugihara ... Inaba, arXiv 2303.14718 | 3.5 kg, 3 tilting ducted fans | PID + wrench allocation | hover, thrust-assisted walking; no walk-to-fly transition | https://arxiv.org/html/2303.14718v3 |
| Anti-gravity walking, Sugihara, Okada, arXiv 2609.07544 (2026) | 0.4 m, 2.0 kg humanoid, 2 vectored rotors | whole-body MPC, thrust-rate input | ceiling walking, 4 steps | https://arxiv.org/html/2609.07544 |
| Jet-HR2, Li et al., arXiv 2108.11557 | 17 kg, 4 ducted fans | thrust vectoring | UNVERIFIED: take-off to 1 m | https://arxiv.org/abs/2108.11557 |
| Harpy capture-point control, Pitroda ... Gharib, arXiv 2411.17727 | thruster-assisted biped | capture point + QP | no (simulation) | https://arxiv.org/abs/2411.17727 |
| Harpy thrust/torque identification, Cahill, arXiv 2504.20313 | Harpy, 2 thrusters | sys-id | thruster tests | https://arxiv.org/abs/2504.20313 |
| Husky Carbon, Wang, Ramezani, arXiv 2508.03003 | thruster-assisted quadruped | Raibert legs + thruster MPC, learned contact residual | yes | https://arxiv.org/abs/2508.03003 |
| SPIDAR, Zhao, Anzai, Nishio, arXiv 2301.04050 | 15.2 kg flying quadruped, 8 vectored rotor modules | model-based | creeping gait, flight, gait then take-off | https://ar5iv.labs.arxiv.org/html/2301.04050 |
| M4 Morphobot, Sihite ... Gharib, Nature Communications 2023 | wheels / legs / thrusters | not RL as far as read | multi-modal | https://authors.library.caltech.edu/records/pch7d-bq204 |
| ATMO, Mandralis, Murray, Gharib, arXiv 2506.14039 | 5.5 kg morphing quadrotor (wheeled) | **RL (PPO) vs MPC** | morphing landings | https://arxiv.org/html/2506.14039v1 |
| DoubleBee, Li ... Cao, arXiv 2603.26687 (2026) | 2 propellers, 2 tilt servos, 2 wheels | **RL, one policy** | stairs, about 38 % energy saved | https://arxiv.org/html/2603.26687v1 |

Searched and not found: RL on Harpy or Husky; a KAIST thruster-assisted quadruped; a jetpack humanoid trained by RL
with hardware results; a Disney character robot with flight (only an IEEE Spectrum article on fan-assisted landing).

## How the RL papers set it up

**L'Erario et al. (iRonCub).** Isaac Gym, 4096 envs, PPO, 60 Hz. One policy outputs 23 joint targets and a thrust
input per jet, commanded as thrust RATE and integrated. Observations include the current thrust and base height.
Command: waypoints at 0.8 m/s; no mode flag, no vertical-velocity command. No transition reward: walking and flying
come from motion priors (human mocap for walking, trajectory optimisation for flight) and switching emerges from a
thrust penalty. Thruster: an LSTM fitted to load-cell data, with a 15 % minimum throttle. No domain randomisation or
curriculum reported.

**ATMO (the best-documented sim-to-real recipe).** Isaac Lab, PPO, 50 Hz. Actions: 4 normalised rotor commands + tilt
velocity. Observations include the last 10 actions; the critic also sees true rotor speeds and disturbances. Motor:
first-order lag 0.15 s, randomised 0.10-0.20 s; 20 ms observation delay; thrust and moment coefficients +-20 %.
Without lag, delay and randomisation the policy did not transfer. Landing rewards: descent-rate tracking,
contact-impulse penalty, thrust-while-in-contact penalty.

**DoubleBee.** Isaac Lab, thrust as an external force, one 6-D policy, actuator lag / delay / rate limits, +-20 % on
mass, friction, thrust. An energy penalty gives thrust-assisted driving without mode labels.

No RL paper found models battery sag.

## Code

| Repo | Licence | Contents |
|---|---|---|
| https://github.com/ami-iit/ironcub-mk1-software | BSD-3 | iRonCub URDFs, Gazebo jet plugins, Simulink flight controllers; no RL |
| https://github.com/ami-iit/ironcub-models | BSD-3 | iRonCub models incl. Mk3 |
| https://github.com/mandralis/IsaacLab | BSD-3 | Isaac Lab fork linked by the ATMO paper |
| https://github.com/arplaboratory/learning-to-fly | MIT | Learning to Fly in Seconds: direct motor commands, action history, motor lag, reward curriculum |
| https://github.com/btx0424/OmniDrones | MIT | Isaac Sim drone benchmark; hosts SimpleFlight |
| https://github.com/ntnu-arl/aerial_gym_simulator | BSD-3 | arbitrary multirotor configurations, GPU controllers |
| https://github.com/utiasDSL/gym-pybullet-drones | MIT | PyBullet drone environments |
| https://github.com/utiasDSL/crazyflow | MIT | batched differentiable drone simulation |

No public code: L'Erario walk+fly, DoubleBee, both Sugihara papers, Harpy/Husky, SPIDAR, M4 (on request), TU Delft
end-to-end racing.

Fan-command evidence from quadrotors: collective thrust + body rates transfers more robustly than per-rotor thrust
(Kaufmann, Bauersfeld, Scaramuzza, ICRA 2022, https://arxiv.org/abs/2202.10796; SimpleFlight,
https://arxiv.org/html/2412.11764v4). Direct motor commands are slightly faster but have a larger reality gap (TU
Delft); they work if the motor lag is modelled and action history is observed (Learning to Fly in Seconds).

## What this means for the flying duck

- One policy over legs and fans is the supported RL route; separate leg and fan controllers is what the model-based
  labs do and is our classical baseline.
- For the fans, prefer collective thrust + body rates (or a wrench) with an inner loop for transfer; if commanding fans
  directly, command thrust rate or lag-filtered targets.
- A velocity command with a vertical component, as this project uses, has no published precedent for a walk/fly robot.
- Transitions: a thrust or energy penalty can make switching emerge; ATMO's landing rewards are reusable.
- Sim-to-real: model rotor lag and observation delay, give the actor action history and the critic true rotor state,
  randomise the thrust coefficient +-20 %, penalise action differences, enforce an idle-throttle floor.
- Battery sag in training would be new. Ground effect is unmodelled everywhere found.
- Read first: arXiv 2309.12784 (formulation), arXiv 2506.14039 with the mandralis/IsaacLab fork (sim-to-real recipe),
  arXiv 2412.11764 or learning-to-fly (fan side). Nearest hardware at this scale: Sugihara and Okada 2026.
