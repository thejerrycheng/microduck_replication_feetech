# Sole-floor friction: what the policies actually trained against (2026-11-19)

User report: the deployed walker falls on the real floor, attributed to friction. Measured in the training env
(`scripts/measure_foot_friction.py`, 96 envs, sole friction set per environment, friction read from the solver's own
contact buffer `mjw_data.contact.friction` after the robot stands 0.6 s):

| sole mu set | contact friction, old env (`mu_eff_velstand_asis.json`) | contact friction, `*-LowMu` env (`mu_eff_velstand_lowmu_task.json`) |
|---|---|---|
| 0.2 | 1.0 | 0.2 |
| 0.4 | 1.0 | 0.4 |
| 0.6 | 1.0 | 0.6 |
| 0.8 | 1.0 | 0.8 |
| 1.0 | 1.0 | 1.0 |
| 1.2 | 1.2 | 1.2 |

Cause: the flat terrain is authored at friction 1.0 and Newton's MuJoCo solver resolves a contact's friction as the
MAX of the two geoms unless their `geom_priority` differ (`newton/_src/solvers/mujoco/kernels.py::contact_params`,
same rule as `mujoco_warp/_src/collision_core.py`). The official `foot_friction` event (0.7-1.3) and the replica
widening (0.5-1.3) therefore only ever RAISED the contact friction; no policy here has seen a floor below 1.0.

Fix (`microduck_rl.mdp.set_sole_contact_priority`, startup event in `apply_slippery_floor`): write `geom_priority = 1`
for the two sole geoms into the solver's geom table, so the contact takes the sole's draw alone. The USD route
(`mjc:priority` through `collision_props`) does not work on these assets: the spawner skips the sole prims as
instanced ("Skipping fragment updates on instanced prims matched by ..."). The physical pull test in the script
(force ramp at the feet) is NOT a friction measurement on this robot -- the feet move 5 mm at 0.21 of the normal load
at every setting because the standing posture leans before the soles slide; it is kept as a sanity trace only.

Current walker (`velstand_21495`, vx 0.3, 8 s, 16 envs per friction, effective friction via the LowMu env):

| contact friction | fallen | contact-foot speed p50 / p90 / p99 (m/s) |
|---|---|---|
| 0.2 | 0.19 | 0.067 / 0.233 / 0.575 |
| 0.4 | 0.12 | 0.048 / 0.120 / 0.325 |
| 0.6 | 0.06 | 0.050 / 0.121 / 0.338 |
| 0.8 | 0.00 | 0.049 / 0.113 / 0.308 |
| 1.0 | 0.06 | 0.049 / 0.118 / 0.320 |
| 1.2 | 0.00 | 0.053 / 0.124 / 0.337 |

The fall rate rises as the floor gets slippery and the foot slides (p90 0.12 -> 0.23 m/s at 0.2), which is the
reported failure. The ankle-link speed in contact is ~0.05 m/s at the median and 0.12 at p90 even when no slip is
possible (the link rolls through every step), so the thresholded slip cost starts at 0.20 m/s.

Training (`scripts/train_friction.sh`, unit `microduck-friction`): `Microduck-VelStand-Flat-HD1910-AllColl-LowMu` from
velstand_21495 and `Microduck-RunV3-Flat-HD1910-AllColl-LowMu` from run_v3_5999, 2000 iterations each, sole friction
0.25-1.2 re-drawn per sole at every reset, servo P 5-8 (kp_scale 1.0-1.6), entropy 0.001 / lr 5e-4, slip cost
`foot_slip_over_threshold` (threshold 0.20 m/s, weight -20, squared excess) with the always-on `foot_slip` at 0.
Graded afterwards under the LowMu env at fixed friction 0.3 / 0.6 / 1.0 and fixed P 5 / 8, base vs every checkpoint.
