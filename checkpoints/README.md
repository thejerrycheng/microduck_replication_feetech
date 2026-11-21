# Selected checkpoints

Each folder holds the RSL-RL checkpoint (`model_N.pt`, resumable with `--checkpoint`), the TorchScript actor
(`policy.pt`, mean action, normaliser baked in) and the run's `env.yaml`/`agent.yaml`. `INDEX.json` maps each
to its training run. The matching single-file ONNX exports are in `../exports/`. Picks are by measured
deterministic performance, not recency (see `docs/*.json` and the project log).

The ONNX files in `../exports/` are self-contained (weights inline). Isaac Lab's exporter writes `policy.onnx` +
`policy.onnx.data` (external weights); the nine exports made from it before 2026-11-18 were the 14 kB stub that
references a `.data` file not shipped with it, and have been rebuilt inline (`onnx.save_model(..., save_as_external_data=False)`).
The per-checkpoint `policy.onnx` here still needs its `.data` beside it.

## 2026-11-20 rounds (4090 laptop)

Grades: NOMINAL (P x1.0, 7.6-7.8 V, sole friction 1.0, 20 ms loop) / STRESS2 (P x1.6, 6.4 V, friction 0.15, 40 ms);
the get-ups also WEAK (P x0.85, servo friction x2, 6.2 V, sole 0.4, 35 ms). JSONs in `docs/standup`, `docs/walk95`,
`docs/anim`, `docs/real2`. Targets (user): walking >= 95 %, every other policy >= 90 %, in both conditions.

| Folder | What | Result |
|---|---|---|
| `standup_r4_18499` | get-up round 4 (StandUpRobust) | gets up 99-100 % nominal/stress, 96-99 % weak; shoved stand 88 / 69 / 30 % |
| `standup_r5_21498` | round 5 (StandUpRobustStress) | gets up 100 % nominal/stress, 89-95 % weak; shoved stand 79 / 72 / 19 % |
| `standup_13248` | round 2 pick | gets up 98-100 % nominal, 96-99 % stress, 59-68 % weak |
| `velstand_r2_26748` | walk + get-up, round 2 | walking falls 2-6 % nominal, 98-100 % stress |
| `velstand_stress_30747` | velstand, hard-corner DR mix | walking falls 0-2 % nominal, 100 % stress |
| `openmd_animwalk_stress_6998` | OpenMD walk under the emotion clips, hard-corner mix | clips standing 98 % nominal / 74 % stress; walking falls 6-38 % nominal, 92-100 % stress |
| `openmd_animwalk_2999` | OpenMD walk under the clips | clips standing 96 % nominal / 79 % hard |
| `openmd_walk_ft_2999` | OpenMD walk fine-tune | stopped walking (0.03 m/s); kept for reference only |
| `openmd_stand_ft_2500` | OpenMD stand fine-tune (crashed at 2500) | |
| `sitstand_levelhead_8497` | sit/stand, level head | falls 2 % / 0 %, head 3.3 deg off level |
| `sitstand_anim_9750` | sit/stand tracking the seated clips | falls 0 % / 0 % |
| `roulade_recover_14000`, `roulade_head_13249` | flip | flip + stand 69 % nominal / 74 % stress (14000) |
| `jump_6748` | jump round 2 | landing 25 % / 1 % |
| `sprint_max_9497` | sprint | falls 2 % / 4 % (runs ~0.86 m/s whatever the command) |
| `ball_kick_7998`, `ball_kick_9750` | right-foot ball kick rounds 2-3 | |
| `openmd/` | the OpenMicroDuck originals (ONNX + imported `model_0.pt`) | |

Inline single-file ONNX for each: `../exports/<folder>.onnx`.
