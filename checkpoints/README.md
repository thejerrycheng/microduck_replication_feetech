# Selected checkpoints

Each folder holds the RSL-RL checkpoint (`model_N.pt`, resumable with `--checkpoint`), the TorchScript actor
(`policy.pt`, mean action, normaliser baked in) and the run's `env.yaml`/`agent.yaml`. `INDEX.json` maps each
to its training run. The matching single-file ONNX exports are in `../exports/`. Picks are by measured
deterministic performance, not recency (see `docs/*.json` and the project log).

The ONNX files in `../exports/` are self-contained (weights inline). Isaac Lab's exporter writes `policy.onnx` +
`policy.onnx.data` (external weights); the nine exports made from it before 2026-11-18 were the 14 kB stub that
references a `.data` file not shipped with it, and have been rebuilt inline (`onnx.save_model(..., save_as_external_data=False)`).
The per-checkpoint `policy.onnx` here still needs its `.data` beside it.
