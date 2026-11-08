# Selected checkpoints

Each folder holds the RSL-RL checkpoint (`model_N.pt`, resumable with `--checkpoint`), the TorchScript actor
(`policy.pt`, mean action, normaliser baked in) and the run's `env.yaml`/`agent.yaml`. `INDEX.json` maps each
to its training run. The matching single-file ONNX exports are in `../exports/`. Picks are by measured
deterministic performance, not recency (see `docs/*.json` and the project log).
