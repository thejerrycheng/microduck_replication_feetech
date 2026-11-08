#!/usr/bin/env bash
# Deterministic (mean-action) posture evaluation of a sit run's checkpoints as they appear.
# Usage: scripts/eval_sit_checkpoints.sh <run dir> <tag> <ckpt...>   -> docs/posture_<tag>_<ckpt>.json
set -u
cd "$(dirname "$0")/.."
RUN=$(readlink -f "$1"); TAG=$2; shift 2
MIN_FREE_MB=${MIN_FREE_MB:-6500}
for ck in "$@"; do
  until [ -f "$RUN/model_$ck.pt" ]; do sleep 300; done
  while [ "$(free -m | awk 'NR==2{print $7}')" -lt "$MIN_FREE_MB" ]; do sleep 60; done
  uv run --project external/IsaacLab python scripts/eval_posture.py --task Microduck-Sit-Flat-HD1910 --run "$RUN" --checkpoint "model_$ck.pt" \
    --num_envs 16 presets=backlash --out "docs/posture_${TAG}_${ck}.json" > "logs/eval_posture_${TAG}_${ck}.log" 2>&1
  echo "[$(date +%F_%T)] sit posture $TAG $ck exit $?" >> logs/evals.log
done
