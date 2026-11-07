#!/usr/bin/env bash
# Achieved-vs-commanded speed table for a velocity policy: scripts/eval_run_speeds.sh <run dir> <task> <ckpt...>
# Each (checkpoint, vx) rollout waits for MIN_FREE_MB of RAM (training has priority on this shared machine).
set -u
cd "$(dirname "$0")/.."
RUN=$(readlink -f "$1"); TASK=$2; shift 2
MIN_FREE_MB=${MIN_FREE_MB:-6500}; SPEEDS=${SPEEDS:-"0.3 0.5 0.7 0.9 1.1"}
for ck in "$@"; do
  for vx in $SPEEDS; do
    while [ "$(free -m | awk 'NR==2{print $7}')" -lt "$MIN_FREE_MB" ]; do sleep 30; done
    uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --task "$TASK" --run "$RUN" --checkpoint "model_$ck.pt" \
      --vx "$vx" --num_envs 32 presets=backlash --out "docs/run_speed_${ck}_vx${vx}.json" > "logs/eval_run_speed_${ck}_${vx}.log" 2>&1
    echo "[$(date +%F_%T)] speed eval $ck vx=$vx exit $?" >> logs/evals.log
  done
done
echo "[$(date +%F_%T)] run speed eval done" >> logs/evals.log
