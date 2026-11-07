#!/usr/bin/env bash
# Run the pending evaluations one at a time, each only when >= MIN_FREE_MB of RAM is available, and stop
# an evaluation if free RAM drops under STOP_MB (training always has priority on this shared machine).
set -u
cd "$(dirname "$0")/.."
MIN_FREE_MB=${MIN_FREE_MB:-7000}; STOP_MB=${STOP_MB:-1000}
SIT_RUN=$(ls -d logs/rsl_rl/microduck_sit_flat_hd1910_backlash/2026-11-06_19-47-46)
WALK_RUN=$(ls -d logs/rsl_rl/microduck_walk_flat_hd1910_backlash/2026-11-06_11-44-05)
run_gated() {  # name cmd...
  local name=$1; shift
  while [ "$(free -m | awk 'NR==2{print $7}')" -lt "$MIN_FREE_MB" ]; do sleep 30; done
  echo "[$(date +%T)] start $name" >> logs/evals.log
  "$@" > "logs/eval_$name.log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(free -m | awk 'NR==2{print $7}')" -lt "$STOP_MB" ]; then kill $pid; echo "[$(date +%T)] $name stopped: low RAM" >> logs/evals.log; sleep 300; run_gated "$name" "$@"; return; fi
    sleep 5
  done
  echo "[$(date +%T)] end $name" >> logs/evals.log
}
latest_sit=$(ls $SIT_RUN/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
run_gated sit_posture uv run --project external/IsaacLab python scripts/eval_posture.py --task Microduck-Sit-Flat-HD1910 --run "$SIT_RUN" --checkpoint model_$latest_sit.pt --num_envs 16 presets=backlash --out docs/posture_sit_$latest_sit.json
for ck in 2000 9500; do
  run_gated gait_$ck uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --run "$WALK_RUN" --checkpoint model_$ck.pt presets=backlash --out docs/gait_walk_$ck.json
done
echo "[$(date +%T)] all evaluations done" >> logs/evals.log
