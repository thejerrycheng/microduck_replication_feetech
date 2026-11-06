#!/usr/bin/env bash
# Train walk -> sit -> stand sequentially (one Newton trainer at a time: the GPU and RAM are shared).
# Each run records a 400-step clip of 16 robots every 250 iterations and saves a checkpoint every 250.
# Usage: scripts/train_all.sh [WALK_ITERS] [SIT_ITERS] [STAND_ITERS]   (defaults 15000 15000 15000)
set -u
cd "$(dirname "$0")/.."
WALK_ITERS=${1:-15000}; SIT_ITERS=${2:-15000}; STAND_ITERS=${3:-15000}
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
run() {  # name task iterations
  echo "[$(date +%F_%T)] start $1 ($3 iterations)" | tee -a logs/train_all.log
  uv run --project external/IsaacLab python scripts/train.py --task "$2" --max_iterations "$3" $COMMON > "logs/train_$1.log" 2>&1
  echo "[$(date +%F_%T)] end $1 exit=$?" | tee -a logs/train_all.log
}
mkdir -p logs
run walk  Microduck-Velocity-Flat-HD1910 "$WALK_ITERS"
run sit   Microduck-Sit-Flat-HD1910      "$SIT_ITERS"
run stand Microduck-Stand-Flat-HD1910    "$STAND_ITERS"
