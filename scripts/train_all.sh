#!/usr/bin/env bash
# Train walk -> sit -> stand sequentially (one Newton trainer at a time: the GPU and RAM are shared).
# Each run records a 400-step clip of 16 robots every 250 iterations and saves a checkpoint every 250.
# Usage: scripts/train_all.sh [WALK_ITERS] [SIT_ITERS] [STAND_ITERS]   (defaults 15000 15000 15000)
# Launch it detached from any terminal or agent session:
#   systemd-run --user --unit microduck-train --working-directory "$PWD" scripts/train_all.sh
set -u
cd "$(dirname "$0")/.."
WALK_ITERS=${1:-15000}; SIT_ITERS=${2:-15000}; STAND_ITERS=${3:-15000}
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
run() {  # name task iterations [extra args...]
  local name=$1 task=$2 iters=$3; shift 3
  echo "[$(date +%F_%T)] start $name ($iters iterations) $*" | tee -a logs/train_all.log
  uv run --project external/IsaacLab python scripts/train.py --task "$task" --max_iterations "$iters" $COMMON "$@" >> "logs/train_$name.log" 2>&1
  echo "[$(date +%F_%T)] end $name exit=$?" | tee -a logs/train_all.log
}
mkdir -p logs
# WALK_RESUME="--checkpoint /abs/path/model_N.pt" continues an interrupted walk run (pass the REMAINING iterations:
# RSL-RL adds --max_iterations to the checkpoint's iteration count).
# SKIP_WALK=1 skips the walk stage; SIT_RESUME / STAND_RESUME="--checkpoint /abs/path/model_N.pt" resume a stage.
# NOTE: when resuming, RSL-RL ADDS the iteration count to the checkpoint's, so pass the REMAINING iterations.
[ -n "${SKIP_WALK:-}" ] || run walk  Microduck-Velocity-Flat-HD1910 "$WALK_ITERS" ${WALK_RESUME:-}
run sit   Microduck-Sit-Flat-HD1910      "$SIT_ITERS"   ${SIT_RESUME:-}
# RUN_AFTER_SIT=1 trains the running policy (official sprint budget, RUN_ITERS, default 3000) before stand.
[ -z "${RUN_AFTER_SIT:-}" ] || run run Microduck-Run-Flat-HD1910 "${RUN_ITERS:-3000}" --agent rsl_rl_with_symmetry_cfg_entry_point
run stand Microduck-Stand-Flat-HD1910    "$STAND_ITERS" ${STAND_RESUME:-}
