#!/usr/bin/env bash
# Stand in two phases: discovery at the recipe's entropy 0.01 (PHASE1_ITERS), then exploitation at 0.001 for the
# remainder (resumed from the phase-1 checkpoint; RSL-RL ADDS --max_iterations to the checkpoint's count).
# Logs "end stand" only once, at the very end, so a chain waiting on it does not fire between phases.
set -u
cd "$(dirname "$0")/.."
P1=${PHASE1_ITERS:-6000}; TOTAL=${TOTAL_ITERS:-15000}; TASK=${TASK:-Microduck-Stand-Flat-HD1910}; DIR=${DIR:-microduck_stand_flat_hd1910_backlash}
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
echo "[$(date +%F_%T)] start stand ($TASK two-phase: $P1 @ entropy 0.01, then $((TOTAL-P1)) @ 0.001)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task "$TASK" --max_iterations "$P1" $COMMON >> logs/train_stand.log 2>&1
echo "[$(date +%F_%T)] stand phase1 finished exit=$?" | tee -a logs/train_all.log
RUN=$(ls -td logs/rsl_rl/$DIR/*/ | head -1)
CK=$(ls "$RUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
echo "[$(date +%F_%T)] stand phase2 from $RUN model_$CK.pt" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task "$TASK" --max_iterations $((TOTAL-P1)) $COMMON \
  --checkpoint "$(readlink -f "$RUN")/model_$CK.pt" agent.algorithm.entropy_coef=0.001 >> logs/train_stand.log 2>&1
echo "[$(date +%F_%T)] end stand exit=$?" | tee -a logs/train_all.log
