#!/usr/bin/env bash
# Stand in two phases: discovery at the recipe's entropy 0.01 (PHASE1_ITERS), then exploitation at 0.001 for the
# remainder (resumed from the phase-1 checkpoint; RSL-RL ADDS --max_iterations to the checkpoint's count).
# Logs "end stand" only once, at the very end, so a chain waiting on it does not fire between phases.
set -u
cd "$(dirname "$0")/.."
P1=${PHASE1_ITERS:-6000}; TOTAL=${TOTAL_ITERS:-15000}; TASK=${TASK:-Microduck-Stand-Flat-HD1910}; DIR=${DIR:-microduck_stand_flat_hd1910_backlash}
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
# RESUME: the newest run directory of $DIR that holds a checkpoint decides where to continue (a chain restart
# must not re-run a phase; trainers save model_{N-1} as their final checkpoint, hence the -1 margins).
RUN=""; CK=-1
for d in $(ls -td logs/rsl_rl/$DIR/*/ 2>/dev/null); do
  if ls "$d"/model_*.pt >/dev/null 2>&1; then RUN="$d"; CK=$(ls "$d"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1); break; fi
done
if [ "$CK" -lt $((P1 - 1)) ]; then
  if [ "$CK" -ge 0 ]; then
    echo "[$(date +%F_%T)] resume stand phase1 from $RUN model_$CK.pt ($((P1 - CK)) its left)" | tee -a logs/train_all.log
    uv run --project external/IsaacLab python scripts/train.py --task "$TASK" --max_iterations $((P1 - CK)) $COMMON \
      --checkpoint "$(readlink -f "$RUN")/model_$CK.pt" >> logs/train_stand.log 2>&1
  else
    echo "[$(date +%F_%T)] start stand ($TASK two-phase: $P1 @ entropy 0.01, then $((TOTAL-P1)) @ 0.001)" | tee -a logs/train_all.log
    uv run --project external/IsaacLab python scripts/train.py --task "$TASK" --max_iterations "$P1" $COMMON >> logs/train_stand.log 2>&1
  fi
  echo "[$(date +%F_%T)] stand phase1 finished exit=$?" | tee -a logs/train_all.log
  RUN=""; for d in $(ls -td logs/rsl_rl/$DIR/*/); do ls "$d"/model_*.pt >/dev/null 2>&1 && { RUN="$d"; break; }; done
  CK=$(ls "$RUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
fi
if [ "$CK" -lt $((TOTAL - 1)) ]; then
  echo "[$(date +%F_%T)] stand phase2 from $RUN model_$CK.pt ($((TOTAL - CK)) its)" | tee -a logs/train_all.log
  uv run --project external/IsaacLab python scripts/train.py --task "$TASK" --max_iterations $((TOTAL - CK)) $COMMON \
    --checkpoint "$(readlink -f "$RUN")/model_$CK.pt" agent.algorithm.entropy_coef=0.001 >> logs/train_stand.log 2>&1
fi
echo "[$(date +%F_%T)] end stand exit=$?" | tee -a logs/train_all.log
