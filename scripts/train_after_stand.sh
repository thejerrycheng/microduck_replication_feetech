#!/usr/bin/env bash
# After the stand run ends: train the front flip (roulade), evaluate its checkpoints, then the jump if registered.
# Launch: SINCE=$(date +%F_%T) systemd-run --user --unit microduck-after-stand --working-directory "$PWD" --setenv=SINCE="$SINCE" scripts/train_after_stand.sh
set -u
cd "$(dirname "$0")/.."
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
until grep "end stand" logs/train_all.log 2>/dev/null | awk -v s="[$SINCE" '$1 > s' | grep -q .; do sleep 120; done
sleep 30
# Run v3: the full-collision run with the mirror loss (timing asymmetry: air time L/R 0.89-0.93 on run v2).
# Warm start from run v2 (the policy that still runs the whole band), NOT from the full-collision fine-tune,
# which had already given up the fast commands; the v3 env reopens the speed band by curriculum.
RUN2=/home/jerrycheng/microduck/logs/rsl_rl/microduck_run_flat_hd1910_backlash/2026-11-07_19-05-22
RUN2_CK=2999
echo "[$(date +%F_%T)] start run-v3 (mirror loss + reopened speed curriculum, 3000 its from run v2 $RUN2_CK)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-RunV3-Flat-HD1910-AllColl --agent rsl_rl_with_symmetry_cfg_entry_point \
  --max_iterations 3000 $COMMON --checkpoint "$RUN2/model_$RUN2_CK.pt" >> logs/train_run_v3.log 2>&1
echo "[$(date +%F_%T)] end run-v3 exit=$?" | tee -a logs/train_all.log
RUN3=$(ls -td logs/rsl_rl/microduck_run_flat_allcoll_mirror_hd1910_backlash/*/ | head -1)
RUN3_CK=$(ls "$RUN3"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
SPEEDS="0.3 0.5 0.7 0.9 1.1" MIN_FREE_MB=3000 scripts/eval_run_speeds.sh "$RUN3" Microduck-RunV3-Flat-HD1910-AllColl "$RUN3_CK"
for vx in 0.3 0.5 0.7 0.9 1.1; do mv -f "docs/run_speed_${RUN3_CK}_vx$vx.json" "docs/run_v3_speed_${RUN3_CK}_vx$vx.json" 2>/dev/null; done
echo "[$(date +%F_%T)] run-v3 speed table done" >> logs/evals.log
echo "[$(date +%F_%T)] start roulade (front flip, ${ROULADE_ITERS:-8000} iterations)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-Roulade-Flat-HD1910 --agent rsl_rl_with_symmetry_cfg_entry_point \
  --max_iterations "${ROULADE_ITERS:-8000}" $COMMON >> logs/train_roulade.log 2>&1 &
TRAIN_PID=$!
sleep 240
RUN=$(ls -td logs/rsl_rl/microduck_roulade_flat_hd1910_backlash/*/ | head -1)
scripts/eval_roulade_checkpoints.sh "$RUN" roulade 1000 2000 3000 4000 5000 6000 7000 8000 &
EVAL_PID=$!
wait $TRAIN_PID
echo "[$(date +%F_%T)] end roulade exit=$?" | tee -a logs/train_all.log
LAST=$(ls "$RUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
[ -f "$RUN/model_8000.pt" ] || ln "$RUN/model_$LAST.pt" "$RUN/model_8000.pt"
wait $EVAL_PID
if uv run --project external/IsaacLab python -c "import gymnasium as gym, microduck_rl; gym.spec('Microduck-Jump-Flat-HD1910')" >/dev/null 2>&1; then
  echo "[$(date +%F_%T)] start jump (${JUMP_ITERS:-5000} iterations)" | tee -a logs/train_all.log
  uv run --project external/IsaacLab python scripts/train.py --task Microduck-Jump-Flat-HD1910 --agent rsl_rl_with_symmetry_cfg_entry_point \
    --max_iterations "${JUMP_ITERS:-5000}" $COMMON >> logs/train_jump.log 2>&1 &
  JPID=$!
  sleep 240
  JRUN=$(ls -td logs/rsl_rl/microduck_jump_flat_hd1910_backlash/*/ | head -1)
  scripts/eval_jump_checkpoints.sh "$JRUN" jump 500 1000 1500 2000 3000 4000 5000 &
  JEVAL=$!
  wait $JPID
  echo "[$(date +%F_%T)] end jump exit=$?" | tee -a logs/train_all.log
  JLAST=$(ls "$JRUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
  [ -f "$JRUN/model_5000.pt" ] || ln "$JRUN/model_$JLAST.pt" "$JRUN/model_5000.pt"
  wait $JEVAL
else
  echo "[$(date +%F_%T)] jump task not registered; skipped" | tee -a logs/train_all.log
fi
