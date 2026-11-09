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
START_CK="$RUN2/model_$RUN2_CK.pt"; ITERS=3000
# RESUME run v3 if an earlier attempt left checkpoints (the chain unit was restarted mid-stage once).
PREV=""
for d in $(ls -td logs/rsl_rl/microduck_run_flat_allcoll_mirror_hd1910_backlash/*/ 2>/dev/null); do
  if ls "$d"/model_*.pt >/dev/null 2>&1; then PREV="$d"; break; fi
done
if [ -n "$PREV" ]; then
  PREV_CK=$(ls "$PREV"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
  if [ "$PREV_CK" -lt 5999 ]; then START_CK="$(readlink -f "$PREV")/model_$PREV_CK.pt"; ITERS=$((5999 - PREV_CK)); fi
fi
echo "[$(date +%F_%T)] start run-v3 (mirror loss + reopened speed curriculum, $ITERS its from $START_CK)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-RunV3-Flat-HD1910-AllColl --agent rsl_rl_with_symmetry_cfg_entry_point \
  --max_iterations $ITERS $COMMON --checkpoint "$START_CK" >> logs/train_run_v3.log 2>&1
echo "[$(date +%F_%T)] end run-v3 exit=$?" | tee -a logs/train_all.log
RUN3=$(ls -td logs/rsl_rl/microduck_run_flat_allcoll_mirror_hd1910_backlash/*/ | head -1)
RUN3_CK=$(ls "$RUN3"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
SPEEDS="0.3 0.5 0.7 0.9 1.1" MIN_FREE_MB=3000 scripts/eval_run_speeds.sh "$RUN3" Microduck-RunV3-Flat-HD1910-AllColl "$RUN3_CK"
for vx in 0.3 0.5 0.7 0.9 1.1; do mv -f "docs/run_speed_${RUN3_CK}_vx$vx.json" "docs/run_v3_speed_${RUN3_CK}_vx$vx.json" 2>/dev/null; done
echo "[$(date +%F_%T)] run-v3 speed table done" >> logs/evals.log
# Walk v3: full-collision walk with more standing/turning environments, an always-on trunk-height cost (no
# squat at stop / turn onset) and the mirror loss.
WALK2=/home/jerrycheng/microduck/logs/rsl_rl/microduck_walk_flat_allcoll_hd1910_backlash/2026-11-08_14-55-24
echo "[$(date +%F_%T)] start walk-v3 (height cost + standing envs + mirror loss, 1500 its from walk-allcoll 16498)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-WalkV3-Flat-HD1910-AllColl --agent rsl_rl_with_symmetry_cfg_entry_point \
  --max_iterations 1500 $COMMON --checkpoint "$WALK2/model_16498.pt" >> logs/train_walk_v3.log 2>&1
echo "[$(date +%F_%T)] end walk-v3 exit=$?" | tee -a logs/train_all.log
WALK3=$(ls -td logs/rsl_rl/microduck_walk_flat_allcoll_mirror_hd1910_backlash/*/ | head -1)
WALK3_CK=$(ls "$WALK3"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
uv run --project external/IsaacLab python scripts/play.py --rl_library rsl_rl --task Microduck-WalkV3-Flat-HD1910-AllColl --num_envs 4 presets=backlash \
  --checkpoint "$(readlink -f "$WALK3")/model_$WALK3_CK.pt" --video --video_length 1 --video_interval 1000000 > logs/export_walk_v3.log 2>&1
mkdir -p "$WALK3/exported_model_$WALK3_CK"; cp "$WALK3"/exported/policy.* "$WALK3/exported_model_$WALK3_CK/" 2>/dev/null
uv run --project external/IsaacLab python scripts/eval_squat.py --policy "$WALK3/exported_model_$WALK3_CK/policy.pt" presets=backlash --out "docs/squat_walk_v3_$WALK3_CK.json" > logs/eval_squat_walk_v3.log 2>&1
uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --task Microduck-WalkV3-Flat-HD1910-AllColl --run "$(readlink -f "$WALK3")" --checkpoint "model_$WALK3_CK.pt" \
  --vx 0.3 --num_envs 64 presets=backlash --out "docs/gait_walk_v3_$WALK3_CK.json" > logs/eval_gait_walk_v3.log 2>&1
echo "[$(date +%F_%T)] walk-v3 squat+gait evals done" >> logs/evals.log
# StandUp (get-up + robust stand), if registered by then.
if uv run --project external/IsaacLab python -c "import gymnasium as gym, microduck_rl; gym.spec('Microduck-StandUp-Flat-HD1910')" >/dev/null 2>&1; then
  # Bank of real fall end-states for the stand-up start distribution (runs while no trainer is up).
  if [ ! -f docs/fall_bank.json ]; then
    uv run --project external/IsaacLab python scripts/capture_fall_states.py --target 300 --num_envs 32 presets=backlash > logs/capture_fall_states.log 2>&1
    echo "[$(date +%F_%T)] fall bank capture exit=$? ($(python3 -c "import json;print(len(json.load(open('docs/fall_bank.json'))))" 2>/dev/null || echo 0) states)" | tee -a logs/train_all.log
  fi
  echo "[$(date +%F_%T)] start standup (${STANDUP_ITERS:-10000} its, two-phase entropy)" | tee -a logs/train_all.log
  TASK=Microduck-StandUp-Flat-HD1910 DIR=microduck_standup_flat_hd1910_backlash PHASE1_ITERS=7000 TOTAL_ITERS=${STANDUP_ITERS:-10000} scripts/train_stand_two_phase.sh
  echo "[$(date +%F_%T)] end standup" | tee -a logs/train_all.log
else
  echo "[$(date +%F_%T)] standup task not registered; skipped" | tee -a logs/train_all.log
fi
# Sprint-max (hardware top speed) after the stand-up: fine-tune from run v3's last checkpoint.
RUN3=$(ls -td logs/rsl_rl/microduck_run_flat_allcoll_mirror_hd1910_backlash/*/ | head -1)
RUN3_CK=$(ls "$RUN3"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
echo "[$(date +%F_%T)] start sprint-max (2000 its from run v3 $RUN3_CK)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-SprintMax-Flat-HD1910-AllColl --agent rsl_rl_with_symmetry_cfg_entry_point \
  --max_iterations 2000 $COMMON --checkpoint "$(readlink -f "$RUN3")/model_$RUN3_CK.pt" >> logs/train_sprint_max.log 2>&1
echo "[$(date +%F_%T)] end sprint-max exit=$?" | tee -a logs/train_all.log
SP=$(ls -td logs/rsl_rl/microduck_sprint_max_allcoll_hd1910_backlash/*/ | head -1)
SP_CK=$(ls "$SP"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
SPEEDS="0.7 1.0 1.2 1.4" MIN_FREE_MB=3000 scripts/eval_run_speeds.sh "$SP" Microduck-SprintMax-Flat-HD1910-AllColl "$SP_CK"
for vx in 0.7 1.0 1.2 1.4; do mv -f "docs/run_speed_${SP_CK}_vx$vx.json" "docs/sprint_max_speed_${SP_CK}_vx$vx.json" 2>/dev/null; done
echo "[$(date +%F_%T)] sprint-max speed table done" >> logs/evals.log
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
