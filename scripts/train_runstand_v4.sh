#!/usr/bin/env bash
# RunStand v4 (v2 + upright-gated tracking), queued behind the v3 unit: waits for `microduck-runstand` to finish, trains
# 2000 iterations from run v3 5999 (expert standup 9999, anchor run v3 5999), then the get-up battery, the speed table
# and the push-while-running test on the final checkpoint plus any in RUNSTAND_EVAL_CKPTS.
# Launch: systemd-run --user --unit microduck-runstand-v4 --working-directory "$PWD" scripts/train_runstand_v4.sh
set -u
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet microduck-runstand-v3; do sleep 60; done
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
latest_with_ckpt() { for d in $(ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null); do ls "$d"/model_*.pt >/dev/null 2>&1 && { echo "$d"; return; }; done; }
last_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1; }
TASK=Microduck-RunStandV4-Flat-HD1910-AllColl; EXP=microduck_runstand_v4_allcoll_hd1910_backlash
ANCHOR=${RUNSTAND_ANCHOR:-/home/jerrycheng/microduck/logs/rsl_rl/microduck_run_flat_allcoll_mirror_hd1910_backlash/2026-11-09_13-40-30/model_5999.pt}
EXPERT=${RUNSTAND_EXPERT:-/home/jerrycheng/microduck/logs/rsl_rl/microduck_standup_flat_hd1910_backlash/2026-11-09_07-08-09/model_9999.pt}
START_CK="$ANCHOR"; END=$((6000 + 2000)); ITERS=2000
PREV=$(latest_with_ckpt $EXP)
if [ -n "$PREV" ]; then PREV_CK=$(last_ckpt "$PREV"); if [ "$PREV_CK" -lt $((END - 1)) ]; then START_CK="$(readlink -f "$PREV")/model_$PREV_CK.pt"; ITERS=$((END - PREV_CK)); else ITERS=0; fi; fi
gpu_gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "${MIN_GPU_FREE_MB:-7500}" ]; do sleep 60; done; }
prune_empty() { for d in logs/rsl_rl/$EXP/*/; do ls "$d"model_*.pt >/dev/null 2>&1 || rm -rf "$d"; done; }
attempt=0
while [ "$ITERS" -gt 0 ] && [ "$attempt" -lt 6 ]; do
  attempt=$((attempt + 1))
  prune_empty
  gpu_gate
  echo "[$(date +%F_%T)] start runstand-v4 attempt $attempt ($ITERS its from $START_CK; expert $EXPERT; anchor $ANCHOR)" | tee -a logs/train_all.log
  uv run --project external/IsaacLab python scripts/train.py --task $TASK --agent rsl_rl_with_symmetry_cfg_entry_point \
    --max_iterations $ITERS $COMMON --checkpoint "$START_CK" \
    agent.algorithm.bc_cfg.expert_checkpoint="$EXPERT" agent.algorithm.bc_cfg.anchor_checkpoint="$ANCHOR" >> logs/train_runstand_v4.log 2>&1
  echo "[$(date +%F_%T)] end runstand-v4 attempt $attempt exit=$?" | tee -a logs/train_all.log
  # done when the final checkpoint exists; otherwise resume from whatever was saved (or retry from scratch)
  PREV=$(latest_with_ckpt $EXP)
  if [ -n "$PREV" ]; then PREV_CK=$(last_ckpt "$PREV"); if [ "$PREV_CK" -lt $((END - 1)) ]; then START_CK="$(readlink -f "$PREV")/model_$PREV_CK.pt"; ITERS=$((END - PREV_CK)); else ITERS=0; fi; fi
  [ "$ITERS" -gt 0 ] && sleep 120
done
[ "$ITERS" -gt 0 ] && echo "[$(date +%F_%T)] runstand-v4 gave up after $attempt attempts" | tee -a logs/train_all.log
RS=$(latest_with_ckpt $EXP); RS_CK=$(last_ckpt "$RS")
[ -n "$RS" ] || exit 1
gate() { while [ "$(free -m | awk 'NR==2{print $7}')" -lt 3300 ] || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ]; do sleep 30; done; }
mkdir -p docs/push
for ck in ${RUNSTAND_EVAL_CKPTS:-$RS_CK}; do
  gate
  uv run --project external/IsaacLab python scripts/eval_standup.py --task $TASK --run "$(readlink -f "$RS")" --checkpoint "model_$ck.pt" \
    --num_envs 32 presets=backlash --out "docs/standup_runstand_v4_$ck.json" > "logs/eval_standup_runstand_v4_$ck.log" 2>&1
  echo "[$(date +%F_%T)] runstand-v4 get-up eval $ck exit $?" >> logs/evals.log
  gate
  uv run --project external/IsaacLab python scripts/push_test.py --task $TASK --policy "runstand_v4_$ck" --policy_path "$(readlink -f "$RS")/exported_model_$ck/policy.pt" \
    --vx 0.5 --num_envs 32 --no_video --out docs/push presets=backlash > "logs/push_runstand_v4_$ck.log" 2>&1
  echo "[$(date +%F_%T)] runstand-v4 push test $ck exit $?" >> logs/evals.log
  SPEEDS="0.3 0.5 0.7 0.9 1.1" MIN_FREE_MB=3300 scripts/eval_run_speeds.sh "$RS" $TASK "$ck"
  for vx in 0.3 0.5 0.7 0.9 1.1; do [ -f "docs/run_speed_${ck}_vx${vx}.json" ] && mv "docs/run_speed_${ck}_vx${vx}.json" "docs/runstand_v4_speed_${ck}_vx${vx}.json"; done
done
echo "[$(date +%F_%T)] runstand-v4 evals done" >> logs/evals.log
