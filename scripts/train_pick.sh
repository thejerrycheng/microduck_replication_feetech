#!/usr/bin/env bash
# Pick-up policy: walk to a randomised object and lift it with the beak. Warm start velstand_21495 (same 61/14 contract),
# PICK_ITERS iterations with the firmware P drawn per servo in [5, 8] (the gain decision of 2026-11-17), then the
# deterministic pick evaluation on every 500th checkpoint and the last. Queued behind the gain fine-tune unit (one
# 4096-env trainer at a time). Launch: systemd-run --user --unit microduck-pick --working-directory "$PWD" scripts/train_pick.sh
set -u
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet microduck-finetune-gains; do sleep 60; done
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
KP_TRAIN="env.events.randomize_joint_friction.params.kp_scale_range=[1.0,1.6]"
TASK=Microduck-Pick-Flat-HD1910-AllColl; EXP=microduck_pick_allcoll_hd1910_backlash
START="$(readlink -f checkpoints/velstand_21495)/model_21495.pt"; START_IT=21495
PICK_ITERS=${PICK_ITERS:-3000}; END=$((START_IT + PICK_ITERS - 1))
latest_with_ckpt() { for d in $(ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null); do ls "$d"/model_*.pt >/dev/null 2>&1 && { echo "$d"; return; }; done; }
last_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1; }
gpu_gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "${MIN_GPU_FREE_MB:-7500}" ]; do sleep 60; done; }
gate() { while [ "$(free -m | awk 'NR==2{print $7}')" -lt 3300 ] || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ]; do sleep 30; done; }
prune_empty() { for d in logs/rsl_rl/$EXP/*/; do [ -d "$d" ] && { ls "$d"model_*.pt >/dev/null 2>&1 || rm -rf "$d"; }; done; }
START_CK="$START"; ITERS=$PICK_ITERS
PREV=$(latest_with_ckpt $EXP)
if [ -n "$PREV" ]; then PREV_CK=$(last_ckpt "$PREV"); if [ "$PREV_CK" -lt "$END" ]; then START_CK="$(readlink -f "$PREV")/model_$PREV_CK.pt"; ITERS=$((END - PREV_CK)); else ITERS=0; fi; fi
attempt=0
while [ "$ITERS" -gt 0 ] && [ "$attempt" -lt 6 ]; do
  attempt=$((attempt + 1)); prune_empty; gpu_gate
  echo "[$(date +%F_%T)] start pick attempt $attempt ($ITERS its from $START_CK)" | tee -a logs/train_all.log
  uv run --project external/IsaacLab python scripts/train.py --task $TASK --max_iterations $ITERS $COMMON --checkpoint "$START_CK" "$KP_TRAIN" >> logs/train_pick.log 2>&1
  echo "[$(date +%F_%T)] end pick attempt $attempt exit=$?" | tee -a logs/train_all.log
  PREV=$(latest_with_ckpt $EXP)
  if [ -n "$PREV" ]; then PREV_CK=$(last_ckpt "$PREV"); if [ "$PREV_CK" -lt "$END" ]; then START_CK="$(readlink -f "$PREV")/model_$PREV_CK.pt"; ITERS=$((END - PREV_CK)); else ITERS=0; fi; fi
  [ "$ITERS" -gt 0 ] && sleep 120
done
[ "$ITERS" -gt 0 ] && echo "[$(date +%F_%T)] pick gave up after $attempt attempts" | tee -a logs/train_all.log
RUN=$(latest_with_ckpt $EXP); [ -n "$RUN" ] || exit 1
for ck in $(ls "$RUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | awk -v s=$START_IT -v e=$END '($1-s)%500==499 || $1==e'); do
  gate
  uv run --project external/IsaacLab python scripts/eval_pick.py --task $TASK --run "$(readlink -f "$RUN")" --checkpoint "model_$ck.pt" --num_envs 32 presets=backlash "$KP_TRAIN" --out "docs/pick_$ck.json" > "logs/eval_pick_$ck.log" 2>&1
  echo "[$(date +%F_%T)] pick eval $ck exit $?" >> logs/evals.log
done
echo "[$(date +%F_%T)] pick evals done" >> logs/evals.log
