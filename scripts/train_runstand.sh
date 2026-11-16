#!/usr/bin/env bash
# RunStand: run v3 with fall recovery folded in (VelStand recipe on the run env). Train 2000 iterations from run v3
# 5999 with standup 9999 as the recovery expert and run v3 5999 as the upright anchor, then the deterministic
# batteries: get-up families (eval_standup), achieved speed at 0.3-1.1 m/s, and the push test while running.
# Launch: systemd-run --user --unit microduck-runstand --working-directory "$PWD" scripts/train_runstand.sh
# Resumes from the latest RunStand checkpoint; skips training when the final checkpoint exists.
set -u
cd "$(dirname "$0")/.."
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
latest_with_ckpt() { for d in $(ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null); do ls "$d"/model_*.pt >/dev/null 2>&1 && { echo "$d"; return; }; done; }
last_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1; }

ANCHOR=${RUNSTAND_ANCHOR:-/home/jerrycheng/microduck/logs/rsl_rl/microduck_run_flat_allcoll_mirror_hd1910_backlash/2026-11-09_13-40-30/model_5999.pt}
EXPERT=${RUNSTAND_EXPERT:-/home/jerrycheng/microduck/logs/rsl_rl/microduck_standup_flat_hd1910_backlash/2026-11-09_07-08-09/model_9999.pt}
START_CK="$ANCHOR"; END=$((6000 + 2000)); ITERS=2000
PREV=$(latest_with_ckpt microduck_runstand_allcoll_hd1910_backlash)
if [ -n "$PREV" ]; then PREV_CK=$(last_ckpt "$PREV"); if [ "$PREV_CK" -lt $((END - 1)) ]; then START_CK="$(readlink -f "$PREV")/model_$PREV_CK.pt"; ITERS=$((END - PREV_CK)); else ITERS=0; fi; fi
if [ "$ITERS" -gt 0 ]; then
  echo "[$(date +%F_%T)] start runstand ($ITERS its from $START_CK; expert $EXPERT; anchor $ANCHOR)" | tee -a logs/train_all.log
  uv run --project external/IsaacLab python scripts/train.py --task Microduck-RunStand-Flat-HD1910-AllColl --agent rsl_rl_with_symmetry_cfg_entry_point \
    --max_iterations $ITERS $COMMON --checkpoint "$START_CK" \
    agent.algorithm.bc_cfg.expert_checkpoint="$EXPERT" agent.algorithm.bc_cfg.anchor_checkpoint="$ANCHOR" >> logs/train_runstand.log 2>&1
  echo "[$(date +%F_%T)] end runstand exit=$?" | tee -a logs/train_all.log
else
  echo "[$(date +%F_%T)] runstand already complete ($PREV); skipped" | tee -a logs/train_all.log
fi

RS=$(latest_with_ckpt microduck_runstand_allcoll_hd1910_backlash); RS_CK=$(last_ckpt "$RS")
[ -n "$RS" ] || exit 1
gate() { while [ "$(free -m | awk 'NR==2{print $7}')" -lt 3300 ]; do sleep 30; done; }
for ck in ${RUNSTAND_EVAL_CKPTS:-$RS_CK}; do
  gate
  uv run --project external/IsaacLab python scripts/eval_standup.py --task Microduck-RunStand-Flat-HD1910-AllColl --run "$(readlink -f "$RS")" --checkpoint "model_$ck.pt" \
    --num_envs 32 presets=backlash --out "docs/standup_runstand_$ck.json" > "logs/eval_standup_runstand_$ck.log" 2>&1
  echo "[$(date +%F_%T)] runstand get-up eval $ck exit $?" >> logs/evals.log
  SPEEDS="0.3 0.5 0.7 0.9 1.1" MIN_FREE_MB=3300 scripts/eval_run_speeds.sh "$RS" Microduck-RunStand-Flat-HD1910-AllColl "$ck"
  for vx in 0.3 0.5 0.7 0.9 1.1; do [ -f "docs/run_speed_${ck}_vx${vx}.json" ] && mv "docs/run_speed_${ck}_vx${vx}.json" "docs/runstand_speed_${ck}_vx${vx}.json"; done
done
echo "[$(date +%F_%T)] runstand evals done" >> logs/evals.log
