#!/usr/bin/env bash
# Roll + stand straight: fine-tune roulade 6000 with the post-roll head-home term, then grade every checkpoint
# on a fresh 96-env block (sequentially, in this one script).
set -u
cd "$(dirname "$0")/.."
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
SRC=/home/jerrycheng/microduck/logs/rsl_rl/microduck_roulade_flat_hd1910_backlash/2026-11-09_16-40-57/model_6000.pt
echo "[$(date +%F_%T)] start roulade-head (2000 its from $SRC)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-RouladeHead-Flat-HD1910 --agent rsl_rl_with_symmetry_cfg_entry_point \
  --max_iterations 2000 $COMMON --checkpoint "$SRC" >> logs/train_roulade_head.log 2>&1
echo "[$(date +%F_%T)] end roulade-head exit=$?" | tee -a logs/train_all.log
RUN=""; for d in $(ls -td logs/rsl_rl/microduck_roulade_head_flat_hd1910_backlash/*/); do ls "$d"/model_*.pt >/dev/null 2>&1 && { RUN="$(readlink -f "$d")"; break; }; done
LAST=$(ls "$RUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
for ck in 6500 7000 7500 $LAST; do
  [ -f "$RUN/model_$ck.pt" ] || continue
  uv run --project external/IsaacLab python scripts/eval_roulade.py --task Microduck-RouladeHead-Flat-HD1910 --run "$RUN" --checkpoint "model_$ck.pt" --num_envs 96 presets=backlash \
    --out "docs/roulade_head_confirm96_$ck.json" > "logs/eval_roulade_head_$ck.log" 2>&1
  echo "[$(date +%F_%T)] roulade-head eval $ck exit $?" >> logs/evals.log
done
echo "[$(date +%F_%T)] roulade-head evals done" | tee -a logs/train_all.log
