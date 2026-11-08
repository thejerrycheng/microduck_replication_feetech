#!/usr/bin/env bash
# 1) fine-tune walk and run on the full-collision model (warm starts), 2) resume sit/stand phase 2 from its last
# checkpoint, 3) log "end stand" so the roulade/jump chain fires. One trainer at a time.
set -u
cd "$(dirname "$0")/.."
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
WALK_CK=/home/jerrycheng/microduck/logs/rsl_rl/microduck_walk_flat_symmetry_hd1910_backlash/2026-11-07_09-39-14/model_15000.pt
RUN_CK=/home/jerrycheng/microduck/logs/rsl_rl/microduck_run_flat_hd1910_backlash/2026-11-07_19-05-22/model_2999.pt
STAND_CK=/home/jerrycheng/microduck/logs/rsl_rl/microduck_sitstand_flat_hd1910_backlash/2026-11-08_13-55-30/model_7250.pt
echo "[$(date +%F_%T)] start walk-allcoll fine-tune (${FT_ITERS:-1500} its from walk 15000)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-Velocity-Flat-HD1910-AllColl --agent rsl_rl_with_symmetry_cfg_entry_point \
  --max_iterations "${FT_ITERS:-1500}" $COMMON --checkpoint "$WALK_CK" >> logs/train_walk_allcoll.log 2>&1
echo "[$(date +%F_%T)] end walk-allcoll exit=$?" | tee -a logs/train_all.log
echo "[$(date +%F_%T)] start run-allcoll fine-tune (${FT_ITERS:-1500} its from run 2999)" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-RunV2-Flat-HD1910-AllColl --agent rsl_rl_with_symmetry_cfg_entry_point \
  --max_iterations "${FT_ITERS:-1500}" $COMMON --checkpoint "$RUN_CK" >> logs/train_run_allcoll.log 2>&1
echo "[$(date +%F_%T)] end run-allcoll exit=$?" | tee -a logs/train_all.log
echo "[$(date +%F_%T)] resume stand phase2 from model_7250 (OOM-killed 14:50), $((14999-7250)) its @ entropy 0.001" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-SitStand-Flat-HD1910 --max_iterations $((14999-7250)) $COMMON \
  --checkpoint "$STAND_CK" agent.algorithm.entropy_coef=0.001 >> logs/train_stand.log 2>&1
echo "[$(date +%F_%T)] end stand exit=$?" | tee -a logs/train_all.log
