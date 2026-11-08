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
# Demo takes with the fine-tuned policies, while NO trainer is running (video frames live in RAM).
WALK_RUN=$(ls -td logs/rsl_rl/microduck_walk_flat_allcoll_hd1910_backlash/*/ | head -1)
RUN_RUN=$(ls -td logs/rsl_rl/microduck_run_flat_allcoll_hd1910_backlash/*/ | head -1)
WALK_LAST=$(ls "$WALK_RUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
RUN_LAST=$(ls "$RUN_RUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
for ck in "$WALK_RUN:$WALK_LAST:Microduck-Velocity-Flat-HD1910-AllColl" "$RUN_RUN:$RUN_LAST:Microduck-RunV2-Flat-HD1910-AllColl"; do
  IFS=: read -r rd n task <<< "$ck"
  uv run --project external/IsaacLab python scripts/play.py --rl_library rsl_rl --task "$task" --num_envs 4 presets=backlash \
    --checkpoint "$(readlink -f "$rd")/model_$n.pt" --video --video_length 1 --video_interval 1000000 > logs/export_allcoll.log 2>&1
  mkdir -p "$rd/exported_model_$n"; cp "$rd"/exported/policy.* "$rd/exported_model_$n/" 2>/dev/null
done
echo "[$(date +%F_%T)] demo takes with walk-allcoll $WALK_LAST and run-allcoll $RUN_LAST" | tee -a logs/train_all.log
for seed in 1 2 3 4 5; do
  uv run --project external/IsaacLab python scripts/demo_sequence.py presets=backlash --out rollouts/demo_allcoll_seed$seed --seed $seed \
    --walk "$WALK_RUN/exported_model_$WALK_LAST/policy.pt" --run "$RUN_RUN/exported_model_$RUN_LAST/policy.pt" > logs/demo_allcoll_$seed.log 2>&1
  grep -q "^CLEAN" logs/demo_allcoll_$seed.log && { echo "[$(date +%F_%T)] demo CLEAN at seed $seed" | tee -a logs/train_all.log; break; }
  echo "[$(date +%F_%T)] demo seed $seed fell" | tee -a logs/train_all.log
done
echo "[$(date +%F_%T)] demo takes done" >> logs/evals.log
echo "[$(date +%F_%T)] resume stand phase2 from model_7250 (OOM-killed 14:50), $((14999-7250)) its @ entropy 0.001" | tee -a logs/train_all.log
uv run --project external/IsaacLab python scripts/train.py --task Microduck-SitStand-Flat-HD1910 --max_iterations $((14999-7250)) $COMMON \
  --checkpoint "$STAND_CK" agent.algorithm.entropy_coef=0.001 >> logs/train_stand.log 2>&1
echo "[$(date +%F_%T)] end stand exit=$?" | tee -a logs/train_all.log
