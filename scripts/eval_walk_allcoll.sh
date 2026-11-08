#!/usr/bin/env bash
# Gait/fall check of the full-collision walk fine-tune at its last checkpoint, once it exists (RAM-gated, no video).
set -u
cd "$(dirname "$0")/.."
until grep -q "end walk-allcoll" logs/train_all.log; do sleep 120; done
RUN=$(ls -td logs/rsl_rl/microduck_walk_flat_allcoll_hd1910_backlash/*/ | head -1)
CK=$(ls "$RUN"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1)
while [ "$(free -m | awk 'NR==2{print $7}')" -lt 5000 ]; do sleep 60; done
uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --task Microduck-Velocity-Flat-HD1910-AllColl --run "$(readlink -f "$RUN")" --checkpoint "model_$CK.pt" \
  --vx 0.3 --num_envs 64 presets=backlash --out "docs/gait_walk_allcoll_$CK.json" > "logs/eval_gait_walk_allcoll_$CK.log" 2>&1
echo "[$(date +%F_%T)] walk-allcoll gait $CK exit $?" >> logs/evals.log
