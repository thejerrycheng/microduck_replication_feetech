#!/usr/bin/env bash
# RunStand v4 final-checkpoint evaluations (the training unit's retry loop expected model_7999; the last checkpoint of a
# 5999 + 2000 warm start is model_7998). Battery, push test, speed table on 7998 (or RUNSTAND_EVAL_CKPTS).
# Launch: systemd-run --user --unit microduck-runstand-v4-eval --working-directory "$PWD" scripts/eval_runstand_v4_final.sh
set -u
cd "$(dirname "$0")/.."
TASK=Microduck-RunStandV4-Flat-HD1910-AllColl; EXP=microduck_runstand_v4_allcoll_hd1910_backlash
RS=$(readlink -f "$(ls -td logs/rsl_rl/$EXP/*/ | head -1)")
gate() { while [ "$(free -m | awk 'NR==2{print $7}')" -lt 3300 ] || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ]; do sleep 30; done; }
mkdir -p docs/push
for ck in ${RUNSTAND_EVAL_CKPTS:-7998}; do
  gate
  uv run --project external/IsaacLab python scripts/eval_standup.py --task $TASK --run "$RS" --checkpoint "model_$ck.pt" \
    --num_envs 32 presets=backlash --out "docs/standup_runstand_v4_$ck.json" > "logs/eval_standup_runstand_v4_$ck.log" 2>&1
  echo "[$(date +%F_%T)] runstand-v4 get-up eval $ck exit $?" >> logs/evals.log
  gate
  uv run --project external/IsaacLab python scripts/push_test.py --task $TASK --policy "runstand_v4_$ck" --policy_path "$RS/exported_model_$ck/policy.pt" \
    --vx 0.5 --num_envs 32 --no_video --out docs/push presets=backlash > "logs/push_runstand_v4_$ck.log" 2>&1
  echo "[$(date +%F_%T)] runstand-v4 push test $ck exit $?" >> logs/evals.log
  SPEEDS="0.3 0.5 0.7 0.9 1.1" MIN_FREE_MB=3300 scripts/eval_run_speeds.sh "$RS" $TASK "$ck"
  for vx in 0.3 0.5 0.7 0.9 1.1; do [ -f "docs/run_speed_${ck}_vx${vx}.json" ] && mv "docs/run_speed_${ck}_vx${vx}.json" "docs/runstand_v4_speed_${ck}_vx${vx}.json"; done
done
echo "[$(date +%F_%T)] runstand-v4 evals done" >> logs/evals.log
