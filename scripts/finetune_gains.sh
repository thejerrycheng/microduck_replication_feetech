#!/usr/bin/env bash
# Firmware-gain fine-tune of every deployable policy (user decision 2026-11-17: "fine-tune all policies for P 5-8, D 0").
# The identified HD1910 law has a P path only (BAM, kp_fw 5); real runtimes write KP 5 (stored), 6 (vendor run gain) or
# 8 (duckyduck walkers). Each policy is warm-started from its export and trained GAIN_ITERS iterations with the per-servo
# firmware P drawn uniformly in [5, 8] (kp_scale 1.0-1.6 on kp_fw 5) instead of 5 x [0.85, 1.15]; baseline and fine-tune
# are then graded DETERMINISTICALLY at fixed P 5 and fixed P 8 with the same evaluators the originals were accepted on.
# Results: docs/gains/<policy>_<base|ft>_p<5|8>_<eval>.json. Nothing is exported or replaced here: the pick is by hand.
# Launch: systemd-run --user --unit microduck-finetune-gains --working-directory "$PWD" scripts/finetune_gains.sh
set -u
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet microduck-runstand-v4; do sleep 60; done
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
KP_TRAIN="env.events.randomize_joint_friction.params.kp_scale_range=[1.0,1.6]"
kp_fixed() { echo "env.events.randomize_joint_friction.params.kp_scale_range=[$1,$1]"; }   # 1.0 -> P 5, 1.6 -> P 8
GAIN_ITERS=${GAIN_ITERS:-1000}
SYM="--agent rsl_rl_with_symmetry_cfg_entry_point"
POLICIES=${POLICIES:-"velstand run_v3 sprint_max sitstand_fast standup roulade_head jump"}
declare -A TASK=( [velstand]=Microduck-VelStand-Flat-HD1910-AllColl [run_v3]=Microduck-RunV3-Flat-HD1910-AllColl
  [sprint_max]=Microduck-SprintMax-Flat-HD1910-AllColl [sitstand_fast]=Microduck-SitStandFast-Flat-HD1910
  [standup]=Microduck-StandUp-Flat-HD1910 [roulade_head]=Microduck-RouladeHead-Flat-HD1910 [jump]=Microduck-Jump-Flat-HD1910 )
declare -A AGENT=( [velstand]="$SYM" [run_v3]="$SYM" [sprint_max]="$SYM" [sitstand_fast]="" [standup]="" [roulade_head]="$SYM" [jump]="$SYM" )
declare -A DIR=( [velstand]=velstand_21495 [run_v3]=run_v3_5999 [sprint_max]=sprint_max_7998 [sitstand_fast]=sitstand_fast_5499
  [standup]=standup_9999 [roulade_head]=roulade_head_7000 [jump]=jump_1500 )
declare -A CK=( [velstand]=21495 [run_v3]=5999 [sprint_max]=7998 [sitstand_fast]=5499 [standup]=9999 [roulade_head]=7000 [jump]=1500 )
gpu_gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "${MIN_GPU_FREE_MB:-7500}" ]; do sleep 60; done; }
gate() { while [ "$(free -m | awk 'NR==2{print $7}')" -lt 3300 ] || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ]; do sleep 30; done; }
last_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1; }
# newest run dir (any experiment) created after $1 (epoch seconds) that holds a checkpoint
newest_since() { for d in $(ls -td logs/rsl_rl/*/*/ 2>/dev/null); do [ "$(stat -c %Y "$d")" -ge "$1" ] || continue; ls "$d"/model_*.pt >/dev/null 2>&1 && { readlink -f "$d"; return; }; done; }
agent_val() { grep -m1 -E "^\s*$2:" "checkpoints/${DIR[$1]}/agent.yaml" | awk '{print $2}'; }
P=""   # set per call to 5 or 8
run_eval() { # run_eval <name> <run dir> <ckpt> <tag>
  local name=$1 run=$2 ck=$3 tag=$4 task=${TASK[$1]} scale out exported
  for P in 5 8; do
    scale=$([ "$P" = 5 ] && echo 1.0 || echo 1.6); out="docs/gains/${name}_${tag}_p${P}"; exported="$run/exported_model_$ck/policy.pt"
    case $name in
      velstand)
        gate; uv run --project external/IsaacLab python scripts/eval_standup.py --task $task --run "$run" --checkpoint "model_$ck.pt" --zero_twist --num_envs 32 presets=backlash "$(kp_fixed $scale)" --out "${out}_standup.json" > "logs/gains_${name}_${tag}_p${P}_standup.log" 2>&1
        gate; uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --task $task --run "$run" --checkpoint "model_$ck.pt" --vx 0.3 --num_envs 64 presets=backlash "$(kp_fixed $scale)" --out "${out}_gait03.json" > "logs/gains_${name}_${tag}_p${P}_gait.log" 2>&1
        gate; uv run --project external/IsaacLab python scripts/eval_squat.py --task $task --policy "$exported" presets=backlash "$(kp_fixed $scale)" --out "${out}_squat.json" > "logs/gains_${name}_${tag}_p${P}_squat.log" 2>&1 ;;
      run_v3)
        for vx in 0.5 0.9; do gate; uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --task $task --run "$run" --checkpoint "model_$ck.pt" --vx $vx --num_envs 32 presets=backlash "$(kp_fixed $scale)" --out "${out}_gait${vx/./}.json" > "logs/gains_${name}_${tag}_p${P}_gait${vx/./}.log" 2>&1; done ;;
      sprint_max)
        gate; uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --task $task --run "$run" --checkpoint "model_$ck.pt" --vx 1.2 --num_envs 32 presets=backlash "$(kp_fixed $scale)" --out "${out}_gait12.json" > "logs/gains_${name}_${tag}_p${P}_gait12.log" 2>&1 ;;
      sitstand_fast)
        gate; uv run --project external/IsaacLab python scripts/eval_posture.py --task $task --run "$run" --checkpoint "model_$ck.pt" --num_envs 32 presets=backlash "$(kp_fixed $scale)" --out "${out}_posture.json" > "logs/gains_${name}_${tag}_p${P}_posture.log" 2>&1
        gate; uv run --project external/IsaacLab python scripts/eval_transition.py --task $task --policy "$exported" presets=backlash "$(kp_fixed $scale)" --out "${out}_transition.json" > "logs/gains_${name}_${tag}_p${P}_transition.log" 2>&1 ;;
      standup)
        gate; uv run --project external/IsaacLab python scripts/eval_standup.py --task $task --run "$run" --checkpoint "model_$ck.pt" --num_envs 32 presets=backlash "$(kp_fixed $scale)" --out "${out}_standup.json" > "logs/gains_${name}_${tag}_p${P}_standup.log" 2>&1 ;;
      roulade_head)
        gate; uv run --project external/IsaacLab python scripts/eval_roulade.py --task $task --run "$run" --checkpoint "model_$ck.pt" --num_envs 32 presets=backlash "$(kp_fixed $scale)" --out "${out}_roulade.json" > "logs/gains_${name}_${tag}_p${P}_roulade.log" 2>&1 ;;
      jump)
        gate; uv run --project external/IsaacLab python scripts/eval_jump.py --task $task --run "$run" --checkpoint "model_$ck.pt" --num_envs 32 presets=backlash "$(kp_fixed $scale)" --out "${out}_jump.json" > "logs/gains_${name}_${tag}_p${P}_jump.log" 2>&1 ;;
    esac
    echo "[$(date +%F_%T)] gains eval $name $tag P$P done" >> logs/evals.log
  done
}
echo "[$(date +%F_%T)] finetune-gains: baselines" | tee -a logs/train_all.log
for name in $POLICIES; do ls docs/gains/${name}_base_p8_*.json >/dev/null 2>&1 || run_eval "$name" "$(readlink -f checkpoints/${DIR[$name]})" "${CK[$name]}" base; done
for name in $POLICIES; do
  task=${TASK[$name]}; ck=${CK[$name]}; start="$(readlink -f checkpoints/${DIR[$name]})/model_$ck.pt"
  extra="agent.algorithm.entropy_coef=$(agent_val "$name" entropy_coef) agent.algorithm.learning_rate=$(agent_val "$name" learning_rate)"
  [ "$name" = velstand ] && extra="$extra agent.algorithm.bc_cfg.expert_checkpoint=$(readlink -f checkpoints/standup_9999)/model_9999.pt agent.algorithm.bc_cfg.anchor_checkpoint=$start"
  attempt=0; run=""
  while [ -z "$run" ] && [ "$attempt" -lt 4 ]; do
    attempt=$((attempt + 1)); since=$(date +%s); gpu_gate
    echo "[$(date +%F_%T)] start gains-$name attempt $attempt ($GAIN_ITERS its from $start; $extra)" | tee -a logs/train_all.log
    uv run --project external/IsaacLab python scripts/train.py --task $task ${AGENT[$name]} --max_iterations $GAIN_ITERS $COMMON \
      --checkpoint "$start" "$KP_TRAIN" $extra >> "logs/train_gains_$name.log" 2>&1
    echo "[$(date +%F_%T)] end gains-$name attempt $attempt exit=$?" | tee -a logs/train_all.log
    run=$(newest_since "$since"); [ -n "$run" ] && [ "$(last_ckpt "$run")" -ge $((ck + GAIN_ITERS - 1)) ] || { run=""; sleep 120; }
  done
  [ -n "$run" ] || { echo "[$(date +%F_%T)] gains-$name gave up" | tee -a logs/train_all.log; continue; }
  echo "$name $run $(last_ckpt "$run")" >> docs/gains/RUNS.txt
  run_eval "$name" "$run" "$(last_ckpt "$run")" ft
done
echo "[$(date +%F_%T)] finetune-gains done" | tee -a logs/train_all.log
