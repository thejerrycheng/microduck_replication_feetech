#!/usr/bin/env bash
# Real-duck fine-tunes of all seven deployed policies (user request 2026-11-19, priority): the training now randomises
# what the duck measured on 2026-11-18 -- supply 6.4-8.2 V with sag floor 6.2 V, command delay 15-40 ms, control period
# 20-40 ms (50 down to 25 Hz, ticks held), servo P 5-8, servo friction x0.8-1.5, joint speed quantised to 0.077 rad/s,
# read dropout 0.5 %, trunk (battery) mass x0.9-1.6, sole friction 0.25-1.2 effective (soles own the contact), the
# locomotion slip cost above 0.2 m/s, stronger pushes for VelStand, the jump flag on a 0.36 s clock. Every policy is
# warm-started from its current pick and graded under the same env at a NOMINAL condition (P 5, 7.6-7.8 V, friction 1.0,
# 50 Hz) and a STRESS condition (P 8, 6.4 V, friction 0.3, 25 Hz) before and after.
# Launch: systemd-run --user --unit microduck-real -p WorkingDirectory="$PWD" scripts/train_real.sh
set -u
cd "$(dirname "$0")/.."
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
SYM="--agent rsl_rl_with_symmetry_cfg_entry_point"
ITERS=${REAL_ITERS:-2000}
POLICIES=${POLICIES:-"velstand standup run_v3 sitstand_fast jump roulade_head sprint_max"}
declare -A TASK=( [velstand]=Microduck-VelStand-Flat-HD1910-AllColl [run_v3]=Microduck-RunV3-Flat-HD1910-AllColl [sprint_max]=Microduck-SprintMax-Flat-HD1910-AllColl
  [sitstand_fast]=Microduck-SitStandFast-Flat-HD1910 [standup]=Microduck-StandUp-Flat-HD1910 [roulade_head]=Microduck-RouladeHead-Flat-HD1910 [jump]=Microduck-Jump-Flat-HD1910 )
declare -A AGENT=( [velstand]="$SYM" [run_v3]="$SYM" [sprint_max]="$SYM" [sitstand_fast]="" [standup]="" [roulade_head]="$SYM" [jump]="$SYM" )
declare -A DIR=( [velstand]=velstand_21495 [run_v3]=run_v3_5999 [sprint_max]=sprint_max_7998 [sitstand_fast]=sitstand_fast_5499 [standup]=standup_9999 [roulade_head]=roulade_head_kp58_7999 [jump]=jump_kp58_2250 )
declare -A CK=( [velstand]=21495 [run_v3]=5999 [sprint_max]=7998 [sitstand_fast]=5499 [standup]=9999 [roulade_head]=7999 [jump]=2250 )
# fine-tune settings: entropy 0.001 everywhere (0.01 collapsed the run and sprint gaits, docs/gains/NOTES.md)
declare -A EXTRA=( [velstand]="agent.algorithm.entropy_coef=0.001 agent.algorithm.learning_rate=0.0005" [run_v3]="agent.algorithm.entropy_coef=0.001 agent.algorithm.learning_rate=0.0005"
  [sprint_max]="agent.algorithm.entropy_coef=0.001 agent.algorithm.learning_rate=0.0005" [sitstand_fast]="agent.algorithm.entropy_coef=0.001 agent.algorithm.learning_rate=0.001"
  [standup]="agent.algorithm.entropy_coef=0.001 agent.algorithm.learning_rate=0.001" [roulade_head]="agent.algorithm.entropy_coef=0.001 agent.algorithm.learning_rate=0.001" [jump]="agent.algorithm.entropy_coef=0.001 agent.algorithm.learning_rate=0.001" )
NOMINAL="env.events.randomize_joint_friction.params.kp_scale_range=[1.0,1.0] env.scene.robot.actuators.servos.vin_range=[7.6,7.8] env.events.foot_friction.params.static_friction_range=[1.0,1.0] env.events.foot_friction.params.dynamic_friction_range=[1.0,1.0] env.events.foot_friction_patch.params.static_friction_range=[1.0,1.0] env.events.foot_friction_patch.params.dynamic_friction_range=[1.0,1.0] env.actions.joint_pos.control_period_range_ms=[20,20]"
STRESS="env.events.randomize_joint_friction.params.kp_scale_range=[1.6,1.6] env.scene.robot.actuators.servos.vin_range=[6.4,6.4] env.events.foot_friction.params.static_friction_range=[0.3,0.3] env.events.foot_friction.params.dynamic_friction_range=[0.3,0.3] env.events.foot_friction_patch.params.static_friction_range=[0.3,0.3] env.events.foot_friction_patch.params.dynamic_friction_range=[0.3,0.3] env.actions.joint_pos.control_period_range_ms=[40,40]"
gpu_gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "${MIN_GPU_FREE_MB:-7500}" ]; do sleep 60; done; }
gate() { while [ "$(free -m | awk 'NR==2{print $7}')" -lt 3300 ] || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ]; do sleep 30; done; }
last_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1; }
newest_since() { for d in $(ls -td logs/rsl_rl/*/*/ 2>/dev/null); do [ "$(stat -c %Y "$d")" -ge "$1" ] || continue; ls "$d"/model_*.pt >/dev/null 2>&1 && { readlink -f "$d"; return; }; done; }
mkdir -p docs/real
ev() { # name run ck tag cond suffix script args...
  local name=$1 run=$2 ck=$3 tag=$4 cond=$5 suffix=$6; shift 6
  local out=docs/real/${name}_${tag}_${cond}_${suffix}.json; [ -f "$out" ] && return 0
  local ovr; [ "$cond" = nominal ] && ovr=$NOMINAL || ovr=$STRESS
  gate; uv run --project external/IsaacLab python "$@" --task ${TASK[$name]} --run "$run" --checkpoint "model_$ck.pt" --num_envs 96 presets=backlash $ovr --out "$out" > "logs/real_${name}_${tag}_${cond}_${suffix}.log" 2>&1
  echo "[$(date +%F_%T)] real eval $name $tag $cond $suffix exit=$?" >> logs/evals.log
}
run_eval() { # name run ck tag
  local name=$1 run=$2 ck=$3 tag=$4
  for cond in nominal stress; do
    case $name in
      velstand)      ev $name "$run" $ck $tag $cond gait03 scripts/eval_gait_symmetry.py --vx 0.3; ev $name "$run" $ck $tag $cond standup scripts/eval_standup.py --zero_twist ;;
      run_v3)        ev $name "$run" $ck $tag $cond gait05 scripts/eval_gait_symmetry.py --vx 0.5; ev $name "$run" $ck $tag $cond gait09 scripts/eval_gait_symmetry.py --vx 0.9 ;;
      sprint_max)    ev $name "$run" $ck $tag $cond gait12 scripts/eval_gait_symmetry.py --vx 1.2 ;;
      sitstand_fast) ev $name "$run" $ck $tag $cond posture scripts/eval_posture.py ;;
      standup)       ev $name "$run" $ck $tag $cond standup scripts/eval_standup.py ;;
      roulade_head)  ev $name "$run" $ck $tag $cond roulade scripts/eval_roulade.py ;;
      jump)          ev $name "$run" $ck $tag $cond jump scripts/eval_jump.py ;;
    esac
  done
}
echo "[$(date +%F_%T)] real chain start: $POLICIES ($ITERS its each)" | tee -a logs/train_all.log
for name in $POLICIES; do run_eval "$name" "$(readlink -f checkpoints/${DIR[$name]})" "${CK[$name]}" base; done
echo "[$(date +%F_%T)] real chain: baselines done" | tee -a logs/train_all.log
for name in $POLICIES; do
  task=${TASK[$name]}; ck=${CK[$name]}; start="$(readlink -f checkpoints/${DIR[$name]})/model_$ck.pt"; extra=${EXTRA[$name]}
  [ "$name" = velstand ] && extra="$extra agent.algorithm.bc_cfg.expert_checkpoint=$(readlink -f checkpoints/standup_9999)/model_9999.pt agent.algorithm.bc_cfg.anchor_checkpoint=$start"
  attempt=0; run=""
  while [ -z "$run" ] && [ "$attempt" -lt 4 ]; do
    attempt=$((attempt + 1)); since=$(date +%s); gpu_gate
    echo "[$(date +%F_%T)] start real-$name attempt $attempt ($ITERS its from $start; $extra)" | tee -a logs/train_all.log
    uv run --project external/IsaacLab python scripts/train.py --task $task ${AGENT[$name]} --max_iterations $ITERS $COMMON --checkpoint "$start" $extra >> "logs/train_real_$name.log" 2>&1
    echo "[$(date +%F_%T)] end real-$name attempt $attempt exit=$?" | tee -a logs/train_all.log
    run=$(newest_since "$since"); [ -n "$run" ] && [ "$(last_ckpt "$run")" -ge $((ck + ITERS - 1)) ] || { run=""; sleep 120; }
  done
  [ -n "$run" ] || { echo "[$(date +%F_%T)] real-$name gave up" | tee -a logs/train_all.log; continue; }
  echo "$name $run $(last_ckpt "$run")" >> docs/real/RUNS.txt
  run_eval "$name" "$run" "$(last_ckpt "$run")" ft
done
echo "[$(date +%F_%T)] real chain done" | tee -a logs/train_all.log
