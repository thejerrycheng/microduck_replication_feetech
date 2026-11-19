#!/bin/bash
# grade_real_ckpts.sh <policy> [run_dir]: base + every saved checkpoint of the real-duck fine-tune, 96 envs, NOMINAL and
# STRESS (the conditions of scripts/train_real.sh). Run dir from docs/real/RUNS.txt unless given. Skips existing files.
set -u
name=$1; cd /home/jerrycheng/microduck
declare -A TASK=( [velstand]=Microduck-VelStand-Flat-HD1910-AllColl [run_v3]=Microduck-RunV3-Flat-HD1910-AllColl [sprint_max]=Microduck-SprintMax-Flat-HD1910-AllColl
  [sitstand_fast]=Microduck-SitStandFast-Flat-HD1910 [standup]=Microduck-StandUp-Flat-HD1910 [roulade_head]=Microduck-RouladeHead-Flat-HD1910 [jump]=Microduck-Jump-Flat-HD1910 )
declare -A DIR=( [velstand]=velstand_21495 [run_v3]=run_v3_5999 [sprint_max]=sprint_max_7998 [sitstand_fast]=sitstand_fast_5499 [standup]=standup_9999 [roulade_head]=roulade_head_kp58_7999 [jump]=jump_kp58_2250 )
declare -A CK=( [velstand]=21495 [run_v3]=5999 [sprint_max]=7998 [sitstand_fast]=5499 [standup]=9999 [roulade_head]=7999 [jump]=2250 )
NOMINAL=$(grep -oP '^NOMINAL="\K[^"]+' scripts/train_real.sh); STRESS=$(grep -oP '^STRESS="\K[^"]+' scripts/train_real.sh)
T=${TASK[$name]}; B=$(readlink -f checkpoints/${DIR[$name]}); R=${2:-$(awk -v n="$name" '$1==n{print $2}' docs/real/RUNS.txt | tail -1)}
[ -n "$R" ] || { echo "no run for $name"; exit 1; }
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ] || [ "$(free -m | awk '/Mem:/{print $7}')" -lt 3300 ]; do sleep 30; done; }
out_dir=docs/real/${name}_ckpts; mkdir -p "$out_dir"
specs=("$B:${CK[$name]}:base"); for f in $(ls "$R"/model_*.pt | sort -t_ -k2 -n); do c=$(basename "$f" .pt); c=${c#model_}; [ "$c" -gt "${CK[$name]}" ] && specs+=("$R:$c:ft"); done
echo "grading $name: ${specs[*]}"
one() { # run ck tag cond suffix script args...
  local run=$1 ck=$2 tag=$3 cond=$4 suffix=$5; shift 5
  local out=$out_dir/${tag}_${ck}_${cond}_${suffix}_n96.json; [ -f "$out" ] && return 0
  local ovr; [ "$cond" = nominal ] && ovr=$NOMINAL || ovr=$STRESS
  gate; uv run --project external/IsaacLab python "$@" --task $T --run "$run" --checkpoint "model_$ck.pt" --num_envs 96 presets=backlash $ovr --out "$out" > "logs/gradereal_${name}_${tag}_${ck}_${cond}_${suffix}.log" 2>&1
  echo "$(date +%T) $name $tag $ck $cond $suffix exit=$?"
}
for cond in stress nominal; do for spec in "${specs[@]}"; do IFS=: read run ck tag <<< "$spec"
  case $name in
    velstand)      one "$run" $ck $tag $cond standup scripts/eval_standup.py --zero_twist; one "$run" $ck $tag $cond gait03 scripts/eval_gait_symmetry.py --vx 0.3 ;;
    standup)       one "$run" $ck $tag $cond standup scripts/eval_standup.py ;;
    run_v3)        one "$run" $ck $tag $cond gait05 scripts/eval_gait_symmetry.py --vx 0.5; one "$run" $ck $tag $cond gait09 scripts/eval_gait_symmetry.py --vx 0.9 ;;
    sprint_max)    one "$run" $ck $tag $cond gait12 scripts/eval_gait_symmetry.py --vx 1.2 ;;
    sitstand_fast) one "$run" $ck $tag $cond posture scripts/eval_posture.py ;;
    roulade_head)  one "$run" $ck $tag $cond roulade scripts/eval_roulade.py ;;
    jump)          one "$run" $ck $tag $cond jump scripts/eval_jump.py ;;
  esac
done; done
echo "DONE $name"
