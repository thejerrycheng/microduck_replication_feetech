#!/bin/bash
# grade_gains_ckpts.sh <name>: grade the base checkpoint and every saved checkpoint of the P 5-8 fine-tune at 96 envs, P5 and P8 (docs/gains/<name>_ckpts/).
# Run dir comes from docs/gains/RUNS.txt (written by the chain when the fine-tune finishes).
set -u
name=$1; cd /home/jerrycheng/microduck
declare -A TASK=( [run_v3]=Microduck-RunV3-Flat-HD1910-AllColl [sprint_max]=Microduck-SprintMax-Flat-HD1910-AllColl [sitstand_fast]=Microduck-SitStandFast-Flat-HD1910 [standup]=Microduck-StandUp-Flat-HD1910 [roulade_head]=Microduck-RouladeHead-Flat-HD1910 [jump]=Microduck-Jump-Flat-HD1910 )
declare -A DIR=( [run_v3]=run_v3_5999 [sprint_max]=sprint_max_7998 [sitstand_fast]=sitstand_fast_5499 [standup]=standup_9999 [roulade_head]=roulade_head_7000 [jump]=jump_1500 )
declare -A CK=( [run_v3]=5999 [sprint_max]=7998 [sitstand_fast]=5499 [standup]=9999 [roulade_head]=7000 [jump]=1500 )
T=${TASK[$name]}; B=$(readlink -f checkpoints/${DIR[$name]}); R=$(awk -v n="$name" '$1==n{print $2}' docs/gains/RUNS.txt | tail -1)
[ -n "$R" ] || { echo "no run for $name in RUNS.txt"; exit 1; }
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ] || [ "$(free -m | awk '/Mem:/{print $7}')" -lt 3300 ]; do sleep 30; done; }
out_dir=docs/gains/${name}_ckpts; mkdir -p "$out_dir"
specs=("$B:${CK[$name]}:base"); for f in $(ls "$R"/model_*.pt | sort -t_ -k2 -n); do c=$(basename "$f" .pt); c=${c#model_}; [ "$c" -gt "${CK[$name]}" ] && specs+=("$R:$c:ft"); done
echo "grading $name: ${specs[*]}"
run_one() { # run ck tag P kp suffix cmd...
  local out=$out_dir/${3}_${2}_p${4}_${6}_n96.json; [ -f "$out" ] && return 0
  gate; uv run --project external/IsaacLab python "${@:7}" --task $T --run "$1" --checkpoint "model_$2.pt" --num_envs 96 presets=backlash "$5" --out "$out" > "logs/grade_${name}_${3}_${2}_p${4}_${6}.log" 2>&1
  echo "$(date +%T) $name $3 $2 P$4 $6 exit=$?"
}
for P in 5 8; do s=$([ $P = 5 ] && echo 1.0 || echo 1.6); KP="env.events.randomize_joint_friction.params.kp_scale_range=[$s,$s]"
  for spec in "${specs[@]}"; do IFS=: read run ck tag <<< "$spec"
    case $name in
      roulade_head) run_one "$run" $ck $tag $P "$KP" roulade scripts/eval_roulade.py ;;
      jump)         run_one "$run" $ck $tag $P "$KP" jump scripts/eval_jump.py ;;
      standup)      run_one "$run" $ck $tag $P "$KP" standup scripts/eval_standup.py ;;
      sprint_max)   run_one "$run" $ck $tag $P "$KP" gait12 scripts/eval_gait_symmetry.py --vx 1.2 ;;
      run_v3)       run_one "$run" $ck $tag $P "$KP" gait05 scripts/eval_gait_symmetry.py --vx 0.5; run_one "$run" $ck $tag $P "$KP" gait09 scripts/eval_gait_symmetry.py --vx 0.9 ;;
      sitstand_fast) run_one "$run" $ck $tag $P "$KP" posture scripts/eval_posture.py ;;
    esac
  done
done
echo "DONE $name"
