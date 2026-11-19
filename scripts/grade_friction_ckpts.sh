#!/bin/bash
# grade_friction_ckpts.sh <velstand|run_v3>: base + every saved checkpoint of the slippery-floor fine-tune, 96 envs,
# under the *-LowMu task at FIXED sole friction (0.3, 0.6, 1.0) and FIXED P (5, 8). Run dir from docs/friction/RUNS.txt.
set -u
name=$1; cd /home/jerrycheng/microduck
declare -A TASK=( [velstand]=Microduck-VelStand-Flat-HD1910-AllColl-LowMu [run_v3]=Microduck-RunV3-Flat-HD1910-AllColl-LowMu )
declare -A DIR=( [velstand]=velstand_21495 [run_v3]=run_v3_5999 ); declare -A CK=( [velstand]=21495 [run_v3]=5999 )
T=${TASK[$name]}; B=$(readlink -f checkpoints/${DIR[$name]}); R=$(awk -v n="$name" '$1==n{print $2}' docs/friction/RUNS.txt | tail -1)
[ -n "$R" ] || { echo "no run for $name in docs/friction/RUNS.txt"; exit 1; }
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ] || [ "$(free -m | awk '/Mem:/{print $7}')" -lt 3300 ]; do sleep 30; done; }
out_dir=docs/friction/${name}_ckpts; mkdir -p "$out_dir"
specs=("$B:${CK[$name]}:base"); for f in $(ls "$R"/model_*.pt | sort -t_ -k2 -n); do c=$(basename "$f" .pt); c=${c#model_}; [ "$c" -gt "${CK[$name]}" ] && specs+=("$R:$c:ft"); done
echo "grading $name: ${specs[*]}"
run_one() { # run ck tag P mu suffix cmd...
  local out=$out_dir/${3}_${2}_mu${5/./}_p${4}_${6}_n96.json; [ -f "$out" ] && return 0
  local s=$([ $4 = 5 ] && echo 1.0 || echo 1.6)
  gate; uv run --project external/IsaacLab python "${@:7}" --task $T --run "$1" --checkpoint "model_$2.pt" --num_envs 96 presets=backlash \
    "env.events.randomize_joint_friction.params.kp_scale_range=[$s,$s]" "env.events.foot_friction.params.static_friction_range=[$5,$5]" "env.events.foot_friction.params.dynamic_friction_range=[$5,$5]" \
    --out "$out" > "logs/gradefr_${name}_${3}_${2}_mu${5/./}_p${4}_${6}.log" 2>&1
  echo "$(date +%T) $name $3 $2 mu$5 P$4 $6 exit=$?"
}
for mu in 0.3 0.6 1.0; do for P in 5 8; do for spec in "${specs[@]}"; do IFS=: read run ck tag <<< "$spec"
  case $name in
    velstand) run_one "$run" $ck $tag $P $mu gait03 scripts/eval_gait_symmetry.py --vx 0.3; run_one "$run" $ck $tag $P $mu standup scripts/eval_standup.py --zero_twist ;;
    run_v3)   run_one "$run" $ck $tag $P $mu gait05 scripts/eval_gait_symmetry.py --vx 0.5; run_one "$run" $ck $tag $P $mu gait09 scripts/eval_gait_symmetry.py --vx 0.9 ;;
  esac
done; done; done
echo "DONE $name"
