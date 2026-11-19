#!/usr/bin/env bash
# Slippery-floor fine-tunes (user request 2026-11-19: "the current policy falls due to friction; decrease the friction DR,
# rigorously randomise the foot friction, penalise only when the foot slips too much; retrain walking and running").
# Measured first (docs/friction/mu_eff_velstand_asis.json): the floor is authored at 1.0 and Newton's MuJoCo solver takes
# the MAX of sole and floor friction, so the old 0.5-1.3 sole draw never produced a contact below 1.0. The *-LowMu tasks
# give the soles mjc:priority 1 (their draw IS the contact friction), draw 0.25-1.2 per sole at every reset, and replace
# the always-on slip cost by one that charges only above 0.2 m/s. Warm starts: velstand_21495 and run_v3_5999, with the
# real-robot servo P range (kp_scale 1.0-1.6 = P 5-8), entropy 0.001 / lr 5e-4 (the 0.01 fine-tunes collapsed the gaits).
# Launch: systemd-run --user --unit microduck-friction -p WorkingDirectory="$PWD" scripts/train_friction.sh
set -u
cd "$(dirname "$0")/.."
COMMON="--rl_library rsl_rl --num_envs 4096 presets=backlash --video --video_length 400 --video_interval 6000 --max_visible_envs 16"
KP_TRAIN="env.events.randomize_joint_friction.params.kp_scale_range=[1.0,1.6]"
kp_fixed() { echo "env.events.randomize_joint_friction.params.kp_scale_range=[$1,$1]"; }   # 1.0 -> P 5, 1.6 -> P 8
mu_fixed() { echo "env.events.foot_friction.params.static_friction_range=[$1,$1] env.events.foot_friction.params.dynamic_friction_range=[$1,$1]"; }
ITERS=${FRICTION_ITERS:-2000}
SYM="--agent rsl_rl_with_symmetry_cfg_entry_point"
POLICIES=${POLICIES:-"velstand run_v3"}
declare -A TASK=( [velstand]=Microduck-VelStand-Flat-HD1910-AllColl-LowMu [run_v3]=Microduck-RunV3-Flat-HD1910-AllColl-LowMu )
declare -A DIR=( [velstand]=velstand_21495 [run_v3]=run_v3_5999 )
declare -A CK=( [velstand]=21495 [run_v3]=5999 )
gpu_gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "${MIN_GPU_FREE_MB:-7500}" ]; do sleep 60; done; }
gate() { while [ "$(free -m | awk 'NR==2{print $7}')" -lt 3300 ] || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 1500 ]; do sleep 30; done; }
last_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1; }
newest_since() { for d in $(ls -td logs/rsl_rl/*/*/ 2>/dev/null); do [ "$(stat -c %Y "$d")" -ge "$1" ] || continue; ls "$d"/model_*.pt >/dev/null 2>&1 && { readlink -f "$d"; return; }; done; }
mkdir -p docs/friction
# final-checkpoint evaluation under the LowMu task at FIXED sole friction and FIXED P (96 envs)
run_eval() { # name run ck tag
  local name=$1 run=$2 ck=$3 tag=$4 task=${TASK[$1]}
  for mu in 0.3 0.6 1.0; do for P in 5 8; do
    local scale=$([ $P = 5 ] && echo 1.0 || echo 1.6); local out=docs/friction/${name}_${tag}_mu${mu/./}_p${P}
    case $name in
      velstand)
        gate; uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --task $task --run "$run" --checkpoint "model_$ck.pt" --vx 0.3 --num_envs 96 presets=backlash $(kp_fixed $scale) $(mu_fixed $mu) --out "${out}_gait03.json" > "logs/friction_${name}_${tag}_mu${mu/./}_p${P}_gait.log" 2>&1
        gate; uv run --project external/IsaacLab python scripts/eval_standup.py --task $task --run "$run" --checkpoint "model_$ck.pt" --zero_twist --num_envs 96 presets=backlash $(kp_fixed $scale) $(mu_fixed $mu) --out "${out}_standup.json" > "logs/friction_${name}_${tag}_mu${mu/./}_p${P}_standup.log" 2>&1 ;;
      run_v3)
        for vx in 0.5 0.9; do gate; uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --task $task --run "$run" --checkpoint "model_$ck.pt" --vx $vx --num_envs 96 presets=backlash $(kp_fixed $scale) $(mu_fixed $mu) --out "${out}_gait${vx/./}.json" > "logs/friction_${name}_${tag}_mu${mu/./}_p${P}_gait${vx/./}.log" 2>&1; done ;;
    esac
    echo "[$(date +%F_%T)] friction eval $name $tag mu$mu P$P done" >> logs/evals.log
  done; done
}
echo "[$(date +%F_%T)] friction chain start: $POLICIES ($ITERS its each)" | tee -a logs/train_all.log
for name in $POLICIES; do ls docs/friction/${name}_base_mu10_p8_*.json >/dev/null 2>&1 || run_eval "$name" "$(readlink -f checkpoints/${DIR[$name]})" "${CK[$name]}" base; done
for name in $POLICIES; do
  task=${TASK[$name]}; ck=${CK[$name]}; start="$(readlink -f checkpoints/${DIR[$name]})/model_$ck.pt"; extra=""
  [ "$name" = velstand ] && extra="agent.algorithm.bc_cfg.expert_checkpoint=$(readlink -f checkpoints/standup_9999)/model_9999.pt agent.algorithm.bc_cfg.anchor_checkpoint=$start"
  attempt=0; run=""
  while [ -z "$run" ] && [ "$attempt" -lt 4 ]; do
    attempt=$((attempt + 1)); since=$(date +%s); gpu_gate
    echo "[$(date +%F_%T)] start friction-$name attempt $attempt ($ITERS its from $start; $KP_TRAIN $extra)" | tee -a logs/train_all.log
    uv run --project external/IsaacLab python scripts/train.py --task $task $SYM --max_iterations $ITERS $COMMON --checkpoint "$start" "$KP_TRAIN" $extra >> "logs/train_friction_$name.log" 2>&1
    echo "[$(date +%F_%T)] end friction-$name attempt $attempt exit=$?" | tee -a logs/train_all.log
    run=$(newest_since "$since"); [ -n "$run" ] && [ "$(last_ckpt "$run")" -ge $((ck + ITERS - 1)) ] || { run=""; sleep 120; }
  done
  [ -n "$run" ] || { echo "[$(date +%F_%T)] friction-$name gave up" | tee -a logs/train_all.log; continue; }
  echo "$name $run $(last_ckpt "$run")" >> docs/friction/RUNS.txt
  run_eval "$name" "$run" "$(last_ckpt "$run")" ft
done
echo "[$(date +%F_%T)] friction chain done" | tee -a logs/train_all.log
