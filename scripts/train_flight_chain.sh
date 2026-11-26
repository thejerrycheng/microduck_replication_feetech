#!/usr/bin/env bash
# Flying-duck training chain for the GPU desktop (docs/duck_flight.tex, "Training chain").
#   -  plant check (scripts/check_flight_plant.py): pack mass on the trunk, thrust against weight
#   0  smoke test of the three new tasks (64 envs, 3 iterations): catches config / API errors in minutes
#   1  walk with the flight hardware   (fine-tune of VelStand 21495, mirror loss)
#   2  run with the flight hardware    (fine-tune of run v3 5999, mirror loss)
#   3  run -> take-off                 (from scratch: 18 actions)
#   4  play each final checkpoint: exports policy.onnx and records a clip
# Stage 0 and the plant check pass on the 5080 (2026-11-26). Each training stage waits for free GPU memory.
# Launch: systemd-run --user --unit microduck-flight-chain --working-directory "$PWD" scripts/train_flight_chain.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
# The package is installed editable from another checkout on the GPU machine: PYTHONPATH makes THIS tree win.
export PYTHONPATH="$PWD"
UV="external/IsaacLab/.venv/bin/python"
COMMON="--rl_library rsl_rl presets=backlash"
SYM="--agent rsl_rl_with_symmetry_cfg_entry_point"
N=${NUM_ENVS:-4096}
VIDEO="--video --video_length 400 --video_interval 6000"
log() { echo "[$(date +%F_%T)] $*" | tee -a logs/flight_chain.log; }
newest() { ls -td logs/rsl_rl/"$1"*/*/ 2>/dev/null | head -1; }
last_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_//; s/.pt//' | sort -n | tail -1; }
# other projects train on this GPU: wait for room instead of pushing them out of memory (same gate as finetune_gains.sh)
gate() { local said=""; while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "${MIN_GPU_FREE_MB:-7500}" ] || [ "$(free -m | awk 'NR==2{print $7}')" -lt "${MIN_RAM_FREE_MB:-6000}" ]; do
    [ -z "$said" ] && { log "waiting for GPU >= ${MIN_GPU_FREE_MB:-7500} MB and RAM >= ${MIN_RAM_FREE_MB:-6000} MB free"; said=1; }; sleep 60; done; }

if [ -z "${SKIP_CHECK:-}" ]; then
  log "plant check"
  $UV scripts/check_flight_plant.py presets=backlash --out docs/flight_isaac_check.json > logs/flight_check.log 2>&1 || { log "PLANT CHECK FAILED (logs/flight_check.log)"; exit 1; }
  grep "\[check\]" logs/flight_check.log | tee -a logs/flight_chain.log
fi
if [ -z "${SKIP_SMOKE:-}" ]; then
  for t in Microduck-WalkPack-Flat-HD1910-AllColl Microduck-RunPack-Flat-HD1910-AllColl Microduck-Takeoff-Flat-HD1910-AllColl; do
    log "smoke $t"
    $UV scripts/train.py $COMMON --task $t --num_envs 64 --max_iterations 3 > "logs/flight_smoke_$t.log" 2>&1 || { log "SMOKE FAILED $t (logs/flight_smoke_$t.log)"; exit 1; }
  done
fi
if [ -z "${SKIP_WALK:-}" ]; then
  gate
  log "walk + pack: fine-tune from VelStand 21495, ${WALK_ITERS:-3000} iterations"
  $UV scripts/train.py $COMMON $SYM --task Microduck-WalkPack-Flat-HD1910-AllColl --num_envs $N $VIDEO \
      --checkpoint "$PWD/checkpoints/velstand_21495/model_21495.pt" --max_iterations ${WALK_ITERS:-3000} > logs/flight_walkpack.log 2>&1 || log "walk+pack FAILED"
fi
if [ -z "${SKIP_RUN:-}" ]; then
  gate
  log "run + pack: fine-tune from run v3 5999, ${RUN_ITERS:-3000} iterations"
  $UV scripts/train.py $COMMON $SYM --task Microduck-RunPack-Flat-HD1910-AllColl --num_envs $N $VIDEO \
      --checkpoint "$PWD/checkpoints/run_v3_5999/model_5999.pt" --max_iterations ${RUN_ITERS:-3000} > logs/flight_runpack.log 2>&1 || log "run+pack FAILED"
fi
if [ -z "${SKIP_TAKEOFF:-}" ]; then
  gate
  log "take-off: from scratch, ${TAKEOFF_ITERS:-8000} iterations"
  $UV scripts/train.py $COMMON --task Microduck-Takeoff-Flat-HD1910-AllColl --num_envs $N $VIDEO \
      --max_iterations ${TAKEOFF_ITERS:-8000} > logs/flight_takeoff.log 2>&1 || log "take-off FAILED"
fi
for pair in "Microduck-WalkPack-Flat-HD1910-AllColl microduck_walk_pack_allcoll $SYM" "Microduck-RunPack-Flat-HD1910-AllColl microduck_run_pack_allcoll $SYM" "Microduck-Takeoff-Flat-HD1910-AllColl microduck_takeoff_allcoll"; do
  set -- $pair; task=$1; exp=$2; shift 2
  run=$(newest "$exp"); [ -n "$run" ] || { log "no run for $exp"; continue; }
  ck=$(last_ckpt "$run")
  log "play $task $run model_$ck.pt"
  $UV scripts/play.py $COMMON "$@" --task $task --checkpoint "$(readlink -f "$run")/model_$ck.pt" --num_envs 4 --video --video_length 600 > "logs/flight_play_$exp.log" 2>&1 || log "play FAILED $task"
done
log "chain done"
