#!/usr/bin/env bash
# Record a velocity policy under fixed commands (4 robots, 10 s each), labelled, into rollouts/<run>/speeds/.
# Usage: scripts/record_speeds.sh <run dir> <model_N.pt> [TASK] [preset] [commands...]
#   commands: label:vx:vy:wz  (defaults: a walking set for Velocity tasks, a sprint sweep for the Run task)
set -u
cd "$(dirname "$0")/.."
RUN=$(readlink -f "$1"); CK=$2; TASK=${3:-Microduck-Velocity-Flat-HD1910}; PRESET=${4:-presets=backlash}; shift 4 2>/dev/null || shift $#
OUT=rollouts/$(basename "$(dirname "$RUN")")/$(basename "$RUN")/speeds; mkdir -p "$OUT/labeled"
case "$TASK" in
  *Run*) DEFAULTS="forward_0.30ms:0.30:0:0 forward_0.50ms:0.50:0:0 forward_0.70ms:0.70:0:0 forward_0.90ms:0.90:0:0 forward_1.10ms:1.10:0:0"; KIND="RUN";;
  *)     DEFAULTS="forward_0.15ms:0.15:0:0 forward_0.30ms:0.30:0:0 forward_0.40ms:0.40:0:0 forward_0.60ms:0.60:0:0 backward_0.20ms:-0.20:0:0 sideways_0.20ms:0:0.20:0 turn_1.0rads:0:0:1.0"; KIND="WALK";;
esac
CMDS=${*:-$DEFAULTS}
for spec in $CMDS; do
  IFS=: read -r label vx vy wz <<< "$spec"
  before=$(date +%s)
  uv run --project external/IsaacLab python scripts/play.py --rl_library rsl_rl --task "$TASK" \
    --checkpoint "$RUN/$CK" --num_envs 4 $PRESET --video --video_length 500 --video_interval 1000000 \
    env.commands.base_velocity.ranges.lin_vel_x="[$vx,$vx]" env.commands.base_velocity.ranges.lin_vel_y="[$vy,$vy]" \
    env.commands.base_velocity.ranges.ang_vel_z="[$wz,$wz]" env.commands.base_velocity.rel_standing_envs=0.0 \
    env.commands.base_velocity.rel_forward_envs=0.0 env.commands.base_velocity.rel_turn_in_place_envs=0.0 \
    > "logs/record_speed_$label.log" 2>&1
  clip=$(find "$RUN/videos/play" -name "*.mp4" -newermt "@$before" -size +10k | head -1)
  if [ -n "$clip" ]; then
    mv "$clip" "$OUT/${CK%.pt}_$label.mp4"
    scripts/label_clip.sh "$OUT/${CK%.pt}_$label.mp4" "$OUT/labeled/${KIND,,}_${CK%.pt}_$label.mp4" \
      "$KIND policy  checkpoint ${CK#model_}  ($TASK)" "commanded: vx=$vx m/s  vy=$vy m/s  yaw=$wz rad/s   |  4 robots, 10 s, deterministic" \
      && echo "[$(date +%T)] $label -> $OUT/labeled/${KIND,,}_${CK%.pt}_$label.mp4"
  else
    echo "[$(date +%T)] $label FAILED (see logs/record_speed_$label.log)"
  fi
done
