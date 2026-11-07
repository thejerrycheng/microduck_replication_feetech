#!/usr/bin/env bash
# Record a walking checkpoint under fixed velocity commands (4 robots, 10 s each) into rollouts/<run>/speeds/.
# Usage: scripts/record_speeds.sh <run dir> <model_N.pt> [presets=backlash]
set -u
cd "$(dirname "$0")/.."
RUN=$(readlink -f "$1"); CK=$2; PRESET=${3:-presets=backlash}
OUT=rollouts/$(basename "$(dirname "$RUN")")/$(basename "$RUN")/speeds; mkdir -p "$OUT"
record() {  # label vx vy wz
  local label=$1 vx=$2 vy=$3 wz=$4
  local before=$(date +%s)
  uv run --project external/IsaacLab python scripts/play.py --rl_library rsl_rl --task Microduck-Velocity-Flat-HD1910 \
    --checkpoint "$RUN/$CK" --num_envs 4 $PRESET --video --video_length 500 --video_interval 1000000 \
    env.commands.base_velocity.ranges.lin_vel_x="[$vx,$vx]" env.commands.base_velocity.ranges.lin_vel_y="[$vy,$vy]" \
    env.commands.base_velocity.ranges.ang_vel_z="[$wz,$wz]" env.commands.base_velocity.rel_standing_envs=0.0 \
    env.commands.base_velocity.rel_forward_envs=0.0 env.commands.base_velocity.rel_turn_in_place_envs=0.0 \
    > "logs/record_speed_$label.log" 2>&1
  local clip=$(find "$RUN/videos/play" -name "*.mp4" -newermt "@$before" | head -1)
  [ -n "$clip" ] && mv "$clip" "$OUT/${CK%.pt}_$label.mp4" && echo "[$(date +%T)] $label -> $OUT/${CK%.pt}_$label.mp4" || echo "[$(date +%T)] $label FAILED (see logs/record_speed_$label.log)"
}
record forward_0.15ms 0.15 0 0
record forward_0.30ms 0.30 0 0
record forward_0.40ms 0.40 0 0
record backward_0.20ms -0.20 0 0
record sideways_0.20ms 0 0.20 0
record turn_1.0rads 0 0 1.0
