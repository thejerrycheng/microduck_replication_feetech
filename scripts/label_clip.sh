#!/usr/bin/env bash
# Burn a caption into a rollout clip: scripts/label_clip.sh <in.mp4> <out.mp4> "<line 1>" ["<line 2>"]
set -eu
IN=$1; OUT=$2; L1=$3; L2=${4:-}
FF=$(uv run --project "$(dirname "$0")/../external/IsaacLab" python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
esc() { printf '%s' "$1" | sed "s/'/\\\\'/g; s/:/\\\\:/g"; }
VF="drawtext=fontfile=$FONT:text='$(esc "$L1")':fontcolor=white:fontsize=44:box=1:boxcolor=black@0.55:boxborderw=14:x=30:y=30"
[ -n "$L2" ] && VF="$VF,drawtext=fontfile=$FONT:text='$(esc "$L2")':fontcolor=white:fontsize=34:box=1:boxcolor=black@0.55:boxborderw=12:x=30:y=110"
"$FF" -loglevel error -y -i "$IN" -vf "$VF" -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -an "$OUT"
