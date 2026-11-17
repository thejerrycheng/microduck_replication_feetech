#!/bin/bash
# Duck Live: the HD1910 Microduck, driven by keyboard, as a Mac app; the simulation runs on this Mac (CPU).
# The same launcher conventions as WUJI Live (~/Desktop/WUJI_Tools/rl/tools/mac_app) and M2 Live.
#
#   app/ducklive.sh                  open the app (installs it first if needed)
#   app/ducklive.sh --browser        in the default browser instead of an app window (keys via the page; see README)
#   app/ducklive.sh --serve          no window: run the server and print the page URL (tests, another browser)
#   app/ducklive.sh --install-app    (re)build "Duck Live.app" into /Applications (double-click / Spotlight)
#   app/ducklive.sh --sync           pull newer ONNX exports from the 5080 (ssh 5080) and GitHub, then open
#
# First run: setup.sh creates ~/.venvs/microduck_live (MuJoCo, onnxruntime, BAM, pywebview). Logs: logs/app/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${DUCK_PYTHON:-$HOME/.venvs/microduck_live/bin/python}"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$HOME/anaconda3/bin:$PATH"
[[ -x $PY ]] && "$PY" -c "import mujoco, onnxruntime, bam, websockets" 2>/dev/null || "$HERE/setup.sh"
case "${1:-}" in
  --install-app) exec "$HERE/build_app.sh" ;;
  --sync) "$HERE/sync_policies.sh"; shift; exec "$0" "$@" ;;
  --browser) cd "$REPO" && "$PY" "$HERE/server.py" "${@:2}" | while read -r a b c; do
      if [[ $a == DUCK_LIVE_READY ]]; then echo "http://127.0.0.1:$b/?token=$c"; open "http://127.0.0.1:$b/?token=$c"; fi; done ;;
  --serve) cd "$REPO" && exec "$PY" "$HERE/server.py" "${@:2}" ;;
  *) [[ -d "/Applications/Duck Live.app" ]] || "$HERE/build_app.sh"; open "/Applications/Duck Live.app" ;;
esac
