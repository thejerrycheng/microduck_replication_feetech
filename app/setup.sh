#!/bin/bash
# One-time setup of the Mac app's Python environment (~/.venvs/microduck_live):
# MuJoCo (C, CPU), onnxruntime (the 61->14 actors), Rhoban BAM (the HD1910 servo model the policies trained on),
# websockets (page <-> engine), pywebview/pyobjc (fallback window), matplotlib (bench figures).
set -euo pipefail
V="${DUCK_VENV:-$HOME/.venvs/microduck_live}"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$HOME/anaconda3/bin:$PATH"
command -v uv >/dev/null || { echo "needs uv (brew install uv)"; exit 1; }
[ -x "$V/bin/python" ] || uv venv -p 3.12 "$V"
VIRTUAL_ENV="$V" uv pip install -q "mujoco>=3.3" onnxruntime onnx numpy websockets "better-actuator-models[mujoco]" \
  pywebview pyobjc-framework-Cocoa matplotlib        # onnx: repack exports whose weights sit in a .data sidecar (sync_policies.sh)
"$V/bin/python" -c "import mujoco, onnxruntime, bam, websockets; print('ready:', '$V', 'mujoco', mujoco.__version__)"
echo "now: app/ducklive.sh"
