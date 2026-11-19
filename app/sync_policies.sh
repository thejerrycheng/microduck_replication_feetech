#!/bin/bash
# Pull the newest deployable policies: `git pull` (the exports and checkpoints tracked on GitHub), repackage any
# export whose weights sit in a `policy.onnx.data` sidecar (Isaac Lab's exporter writes one; the exports/ copy alone
# cannot load), then any newer ONNX export the 5080 has under its checkout (exports/*.onnx and every
# exported_model_*/policy.onnx(+.data) of the training runs, renamed <experiment>_iter<N>.onnx).
# ssh alias: 5080 (DUCK_5080), remote checkout: DUCK_5080_REPO. From a network that blocks port 22 (see ~/.claude
# memory "ssh-blocked-from-china"): DUCK_SSH='ssh -o ProxyCommand=...'. --local: git pull + repackage only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${DUCK_VENV:-$HOME/.venvs/microduck_live}/bin/python"
H="${DUCK_5080:-5080}"; RR="${DUCK_5080_REPO:-~/microduck_replication_feetech}"
SSH="${DUCK_SSH:-ssh}"
cd "$REPO"
git pull --ff-only || echo "[sync] git pull failed (offline?) -- continuing"
mkdir -p exports

repack() {   # repack <src policy.onnx (with sidecar next to it)> <dst export>: one self-contained file
  "$PY" - "$1" "$2" <<'EOF'
import sys, onnx
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=True)
for t in m.graph.initializer:
    if t.HasField("data_location"):
        t.ClearField("data_location"); del t.external_data[:]
onnx.save_model(m, dst, save_as_external_data=False)
print("[sync] repacked", dst)
EOF
}

# exports/ that reference a sidecar: rebuild them from checkpoints/<run>/policy.onnx(+.data)
for f in exports/*.onnx; do
  if "$PY" -c "import onnxruntime as o,sys; o.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])" "$f" 2>/dev/null; then continue; fi
  it="$(sed -E 's/.*iter([0-9]+)\.onnx/\1/' <<<"$f")"
  src="$(ls checkpoints/*_"$it"/policy.onnx 2>/dev/null | head -1 || true)"
  if [[ -n $src && -f ${src}.data ]]; then repack "$src" "$f"; else echo "[sync] $f cannot load and no checkpoint sidecar found"; fi
done

if [[ ${1:-} == --local ]]; then ls -la exports/*.onnx | awk '{print $5, $9}'; exit 0; fi

$SSH "$H" "cd $RR 2>/dev/null && ls exports/*.onnx 2>/dev/null; cd $RR/external/IsaacLab/logs/rsl_rl 2>/dev/null && for f in */*/exported_model_*/policy.onnx; do echo \"RUN \$f\"; done" | while read -r kind path; do
  if [[ $kind == RUN ]]; then
    exp="${path%%/*}"; it="${path##*exported_model_}"; it="${it%%/*}"
    name="${exp#microduck_}"; name="${name%_hd1910_backlash}"; name="${name%_backlash}"; name="${name/_flat/}"
    dst="exports/5080_${name}_iter${it}.onnx"
    if [[ ! -f $dst ]]; then
      echo "[sync] $dst"; tmp="$(mktemp -d)"
      $SSH "$H" "cat $RR/external/IsaacLab/logs/rsl_rl/$path" > "$tmp/policy.onnx"
      $SSH "$H" "cat $RR/external/IsaacLab/logs/rsl_rl/${path}.data 2>/dev/null" > "$tmp/policy.onnx.data" || true
      [[ -s $tmp/policy.onnx.data ]] && repack "$tmp/policy.onnx" "$dst" || mv "$tmp/policy.onnx" "$dst"
      rm -rf "$tmp"
    fi
  else
    f="$kind"; dst="exports/$(basename "$f")"
    [[ -f $dst ]] || { echo "[sync] $dst"; $SSH "$H" "cat $RR/$f" > "$dst.part" && mv "$dst.part" "$dst"; }
  fi
done
ls -la exports/*.onnx | awk '{print $5, $9}'
