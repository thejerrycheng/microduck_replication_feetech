#!/usr/bin/env bash
# Claude Code PostToolUse hook: when a *.tex file is edited/written, rebuild its
# PDF so the PDF always tracks the source. Reads the hook JSON event on stdin.
# Registered in <project>/.claude/settings.json (matcher Edit|Write|MultiEdit).
set -euo pipefail

event="$(cat)"
path="$(printf '%s' "$event" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null || true)"

case "$path" in
  *.tex)
    dir="$(dirname "$path")"; file="$(basename "$path")"
    if command -v latexmk >/dev/null 2>&1; then
      if ( cd "$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error "$file" >/tmp/latexmk_hook.log 2>&1 ); then
        echo "[hook] rebuilt ${file%.tex}.pdf"
      else
        echo "[hook] latexmk failed for $file (see /tmp/latexmk_hook.log)" >&2
      fi
    else
      echo "[hook] latexmk not found; skipping PDF rebuild" >&2
    fi
    ;;
esac
