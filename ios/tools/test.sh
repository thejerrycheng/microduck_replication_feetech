#!/bin/bash
# The whole simulator test: starts the mock studio with its stand-in duck (mock_studio.py; port 8780, so a studio
# already open on 8770 is left alone) and the Duck Live engine, both announced on the network, and a stand-in for the
# duck's factory runtime (mock_runtime.py); runs every UI test in light and dark appearance; then stops the servers, checks
# the offline screen, starts them again and checks the app finds them. Screenshots -> docs/figures/ios/, log and
# METRIC lines -> logs/ios/.     ios/tools/test.sh [device name]
set -u
cd "$(dirname "$0")/.."; ROOT=$(cd .. && pwd); PY=~/.venvs/microduck_live/bin/python
DEV="${1:-iPhone 17 Pro}"; SHOTS=$ROOT/docs/figures/ios; LOG=$ROOT/logs/ios; mkdir -p "$SHOTS" "$LOG"
up() {
  (cd "$ROOT" && PYTHONUNBUFFERED=1 nohup $PY ios/tools/mock_studio.py --port 8780 --log "$LOG/pilot_mock.jsonl" > "$LOG/studio.log" 2>&1 & echo $! > "$LOG/studio.pid")
  (cd "$ROOT" && nohup $PY app/server.py --lan > "$LOG/live.log" 2>&1 & echo $! > "$LOG/live.pid")
  (cd "$ROOT" && PYTHONUNBUFFERED=1 nohup $PY ios/tools/mock_runtime.py ${RTLOG---log "$LOG/runtime_mock.jsonl"} > "$LOG/runtime.log" 2>&1 & echo $! > "$LOG/runtime.pid")
}
down() { for f in studio live runtime; do [ -f "$LOG/$f.pid" ] && kill "$(cat "$LOG/$f.pid")" 2>/dev/null; rm -f "$LOG/$f.pid"; done; pkill -f "mock_studio.py --port 8780"; pkill -f "server.py --lan"; pkill -f "mock_runtime.py"; }
run() { TEST_RUNNER_SHOTS=$SHOTS TEST_RUNNER_PILOTLOG=$LOG/pilot_mock.jsonl "$@" xcodebuild -project Duck.xcodeproj -scheme Duck -destination "platform=iOS Simulator,name=$DEV" -derivedDataPath build "${ARGS[@]}" 2>&1; }
{ [ -d Duck.xcodeproj ] || xcodegen generate >/dev/null; } && xcrun simctl boot "$DEV" 2>/dev/null
down; rm -f "$LOG/pilot_mock.jsonl"; up; sleep 9
: > "$LOG/test.log"
xcrun simctl ui "$DEV" appearance light
ARGS=(test); run env >> "$LOG/test.log"
xcrun simctl ui "$DEV" appearance dark
ARGS=(test-without-building -only-testing:DuckUITests/DuckUITests/testCalibrate -only-testing:DuckUITests/DuckUITests/testLinks -only-testing:DuckUITests/DuckUITests/testServo -only-testing:DuckUITests/DuckUITests/testSensors -only-testing:DuckUITests/DuckUITests/testPlay -only-testing:DuckUITests/DuckUITests/testRunPlay -only-testing:DuckUITests/DuckUITests/testPilot)
run env TEST_RUNNER_DARK=1 >> "$LOG/test.log"
xcrun simctl ui "$DEV" appearance light
down; sleep 3
(sleep 50; RTLOG= up) &          # this second start keeps the first one's log for plot_runtime.py
ARGS=(test-without-building -only-testing:DuckUITests/DuckUITests/testNoDuck -only-testing:DuckUITests/DuckUITests/testOffline); run env TEST_RUNNER_OFFLINE=1 >> "$LOG/test.log"
down
grep -E "METRIC|error:|Test Case.*(passed|failed)|Executed .* tests" "$LOG/test.log"
! grep -qE "Test Case.*failed|error:" "$LOG/test.log"
