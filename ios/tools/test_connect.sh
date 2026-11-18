#!/bin/bash
# Connection test against the REAL duck, from the simulator, read-only (no servo command is sent; the app only opens
# the duck's servo bus and reads). Cold-launches the app N times for each way of finding the duck (all together,
# Bonjour alone, its name alone, the network sweep alone) and times launch -> found -> connected with no tap; then
# stops the duck's server for 8 s under a running app, D times, and times lost -> back.
#     ios/tools/test_connect.sh [N=5] [D=3]      -> logs/ios/connect.log; then tools/plot_connect.py
set -u
cd "$(dirname "$0")/.."; ROOT=$(cd .. && pwd); DEV="iPhone 17 Pro"; LOG=$ROOT/logs/ios/connect.log; N=${1:-5}; D=${2:-3}
mkdir -p "$ROOT/logs/ios"; [ -n "${APPEND:-}" ] || : > "$LOG"
curl -s -m 5 http://radxa-zero3.local:8770/api/state | grep -q '"mock": false' || { echo "the duck's studio does not answer: firmware/duck_serve.sh"; exit 1; }
xcrun simctl boot "$DEV" 2>/dev/null
run() { TEST_RUNNER_REAL=1 TEST_RUNNER_SHOTS=$ROOT/docs/figures/ios "$@" xcodebuild -project Duck.xcodeproj -scheme Duck -destination "platform=iOS Simulator,name=$DEV" \
  -derivedDataPath build $BUILD -only-testing:DuckUITests/DuckUITests/testRealDuck 2>&1 | grep --line-buffered -E "METRIC|READY|SENSOR|NOTE|error:|Test Case.*(passed|failed)|timed out"; }
BUILD=${BUILD:-test}
for find in all bonjour name sweep; do for ((i = 1; i <= N; i++)); do
  xcrun simctl spawn "$DEV" defaults delete com.microduck.Duck duckHost >/dev/null 2>&1      # no remembered address: a first meeting every time
  echo "TRIAL find=$find i=$i" >> "$LOG"
  if [ $find = all ]; then [ "$i" = 1 ] && run env TEST_RUNNER_SHOT=1 >> "$LOG" || run env >> "$LOG"; else run env TEST_RUNNER_FIND=$find >> "$LOG"; fi
  BUILD=test-without-building
done; done
for ((i = 1; i <= D; i++)); do
  echo "TRIAL find=drop i=$i" >> "$LOG"
  (while ! grep -q READY_FOR_DROP "$LOG.drop" 2>/dev/null; do sleep 0.5; done
   ssh -o BatchMode=yes microduck 'systemctl --user stop duck-studio; sleep 8; systemctl --user start duck-studio') &
  : > "$LOG.drop"; run env TEST_RUNNER_DROP=1 | tee "$LOG.drop" >> "$LOG"; wait
done
rm -f "$LOG.drop"
grep -cE "Test Case.*passed" "$LOG"; grep -E "Test Case.*failed|error:|timed out" "$LOG"
! grep -qE "Test Case.*failed|error:" "$LOG"
