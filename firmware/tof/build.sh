#!/bin/sh
# Build the two ToF readers on the duck: sh build.sh <output dir>. Needs gcc; takes about half a minute.
set -e
cd "$(dirname "$0")"
out=${1:-$HOME/microduck}
mkdir -p "$out"
gcc -O2 -w -DGEN=5 -DTOF_PLATFORM=VL53L5CX_Platform -Ivl53l5cx main.c platform.c probe.c vl53l5cx/shim.c vl53l5cx/vl53l5cx_api.c -o "$out/tof_vl53l5cx"
gcc -O2 -w -DGEN=8 -DTOF_PLATFORM=VL53L8CX_Platform -Ivl53l8cx main.c platform.c probe.c vl53l8cx/shim.c vl53l8cx/vl53l8cx_api.c -o "$out/tof_vl53l8cx"
echo "built $out/tof_vl53l5cx $out/tof_vl53l8cx"
