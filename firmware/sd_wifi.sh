#!/bin/bash
# Put Wi-Fi credentials on a Radxa Zero 3W microSD so the duck's computer joins the network on its next boot.
#
#     sudo bash firmware/sd_wifi.sh            # card in the Mac's reader
#     sudo bash firmware/sd_wifi.sh disk7      # a specific disk
#
# Radxa OS runs `before.txt` from the card's small `config` partition once at boot, then deletes it (rsetup).
# Its parser splits on spaces and knows no quotes, so a network name or password with a space cannot be given this way.
set -euo pipefail
disk=${1:-$(system_profiler SPCardReaderDataType 2>/dev/null | awk '/BSD Name: disk[0-9]+$/ {print $3; exit}')}
[ -n "$disk" ] || { echo "no SD card found; pass the disk, e.g. disk4"; exit 1; }
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
dir=$(mktemp -d)
trap 'umount "$dir" 2>/dev/null; rmdir "$dir" 2>/dev/null' EXIT
mount -t msdos "/dev/${disk}s1" "$dir"
grep -q rsetup "$dir/config.txt" || { echo "/dev/${disk}s1 is not a Radxa config partition"; exit 1; }

read -r -p "Wi-Fi name: " ssid
read -r -s -p "Wi-Fi password: " pass; echo
case "$ssid$pass" in *" "*) echo "spaces are not supported here"; exit 1;; esac
[ -n "$ssid" ] || { echo "no name given"; exit 1; }

printf 'no_fail\nconnect_wi-fi %s %s\nenable_service ssh\n' "$ssid" "$pass" > "$dir/before.txt"
umount "$dir"
diskutil eject "/dev/$disk" >/dev/null
echo "Done. Card ejected: put it in the board and power on."
