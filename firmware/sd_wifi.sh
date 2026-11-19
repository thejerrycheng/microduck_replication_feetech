#!/bin/bash
# Set up a Radxa Zero 3W microSD from the Mac, for the duck's next boot: Wi-Fi credentials, and USB networking.
#
#     sudo bash firmware/sd_wifi.sh            # ask for a Wi-Fi name and password; also turn USB networking on
#     sudo bash firmware/sd_wifi.sh usb        # USB networking only, no questions
#     sudo bash firmware/sd_wifi.sh usb disk7  # ... on a specific disk
#     sudo bash firmware/sd_wifi.sh check      # show what is on the config partition; change nothing
#     sudo bash firmware/sd_wifi.sh clear      # remove a pending before.txt, so the next boot runs none
#
# Radxa OS runs `before.txt` from the card's small `config` partition once at boot, then deletes it (rsetup).
# Its parser splits on spaces and knows no quotes, so a network name or password with a space cannot be given this way.
# USB networking = Radxa's `radxa-ncm` gadget on the OTG port: the duck then shows up on the Mac as a network
# adapter over the same cable that powers it, with no Wi-Fi involved.
set -euo pipefail
mode=wifi; case "${1:-}" in usb|check|clear) mode=$1; shift;; esac
disk=${1:-$(system_profiler SPCardReaderDataType 2>/dev/null | awk '/BSD Name: disk[0-9]+$/ {print $3; exit}')}
[ -n "$disk" ] || { echo "no SD card found; pass the disk, e.g. disk4"; exit 1; }
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
dir=$(mktemp -d)
trap 'umount "$dir" 2>/dev/null; rmdir "$dir" 2>/dev/null' EXIT
mount -t msdos "/dev/${disk}s1" "$dir"
grep -q rsetup "$dir/config.txt" || { echo "/dev/${disk}s1 is not a Radxa config partition"; exit 1; }

if [ "$mode" = check ] || [ "$mode" = clear ]; then
  ls -la "$dir"
  if [ -f "$dir/before.txt" ]; then
    echo "--- before.txt is still there: the duck has not run it (passwords hidden)"
    sed -E 's/^(connect_wi-fi +[^ ]+) .*/\1 <hidden>/' "$dir/before.txt"
    [ "$mode" = clear ] && rm "$dir/before.txt" && echo "--- removed"
  else
    echo "--- no before.txt: the duck ran it and deleted it, or there never was one"
  fi
  umount "$dir"; diskutil eject "/dev/$disk" >/dev/null; echo "Card ejected."; exit 0
fi

usb="enable_service radxa-ncm@fcc00000.dwc3"
if [ "$mode" = usb ]; then
  printf 'no_fail\n%s\n' "$usb" > "$dir/before.txt"
else
  read -r -p "Wi-Fi name: " ssid
  read -r -s -p "Wi-Fi password: " pass; echo
  case "$ssid$pass" in *" "*) echo "spaces are not supported here"; exit 1;; esac
  [ -n "$ssid" ] || { echo "no name given"; exit 1; }
  printf 'no_fail\nconnect_wi-fi %s %s\nenable_service ssh\n%s\n' "$ssid" "$pass" "$usb" > "$dir/before.txt"
fi
umount "$dir"
diskutil eject "/dev/$disk" >/dev/null
echo "Done. Card ejected: put it in the board and power on."
