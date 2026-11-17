#!/bin/bash
# Write a system image to the microSD card in the Mac's built-in reader. Erases the card.
#
#     sudo bash firmware/sd_flash.sh openmicroduck_image/2026_11_12_openmicroduck.img
#
# It only ever writes to the card in the built-in SD reader: the disk is looked up there, and refused unless macOS
# calls it removable Secure Digital media. Takes a few minutes; Ctrl+T shows progress.
set -euo pipefail
img=${1:?give the .img file}
[ -f "$img" ] || { echo "no such image: $img"; exit 1; }
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
disk=$(system_profiler SPCardReaderDataType 2>/dev/null | awk '/BSD Name: disk[0-9]+$/ {print $3; exit}')
[ -n "$disk" ] || { echo "no SD card in the built-in reader"; exit 1; }
info=$(diskutil info "$disk")
echo "$info" | grep -q "Protocol: *Secure Digital" && echo "$info" | grep -q "Removable Media: *Removable" \
  || { echo "/dev/$disk is not a removable SD card: not touching it"; exit 1; }
size=$(echo "$info" | awk -F'[()]' '/Disk Size/ {print $2}' | awk '{print $1}')
[ "$(stat -f%z "$img")" -le "$size" ] || { echo "the image is larger than the card"; exit 1; }

echo "About to ERASE /dev/$disk ($(echo "$info" | awk -F: '/Disk Size/ {print $2}' | sed 's/ (.*//;s/^ *//') SD card) and write:"
echo "  $img"
read -r -p "Press Return to go on, Ctrl+C to stop. " _
diskutil unmountDisk "/dev/$disk" >/dev/null
dd if="$img" of="/dev/r$disk" bs=4m
sync
diskutil eject "/dev/$disk" >/dev/null
echo "Done. Card ejected: put it in the board and power on."
