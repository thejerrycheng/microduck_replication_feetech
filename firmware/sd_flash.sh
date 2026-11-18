#!/bin/bash
# Write a system image to the microSD card in the Mac's built-in reader. Erases the card.
#
#     sudo bash firmware/sd_flash.sh openmicroduck_image/2026_11_12_openmicroduck.img
#
# It only ever writes to the card in the built-in SD reader: the disk is looked up there, and refused unless macOS
# calls it removable Secure Digital media. Takes a few minutes; Ctrl+T shows progress. Afterwards the card is read
# back and compared with the image, so a card that stores wrong data is caught here and not on the duck.
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
bytes=$(stat -f%z "$img")
if [ $((bytes % 1048576)) = 0 ]; then    # read the card back and compare it with the image, byte for byte
  echo "Verifying: reading the card back (a few minutes)..."
  want=$(shasum -a 256 "$img" | cut -c1-64)
  got=$(dd if="/dev/r$disk" bs=1m count=$((bytes / 1048576)) 2>/dev/null | shasum -a 256 | cut -c1-64)
  [ "$want" = "$got" ] || { echo "VERIFY FAILED: the card does not hold what was written. Do not use this card."; exit 1; }
  echo "Verified: the card matches the image."
else
  echo "note: image size is not a whole number of MiB; not verified"
fi
diskutil eject "/dev/$disk" >/dev/null
echo "Done. Card ejected: put it in the board and power on."
