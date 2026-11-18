#!/bin/bash
# The duck's power-loss protection, on the duck, as root. Installed as /usr/local/sbin/duck-safe by safe_install.sh.
#
#     sudo duck-safe status     # is the system locked, is the card clean
#     sudo duck-safe unlock     # reboot with a writable system: to install, update or change settings
#     sudo duck-safe lock       # reboot with the system read-only again
#
# Locked = the system partition is mounted read-only and every write the running system makes goes to memory
# (overlayroot, tmpfs). Nothing is being written to the card, so cutting the power cannot leave it half-written.
# What must survive a reboot lives on the small /data partition instead: the calibration and the saved Wi-Fi
# networks. Anything else changed while locked is gone at the next boot: unlock first, change it, lock again.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
locked() { [ "$(findmnt -no FSTYPE /)" = overlay ]; }
lower() { if locked; then echo /media/root-ro; else echo ""; fi; }
conf() {                                              # write the setting into the real system, not the memory layer
  local root; root=$(lower)
  [ -z "$root" ] || mount -o remount,rw "$root"
  printf 'overlayroot="%s"\n' "$1" > "$root/etc/overlayroot.local.conf"
  sync
  [ -z "$root" ] || mount -o remount,ro "$root"
}
case "${1:-status}" in
  status)
    dev=$(findmnt -no SOURCE "$(lower)/" | head -1)
    echo "system: $(locked && echo "LOCKED (read-only, writes go to memory)" || echo "UNLOCKED (writable: a power cut can damage it)")"
    echo "next boot: $(grep -q disabled "$(lower)/etc/overlayroot.local.conf" 2>/dev/null && echo unlocked || echo locked)"
    echo "card system partition $dev: mounted $(findmnt -no OPTIONS "$(lower)/" | cut -d, -f1), $(dumpe2fs -h "$dev" 2>/dev/null | grep -E '^Filesystem state' | sed 's/  */ /g')"
    echo "journal needs replay: $(dumpe2fs -h "$dev" 2>/dev/null | grep -q needs_recovery && echo yes || echo no)"
    echo "data: $(findmnt -no SOURCE,FSTYPE,OPTIONS /data 2>/dev/null | cut -c1-60 || true)"
    echo "wifi networks kept on data: $(ls /data/nm 2>/dev/null | wc -l)"
    locked && echo "memory layer in use: $(df -h --output=used /media/root-rw 2>/dev/null | tail -1 | tr -d ' ')"
    echo "storage errors this boot: $(dmesg | grep -ciE 'EXT4-fs error|I/O error|mmc[0-9].*(error|timeout)' || true)"
    ;;
  lock)   conf "tmpfs:recurse=0"; echo "Locking: rebooting."; systemctl reboot ;;
  unlock) conf "disabled";        echo "Unlocking: rebooting."; systemctl reboot ;;
  *) echo "usage: duck-safe status|lock|unlock"; exit 1 ;;
esac
