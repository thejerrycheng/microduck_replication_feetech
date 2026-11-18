#!/bin/bash
# Make the duck safe to switch off at any moment. Run once on the duck, after fresh_install.sh:
#
#     scp firmware/safe_install.sh firmware/duck_safe.sh microduck:/tmp/
#     ssh -t microduck 'sudo bash /tmp/safe_install.sh'
#
# What it does, then reboots:
#   1. /data    a 1 GB ext4 partition in the card's free space, for the two things that must survive: the
#               calibration (/data/calibration.json) and the saved Wi-Fi networks (/data/nm, mounted over
#               NetworkManager's own folder). Both are written rarely, and whole-file-at-a-time
#   2. logs     journald keeps its log in memory; the apt, man-db and trim timers are switched off
#   3. lock     installs Debian's `overlayroot` and sets it to tmpfs: the system partition is mounted read-only and
#               all writes go to memory. `sudo duck-safe unlock` / `lock` switch it (each reboots)
#   4. the studio's login may run `duck-safe` and nothing else new
# Safe to run again. Needs the internet once, for the overlayroot package. For the JoyandAI image on a Radxa Zero 3W.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
user=${SUDO_USER:?run through sudo from the account the studio logs in as}
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$here/duck_safe.sh" ] || { echo "duck_safe.sh must sit beside this script"; exit 1; }
[ "$(findmnt -no FSTYPE /)" != overlay ] || { echo "the system is locked: run 'sudo duck-safe unlock' first, then this again"; exit 1; }
export PATH=$PATH:/sbin:/usr/sbin
rootdev=$(findmnt -no SOURCE /); disk=/dev/$(lsblk -no PKNAME "$rootdev")

# 1. /data
if ! blkid -L duckdata >/dev/null 2>&1; then
  echo "Creating the data partition on $disk..."
  sgdisk -e "$disk" >/dev/null                        # the image's partition table ends where the image did
  sgdisk -n 4:0:+1G -t 4:8300 -c 4:duckdata "$disk" >/dev/null
  partprobe "$disk"; udevadm settle
  mkfs.ext4 -q -F -L duckdata -m 0 "${disk}p4"
  udevadm settle
fi
mkdir -p /data
grep -q "LABEL=duckdata" /etc/fstab || echo "LABEL=duckdata /data ext4 noatime,nofail,x-systemd.device-timeout=8s 0 2" >> /etc/fstab
mountpoint -q /data || mount /data
mkdir -p /data/nm; chmod 700 /data/nm
nm=/etc/NetworkManager/system-connections
if ! mountpoint -q "$nm"; then
  cp -a "$nm"/. /data/nm/ 2>/dev/null || true
  grep -q "/data/nm" /etc/fstab || echo "/data/nm $nm none bind,nofail,x-systemd.requires-mounts-for=/data 0 0" >> /etc/fstab
  mount "$nm"
fi
chown "$user": /data; chmod 755 /data
old=$(getent passwd "$user" | cut -d: -f6)/microduck/calibration.json
[ -f /data/calibration.json ] || { [ -f "$old" ] && cp -p "$old" /data/calibration.json || true; }
sed -i -E '/vfat/{/idle-timeout/!s/x-systemd.automount/x-systemd.automount,x-systemd.idle-timeout=20/}' /etc/fstab   # FAT partitions: mounted only while used

# 2. logs and timers
mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=volatile\nRuntimeMaxUse=24M\n' > /etc/systemd/journald.conf.d/duck.conf
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer man-db.timer fstrim.timer e2scrub_all.timer >/dev/null 2>&1 || true

# 3. the lock
if ! dpkg -s overlayroot >/dev/null 2>&1; then
  echo "Installing overlayroot..."
  apt-get update -qq || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq overlayroot >/dev/null
fi
install -m 755 "$here/duck_safe.sh" /usr/local/sbin/duck-safe
printf 'overlayroot="tmpfs:recurse=0"\n' > /etc/overlayroot.local.conf
update-initramfs -u >/dev/null
built=$(lsinitramfs "/boot/initrd.img-$(uname -r)" | grep -c "scripts/init-bottom/overlayroot" || true)   # counted, not grep -q: that would end the pipe early and read as a failure
[ "$built" -ge 1 ] || { printf 'overlayroot="disabled"\n' > /etc/overlayroot.local.conf; echo "overlayroot did not get into the boot image: NOT locked. Nothing else was undone."; exit 1; }
u-boot-update >/dev/null

# 4. the studio may use the switch
f=/etc/sudoers.d/duck-safe
echo "$user ALL=(root) NOPASSWD: /usr/local/sbin/duck-safe" > "$f.new"; chmod 440 "$f.new"; visudo -cf "$f.new" >/dev/null; mv "$f.new" "$f"

echo "Done: /data ready, logs in memory, system locks from the next boot. Rebooting: back in about 40 s."
sync
systemctl reboot
