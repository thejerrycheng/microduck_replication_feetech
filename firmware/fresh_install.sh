#!/bin/bash
# Everything a freshly flashed duck needs before the studio can use its servos and sensors, in one go. On the duck:
#
#     scp firmware/fresh_install.sh firmware/mode_install.sh microduck:/tmp/
#     ssh -t microduck 'sudo bash /tmp/fresh_install.sh'
#
# For the JoyandAI first-version image on a Radxa Zero 3W with the Zero Robot HAT. It does three things and reboots:
#   1. puts the login user in `dialout` (the servo port /dev/ttyS2) and `robot` (the runtime's sockets)
#   2. switches I2C3 on (header pins 3/5: the HAT's audio codec and ToF socket). The image's I2S overlay switches
#      it off, so the copy is named to sort after that one
#   3. runs mode_install.sh: the studio may pause/resume the runtime and power the duck off; mediad is disabled
# Safe to run again.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
user=${SUDO_USER:?run through sudo from the account the studio logs in as}
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$here/mode_install.sh" ] || { echo "mode_install.sh must sit beside this script"; exit 1; }
usermod -aG robot,dialout "$user"
src=/boot/dtbo/rk3568-i2c3-m0.dtbo.disabled
[ -f /boot/dtbo/zz-i2c3-m0.dtbo ] || { [ -f "$src" ] || { echo "missing $src: not the expected image"; exit 1; }; cp "$src" /boot/dtbo/zz-i2c3-m0.dtbo; }
u-boot-update >/dev/null
bash "$here/mode_install.sh"
echo "Rebooting: the duck is back in about 40 s."
sync
systemctl reboot
