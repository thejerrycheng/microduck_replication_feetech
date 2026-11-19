#!/bin/bash
# Let the studio pause and resume the duck's own runtime without a password. Run once on the duck:
#
#     ssh -t microduck 'sudo bash /tmp/mode_install.sh'
#
# The runtime (robotd, tofd) owns the servo bus and the ToF ranger while it runs, and the studio's setup tools need
# the same devices to themselves. This adds one sudoers file allowing exactly two commands for the login user:
# stopping those two services, and starting them again. Nothing else is granted.
#
# It also switches the runtime's camera service (mediad) off for good. On this board mediad reconfigures the IMX219
# at every boot into a mode the image pipeline then mishandles: it gets 0.1 frames a second of noise itself, and
# leaves the camera unusable for anyone else until the next reboot. With it off the camera boots in its default
# mode, which works. `sudo systemctl enable --now mediad` undoes this.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
user=${SUDO_USER:?run through sudo from the account the studio logs in as}
f=/etc/sudoers.d/duck-studio
cat > "$f.new" <<S
$user ALL=(root) NOPASSWD: /usr/bin/systemctl stop robotd tofd, /usr/bin/systemctl start robotd tofd, /usr/bin/systemctl poweroff, /usr/bin/systemctl reboot
S
chmod 440 "$f.new"
visudo -cf "$f.new" >/dev/null
mv "$f.new" "$f"
systemctl disable --now mediad >/dev/null 2>&1 || true
echo "Done: $user may now stop and start robotd and tofd, and power the duck off or reboot it; mediad is off. Reboot for the camera to come up clean."
