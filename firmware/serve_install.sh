#!/bin/bash
# Start the phone's server (`studio.py --duck --lan`, put on the duck by duck_serve.sh) at every boot. On the duck, once:
#
#     ssh -t microduck 'sudo bash ~/microduck/studio/firmware/serve_install.sh'
#
# One systemd service, `duck-studio`, running as the login user. `sudo systemctl disable --now duck-studio` undoes it.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
user=${SUDO_USER:?run through sudo from the account the studio logs in as}
dir=$(cd "$(dirname "$0")" && pwd)
pkill -u "$user" -f "[s]tudio.py --duck" 2>/dev/null || true
cat > /etc/systemd/system/duck-studio.service <<E
[Unit]
Description=Duck studio for the phone app
After=network.target avahi-daemon.service

[Service]
User=$user
WorkingDirectory=$dir
ExecStart=/usr/bin/python3 -u $dir/studio.py --duck --lan --no-open
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
E
systemctl daemon-reload
systemctl enable --now duck-studio >/dev/null
echo "Done: duck-studio runs now and at every boot (port 8770)."
