#!/bin/bash
# Put the studio on the duck itself, running now and at every boot, so the phone app (ios/) reaches the duck
# directly: on the Wi-Fi, or on the duck's own hotspot, with no Mac in between. Run on the Mac, once per duck
# (and again to update it); needs no root on the duck.
#
#     firmware/duck_serve.sh [ssh host]        # default: microduck
#
# It copies what the studio needs to ~/microduck/studio and installs two systemd *user* services, kept running
# after logout and started at boot by "linger": `duck-agent`, the one program that holds the duck's devices
# (duck_agent.py), and `duck-studio` (`studio.py --duck --lan`, port 8770, low priority), which asks it.
# Undo:  ssh microduck 'systemctl --user disable --now duck-studio duck-agent'
# On a duck locked by safe_install.sh, unlock first (`sudo duck-safe unlock`), or the install is gone at the next boot.
set -euo pipefail
host=${1:-microduck}; cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar czf - firmware/studio.py firmware/duck_agent.py firmware/duck_voice.py firmware/duck_pilot.py firmware/policy_net.py firmware/run_log.py firmware/feetech.py \
  firmware/web app/web/vendor/three.min.js app/web/vendor/OrbitControls.js \
  app/assets/microduck/robot_groundcontact.xml ios/lan.py |
  ssh -o BatchMode=yes "$host" 'mkdir -p ~/microduck/studio && tar xzf - -C ~/microduck/studio 2>/dev/null'
ssh -o BatchMode=yes "$host" 'set -e
  [ "$(findmnt -no FSTYPE /)" != overlay ] || echo "NOTE: the duck is locked; this install lasts until it reboots (sudo duck-safe unlock, then run this again)"
  mkdir -p ~/.config/systemd/user
  cat > ~/.config/systemd/user/duck-studio.service <<E
[Unit]
Description=Duck studio for the phone app
After=duck-agent.service
Wants=duck-agent.service

[Service]
WorkingDirectory=%h/microduck/studio/firmware
ExecStart=/usr/bin/python3 -u studio.py --duck --lan --no-open
Restart=always
RestartSec=3
Nice=10

[Install]
WantedBy=default.target
E
  python3 ~/microduck/studio/firmware/duck_agent.py --install      # the hardware bridge, service duck-agent
  systemctl --user daemon-reload
  systemctl --user enable duck-studio >/dev/null 2>&1
  systemctl --user restart duck-studio
  sleep 6
  systemctl --user is-active duck-agent duck-studio; journalctl --user -u duck-studio -n 5 --no-pager -o cat'
