#!/bin/bash
# Put the studio on the duck itself and start it there, so the phone app (ios/) reaches the duck directly: on the
# Wi-Fi, or on the duck's own hotspot, with no Mac in between. Run on the Mac; needs no root on the duck.
#
#     firmware/duck_serve.sh [ssh host]        # default: microduck
#
# It copies what the studio needs to ~/microduck/studio and runs `studio.py --duck --lan` there (port 8770). That
# lasts until the duck reboots. To start it at every boot, once:   ssh -t microduck 'sudo bash ~/microduck/studio/firmware/serve_install.sh'
set -euo pipefail
host=${1:-microduck}; cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar czf - firmware/studio.py firmware/duck_agent.py firmware/duck_voice.py firmware/feetech.py \
  firmware/serve_install.sh firmware/web app/web/vendor/three.min.js app/web/vendor/OrbitControls.js \
  app/assets/microduck/robot_groundcontact.xml ios/lan.py |
  ssh -o BatchMode=yes "$host" 'mkdir -p ~/microduck/studio && tar xzf - -C ~/microduck/studio 2>/dev/null'
# (the script name is kept out of the remote command line, or pkill -f would match the shell running it)
ssh -o BatchMode=yes "$host" 'if systemctl is-enabled duck-studio >/dev/null 2>&1; then echo "restart it with: sudo systemctl restart duck-studio"; else
  s=studio.py; pkill -f "python3 -u $s --duck" 2>/dev/null; sleep 1
  cd ~/microduck/studio/firmware && (nohup python3 -u $s --duck --lan --no-open > ~/microduck/studio.log 2>&1 < /dev/null &)
  sleep 4; cat ~/microduck/studio.log; fi'
