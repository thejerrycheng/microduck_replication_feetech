#!/bin/bash
# Make the duck reachable three ways, on the duck itself, once:
#
#     ssh -t microduck 'sudo bash /tmp/net_install.sh'        # after copying this file and duck_net.py to /tmp
#
#   1. Wi-Fi     every saved network is retried forever instead of being given up on
#   2. hotspot   always on, alongside the Wi-Fi: network name `microduck`, password `microduck` (or the script's first argument), the duck at
#                10.42.0.1. A phone on it reaches the duck directly and gets the internet through the duck's Wi-Fi
#   3. USB       Radxa's NCM gadget on the OTG port plus a `duck-usb` profile; the duck is then 10.55.0.1 and the
#                Mac gets its address from it over the cable that powers it
#
# On a duck locked by safe_install.sh: `sudo duck-safe unlock` first, run this, then `sudo duck-safe lock`.
# Safe to run again: it replaces its own profiles and service and touches nothing else.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
export PATH=$PATH:/sbin:/usr/sbin
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$here/duck_net.py" ] || { echo "duck_net.py must sit beside this script"; exit 1; }

ssid="microduck"
pw=${1:-microduck}                # the hotspot's password (user, 2026-11-20: "microduck"); another as the first argument
[ ${#pw} -ge 8 ] || { echo "too short"; exit 1; }

# 1. saved Wi-Fi: never stop trying
nmcli -t -f NAME,TYPE con | awk -F: '$NF=="802-11-wireless"{sub(/:802-11-wireless$/,""); print}' | while read -r name; do
  [ "$name" = duck-hotspot ] || nmcli con mod "$name" connection.autoconnect yes connection.autoconnect-retries 0
done

# 2. the hotspot: on its own interface (ap0, made by duck-net), sharing the Wi-Fi's internet
nmcli con delete duck-hotspot >/dev/null 2>&1 || true
nmcli con add type wifi ifname ap0 con-name duck-hotspot autoconnect no ssid "$ssid" \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 ipv4.method shared ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$pw" >/dev/null

# 3. USB networking
nmcli con delete duck-usb >/dev/null 2>&1 || true
nmcli con add type ethernet ifname usb0 con-name duck-usb autoconnect yes \
  ipv4.method shared ipv4.addresses 10.55.0.1/24 ipv6.method link-local >/dev/null
systemctl enable --now radxa-ncm@fcc00000.dwc3 2>/dev/null || echo "note: the USB gadget service did not start; USB networking is not available"

install -m 755 "$here/duck_net.py" /usr/local/bin/duck_net.py
cat > /etc/systemd/system/duck-net.service <<'E'
[Unit]
Description=Duck network: Wi-Fi and its own hotspot together
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
ExecStart=/usr/bin/python3 -u /usr/local/bin/duck_net.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
E
systemctl daemon-reload
systemctl enable duck-net >/dev/null
systemctl restart duck-net
echo "Done. Wi-Fi: saved networks retried forever. Hotspot: $ssid, always on, duck at 10.42.0.1. USB: duck at 10.55.0.1."
echo "In about 20 s: cat /run/duck-net.status"
