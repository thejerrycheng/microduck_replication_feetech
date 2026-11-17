"""Keep the duck reachable: known Wi-Fi if there is one, its own hotspot if not. Runs on the duck as root.

Installed by `net_install.sh` as the `duck-net` service. Three ways in, none depending on the others:

  Wi-Fi     NetworkManager joins any saved network by itself. This script only watches.
  hotspot   If no saved network has been joined WAIT seconds after boot (or after Wi-Fi was lost), it brings up
            the `duck-hotspot` profile: the duck's own network, the duck at 10.42.0.1. While nobody is using the
            hotspot it looks every RETRY seconds for a saved network and goes back to it when one is in range.
  USB       The `radxa-ncm` gadget and the `duck-usb` profile (duck at 10.55.0.1, addresses handed out to the Mac)
            are set up once by the installer and need nothing from this script.

The duck has one radio: it is either on a network or hosting the hotspot, never both. In every case the duck
answers to `radxa-zero3.local`.

    journalctl -u duck-net -f        # what it is doing
"""
import subprocess, time

WAIT, RETRY, TICK = 40, 60, 10          # s offline before the hotspot; s between looks for a known network; loop period
HOTSPOT, DEV = "duck-hotspot", "wlan0"


def nm(*args):
    return subprocess.run(["nmcli", "-t", *args], capture_output=True, text=True, timeout=60).stdout.strip()


def active():
    """The profile wlan0 is running, or "" when it is not connected."""
    for line in nm("-f", "DEVICE,STATE,CONNECTION", "dev").splitlines():
        dev, state, con = (line.split(":", 2) + ["", ""])[:3]
        if dev == DEV and state == "connected":
            return con
    return ""


def saved():
    """Names of the saved client Wi-Fi networks: {profile name: SSID}."""
    out = {}
    for line in nm("-f", "NAME,TYPE", "con").splitlines():
        name, _, kind = line.rpartition(":")
        if kind == "802-11-wireless" and name != HOTSPOT:
            if nm("-g", "802-11-wireless.mode", "con", "show", name) != "ap":
                out[name] = nm("-g", "802-11-wireless.ssid", "con", "show", name)
    return out


def visible():
    return set(nm("-f", "SSID", "dev", "wifi", "list", "ifname", DEV, "--rescan", "yes").splitlines())


def guests():
    """Is anyone on the hotspot? Counted as neighbours on wlan0 the kernel has heard from recently."""
    out = subprocess.run(["ip", "neigh", "show", "dev", DEV], capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines() if l.split()[-1] in ("REACHABLE", "DELAY", "PROBE", "STALE"))


def main():
    offline, looked = None, 0.0
    while True:
        now, con = time.monotonic(), active()
        if con and con != HOTSPOT:                       # on a real network: nothing to do
            offline = None
        elif con == HOTSPOT:
            if now - looked > RETRY and not guests():    # idle hotspot: is a known network back?
                looked = now
                near = visible()
                back = next((name for name, ssid in saved().items() if ssid in near), None)
                if back:
                    print(f"{back} is in range: leaving the hotspot", flush=True)
                    nm("con", "up", back)
        else:
            offline = offline or now
            if now - offline > WAIT:
                print("no known Wi-Fi: starting the hotspot", flush=True)
                nm("con", "up", HOTSPOT)
                offline, looked = None, now
        time.sleep(TICK)


if __name__ == "__main__":
    main()
