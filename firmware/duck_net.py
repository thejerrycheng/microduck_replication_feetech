"""Keep the duck reachable: on a known Wi-Fi and, at the same time, hosting its own hotspot. Runs on the duck as root.

Installed by `net_install.sh` as the `duck-net` service. Three ways in, none depending on the others:

  Wi-Fi     NetworkManager joins any saved network by itself on `wlan0`. This script does not touch it.
  hotspot   Always on, on a second virtual interface `ap0` of the same radio (the chip lists "managed + AP" as a
            valid combination). Network name `microduck`, the duck at 10.42.0.1. NetworkManager's "shared" mode
            hands out addresses and forwards to `wlan0`, so a phone on the hotspot has the internet whenever the duck
            is on a Wi-Fi that has it, and still reaches the duck when it is not.
  USB       The `radxa-ncm` gadget and the `duck-usb` profile (duck at 10.55.0.1) are set up once by the installer.

One radio serves both, so the hotspot follows the Wi-Fi's channel if the chip refuses two (tried in that order),
and a phone on the hotspot may see a short pause when the duck changes Wi-Fi network. In every case the duck
answers to `radxa-zero3.local`. What it is doing, readable by anyone on the duck:

    cat /run/duck-net.status
"""
import subprocess, time

TICK = 10
HOTSPOT, STA, AP = "duck-hotspot", "wlan0", "ap0"


def run(*args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    return r.returncode, (r.stdout + r.stderr).strip()


def state(dev):
    """The profile `dev` is running, "" if it is not connected, None if there is no such interface."""
    for line in run("nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION", "dev")[1].splitlines():
        d, s, con = (line.split(":", 2) + ["", ""])[:3]
        if d == dev:
            return con if s == "connected" else ""
    return None


def channel(dev):
    """(band, channel) `dev` is on, e.g. ("a", 44), or None."""
    for line in run("iw", "dev", dev, "info")[1].splitlines():
        w = line.split()
        if w[:1] == ["channel"]:
            return ("a" if int(w[1]) > 14 else "bg"), int(w[1])
    return None


def note(text, last=[""]):
    if text != last[0]:
        last[0] = text
        print(text, flush=True)
    open("/run/duck-net.status", "w").write(f"{time.strftime('%H:%M:%S')} {text}\n")


def main():
    while True:
        try:
            if state(AP) is None:                            # the second interface is gone after every boot
                code, out = run("iw", "dev", STA, "interface", "add", AP, "type", "__ap")
                if code:
                    note(f"cannot create {AP}: {out}")
                    time.sleep(TICK)
                    continue
                run("nmcli", "dev", "set", AP, "managed", "yes")
                time.sleep(2)
            if state(AP) != HOTSPOT:
                code, out = run("nmcli", "con", "up", HOTSPOT, "ifname", AP)
                if code and channel(STA):                    # refused on its own channel: share the Wi-Fi's
                    band, ch = channel(STA)
                    run("nmcli", "con", "mod", HOTSPOT, "802-11-wireless.band", band, "802-11-wireless.channel", str(ch))
                    code, out = run("nmcli", "con", "up", HOTSPOT, "ifname", AP)
                if code:
                    note(f"hotspot did not start: {out}")
                    time.sleep(TICK)
                    continue
            note(f"wifi: {state(STA) or 'not connected'} {channel(STA) or ''}; hotspot: up {channel(AP) or ''}")
        except Exception as e:
            note(f"error: {e}")
        time.sleep(TICK)


if __name__ == "__main__":
    main()
