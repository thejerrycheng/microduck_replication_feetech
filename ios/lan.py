"""Opt-in network access for the phone app (ios/): the studio and the Duck Live engine listen on 127.0.0.1 by
default; with `--lan` they listen on every interface and announce themselves over Bonjour as `_microduck._tcp`, so
the app finds them with nothing typed. On macOS through dns_sd (in libSystem), on the duck through Avahi (D-Bus);
either way the announcement ends with the process. If neither is there the server still runs, unannounced: the app
also tries the duck's own addresses (radxa-zero3.local, and 10.42.0.1 on its hotspot).

Anyone on the same network can then reach the server. The studio has no login: use `--lan` on a network you trust.
"""
import socket
import sys

TYPE = "_microduck._tcp"
_refs = []                           # keep the registrations alive


def announce(kind, port, **txt):
    """Register `kind` ("studio" | "live") at `port`; extra keys ride in the TXT record. Returns the name used."""
    name = f"Duck {kind} ({socket.gethostname().split('.')[0]})"
    pairs = [f"{k}={v}".encode() for k, v in dict(kind=kind, **txt).items()]
    try:
        if sys.platform == "darwin":
            import ctypes, ctypes.util
            lib = ctypes.CDLL(ctypes.util.find_library("System"))
            rec, ref = b"".join(bytes([len(kv)]) + kv for kv in pairs), ctypes.c_void_p()
            lib.DNSServiceRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p,
                                               ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_char_p,
                                               ctypes.c_void_p, ctypes.c_void_p]
            err = lib.DNSServiceRegister(ctypes.byref(ref), 0, 0, name.encode(), TYPE.encode(), None, None, socket.htons(port),
                                         len(rec), rec, None, None)
            if err:
                raise OSError(f"dns_sd error {err}")
            _refs.append(ref)
        else:
            import dbus
            bus = dbus.SystemBus()
            server = dbus.Interface(bus.get_object("org.freedesktop.Avahi", "/"), "org.freedesktop.Avahi.Server")
            group = dbus.Interface(bus.get_object("org.freedesktop.Avahi", server.EntryGroupNew()), "org.freedesktop.Avahi.EntryGroup")
            group.AddService(dbus.Int32(-1), dbus.Int32(-1), dbus.UInt32(0), name, TYPE, "", "", dbus.UInt16(port),
                             dbus.Array([dbus.ByteArray(kv) for kv in pairs], signature="ay"))
            group.Commit()
            _refs.append((bus, group))
    except Exception as e:           # no Bonjour here: still reachable by address
        return f"{name} (not announced: {e})"
    return name


def address():
    """This machine's address on the local network (for the log line; the app does not need it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
