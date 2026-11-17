"""Opt-in LAN access for the phone app (ios/): the studio and the Duck Live engine listen on 127.0.0.1 by default;
with `--lan` they listen on every interface and announce themselves over Bonjour as `_microduck._tcp`, so the app
finds them with nothing typed. macOS only (dns_sd lives in libSystem); the announcement ends with the process.

Anyone on the same network can then reach the server. The studio has no login: use `--lan` on a network you trust.
"""
import ctypes
import ctypes.util
import socket

TYPE = b"_microduck._tcp"
_refs = []                           # keep the registrations alive


def announce(kind, port, **txt):
    """Register `kind` ("studio" | "live") at `port`; extra keys ride in the TXT record. Returns the name used."""
    lib = ctypes.CDLL(ctypes.util.find_library("System"))
    rec = b"".join(bytes([len(kv)]) + kv for kv in (f"{k}={v}".encode() for k, v in dict(kind=kind, **txt).items()))
    ref, name = ctypes.c_void_p(), f"Duck {kind} ({socket.gethostname().split('.')[0]})".encode()
    lib.DNSServiceRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p,
                                       ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_char_p,
                                       ctypes.c_void_p, ctypes.c_void_p]
    err = lib.DNSServiceRegister(ctypes.byref(ref), 0, 0, name, TYPE, None, None, socket.htons(port), len(rec), rec, None, None)
    if err:
        raise OSError(f"Bonjour registration failed ({err})")
    _refs.append(ref)
    return name.decode()


def address():
    """This machine's address on the local network (for the log line; the app does not need it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
