#!/usr/bin/env python3
"""Printable STLs of the flight frame in their print orientation (flat face on the bed, z = 0 at the bed).

    python3 scripts/tools/flight_print_orient.py      # cad/flight/out/*.stl -> cad/flight/print/stl/*.stl

backpack and nacelles: top face down (the open side and the fan bores face up). Wings: flat root face down.
Cap and rings: as modelled, flat. The orientations are a design choice; no part has been printed.
"""
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC, DST = os.path.join(ROOT, "cad", "flight", "out"), os.path.join(ROOT, "cad", "flight", "print", "stl")
DT = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
RX = lambda deg: np.array([[1, 0, 0], [0, np.cos(np.radians(deg)), -np.sin(np.radians(deg))], [0, np.sin(np.radians(deg)), np.cos(np.radians(deg))]])
# rotation about x that puts the wanted face down
ROT = dict(backpack=RX(180), nacelle_left=RX(180), nacelle_right=RX(180), wing_left=RX(90), wing_right=RX(-90), bottom_cap=RX(0), fan_collar=RX(0), top_lid=RX(180))
os.makedirs(DST, exist_ok=True)
for name, R in ROT.items():
    t = np.fromfile(os.path.join(SRC, name + ".stl"), dtype=DT, offset=84)
    v = t["v"].reshape(-1, 3).astype(np.float64) @ R.T
    v -= [0.5 * (v[:, 0].min() + v[:, 0].max()), 0.5 * (v[:, 1].min() + v[:, 1].max()), v[:, 2].min()]
    tri = v.reshape(-1, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = 0.5 * np.linalg.norm(n, axis=1); n /= np.maximum(2 * area[:, None], 1e-12)
    down = (n[:, 2] < -0.999) & (tri[:, :, 2].max(1) < 0.05)                     # facets lying on the bed
    over = (n[:, 2] < -np.cos(np.radians(40))) & (tri[:, :, 2].min(1) > 0.3)     # steeper than 50 deg from vertical, off the bed
    out = np.zeros(len(tri), dtype=DT); out["v"] = tri; out["n"] = n
    with open(os.path.join(DST, name + ".stl"), "wb") as f:
        f.write(b"microduck flight frame v8, print orientation".ljust(80, b" ")); f.write(np.uint32(len(tri)).tobytes()); out.tofile(f)
    e = v.max(0) - v.min(0)
    print(f"{name:14s} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} mm   on bed {area[down].sum() / 100:6.1f} cm2   overhang >50 deg {area[over].sum() / 100:6.1f} cm2")
