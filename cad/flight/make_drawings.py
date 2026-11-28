#!/usr/bin/env python3
"""2D drawings of the printed parts: three hidden-line views per part with overall dimensions, a feature list and a
title block (material, quantity, print orientation, mass).

    .venv-cad/bin/python cad/flight/make_drawings.py        # after build_parts.py

Writes cad/flight/drawings/<part>.pdf, cad/flight/drawings/flight_frame_drawings.pdf (all sheets) and the assembly
sheet. Views are true projections of the STEP solids (OCC hidden-line removal); dimensions are in millimetres.
"""
import json, math, os, sys
import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from OCP.BRepLib import BRepLib
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools"))
import flight_config as C

OUT = os.path.join(ROOT, "cad", "flight", "out")
DRW = os.path.join(ROOT, "cad", "flight", "drawings")
os.makedirs(DRW, exist_ok=True)
cad = json.load(open(os.path.join(ROOT, "docs", "flight_cad_results.json")))
VIEWS = {"top": ((0, 0, 1), (1, 0, 0)), "front": ((0, -1, 0), (1, 0, 0)), "side": ((1, 0, 0), (0, 1, 0))}
AX = {"top": ("x", "y"), "front": ("x", "z"), "side": ("y", "z")}


def project(shape, view):
    n, xdir = VIEWS[view]
    hlr = HLRBRep_Algo()
    hlr.Add(shape.wrapped)
    hlr.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*n), gp_Dir(*xdir))))
    hlr.Update(); hlr.Hide()
    hs = HLRBRep_HLRToShape(hlr)
    out = {}
    for key, comps in (("visible", (hs.VCompound(), hs.Rg1LineVCompound(), hs.OutLineVCompound())), ("hidden", (hs.HCompound(), hs.OutLineHCompound()))):
        lines = []
        for comp in comps:
            if comp.IsNull():
                continue
            BRepLib.BuildCurves3d_s(comp)
            for e in cq.Shape.cast(comp).Edges():
                k = 2 if e.geomType() == "LINE" else 24
                pts = np.array([[p.x, p.y] for p in (e.positionAt(t) for t in np.linspace(0, 1, k))])
                lines.append(pts)
        out[key] = lines
    return out


def dim(ax, p0, p1, off, text, vertical=False):
    """Dimension line between p0 and p1, offset by `off` (drawing units) perpendicular to it."""
    (x0, y0), (x1, y1) = p0, p1
    if vertical:
        ax.annotate("", (x0 + off, y0), (x0 + off, y1), arrowprops=dict(arrowstyle="<->", lw=0.5, color="#1f4e79"))
        ax.plot([x0, x0 + off], [y0, y0], color="#1f4e79", lw=0.3); ax.plot([x0, x0 + off], [y1, y1], color="#1f4e79", lw=0.3)
        ax.text(x0 + off * 1.08, (y0 + y1) / 2, text, rotation=90, va="center", ha="left" if off > 0 else "right", fontsize=6.5, color="#1f4e79")
    else:
        ax.annotate("", (x0, y0 + off), (x1, y0 + off), arrowprops=dict(arrowstyle="<->", lw=0.5, color="#1f4e79"))
        ax.plot([x0, x0], [y0, y0 + off], color="#1f4e79", lw=0.3); ax.plot([x1, x1], [y0, y0 + off], color="#1f4e79", lw=0.3)
        ax.text((x0 + x1) / 2, y0 + off * 1.08, text, ha="center", va="bottom" if off > 0 else "top", fontsize=6.5, color="#1f4e79")


R_BORE = C.DUCT_OD_MM / 2 + 0.15
g, bay, SJ, ST, TG = cad["wing"], cad["bay"], C.SIDE_JOINT_MM, C.FC_STACK_MM, C.TONGUE_MM
PRINT = "PRINT: PLA, 3 perimeters, 5 top / 4 bottom layers, 15 % gyroid, tree supports"
FEATURES = {
    "backpack": [f"lower bay {C.BAY_MM['depth']:.0f} x {2*C.BAY_MM['half_w']:.0f} inside (pack {C.BATT_SIZE[0]*1e3:.0f} x {C.BATT_SIZE[1]*1e3:.1f} x {C.BATT_SIZE[2]*1e3:.0f}), open at the bottom; wall {C.PACK_WALL_MM:.1f}",
                 f"head from z = {C.BAY_HEAD_MM['z0']:.0f}: bay {C.BAY_HEAD_MM['depth']:.0f} x {2*C.BAY_HEAD_MM['half_w']:.0f} inside for a 30.5 mm stack ({ST['x']:.0f} x {ST['y']:.1f}), open at the top; rear wall 6.5",
                 f"each side wall: 2 x dia 3.4 for the wing-root screws at z = {g['root_z'][0]:.0f} / {g['root_z'][1]:.0f}, dia 6 wire hole between them (x = {g['root_x']:.0f})",
                 "2 x insert pocket dia 4.0 x 5.5 in the rear wall's top face, +-14 from the centre plane (lid screws)",
                 "2 x insert pocket dia 4.0 x 5.5 in the bottom rim (cap screws); slot 20.6 x 2.9 in the front wall for the lid's tab",
                 f"rear slot 14 x 8 at z = {bay['usb_z']:.0f} for USB-C and the serial lead (height to be checked on the board)",
                 "hook 2.5 thick over the trunk roof's rear edge; two fingers on the rear corners (location only)",
                 PRINT + "; top face down"],
    "top_lid": [f"plate {C.PACK_WALL_MM:.1f} thick, outline of the head; 2 x dia 3.4, heads sunk 1.6, +-14 from the centre plane",
                f"4 x post dia 8 x {ST['boss']:.1f} with insert pocket dia 4.0 x 5.5 on a {ST['hole']:.1f} square (FC / ESC stack, M3 x 25 from below)",
                "front block 3.8 x 20 x 5 with a tab 2.5 thick into the front wall's slot",
                PRINT + "; top face down"],
    "bottom_cap": ["plate 3.0 with a 4.0 plug into the lower bay; hook 2.6 thick under the trunk",
                   "2 x dia 3.4, counterbored dia 6, for M3 x 8 into the backpack's rim",
                   PRINT + "; flat"],
    "wing": [f"closed shell lofted through 13 superelliptic sections; root {2*C.BLADE_ROOT_MM['a']:.0f} x {2*C.BLADE_ROOT_MM['b']:.0f}, tip {2*C.BLADE_TIP_MM['a']:.0f} x {2*C.BLADE_TIP_MM['b']:.0f}",
             f"root face flat on the backpack's side wall: 2 x insert pocket dia 4.0 x 5.5 at z = {g['root_z'][0]:.0f} / {g['root_z'][1]:.0f}",
             f"tip mates with the nacelle's inboard flank; tongue {TG['w']:.0f} wide x {TG['t']:.0f} thick under the nacelle, 2 x dia 3.4 counterbored (M3 x 8 from below)",
             f"2 x dia 3.4 through the tip along y, {SJ['dx'][1]-SJ['dx'][0]:.0f} apart; head channel dia {2*SJ['bore']:.1f}, {SJ['grip']:.0f} of material under the head (M3 x 12)",
             PRINT + "; root face down. Right wing is the mirror."],
    "nacelle": [f"stadium pod, {cad['length_x_mm']:.0f} long x {2*(C.DUCT_OD_MM/2+0.15+C.POD_WALL6_MM):.1f} wide x {C.POD_H6_MM:.0f} high, wall {C.POD_WALL6_MM:.1f}",
                f"2 x bore dia {2*R_BORE:.1f} (duct OD {C.DUCT_OD_MM:.1f} + 0.3) from below up to a shoulder, {2*C.FAN_LX*1e3:.0f} apart, axes canted {C.FAN_CANT_DEG:.0f} deg",
                f"air path dia 50.6 through the shoulder with a {C.BELL_MM:.0f} mm bell-mouth lip",
                "2 x insert pocket dia 4.0 x 5.5 in the underside between the fans, 20 apart (tongue screws)",
                f"2 x insert pocket dia 4.0 x 5.5 in the inboard flank at mid height, {SJ['dx'][1]-SJ['dx'][0]:.0f} apart (wing-tip screws)",
                PRINT + "; top face down. Right nacelle is the mirror."],
    "fan_collar": ["tapered ring on the shroud under the nacelle; M2.5 grub screw through the thick upper edge, inboard side",
                   "PRINT: flat, no supports"],
}
SHEETS = [("backpack", "backpack", "Backpack", 1), ("top_lid", "top_lid", "Lid (carries the FC / ESC stack)", 1), ("bottom_cap", "bottom_cap", "Bottom cap", 1),
          ("wing_left", "wing", "Wing, left (mirror for right)", 2), ("nacelle_left", "nacelle", "Nacelle, left (mirror for right)", 2), ("fan_collar", "fan_collar", "Fan ring", 4)]


def sheet(pdf_all, part, feat, title, qty, shape=None, masses=None):
    shape = shape or cq.importers.importStep(os.path.join(OUT, part + ".step")).val()
    bb = shape.BoundingBox()
    lo = dict(x=bb.xmin, y=bb.ymin, z=bb.zmin); size = dict(x=bb.xlen, y=bb.ylen, z=bb.zlen)
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, left=0.04, right=0.98, top=0.93, bottom=0.21, wspace=0.08, hspace=0.12)
    for view, cell in (("top", gs[0, 0]), ("front", gs[1, 0]), ("side", gs[1, 1])):
        ax = fig.add_subplot(cell)
        pr = project(shape, view)
        for l in pr["hidden"]:
            ax.plot(l[:, 0], l[:, 1], color="#9a9a9a", lw=0.3, ls=(0, (3, 2)))
        for l in pr["visible"]:
            ax.plot(l[:, 0], l[:, 1], color="k", lw=0.6)
        a, b = AX[view]
        allp = np.vstack(pr["visible"] + pr["hidden"])
        x0, y0, x1, y1 = allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()
        m = 0.09 * max(x1 - x0, y1 - y0, 20)
        dim(ax, (x0, y0), (x1, y0), -m, f"{size[a]:.1f}")
        dim(ax, (x1, y0), (x1, y1), m, f"{size[b]:.1f}", vertical=True)
        ax.set_aspect("equal"); ax.set_xlim(x0 - 2.2 * m, x1 + 3.0 * m); ax.set_ylim(y0 - 2.6 * m, y1 + 1.2 * m)
        ax.axis("off"); ax.set_title(f"{view} view ({a}-{b})", fontsize=7, loc="left", color="#444")
    axf = fig.add_subplot(gs[0, 1]); axf.axis("off")
    axf.text(0, 1, "Features (mm)", fontsize=8.5, weight="bold", va="top")
    for i, line in enumerate(FEATURES[feat]):
        axf.text(0, 0.9 - i * 0.098, "- " + line, fontsize=6.6, va="top", wrap=True)
    tb = fig.add_axes([0.04, 0.03, 0.94, 0.13]); tb.set_xlim(0, 1); tb.set_ylim(0, 1); tb.set_xticks([]); tb.set_yticks([])
    masses = masses or cad["parts"].get(part, {}).get("mass_g", {})
    for x in (0.36, 0.56, 0.78):
        tb.axvline(x, color="k", lw=0.6)
    tb.text(0.01, 0.62, title, fontsize=11, weight="bold")
    tb.text(0.01, 0.2, f"MicroDuck flight frame v8.1  |  cad/flight/out/{part}.step", fontsize=6.5)
    tb.text(0.37, 0.62, f"Quantity: {qty}", fontsize=9); tb.text(0.37, 0.2, "Units: mm   Tol. +-0.2 (FDM)", fontsize=7)
    tb.text(0.57, 0.62, "Material: PLA (ABS / ASA for heat)", fontsize=7.5)
    tb.text(0.57, 0.2, "Printed mass: " + ", ".join(f"{k} {v:.1f} g" for k, v in masses.items() if k in ("PLA", "ABS")), fontsize=7)
    tb.text(0.79, 0.62, f"Envelope {bb.xlen:.0f} x {bb.ylen:.0f} x {bb.zlen:.0f}", fontsize=8)
    tb.text(0.79, 0.2, "2026-11-27  generated, not to scale", fontsize=7)
    fig.suptitle("Projection: third angle. Dashed = hidden edges. Blue = overall dimensions; feature dimensions in the list (source: flight_config.py).", fontsize=7, y=0.975)
    fig.savefig(os.path.join(DRW, part + ".pdf")); pdf_all.savefig(fig); plt.close(fig)
    print("  sheet", part)


with PdfPages(os.path.join(DRW, "flight_frame_drawings.pdf")) as pdf:
    asm = None
    for p in cad["parts"]:
        s = cq.importers.importStep(os.path.join(OUT, p + ".step"))
        asm = s if asm is None else asm.union(s)
    FEATURES["assembly"] = [f"overall {cad['span_y_mm']:.0f} wide x {cad['length_x_mm']:.0f} long (fans not shown)",
                            "20 x M3 heat-set insert; 12 x M3x8, 4 x M3x12, 4 x M3x25, 4 x M2.5 grub screw",
                            "each wing: 2 x M3x8 from below and 2 x M3x12 through the tip into the nacelle; 2 x M3x8 into the root from inside the bay",
                            "stack under the lid on 4 x M3x25; lid: front tab, 2 x M3x8 at the rear; cap: 2 x M3x8 from below",
                            "order and driver access: assembly guide and scripts/tools/flight_tool_clearance.py",
                            f"frame mass incl. fasteners: PLA {cad['frame_mass_g']['PLA']:.0f} g, ABS {cad['frame_mass_g']['ABS']:.0f} g"]
    sheet(pdf, "flight_frame_assembly", "assembly", "Assembly (printed parts only)", 1, shape=asm.val(), masses={k: v for k, v in cad["frame_mass_g"].items()})
    for part, feat, title, qty in SHEETS:
        sheet(pdf, part, feat, title, qty)
print("wrote cad/flight/drawings/*.pdf")
