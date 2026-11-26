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


H = C.FRAME_H_MM
R_BORE = C.DUCT_OD_MM / 2 + 0.15
FEATURES = {
    "frame": [f"2 x ring bore dia {2*R_BORE:.1f} (duct OD {C.DUCT_OD_MM:.1f} + 0.3), axis canted {C.FAN_CANT_DEG:.0f} deg about the radial line",
              f"ring wall {C.RING_WALL_MM:.1f} nominal ({cad['ring_wall_min_mm']:.2f} min), height {H:.0f}; 2 mm pinch slit, M3 clearance dia 3.4 across it",
              f"ring centres {2*C.FAN_LX*1e3:.0f} apart; ring centre line {C.FAN_LY*1e3:.0f} from the duck's centre plane",
              f"2 x spar {C.SPAR_W_MM:.0f} wide x {H:.0f} deep, inner edges +-{C.SPAR_OFFSET_MM:.2f} from the thrust centroid; run {C.LAP_Y_MM:.1f} past the centre plane",
              "2 x lap-bolt hole dia 3.4 through both spars at +-22 from the centre plane, 6 above the underside",
              "2 x heat-set insert pocket dia 4.0 x 5.5 deep in the rear spar top, +-15.25 (30.5 mm stack pattern)",
              f"rail {C.RAIL_Y_MM[1]-C.RAIL_Y_MM[0]:.1f} wide at {C.RAIL_Y_MM[0]:.1f}-{C.RAIL_Y_MM[1]:.1f} from the centre plane; M2 pilot dia 1.7 for the hook screw",
              "skirt finger 2.6 x 7.5 x 18 below the rail; 2 x gusset under the spars, 18 deep at the rail",
              "PRINT: top face (stack side) on the bed, no supports; 3 perimeters, 6 top/bottom layers, 15 % gyroid"],
    "lift_hook": [f"strap {C.HOOK_STRAP_MM:.1f} thick x {C.HOOK_X_MM[1]-C.HOOK_X_MM[0]:.0f} wide; top flange 2.4 over the rail; lip 2.6 under the shell edge",
                  f"lip reaches {C.SHELL_LEDGE_Y_MM:.1f} from the centre plane; shell underside taken at z = {C.SHELL_BOTTOM_Z_MM:.0f}",
                  "clearance hole dia 2.2 for the M2 retention screw",
                  "PRINT: on its side (profile on the bed), no supports; 100 % infill. Mirror for the other side."],
    "pack_tray": [f"pocket for a {C.BATT_SIZE[0]*1e3:.0f} x {C.BATT_SIZE[1]*1e3:.1f} x {C.BATT_SIZE[2]*1e3:.0f} pack (4S {C.BATT_REF_AH*1e3:.0f} mAh), shelf 2.4, end lip 8 high",
                  "lug 3.0 thick against the rear spar pair: 2 x dia 3.4 at +-22 (the rear lap bolts hold the tray)",
                  "back plate 2.0, roof plate 2.4, 8 mm gusset; 2 strap slots 32 x 3.5; lightening holes dia 9.2",
                  "PRINT: on its side (C profile on the bed), no supports; 3 perimeters, 15 % gyroid"],
}
SHEETS = [("frame_left", "frame", "Frame half, left", 1), ("frame_right", "frame", "Frame half, right (left turned 180 deg; hook pilot differs)", 1),
          ("lift_hook_left", "lift_hook", "Lift hook (left; mirror for right)", 2), ("pack_tray", "pack_tray", "Pack tray", 1)]


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
    tb.text(0.01, 0.2, f"MicroDuck flight frame v3  |  cad/flight/out/{part}.step", fontsize=6.5)
    tb.text(0.37, 0.62, f"Quantity: {qty}", fontsize=9); tb.text(0.37, 0.2, "Units: mm   Tol. +-0.2 (FDM)", fontsize=7)
    tb.text(0.57, 0.62, "Material: ABS / ASA (PLA: prototype)", fontsize=7.5)
    tb.text(0.57, 0.2, "Printed mass: " + ", ".join(f"{k} {v:.1f} g" for k, v in masses.items()), fontsize=7)
    tb.text(0.79, 0.62, f"Envelope {bb.xlen:.0f} x {bb.ylen:.0f} x {bb.zlen:.0f}", fontsize=8)
    tb.text(0.79, 0.2, "2026-11-25  generated, not to scale", fontsize=7)
    fig.suptitle("Projection: third angle. Dashed = hidden edges. Blue = overall dimensions; feature dimensions in the list (source: flight_config.py).", fontsize=7, y=0.975)
    fig.savefig(os.path.join(DRW, part + ".pdf")); pdf_all.savefig(fig); plt.close(fig)
    print("  sheet", part)


with PdfPages(os.path.join(DRW, "flight_frame_drawings.pdf")) as pdf:
    asm = None
    for p in cad["parts"]:
        s = cq.importers.importStep(os.path.join(OUT, p + ".step"))
        asm = s if asm is None else asm.union(s)
    FEATURES["assembly"] = [f"overall {cad['span_y_mm']:.0f} wide x {cad['length_x_mm']:.0f} long (fan rings included, fans not shown)",
                            "4 x M3x25 + nut through the spar laps (rear pair also holds the pack tray)",
                            "4 x M3x12 + nut pinch bolts on the rings; 4 x M3 heat-set inserts for the FC / ESC stack",
                            "2 x M2x8 self-tapper, hook to rail; hook-and-loop strap around pack and tray",
                            "assembly: seat the halves over the roof, bolt the laps, clip both hooks under the shell edge and",
                            "screw them, bolt the tray, insert the fans from above (CW/CCW on diagonals), tighten pinch bolts",
                            f"frame mass incl. fasteners: ABS {cad['frame_mass_g']['ABS']:.0f} g, PLA {cad['frame_mass_g']['PLA']:.0f} g"]
    sheet(pdf, "flight_frame_assembly", "assembly", "Assembly (printed parts only)", 1, shape=asm.val(), masses={k: v for k, v in cad["frame_mass_g"].items()})
    for part, feat, title, qty in SHEETS:
        sheet(pdf, part, feat, title, qty)
print("wrote cad/flight/drawings/*.pdf")
