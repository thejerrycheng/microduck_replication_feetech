#!/usr/bin/env python3
"""Linear-elastic FE check of the bolted frame (both halves, laps bonded), held at the two lift-hook pads.

    .venv-cad/bin/python cad/flight/fea_frame.py        # after build_parts.py

Quadratic tetrahedra (gmsh mesh, scikit-fem). Static: listed full thrust on all four rings. Modal: fans as lumped
masses on the ring bores. Printed material = bulk modulus x PRINT_E_KNOCKDOWN and the printed (not solid) density.
Writes docs/flight_fea_results.json and docs/figures/flight_fea.pdf.
"""
import json, math, os, sys
import cadquery as cq
import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve
from skfem import Basis, ElementTetP2, ElementVector, MeshTet2, MeshTet, asm, condense
from skfem.models.elasticity import lame_parameters, linear_elasticity, linear_stress
from skfem.helpers import sym_grad

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools"))
import flight_config as C

OUT = os.path.join(ROOT, "cad", "flight", "out")
PRINT_E_KNOCKDOWN = 0.6   # 1.2 mm skins + 15 % infill against a solid section (a 12x12 box keeps 59 % of I)
NU = 0.35
MESH_MM = float(os.environ.get("FEA_MESH_MM", 4.0))
cad = json.load(open(os.path.join(ROOT, "docs", "flight_cad_results.json")))
layout = json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))["layout"]

vol_mm3 = sum(cq.importers.importStep(os.path.join(OUT, n + ".step")).val().Volume() for n in ("frame_left", "frame_right"))
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
a_ = gmsh.model.occ.importShapes(os.path.join(OUT, "frame_left.step"))
b_ = gmsh.model.occ.importShapes(os.path.join(OUT, "frame_right.step"))
gmsh.model.occ.fuse(a_, b_)                      # the laps are bolted: bonded here
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 5); gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_MM); gmsh.option.setNumber("Mesh.MeshSizeMin", 0.7)
try:
    gmsh.model.mesh.generate(3)
except Exception as e:
    import re
    for sid in set(int(k) for k in re.findall(r"surface (\d+)", str(e))):
        print("bad surface", sid, [round(v, 2) for v in gmsh.model.getBoundingBox(2, sid)])
    raise
gmsh.write(os.path.join(OUT, "_fea.msh")); gmsh.finalize()
import meshio
mm = meshio.read(os.path.join(OUT, "_fea.msh"))
tets = np.vstack([c.data for c in mm.cells if c.type == "tetra"])
mesh = MeshTet(mm.points.T * 1e-3, tets.T)          # metres
basis = Basis(mesh, ElementVector(ElementTetP2()), intorder=4)
print(f"mesh {MESH_MM} mm: {mesh.t.shape[1]} tets, {basis.N} dofs, solid volume {vol_mm3/1e3:.1f} cm^3")

X = basis.doflocs  # (3, N)
hx0, hx1, hy0, hy1, hz = (v * 1e-3 for v in cad["hook_fix_box_mm"])   # rail top under the lift-hook flanges
ymid = 0.5 * (hy0 + hy1)
# pinned line along x on each rail top (a hook holds the rail down but does not stop it rotating about x)
fixed_nodes = np.where((X[0] >= hx0) & (X[0] <= hx1) & (np.abs(np.abs(X[1]) - ymid) <= 1.2e-3) & (np.abs(X[2] - hz) < 1e-4))[0]
fixed = fixed_nodes
r_bore = (C.DUCT_OD_MM / 2 + 0.15) * 1e-3
fans = np.array(layout["fans"])
results = {"mesh_mm": MESH_MM, "dofs": int(basis.N), "materials": {}}
printed_mass = {m: sum(cad["parts"][k]["mass_g"][m] for k in ("frame_left", "frame_right")) * 1e-3 for m in C.MATERIALS}
T_fan = C.FANS[C.PRIMARY]["thrust_g"] * 1e-3 * C.G
m_fan = C.FANS[C.PRIMARY]["mass_g"] * 1e-3

zdofs = basis.nodal_dofs[2]  # vertex z dofs
vx = mesh.p
bore_nodes = [np.where(np.abs(np.hypot(vx[0] - f[0], vx[1] - f[1]) - r_bore) < 0.9e-3)[0] for f in fans]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
for mat, pm in C.MATERIALS.items():
    E = pm["E"] * PRINT_E_KNOCKDOWN
    rho = printed_mass[mat] / (vol_mm3 * 1e-9)
    lam, mu = lame_parameters(E, NU)
    K = asm(linear_elasticity(lam, mu), basis)
    from skfem import BilinearForm
    from skfem.helpers import dot
    M = asm(BilinearForm(lambda u_, v_, w: rho * dot(u_, v_)), basis)
    # static: thrust shared by the bore nodes of each ring
    f = np.zeros(basis.N)
    Madd = np.zeros(basis.N)
    for bn in bore_nodes:
        for d in range(3):
            Madd[basis.nodal_dofs[d][bn]] += m_fan / len(bn)
        f[basis.nodal_dofs[2][bn]] += T_fan / len(bn)
    D = np.unique(fixed)
    x = np.zeros(basis.N)
    I = np.setdiff1d(np.arange(basis.N), D)
    x[I] = spsolve(K[I][:, I].tocsc(), f[I])
    uz = x[basis.nodal_dofs[2]]
    if mat == 'ABS':
        for (px, py) in ((-18, 36), (-18, 50), (-18, 70), (4, 50), (4, 70), (-11, 102), (-11, 72), (31, 75), (31, 131), (-53, 75), (-53, 131), (-11, 0)):
            k = np.argmin((vx[0] * 1e3 - px) ** 2 + (vx[1] * 1e3 - py) ** 2)
            print(f'   probe ({px},{py}): uz {uz[k]*1e3:.3f} mm')
    tip = [float(uz[bn].mean()) for bn in bore_nodes]
    tilt = []
    for bn, fc in zip(bore_nodes, fans):  # ring tilt: plane fit of uz over the bore nodes
        A_ = np.c_[vx[0][bn] - fc[0], vx[1][bn] - fc[1], np.ones(len(bn))]
        c_ = np.linalg.lstsq(A_, uz[bn], rcond=None)[0]
        tilt.append(math.degrees(math.hypot(c_[0], c_[1])))
        if mat == 'ABS': print(f'   ring at ({fc[0]*1e3:.0f},{fc[1]*1e3:.0f}): lift {c_[2]*1e3:.3f} mm, slope dz/dx {math.degrees(c_[0]):+.2f} deg, dz/dy {math.degrees(c_[1]):+.2f} deg')
    # von Mises at element centroids via projected displacement gradient
    dbasis = basis.with_element(ElementVector(ElementTetP2()))
    ui = basis.interpolate(x)
    eps = sym_grad(ui)
    sig = linear_stress(lam, mu)(eps)
    s = np.array([[sig[i][j] for j in range(3)] for i in range(3)])
    vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2) + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[0, 2] ** 2))
    vm_el = vm.mean(axis=1)
    # modal with the fans as lumped masses
    Mt = (M + sp.diags(Madd)).tocsc()
    lamv, phi = eigsh(K[I][:, I].tocsc(), k=6, M=Mt[I][:, I], sigma=0.0, which="LM")
    freqs = np.sqrt(np.abs(lamv)) / (2 * math.pi)
    results["materials"][mat] = dict(E_eff_gpa=E / 1e9, tip_mm=[t * 1e3 for t in tip], tip_mean_mm=float(np.mean(tip) * 1e3),
                                     ring_tilt_deg=tilt, vm_p999_mpa=float(np.percentile(vm_el, 99.9) / 1e6), vm_max_mpa=float(vm_el.max() / 1e6),
                                     sf_yield=float(pm["yield_pa"] / np.percentile(vm_el, 99.9)), modes_hz=freqs.tolist(),
                                     stiffness_n_per_mm=float(T_fan / (np.mean(tip) * 1e3)))
    r = results["materials"][mat]
    print(f"{mat}: ring lift {r['tip_mean_mm']:.3f} mm at {T_fan:.1f} N per fan ({r['stiffness_n_per_mm']:.0f} N/mm per ring), ring tilt {max(tilt):.3f} deg, "
          f"von Mises p99.9 {r['vm_p999_mpa']:.1f} MPa (SF {r['sf_yield']:.0f}), modes " + ", ".join(f"{q:.0f}" for q in freqs[:4]) + " Hz")
    if mat == "ABS":
        sc = axs[0].scatter(vx[1] * 1e3, vx[0] * 1e3, c=uz * 1e3, s=1.5, cmap="viridis")
        plt.colorbar(sc, ax=axs[0], label="vertical deflection [mm]")
        axs[0].set_aspect("equal"); axs[0].set_xlabel("y [mm]"); axs[0].set_ylabel("x [mm]")
        axs[0].set_title(f"ABS, {T_fan:.1f} N per fan, held at the hook pads", fontsize=8)
    axs[1].plot(range(1, 7), freqs, "o-", label=mat)
axs[1].set_xlabel("mode"); axs[1].set_ylabel("frequency [Hz]"); axs[1].legend(); axs[1].grid(alpha=0.3)
axs[1].set_title("Frame modes with four 76 g fans", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_fea.pdf"))
json.dump(results, open(os.path.join(ROOT, "docs", "flight_fea_results.json"), "w"), indent=1)
for fn in ("_fea.msh",):
    os.remove(os.path.join(OUT, fn))
