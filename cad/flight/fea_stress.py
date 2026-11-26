#!/usr/bin/env python3
"""Stress and strain analysis of the printed flight frame in PLA at 15 % and 25 % infill (linear elastic FE).

    .venv-cad/bin/python cad/flight/fea_stress.py [--mesh 4.0] [--fine 3.0]

MODEL
  Geometry   backpack + both wings + both nacelles from cad/flight/out/*.step, fused (bolted joints bonded).
  Elements   quadratic tetrahedra (gmsh), scikit-fem.
  Material   printed PLA as two phases. Skin: the outer PRINT_SHELL_MM (perimeters and top/bottom layers), solid PLA.
             Core: gyroid infill of relative density rho, a cellular solid with E* = E_s rho^2 and crush strength
             sigma* = 0.3 sigma_s rho^1.5 (Gibson & Ashby, bending-dominated foams). Each element gets the stiffness
             f E_skin + (1 - f) E_core, f = share of its sample points within the skin depth of the surface
             (iso-strain mixture); skin and core stresses are then recovered from the element strain with their own
             moduli. Walls thinner than twice the skin depth are solid, as the slicer prints them.
  Supports   where the yoke bears on the trunk: rear face of the trunk (x fixed on the yoke's front face), the roof
             hook and the lower edge held by the bottom cap (z fixed), the two corner fingers (y fixed).
  Loads      LC1 full thrust: the seller's maximum 950 g per fan on all four shoulders, along the fan axes
             LC2 roll: left pair at maximum, right pair at 5 %         LC3 pitch: front pair at maximum, rear at 5 %
             LC4 hard landing: 6 g downward on the fans (86 g each with collar) and on the frame's own mass
             LC5 handling: 20 N sideways and 20 N rearward on one nacelle's outer end
  Strength   PLA, printed: 50 MPa in the layer plane, 28 MPa across layers (0.55 x), E_s = 3.3 GPa, nu = 0.35,
             skin knock-down 0.9 for raster gaps. The print orientation varies over the part, so the across-layer
             check is applied to the largest tensile principal stress everywhere (conservative).
OUTPUT  docs/flight_stress_results.json, docs/duck_flight_stress_table.tex, docs/figures/flight_stress.pdf
"""
import argparse, json, math, os, sys, time
import gmsh, meshio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu, LinearOperator
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTetP0, ElementTetP2, ElementVector, MeshTet, asm
from skfem.helpers import ddot, dot, sym_grad, trace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools"))
import flight_config as C

OUT = os.path.join(ROOT, "cad", "flight", "out")
ap = argparse.ArgumentParser(); ap.add_argument("--mesh", type=float, default=4.0); ap.add_argument("--fine", type=float, default=3.0)
ap.add_argument("--no_conv", action="store_true")
args = ap.parse_args()
cad = json.load(open(os.path.join(ROOT, "docs", "flight_cad_results.json")))
pack = json.load(open(os.path.join(ROOT, "microduck_rl", "flight", "flight_pack.json")))
E_S, NU, RHO_S = 3.3e9, 0.35, 1240.0
S_XY, S_Z, SKIN_K = 50e6, 28e6, 0.9
T_SKIN = C.PRINT_SHELL_MM * 1e-3
T_MAX = 0.950 * 9.81
M_FAN = (C.FANS[C.PRIMARY]["mass_g"] + cad["parts"]["fan_collar"]["mass_g"]["PLA"]) * 1e-3
T0 = np.array([0.0, 0.0, 0.120])
FANS = [(f["name"], np.array(f["pos"]) + T0, np.array(f["axis"])) for f in pack["fans"]]
R_BORE = (C.DUCT_OD_MM / 2 + 0.15) * 1e-3
PARTS = ("backpack", "wing_left", "wing_right", "nacelle_left", "nacelle_right")


def mesh(size):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    ents = []
    for n in PARTS:
        ents += gmsh.model.occ.importShapes(os.path.join(OUT, n + ".step"))
    gmsh.model.occ.fuse(ents[:1], ents[1:]); gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 5); gmsh.option.setNumber("Mesh.MeshSizeMax", size); gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    fn = os.path.join(OUT, "_stress.msh"); gmsh.write(fn); gmsh.finalize()
    mm = meshio.read(fn); os.remove(fn)
    tets = np.vstack([c.data for c in mm.cells if c.type == "tetra"])
    return MeshTet(np.ascontiguousarray(mm.points.T * 1e-3), np.ascontiguousarray(tets.T))


@BilinearForm
def stiff(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2.0 * w.mu * ddot(eu, ev) + w.lam * trace(eu) * trace(ev)


@BilinearForm
def massf(u, v, w):
    return w.rho * dot(u, v)


def lame(E):
    return E * NU / ((1 + NU) * (1 - 2 * NU)), E / (2 * (1 + NU))


class Model:
    def __init__(self, size):
        t = time.time()
        self.m = mesh(size)
        m = self.m
        from scipy.sparse.csgraph import connected_components
        ed = m.edges; ncomp = connected_components(sp.coo_matrix((np.ones(ed.shape[1]), (ed[0], ed[1])), shape=(m.p.shape[1],) * 2), directed=False)[0]
        print(f'connected bodies in the mesh: {ncomp}', flush=True); assert ncomp == 1, 'parts are not bonded'
        self.basis = Basis(m, ElementVector(ElementTetP2()), intorder=3)
        self.b0 = self.basis.with_element(ElementTetP0())
        # skin fraction per element: sample points (4 vertices-weighted + centroid + 4 face centres) against the boundary
        bf = m.boundary_facets(); tri = m.facets[:, bf]
        P = m.p.T
        a, b, c = P[tri[0]], P[tri[1]], P[tri[2]]
        w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
        surf = np.vstack([wi[0] * a + wi[1] * b + wi[2] * c for wi in w])
        tree = cKDTree(surf)
        V = P[m.t]                                              # (4, nel, 3)
        samples = [V.mean(0)] + [0.7 * V[i] + 0.1 * (V.sum(0) - V[i]) for i in range(4)] + [(V.sum(0) - V[i]) / 3 for i in range(4)]
        self.f_skin = np.mean([tree.query(s)[0] < T_SKIN for s in samples], axis=0)
        self.vol = np.abs(np.einsum("ij,ij->i", V[1] - V[0], np.cross(V[2] - V[0], V[3] - V[0]))) / 6
        X = self.basis.doflocs
        comp = np.zeros(self.basis.N, dtype=int)
        for cdim in range(3):
            comp[self.basis.nodal_dofs[cdim]] = cdim; comp[self.basis.edge_dofs[cdim]] = cdim
        xr = C.TRUNK_X0_MM * 1e-3
        on_front = (np.abs(X[0] - xr) < 2e-4) & (np.abs(X[1]) < 0.035)
        fix = (on_front & (comp == 0))
        fix |= (X[0] > xr + 1e-4) & (X[0] < xr + 0.0125) & (np.abs(X[2] - (C.ROOF_Z_MM + 0.4) * 1e-3) < 2e-4) & (comp == 2)        # roof hook underside
        zb = X[2][on_front].min()
        fix |= on_front & (X[2] < zb + 0.006) & (comp == 2)                                                                     # lower edge: bottom-cap hook
        fix |= (X[0] > xr + 1e-4) & (np.abs(np.abs(X[1]) - (C.TRUNK_WID_MM / 2 + 0.4) * 1e-3) < 2e-4) & (comp == 1)                # corner fingers
        fix |= on_front & (np.abs(X[1]) < 0.004) & (comp == 1)                                                                  # centre line: no lateral slide
        self.D = np.where(fix)[0]; self.I = np.setdiff1d(np.arange(self.basis.N), self.D)
        vx = m.p
        self.shoulder = []
        for name, pos, ax in FANS:
            rel = vx.T - pos; ax_c = rel @ ax; rad = np.linalg.norm(rel - np.outer(ax_c, ax), axis=1)
            top = C.DUCT_LEN_MM * 0.5e-3
            self.shoulder.append(np.where((np.abs(rad - R_BORE) < 1.0e-3) & (ax_c > top - 0.010) & (ax_c < top + 0.0005))[0])
        self.ndz = self.basis.nodal_dofs
        print(f"mesh {size} mm: {m.t.shape[1]} tets, {self.basis.N} dofs, skin share of volume {np.sum(self.f_skin*self.vol)/self.vol.sum()*100:.0f} %, "
              f"shoulder nodes {[len(s) for s in self.shoulder]}, fixed dofs {len(self.D)}, {time.time()-t:.0f} s", flush=True)

    def assemble(self, rho_rel):
        E_core = E_S * rho_rel ** 2
        E_el = self.f_skin * E_S * SKIN_K + (1 - self.f_skin) * E_core
        lam, mu = lame(E_el)
        self.K = asm(stiff, self.basis, lam=self.b0.interpolate(lam), mu=self.b0.interpolate(mu)).tocsc()
        self.rho_el = self.f_skin * RHO_S + (1 - self.f_skin) * RHO_S * rho_rel
        self.M = asm(massf, self.basis, rho=self.b0.interpolate(self.rho_el)).tocsc()
        self.lu = splu(self.K[self.I][:, self.I].tocsc())
        self.E_core, self.E_el = E_core, E_el
        self.mass = float(np.sum(self.rho_el * self.vol))

    def load(self, thrust, g_down=0.0, tip=None):
        f = np.zeros(self.basis.N)
        for (name, pos, ax), nodes, T in zip(FANS, self.shoulder, thrust):
            for cdim in range(3):
                f[self.ndz[cdim][nodes]] += (T * ax[cdim] - (M_FAN * g_down * 9.81 if cdim == 2 else 0.0)) / len(nodes)
        if g_down:
            ones = np.zeros(self.basis.N); ones[self.ndz[2]] = 1.0; ones[self.basis.edge_dofs[2]] = 1.0
            f -= g_down * 9.81 * (self.M @ ones)
        if tip is not None:
            vx = self.m.p
            k = np.argmin((vx[0] - (pack["thrust_centroid_x"])) ** 2 + (vx[1] - vx[1].max()) ** 2 * 4 + (vx[2] - C.FAN_Z) ** 2)
            near = np.where(np.linalg.norm(vx.T - vx[:, k], axis=1) < 0.006)[0]
            for cdim in range(3):
                f[self.ndz[cdim][near]] += tip[cdim] / len(near)
        return f

    def solve(self, f):
        x = np.zeros(self.basis.N); x[self.I] = self.lu.solve(f[self.I]); return x

    def post(self, x):
        eps = sym_grad(self.basis.interpolate(x))
        e = np.array([[np.asarray(eps[i][j]).mean(axis=1) for j in range(3)] for i in range(3)]).transpose(2, 0, 1)     # (nel, 3, 3)

        def stress(E):
            lam, mu = lame(E); tr = np.trace(e, axis1=1, axis2=2)
            return 2 * mu * e + lam * tr[:, None, None] * np.eye(3)

        def vm(s):
            d = s - np.trace(s, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
            return np.sqrt(1.5 * np.einsum("eij,eij->e", d, d))

        s_skin, s_core = stress(E_S * SKIN_K), stress(self.E_core)
        skin, core = self.f_skin > 0.05, self.f_skin < 0.95
        p1 = np.linalg.eigvalsh(s_skin)[:, -1]
        e1 = np.linalg.eigvalsh(e)
        q = lambda a, sel, pct: float(np.percentile(a[sel], pct)) if sel.any() else 0.0
        ux = x[self.ndz[0]], x[self.ndz[1]], x[self.ndz[2]]
        fan = {}
        for (name, pos, ax), nodes in zip(FANS, self.shoulder):
            U = np.array([u[nodes] for u in ux]).T
            Pn = self.m.p.T[nodes] - pos
            A_ = np.c_[Pn, np.ones(len(nodes))]
            coef = np.linalg.lstsq(A_, U @ ax, rcond=None)[0]             # plane fit of the axial displacement: tilt of the seat
            fan[name] = dict(disp_mm=(U.mean(0) * 1e3).tolist(), axial_mm=float((U @ ax).mean() * 1e3), tilt_deg=float(math.degrees(np.linalg.norm(coef[:3] - (coef[:3] @ ax) * ax))))
        Pm = self.m.p; yb = cad["collision"][0]["size"][1]
        rootn = np.where((np.abs(np.abs(Pm[1]) - (yb + 0.004)) < 0.003) & (Pm[2] > 0.125) & (Pm[2] < 0.160))[0]
        self.root_disp_mm = float(np.sqrt(sum(u[rootn] ** 2 for u in ux)).mean() * 1e3) if len(rootn) else float("nan")
        return dict(vm_skin_p999=q(vm(s_skin), skin, 99.9) / 1e6, vm_skin_max=float(vm(s_skin)[skin].max()) / 1e6, p1_skin_p999=q(p1, skin, 99.9) / 1e6,
                    vm_core_p999=q(vm(s_core), core, 99.9) / 1e6, strain_max_p999=q(np.abs(e1).max(1), np.ones(len(e1), bool), 99.9) * 100,
                    disp_max_mm=float(np.sqrt(sum(u ** 2 for u in ux)).max() * 1e3), fans=fan), vm(s_skin)

    def modes(self, k=6):
        madd = np.zeros(self.basis.N)
        for nodes in self.shoulder:
            for cdim in range(3):
                madd[self.ndz[cdim][nodes]] += M_FAN / len(nodes)
        Mt = (self.M + sp.diags(madd)).tocsc()[self.I][:, self.I]
        n = len(self.I)
        op = LinearOperator((n, n), matvec=self.lu.solve)
        lamv = eigsh(self.K[self.I][:, self.I], k=k, M=Mt, sigma=0.0, which="LM", OPinv=op)[0]
        return (np.sqrt(np.abs(lamv)) / (2 * math.pi)).tolist()


CASES = {"LC1 full thrust": dict(thrust=[T_MAX] * 4), "LC2 roll differential": dict(thrust=[T_MAX, T_MAX, 0.05 * T_MAX, 0.05 * T_MAX]),
         "LC3 pitch differential": dict(thrust=[T_MAX, 0.05 * T_MAX, T_MAX, 0.05 * T_MAX]), "LC4 landing 6 g": dict(thrust=[0.0] * 4, g_down=6.0),
         "LC5 handling 20 N": dict(thrust=[0.0] * 4, tip=(-20.0, 20.0, 0.0))}
res = dict(material=dict(E_s_gpa=E_S / 1e9, nu=NU, strength_xy_mpa=S_XY / 1e6, strength_z_mpa=S_Z / 1e6, skin_mm=C.PRINT_SHELL_MM, skin_knockdown=SKIN_K, thrust_max_n=T_MAX, fan_mass_kg=M_FAN),
           fan_order=[f[0] for f in FANS], infill={})
mdl = Model(args.mesh)
res["mesh"] = dict(size_mm=args.mesh, tets=int(mdl.m.t.shape[1]), dofs=int(mdl.basis.N))
field = None
for rho in (0.15, 0.25):
    t = time.time(); mdl.assemble(rho)
    core_strength = 0.3 * S_XY * rho ** 1.5
    r = dict(E_core_mpa=mdl.E_core / 1e6, core_strength_mpa=core_strength / 1e6, fe_mass_g=mdl.mass * 1e3, cases={})
    print(f"\nPLA, {rho*100:.0f} % infill: core modulus {mdl.E_core/1e6:.0f} MPa, core crush strength {core_strength/1e6:.2f} MPa, structure mass in the model {mdl.mass*1e3:.0f} g  (factorised in {time.time()-t:.0f} s)")
    print(f"  {'load case':24s} {'fan defl':>8s} {'tilt':>6s} {'skin vM':>8s} {'skin s1':>8s} {'core vM':>8s} {'strain':>7s} {'SF xy':>6s} {'SF z':>6s} {'SF core':>7s}")
    for name, kw in CASES.items():
        x = mdl.solve(mdl.load(**kw)); p, vmf = mdl.post(x)
        dmax = max(abs(v["axial_mm"]) for v in p["fans"].values()) if "LC5" not in name else p["disp_max_mm"]
        tmax = max(v["tilt_deg"] for v in p["fans"].values())
        p.update(fan_defl_mm=dmax, fan_tilt_deg=tmax, sf_xy=S_XY / 1e6 / max(p["vm_skin_p999"], 1e-9), sf_z=S_Z / 1e6 / max(p["p1_skin_p999"], 1e-9), sf_core=core_strength / 1e6 / max(p["vm_core_p999"], 1e-9))
        r["cases"][name] = p; p["wing_root_disp_mm"] = mdl.root_disp_mm
        print(f"      wing-root displacement {mdl.root_disp_mm:.3f} mm")
        print(f"  {name:24s} {dmax:6.3f}mm {tmax:5.2f}d {p['vm_skin_p999']:6.2f}MPa {p['p1_skin_p999']:6.2f}MPa {p['vm_core_p999']:6.3f}MPa {p['strain_max_p999']:6.3f}% {p['sf_xy']:6.1f} {p['sf_z']:6.1f} {p['sf_core']:7.1f}", flush=True)
        if rho == 0.15 and name.startswith("LC1"):
            field = (mdl.m.p.copy(), mdl.m.t.copy(), vmf.copy(), x[mdl.ndz[2]].copy())
    r["modes_hz"] = mdl.modes()
    r["min_sf"] = min(min(c["sf_xy"], c["sf_z"], c["sf_core"]) for c in r["cases"].values())
    print("  modes with the four fans: " + ", ".join(f"{f:.0f}" for f in r["modes_hz"]) + f" Hz;  lowest safety factor over all cases {r['min_sf']:.1f}")
    res["infill"][f"{int(rho*100)}"] = r
if not args.no_conv:
    t = time.time(); fine = Model(args.fine); fine.assemble(0.15)
    pf, _ = fine.post(fine.solve(fine.load(**CASES["LC1 full thrust"])))
    c0 = res["infill"]["15"]["cases"]["LC1 full thrust"]
    df = max(abs(v["axial_mm"]) for v in pf["fans"].values())
    res["convergence"] = dict(size_mm=args.fine, dofs=int(fine.basis.N), fan_defl_mm=df, vm_skin_p999=pf["vm_skin_p999"], defl_change_pct=(df / c0["fan_defl_mm"] - 1) * 100,
                              stress_change_pct=(pf["vm_skin_p999"] / c0["vm_skin_p999"] - 1) * 100, mode1_hz=fine.modes(2)[0])
    cv = res["convergence"]
    print(f"\nconvergence, LC1 at 15 %: {args.mesh} mm -> {args.fine} mm ({cv['dofs']} dofs): deflection {c0['fan_defl_mm']:.3f} -> {df:.3f} mm ({cv['defl_change_pct']:+.1f} %), "
          f"skin stress {c0['vm_skin_p999']:.2f} -> {pf['vm_skin_p999']:.2f} MPa ({cv['stress_change_pct']:+.1f} %), first mode {res['infill']['15']['modes_hz'][0]:.0f} -> {cv['mode1_hz']:.0f} Hz  ({time.time()-t:.0f} s)")
json.dump(res, open(os.path.join(ROOT, "docs", "flight_stress_results.json"), "w"), indent=1)

rows = []
for rho, r in res["infill"].items():
    for name, c in r["cases"].items():
        rows.append(f"{rho}\\,\\% & {name} & {c['fan_defl_mm']:.2f} & {c['fan_tilt_deg']:.2f} & {c['vm_skin_p999']:.1f} & {c['p1_skin_p999']:.1f} & {c['vm_core_p999']:.2f} & {c['strain_max_p999']:.2f} & {c['sf_xy']:.0f} & {c['sf_z']:.0f} & {c['sf_core']:.1f} \\\\")
open(os.path.join(ROOT, "docs", "duck_flight_stress_table.tex"), "w").write("% generated by cad/flight/fea_stress.py\n" + "\n".join(rows) + "\n")
P, Tt, vmf, uz = field
cen = P[:, Tt].mean(1)
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
sc = axs[0].scatter(cen[1] * 1e3, cen[2] * 1e3, c=np.minimum(vmf / 1e6, np.percentile(vmf / 1e6, 99.9)), s=1.2, cmap="inferno"); plt.colorbar(sc, ax=axs[0], label="skin von Mises [MPa]")
axs[0].set_aspect("equal"); axs[0].set_xlabel("y [mm]"); axs[0].set_ylabel("z [mm]"); axs[0].set_title("LC1 full thrust, PLA 15 %: stress (rear view)", fontsize=8)
sc = axs[1].scatter(P[1] * 1e3, P[2] * 1e3, c=uz * 1e3, s=1.2, cmap="viridis"); plt.colorbar(sc, ax=axs[1], label="vertical deflection [mm]")
axs[1].set_aspect("equal"); axs[1].set_xlabel("y [mm]"); axs[1].set_title("LC1: deflection", fontsize=8)
names = list(CASES); xk = np.arange(len(names))
for i, (rho, r) in enumerate(res["infill"].items()):
    axs[2].bar(xk + (i - 0.5) * 0.38, [min(r["cases"][n]["sf_xy"], r["cases"][n]["sf_z"], r["cases"][n]["sf_core"]) for n in names], 0.38, label=f"{rho} % infill", color=["#2a6f97", "#d1603d"][i])
axs[2].set_yscale("log"); axs[2].axhline(3.0, color="k", ls="--", lw=0.8); axs[2].set_xticks(xk); axs[2].set_xticklabels([n.split()[0] for n in names]); axs[2].set_ylabel("lowest safety factor"); axs[2].legend(fontsize=7)
axs[2].set_title("Safety factor by load case (dashed: 3)", fontsize=8); axs[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_stress.pdf"))
print("wrote docs/flight_stress_results.json, docs/duck_flight_stress_table.tex, docs/figures/flight_stress.pdf")
