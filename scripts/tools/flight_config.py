"""Constants for the flying-duck (jetpack) design study. Pure Python: imported by both
scripts/tools/flight_design.py (sizing, anaconda env) and cad/flight/build_parts.py (CadQuery env).

Frames: the MuJoCo world frame of app/assets/microduck/scene.xml at keyframe INIT
(x forward, y left, z up, ground at z = 0, trunk_base origin at z = 0.120 m). Lengths in metres
unless a name ends in _MM.
"""

G = 9.81
RHO_AIR = 1.225  # kg/m^3, sea level 15 C

# ---------------------------------------------------------------- duck (as built) --------------
MODEL_XML = "app/assets/microduck/scene.xml"
SERVO_EXTRA_KG = 0.005 * 15  # README: HD1910 is +5 g per servo over the modelled XL330, 15 servos
# The real duck has not been weighed (docs/SIM2REAL.md). Sweep range used in the sensitivity plot:
DUCK_MASS_RANGE_KG = (0.75, 1.00)

# ---------------------------------------------------------------- fan candidates ---------------
# thrust_g / power_w: one static operating point. source "measured" = an independent buyer bench
# test; "claimed" = seller listing. mass_g: unit with motor. hub_mm: motor/hub diameter (annulus).
# Missing power -> None (hover power then uses FOM_DEFAULT).
FANS = {
    "XFly 30mm 1108 (3S)":      dict(d_mm=30, hub_mm=12, mass_g=15,  thrust_g=204,  power_w=145, cells=3, source="measured", esc_a=12),
    "XFly 40mm 1413 (3S)":      dict(d_mm=40, hub_mm=16, mass_g=42,  thrust_g=340,  power_w=None, cells=3, source="measured", esc_a=25),
    "FMS 50mm 11b 2627 (3S)":   dict(d_mm=50, hub_mm=26, mass_g=75,  thrust_g=546,  power_w=275.5, cells=3, source="measured", esc_a=40),
    "Powerfun 50mm 4900KV (3S)": dict(d_mm=50, hub_mm=26, mass_g=82, thrust_g=770,  power_w=478, cells=3, source="claimed", esc_a=40),
    "XFly 50mm 2627-4600KV (4S)": dict(d_mm=50, hub_mm=26, mass_g=78, thrust_g=900, power_w=None, cells=4, source="claimed", esc_a=40),
    "QX 50mm QF2611-4000KV (4S)": dict(d_mm=50, hub_mm=26, mass_g=76, thrust_g=950, power_w=16.2 * 34, cells=4, source="claimed", esc_a=40),
    "QX 50mm QF2611-5000KV (4S)": dict(d_mm=50, hub_mm=26, mass_g=76, thrust_g=950, power_w=12.6 * 45, cells=4, source="claimed", esc_a=50),
    "64mm 11b 3500KV (4S)":     dict(d_mm=64, hub_mm=28, mass_g=150, thrust_g=1460, power_w=874, cells=4, source="claimed", esc_a=50),
}
# QX mass: 76 g on three seller pages (racer.lt, qwinout, rc-castle); one eBay listing says 113 g. Weigh on arrival.
CLAIM_DERATE = 0.80      # seller static thrust -> expected bench thrust (assumption; XFly 40 measured/claimed = 0.94)
INSTALL_DERATE = 0.90    # inlet next to the trunk, battery sag under load, wiring drop (assumption)
FOM_DEFAULT = 0.50       # figure of merit when a listing gives no power (the six with data span 0.36-0.62)
PRIMARY = "QX 50mm QF2611-5000KV (4S)"   # sold in CW and CCW; highest thrust on a sagging 4S pack (flight_power.py)
# The 5000KV listing row is at 12.6 V / 45 A (3S); on 4S the ESC current limit holds it at 45 A, its rated maximum.

# ---------------------------------------------------------------- electronics ------------------
ESC_G_PER_A = 0.125      # Holybro Tekko32 F4 4-in-1 60 A AM32: 15.7 g listed, + capacitor = 20 g for 4 x 40 A
FC_G = 10.5              # SpeedyBee F405 V4 30x30 (listed)
WIRING_G = 40.0          # XT60, 12 AWG leads (57 A hover), motor extension wire
BATT_WH_PER_KG = 118.0   # CNHL Black 4S 1300 mAh 100C: 163 g, 19.2 Wh (retailer listings)
BATT_USABLE = 0.80       # discharge to 20 %
BATT_REF_AH = 1.3        # chosen by scripts/tools/flight_power.py (sag + heating + energy, 30 s mission)
TILT_SERVO_G = 40.0      # 2-fan layout only: two tilt servos + bearings for pitch/yaw vectoring
STRUCT_G_FALLBACK = {2: 45.0, 4: 80.0}  # replaced by the CAD mass (docs/flight_cad_results.json) when present

# ---------------------------------------------------------------- layout (4 fans, X) -----------
FAN_LX = 0.042           # fan centre, fore/aft half-spacing about the thrust centroid
FAN_LY = 0.135           # fan centre, lateral half-spacing
FAN_Z = 0.220            # clamp-ring mid-plane: the spars run over the trunk roof (z = 162 mm)
FAN_CANT_DEG = 6.0       # each duct tilted about its radial line: yaw torque from differential thrust
BATT_POS = (-0.0680, 0.0, 0.1070)   # flight pack centre, upright on the back of the trunk
BATT_SIZE = (0.034, 0.0335, 0.074)  # CNHL 4S 1300 mAh, upright inside the backpack
FC_POS = (-0.068, 0.0, 0.1545)    # FC + 4-in-1 ESC stack above the pack, inside the backpack

# ---------------------------------------------------------------- printed parts ----------------
MATERIALS = {  # typical FDM filament data-sheet values; sigma_allow = design stress (in-plane yield / 5)
    "PLA": dict(rho=1240.0, E=3.3e9, yield_pa=55e6, sigma_allow=11e6, hdt_c=55),
    "ABS": dict(rho=1040.0, E=2.2e9, yield_pa=40e6, sigma_allow=8e6, hdt_c=95),
}
TRUNK_LEN_MM = 82.0      # shell outline at z = 130-150 mm, from the model meshes: x in [-46, 35]
TRUNK_WID_MM = 64.0      # y in [-32, 32]
TRUNK_X0_MM = -46.5      # rear face of the shell
COLLAR_FIT_MM = 0.4      # clearance per side between collar and shell
COLLAR_WALL_MM = 3.0
FRAME_H_MM = 18.0        # height of collar, arms and clamp rings (print Z)
DUCT_OD_MM = 54.0        # QX 50 mm shroud OD (rc-castle listing; XFly: 52.5): MEASURE the unit before printing
DUCT_LEN_MM = 39.0       # shroud length with intake ring (rc-castle)
RING_WALL_MM = 2.6       # bore is canted inside a flat ring: wall varies by +-H/2*tan(cant) = 0.63 mm
ARM_W_MM = 10.0          # arm box section width; height = FRAME_H_MM
ARM_WALL_MM = 1.2
EAR_MM = 8.0             # clamp ears at the collar split (front and rear)
HOOK_STRAP_MM = 3.0      # lift-hook strap thickness
HOOK_X_MM = (-21.5, -10.5)   # lift hook position along the trunk (clear of hips and arms in the model)
SHELL_BOTTOM_Z_MM = 120.0    # underside of the trunk shell the hooks bear on
SHELL_LEDGE_Y_MM = 27.5      # hook lip reaches in to here (trunk_base plate edge)
ARM_ANCHOR_X_MM = (-6.0, -32.0)  # arm roots either side of the lift hook; the roof bridge bolts onto them
PRINT_SHELL_MM = 1.2     # slicer: 3 perimeters at 0.4 mm, 6 top/bottom layers at 0.2 mm
PRINT_INFILL = 0.15      # gyroid

# ---------------------------------------------------------------- fan + pack physics priors ----
# Bench point used to calibrate the grey-box fan model until a thrust stand replaces it
# (microduck_rl/flight/fan_model.py, scripts/tools/fan_sysid.py).
FAN_KV = 5000.0
FAN_BENCH_THRUST_G = CLAIM_DERATE * 950.0   # expected static thrust at FAN_BENCH_V at the ESC input (QX table: 950 g)
FAN_BENCH_V = 12.6
FAN_BENCH_I = 45.0                          # A (QX table row for 5000KV: 12.6 V, 45 A)
FAN_I_LIMIT = 45.0                          # ESC phase-current limit = the motor's rated continuous maximum
FAN_VARIANTS = {  # name: (KV, listed thrust g, V, A) -- seller table rows, for the KV comparison
    "QX 3300KV": (3300, 750, 16.2, 22), "QX 4000KV": (4000, 950, 16.2, 34), "QX 4600KV": (4600, 780, 12.6, 32),
    "QX 5000KV": (5000, 950, 12.6, 45), "XFly 4600KV": (4600, 900, 16.5, 40)}
FAN_INSTALL_ETA = 0.95                      # inlet next to the trunk; sag is modelled explicitly, not derated
CELL_MOHM_AH = 6.5                          # cell resistance x capacity [mOhm Ah]: 5 mOhm at 1.3 Ah (charger readings 4.5-7.5 on 75-95C packs)
PACK_OVERHEAD_G = 15.0                      # leads, XT60, wrap: not thermal mass
CELL_CP = 1000.0                            # J/(kg K)
MISSION_S = 30.0                            # design mission: seconds of fan time at hover power
MISSION_DT_MAX_K = 20.0                     # allowed cell temperature rise over the mission
MISSION_SOC = (0.95, 0.20)                  # start / lowest allowed state of charge
TW_MIN_SAG = 1.5                            # thrust/weight at full throttle on the sagging pack, mid charge
COMMERCIAL_4S_MAH = (650, 850, 1050, 1100, 1300, 1500, 1800, 2200)
ROOF_Z_MM = 162.0         # top of the trunk shell
SPAR_W_MM = 7.0           # each spar; two lap side by side over the roof (14 mm pair)
SPAR_OFFSET_MM = 18.3     # spar inner edge from the thrust centroid: leaves the hook zone free on both sides, front pair clears the neck
LAP_Y_MM = 32.4           # spars cross the whole roof and butt on the other half's rail; lap bolts at +-22 mm
RAIL_Y_MM = (32.8, 38.0)  # longitudinal rail beside the shell: ties the spars and carries the lift hook

# ---------------------------------------------------------------- frame v4: hoop + arms + pods ---
# Chosen with scripts/tools/flight_clearance.py: nothing over the roof behind the neck, pods outside both swept volumes.
HOOP_X_MM = (-46.0, -34.0)   # station of the hoop around the trunk (rear, behind the hips, under the battery holder)
HOOP_T_MM = 4.0              # hoop wall (in its plane); width along x = the station width
TRUNK_BOTTOM_Z_MM = 76.0     # underside of the battery holder / power support the hoop passes under (world z, standing)
ARM_ROOT_Z_MM = 150.0        # where the arm leaves the hoop side (world z): high, so hip pitch stays free
ARM_TOP_X_MM = -22.0         # arm slants forward to here at the pod, to sit near the thrust centroid
ARM_T_MM, ARM_D_MM = 14.0, 22.0   # arm section: thickness (along x at the root) x depth (in the lift plane)
POD_H_MM = 16.0              # pod / clamp-ring height

# ---------------------------------------------------------------- frame v5: yoke + nacelles ------
# One rounded backpack box on the trunk's rear face with both arms rooted in its top corners (the two sides' lift
# reacts through the box), arms leaving at roof height (above the thigh's back-swing), smooth twin-fan nacelles.
PACK_WALL_MM = 3.2           # backpack shell wall (2.4 let the wing roots rock: 25 Hz first mode)
PACK_FILLET_MM = 7.0         # backpack outer edge radius
ARM_ROOT_MM = (-62.0, 20.0, 166.0)    # arm root centre on the backpack's top corner (world, y > 0 side)
ARM_SECTIONS_MM = ((9.0, 16.0), (8.0, 13.5), (7.5, 11.0), (7.5, 10.0))   # ellipse half-axes (across, in the lift plane) root -> tip
POD_H5_MM = 24.0             # nacelle height
POD_WALL_MM = 4.5            # nacelle wall around the shroud
POD_FILLET_MM = 5.0
BELL_MM = 3.0                # bell-mouth lip formed by the nacelle above the shroud (the fan's own intake ring is removed)

# ---------------------------------------------------------------- frame v6: backpack + wing trusses + nacelles ---
# The swept single arm of v5 is replaced by two flat triangulated trusses per side (rear and front), a V in plan:
# each nacelle is held at two lugs, the structure is triangulated in the lift plane and in plan, and every wing
# prints flat without supports.
WING_T_MM = 9.0              # wing thickness (print height)
WING_BAR_MM = 7.0            # width of a truss member in the wing's plane
WING_REAR = dict(root_x=-64.0, root_z=(112.0, 164.0))    # root post on the backpack side: x station, z span (world)
WING_FRONT = dict(root_x=-51.0, root_z=(142.0, 164.0))   # shallower: its lower chord must stay above the thigh's arc
LUG_IN_MM = 11.0             # nacelle lugs reach this far inboard of the nacelle's flank
POD_H6_MM = 20.0             # nacelle height
POD_WALL6_MM = 3.4

# v7 (after the first stress analysis of v6: strong enough, but 1.8 mm / 1.6 deg at full thrust and a 25 Hz first mode,
# because the nacelle hung off inboard lugs 37 mm from the thrust line): the wings now carry each nacelle at its tail and
# nose, on the thrust line, on seats two screws wide; the rear wings lie flat on the backpack's rear face.
TAB_R_MM = 15.0              # nacelle nose / tail tabs (discs under the ends, on the centre line)
SEAT_MM = 30.0               # length of a wing's seat under a tab; two screws 18 mm apart
WING_REAR7 = dict(root_y=9.0, root_z=(112.0, 166.0))     # root post on the backpack's REAR face, this far from the centre plane
WING_FRONT7 = dict(root_x=-52.0, root_z=(140.0, 166.0))  # root post on the backpack's side wall

# ---------------------------------------------------------------- frame v8: backpack + blade wings + nacelles ----
# The exposed trusses of v6/v7 read as industrial. v8 carries each nacelle on ONE sculpted hollow wing: a closed shell
# lofted through superelliptic sections, deep at the root on the backpack's side wall, flattening and widening into the
# nacelle's inboard flank. The closed section plus the infill inside it is the stiffening; nothing is exposed.
BLADE_ROOT_MM = dict(x=-68.0, z=141.0, a=13.0, b=18.0)   # root section centre (x, z) and half-sizes (a along x, b along z)
BLADE_TIP_MM = dict(a=27.0, b=10.0, dx=-14.0)   # dx: tip centre behind the nacelle's middle, so the wing stays out of the head's yaw sweep
                      # tip section: as tall as the nacelle allows, wide along its flank
BLADE_EXP = (2.6, 5.0)                                   # superellipse exponent of the sections: root, tip (the tip matches the nacelle's slab)
TONGUE_MM = dict(w=14.0, t=5.0, reach=14.0)              # tongue under the nacelle between the fans: two screws from below
