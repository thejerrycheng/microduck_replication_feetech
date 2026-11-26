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
FAN_LY = 0.102           # fan centre, lateral half-spacing
FAN_Z = 0.172            # clamp-ring mid-plane: the spars run over the trunk roof (z = 162 mm)
FAN_CANT_DEG = 6.0       # each duct tilted about its radial line: yaw torque from differential thrust
BATT_POS = (-0.0665, 0.0, 0.124)   # flight pack centre, upright on the back of the trunk
BATT_SIZE = (0.034, 0.0335, 0.074)  # CNHL 4S 1300 mAh
FC_POS = (-0.013, 0.0, 0.189)     # FC + 4-in-1 ESC stack on the spars

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
