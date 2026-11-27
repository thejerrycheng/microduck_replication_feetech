# Flying MicroDuck: bill of materials

Design: `docs/duck_flight.pdf`. Prices and masses are as listed on the linked pages on 2026-11-25 unless marked
*(search snippet)*, which means the page itself could not be opened. **Taobao was not searched** (the browser
extension was not connected); the search terms at the end are for doing that by hand. Nothing has been ordered.

## Flight hardware

| # | Item | Qty | Listed | Link |
|---|---|---|---|---|
| 1 | QX-Motor 50 mm 12-blade EDF, QF2611 **5000KV**, 3-4S: **2 x CW + 2 x CCW**. 76 g, 3.0 mm shaft, 45 A / 550 W max, shroud OD 54.0 mm, length 39.1 mm | 4 (+1 spare) | US$25 each, US$100 per 4 | [qwinout](https://qwinout.com/products/qx-motor-qf2611-3300-4000kv-4600kv-5000kv-brushless-motor-cw-ccw-3-4s-50mm-12-paddle-edf-ducted-fan-for-fms-rc-plane-helicopter) |
| 1b | Same fan, spec and test table (780 g at 12.6 V / 32 A for 4600KV; 950 g at 16.2 V / 34 A for 4000KV) | - | - | [racer.lt](https://racer.lt/item/qx-motor-50mm-12-blades-ducted-fan-edf-cw-ccw-with-qf2611-3s-4s-brushless-motor-21984.html) |
| 1c | Same family, "Reverse" variant listing with shroud dimensions | - | US$25.93 | [rc-castle](https://www.rc-castle.com/index.php?route=product/product&product_id=8550) |
| 2 | **Flight controller + 4-in-1 ESC, bought as one stack: SpeedyBee F405 V4 + BLS 60A, 30.5 mm.** FC 41.6 x 39.4 x 7.8 mm, 10.5 g, STM32F405, ICM42688P, barometer, microSD blackbox, 6 UARTs, 5 V 3 A and 9 V 3 A BECs, 3-6S; ESC 45.6 x 44 x 8 mm, BLHeli_S, current sensor; joined by one 8-pin cable | 1 | US$75.99 (Team BlackSheep) to US$112.99 | [team-blacksheep](https://team-blacksheep.com/products/product:7493), [board page](https://betaflight.com/docs/wiki/boards/current/SPEEDYBEEF405V4) |
| 2a | Alternative ESC with a firmware current limit (AM32): Holybro Tekko32 F4 4in1 60A, 15.7 g. Its board size was not checked against the 47 x 48 mm bay | 1 | US$79.99, made to order | [holybro](https://holybro.com/products/tekko32-f4-4in1-60a-esc) |
| 2b | Alternative: T-Motor F55A Pro II 55 A AM32, 17.5 g | 1 | US$117.99, sold out when checked | [racedayquads](https://www.racedayquads.com/products/t-motor-f55a-pro-ii-55a-3-6s-am32-4-in-1-esc-w-bec-30x30) |
| 3 | Radio receiver for arming and the kill switch (ExpressLRS, CRSF on the FC's UART2) and its transmitter: not chosen yet | 1 | - | - |
| 4 | 4S 1300 mAh 100C pack, XT60: CNHL Black, 74 x 34 x 33.5 mm, 163 g (one listing says 183 g) *(search snippet)* | 2 | not shown | [chinahobbyline](https://chinahobbyline.com/products/cnhl-black-series-1300mah-14-8v-4s-100c-lipo-battery-with-xt60-plug) |
| 4b | Alternative pack: Tattu R-Line V3 4S 1300 mAh 120C, 77 x 38.5 x 27 mm, 150 g *(search snippet)*: needs the tray pocket changed | 2 | not shown | [progressiverc](https://www.progressiverc.com/products/tattu-r-line-v3-4s-1300mah-120c) |
| 5 | M3 brass heat-set inserts, M3 x 5.7 | 20 (pack of 100) | EUR 5.99 | [3djake](https://www.3djake.com/ruthex/threaded-insert-m3-100-pieces) |
| 6 | M3 socket-head screws: 12 x M3x8 (4 under the nacelles, 4 wing roots, 2 lid, 2 cap), 4 x M3x12 (through the wing tips into the nacelle flanks), 4 x M3x25 with rubber grommets (stack; length to suit the stack's standoffs); 4 x M2.5 grub screws (fan rings) | - | - | any hardware shop |
| 7 | 12 AWG silicone wire, XT60 pair, 20 mm hook-and-loop strap, 1000 uF 35 V low-ESR capacitor on the ESC input | - | - | any hobby shop |
| 8 | PLA filament, about 255 g for the eleven printed parts (3 walls, 15 % gyroid) plus supports | - | - | - |

Rough total for items 1-5 with two packs: about US$300.

Choices that follow from the analysis (`docs/duck_flight.pdf`):

- **5000KV, not 4000KV or the XFly 4600KV.** Sellers quote thrust at 16.2-16.5 V; a loaded 4S pack sits near 12 V.
  At that voltage the 5000KV fan gives thrust/weight 1.67, the 4000KV 1.30 and the XFly 1.15 (all with seller
  thrust derated to 80 %).
- **Set the ESC current limit to 45 A** (AM32): on a full 4S pack the 5000KV motor would otherwise exceed its
  45 A / 550 W rating.
- **Only QX-Motor was found selling both rotation directions.** Order two of each.
- **Measure before printing:** shroud outer diameter (54.0 mm assumed, from one seller) and the trunk shell
  (82 x 64 mm, underside 42 mm below the roof, from the official model). Both are single numbers in
  `scripts/tools/flight_config.py`.

## Bench equipment for system identification

| Item | Listed | Link |
|---|---|---|
| Load cell 10 kg, straight bar (TAL220) | US$12.95 | [sparkfun](https://www.sparkfun.com/load-cell-10kg-straight-bar-tal220.html) |
| HX711 load-cell amplifier (80 Hz: static thrust only) | US$4.95 | [sparkfun](https://www.sparkfun.com/products/13230) |
| Turn-key alternative: Tyto Robotics Series 1585 thrust stand (5 kgf, 55 A) | not shown | [tytorobotics](https://www.tytorobotics.com/pages/series-1580-1585) |

Rotor speed comes from the ESC (bidirectional DShot eRPM), current and voltage from the ESC's sensor through the
flight controller's blackbox log. Procedure and CSV formats: `scripts/tools/fan_sysid.py`.

## Taobao search terms (not yet searched)

- `QX-MOTOR 50mm 涵道 QF2611 5000KV 12叶 正反转`
- `SpeedyBee F405 V4 飞塔 60A`
- `格氏 / CNHL 4S 1300mAh 100C XT60`
- `M3 热熔铜螺母 5.7`

## Flight controller: wiring and mounting

- **Mounting.** The stack hangs under the backpack's removable lid on four M3 x 25 screws with rubber grommets
  (inserts in the lid's posts), flight controller next to the lid, ESC under it, battery below. The lid tucks a tab
  into the front wall and takes two M3 x 8 screws into the thick rear wall; the USB-C port faces a slot in the rear wall
  (slot height is an estimate: check it against the board).
- **Power.** Battery XT60 to the ESC's pads (12 AWG, 1000 uF capacitor across them); the ESC feeds the FC through the
  8-pin cable (battery voltage, current, telemetry, four motor signals). The duck keeps its own supply; only ground is
  shared. The FC's 5 V 3 A BEC could feed the duck's computer instead, but that has not been checked against its draw.
- **Link to the duck's computer (Radxa ZERO 3W, 3.3 V logic).** Three wires, no level shifter: FC `T3` to header
  pin 16 (UART4 RX), FC `R3` from pin 18 (UART4 TX), ground to pin 20. MSP at 115200 baud. Pins 8/10 (UART2) are the
  Radxa's debug console. **Not checked:** whether the Zero Robot HAT leaves pins 16/18 free. Fallback: a USB-C cable
  from the FC to the Radxa (MSP over USB).
- **Firmware.** Betaflight with MSP override: the receiver stays the RC source and owns ARM and the override switch;
  the duck's computer sends `MSP_SET_RAW_RC` for channels 1-4 (`msp_override_channels_mask = 15`) at 25-50 Hz
  ([guide](https://betaflight.com/docs/wiki/guides/current/MSP-Companion-Computer-Control)). Betaflight 2026.6 or later
  hands the channels back to the receiver 300 ms after the frames stop; older versions hold the last values.
- **Current limit.** The BLHeli_S ESC has no current limit of its own. Cap the motor output in Betaflight
  (`motor_output_limit`) at the value where the thrust stand shows 45 A per fan, or use the AM32 ESC (2a).
- Nothing here has been bench-tested.

## Printed parts

STEP and STL in `cad/flight/out/`; print-oriented STL in `cad/flight/print/stl/`; Bambu Studio projects from
`scripts/tools/flight_print_bambu.py`. Assembly guide: https://thejerrycheng.github.io/microduck_assembly.html.
Driver access to every screw: `scripts/tools/flight_tool_clearance.py`.

| Part | Qty | Mass PLA 15 % [g] | Print orientation |
|---|---|---|---|
| `backpack` | 1 | 69.3 | top face down |
| `top_lid` | 1 | 11.0 | top face down |
| `bottom_cap` | 1 | 10.5 | flat |
| `wing_left` | 1 | 36.1 | root face down |
| `wing_right` | 1 | 36.1 | root face down |
| `nacelle_left` | 1 | 38.0 | top face down |
| `nacelle_right` | 1 | 38.0 | top face down |
| `fan_collar` | 4 | 3.9 | flat |
