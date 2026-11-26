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
| 2 | 4-in-1 ESC, 30.5 mm, AM32, current sensor: Holybro Tekko32 F4 4in1 60A, 15.7 g | 1 | US$79.99, made to order | [holybro](https://holybro.com/products/tekko32-f4-4in1-60a-esc) |
| 2b | Alternative: T-Motor F55A Pro II 55 A AM32, 17.5 g | 1 | US$117.99, sold out when checked | [racedayquads](https://www.racedayquads.com/products/t-motor-f55a-pro-ii-55a-3-6s-am32-4-in-1-esc-w-bec-30x30) |
| 3 | Flight controller, 30.5 mm, SD-card blackbox: SpeedyBee F405 V4, 10.5 g *(search snippet)* | 1 | US$38-49 | [getfpv](https://www.getfpv.com/speedybee-f405-v4-flight-controller-30x30.html) |
| 4 | 4S 1300 mAh 100C pack, XT60: CNHL Black, 74 x 34 x 33.5 mm, 163 g (one listing says 183 g) *(search snippet)* | 2 | not shown | [chinahobbyline](https://chinahobbyline.com/products/cnhl-black-series-1300mah-14-8v-4s-100c-lipo-battery-with-xt60-plug) |
| 4b | Alternative pack: Tattu R-Line V3 4S 1300 mAh 120C, 77 x 38.5 x 27 mm, 150 g *(search snippet)*: needs the tray pocket changed | 2 | not shown | [progressiverc](https://www.progressiverc.com/products/tattu-r-line-v3-4s-1300mah-120c) |
| 5 | M3 brass heat-set inserts, M3 x 5.7 | 4 (pack of 100) | EUR 5.99 | [3djake](https://www.3djake.com/ruthex/threaded-insert-m3-100-pieces) |
| 6 | M3 socket screws: 4 x M3x25 (laps), 4 x M3x12 (ring pinch), 4 x M3x8 (stack) + 8 nuts; 2 x M2x8 self-tappers (the duck's BOM already has M2 self-tappers) | - | from US$0.94 per 20 *(search snippet; M3x25 not confirmed there)* | [bambulab](https://us.store.bambulab.com/products/m3-socket-head-cap-machine-screws-shcs-1) |
| 7 | 12 AWG silicone wire, XT60 pair, 20 mm hook-and-loop strap, 1000 uF 35 V low-ESR capacitor on the ESC input | - | - | any hobby shop |
| 8 | ABS or ASA filament, about 150 g for the five printed parts | - | - | - |

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
- `四合一电调 60A AM32 30.5`
- `F405 飞控 30.5 黑匣子`
- `格氏 / CNHL 4S 1300mAh 100C XT60`
- `M3 热熔铜螺母 5.7`

## Printed parts

STEP and STL in `cad/flight/out/`, drawings in `cad/flight/drawings/flight_frame_drawings.pdf`.

| Part | Qty | Print orientation |
|---|---|---|
| `frame_left`, `frame_right` | 1 each | top face on the bed, no supports, 3 perimeters, 15 % gyroid |
| `lift_hook_left`, `lift_hook_right` | 1 each | on the side, 100 % infill |
| `pack_tray` | 1 | on the side, no supports |
