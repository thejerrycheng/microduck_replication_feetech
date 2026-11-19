# Duck (iPhone)

The servo studio and Duck Live on a phone: one screen, two modes. Design and measurements: `docs/duck_ios_app.pdf`.

```bash
firmware/studio.py --lan      # on the Mac; add --mock for no hardware
app/server.py --lan
cd ios && DEVELOPMENT_TEAM=<your team id> xcodegen generate && open Duck.xcodeproj     # run on the phone
```

The phone finds both servers over Bonjour; allow "local network" when iOS asks.

**Straight to the duck (no Mac).** `firmware/duck_serve.sh` (run once on the Mac, no root) copies the studio to
the duck and installs it as a systemd user service (`studio.py --duck --lan`, port 8770) that starts at every boot.
The app then finds the duck itself: over Bonjour, at the address it last answered on, at `radxa-zero3.local` on the
Wi-Fi, at `10.42.0.1` on the duck's own hotspot, or by sweeping port 8770 across the phone's /24 network (every 15 s
while nothing is found, and when Connect is tapped). Once found it connects the duck's computer and servo bus by
itself. With no duck in reach the app says "Duck can't be reached"; it never shows a command. A duck found this way
wins over a studio on the Mac. Tests against the real duck: `tools/test_connect.sh` (connection, read-only) and
`tools/check_codec.py` (camera and microphone decoding). The hotspot itself is
`firmware/net_install.sh` (also once, with sudo). On the duck the studio never takes the servos by itself: it leaves
the duck's runtime alone until Connect is tapped, and in Run mode it asks for Setup first.

| | |
|---|---|
| top bar | **Duck** (the real one, default) or **Sim**. Then that one's lights, the battery and the temperature (orange when warm, red when hot); tap them for details: host, the bus and computer switches, **Calibrate**, **Setup / Run**, the computer's load, the processor and hottest-servo temperatures, the battery voltage. **Connect** appears while the real duck is not connected. |
| 🐦 Duck | the duck as the studio draws it, and all 15 servos in a grid. Tap a servo (on the duck or in the grid) for its plots, joint angle, torque, slider with ‹ › nudges, Test, Zero, P / D (hold a gain for the replica's value). The round buttons on the left are the sensors. |
| 🎮 Play | left stick: walk forward / back, sidestep left / right. Right slider: turn. Speed slider, sit, run (straight only), jump, flip, the emotions. With **Duck**: also power. |

Play with **Sim** drives the simulated duck. Play with **Duck** drives the real one:

- **Setup** (the default): the policies trained for this duck, run on the duck by its pilot (`firmware/duck_pilot.py`).
  The same pad as Sim plus **power**: on (asks first: the legs move) the policies take the joints; off (asks first:
  the duck drops) every joint goes limp. The emotions are the duck's own animations (head, beak, sound) and play with
  the power off; tap the one playing to stop it. Needs the joints and the IMU calibrated and the policy pack on the
  duck (a Mac studio sends it once).
- **Run** (the Mode switch under the top bar): the factory runtime (Pollen's `robotd`, port 8090 on the duck). Power
  (the duck stands up), walk, the sticks, its skills. The wrench returns to Setup.

The IMU window (Duck tab) calibrates the IMU as the studio does: **1 Upright**, then **2 Beak up** (on its back),
then ↺ / X / Y / Z to turn the result if the front points the wrong way.

`tools/mock_studio.py` is the mock studio with a stand-in duck (the real pilot code and policies, a still IMU, no
servo) and `tools/mock_runtime.py` stands in for the factory runtime; `tools/runtime_probe.py` reads the real runtime
without powering a joint. The app has not yet moved the real duck.
`--lan` opens the server to everyone on the network: the studio has no login.
A real studio always wins over a `--mock` one; the tests launch the app with `-mockOnly 1` so they can never touch real servos.

```bash
ios/tools/test.sh             # simulator UI tests: light, dark, offline -> docs/figures/ios/, logs/ios/test.log
ios/tools/plot_tests.py       # -> docs/figures/duck_ios_drive.pdf, docs/duck_ios_numbers.tex
ios/tools/bench_stick.py      # how the engine follows the sticks -> docs/figures/duck_ios_stick.pdf
ios/tools/plot_pilot.py       # Play on the duck's own policies -> docs/figures/duck_ios_pilot.pdf
```

| File | |
|---|---|
| `lan.py` | Bonjour announcement used by both servers' `--lan` |
| `Duck/Link.swift` | discovery, the studio and engine clients, app state |
| `Duck/Scene.swift` | loads `firmware/web/duck.bin`, the SceneKit duck in its two looks |
| `Duck/Views.swift` | the screen |
| `DuckUITests/` | the tests |
