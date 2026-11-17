# Duck (iPhone)

The servo studio and Duck Live on a phone: one screen, three modes. Design and measurements: `docs/duck_ios_app.pdf`.

```bash
firmware/studio.py --lan      # on the Mac; add --mock for no hardware
app/server.py --lan
cd ios && xcodegen generate && open Duck.xcodeproj     # choose your team under Signing, run on the phone
```

The phone finds both servers over Bonjour; allow "local network" when iOS asks.

| | |
|---|---|
| top pill | studio · servo bus · duck's computer · engine. Tap it for the two switches (bus, duck). |
| 🐦 Duck | the duck as the studio draws it. Tap a servo (on the duck or in the row) for its plots, torque, slider, Test, Zero, P / D (hold a gain for the replica's value). The round buttons on the left are the sensors. |
| 🎮 Play | joystick: forward / back walk, left / right turn. Sit, run (straight only), jump, flip, the speed level, the emotions. |
| 🧠 Policy | the export each skill runs; picking one loads it into the engine. |

Play and Policy drive the simulated duck (the engine); the real duck has no policy runtime here yet.
`--lan` opens the server to everyone on the network: the studio has no login.

```bash
ios/tools/test.sh             # simulator UI tests: light, dark, offline -> docs/figures/ios/, logs/ios/test.log
ios/tools/plot_tests.py       # -> docs/figures/duck_ios_drive.pdf, docs/duck_ios_numbers.tex
```

| File | |
|---|---|
| `lan.py` | Bonjour announcement used by both servers' `--lan` |
| `Duck/Link.swift` | discovery, the studio and engine clients, app state |
| `Duck/Scene.swift` | loads `firmware/web/duck.bin`, the SceneKit duck in its two looks |
| `Duck/Views.swift` | the screen |
| `DuckUITests/` | the tests |
