// One screen: the duck, a connection pill, three modes. Same glass, colours and spring as the servo studio
// (firmware/web/index.html).
import SwiftUI

let morph = Animation.spring(duration: 0.5, bounce: 0.28)

@main struct DuckApp: App {
    @State private var duck = Duck()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            Root().environment(duck)
                .task {
                    if let m = UserDefaults.standard.string(forKey: "mode").flatMap(Mode.init) { duck.mode = m }     // `-mode play`
                    duck.start()
                }
                .onChange(of: phase) { if phase != .active { duck.pause() } }
        }
    }
}

struct Root: View {
    @Environment(Duck.self) private var duck
    @Environment(\.colorScheme) private var scheme
    @State private var links = false
    @Namespace private var ns

    private var lift: CGFloat {
        switch duck.mode {
        case .duck: duck.sel == nil ? -10 : -118
        case .play: -86
        case .policy: -125
        }
    }

    var body: some View {
        ZStack {
            Backdrop()
            DuckView(duck: duck).offset(y: lift).ignoresSafeArea()
            if links { Color.clear.contentShape(Rectangle()).onTapGesture { links = false }.ignoresSafeArea() }
            VStack(spacing: 10) {
                LinkPill(open: $links)
                if duck.mode == .duck { if !duck.ids.isEmpty { Found(ns: ns).transition(.blurReplace) } }
                else if duck.liveUp { StateChip().transition(.blurReplace) }
                if let t = duck.toast { Text(t).font(.callout.weight(.medium)).padding(.horizontal, 16).padding(.vertical, 9)
                    .glassEffect(.regular, in: .capsule).transition(.blurReplace).accessibilityIdentifier("toast") }
                Spacer(minLength: 0)
                Group {
                    switch duck.mode {
                    case .duck: if duck.sel != nil { ServoCard() }
                    case .play: PlayCard()
                    case .policy: PolicyCard()
                    }
                }
                .transition(.blurReplace.combined(with: .scale(0.85, anchor: .bottom)))
                ModeBar(ns: ns)
            }
            .padding(.horizontal, 14).padding(.bottom, 4)
            if duck.mode == .duck && duck.sel == nil && !links {
                Dock().frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading).padding(.leading, 14)
                    .transition(.blurReplace.combined(with: .move(edge: .leading)))
            }
        }
        .animation(morph, value: duck.mode).animation(morph, value: duck.sel).animation(morph, value: duck.ids)
        .animation(morph, value: duck.toast).animation(morph, value: links).animation(morph, value: duck.liveUp).animation(morph, value: duck.open)
        .onChange(of: scheme, initial: true) { duck.dark = scheme == .dark }
    }
}

// ---- the look ----------------------------------------------------------------------------------------------------------
struct Backdrop: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let dark = scheme == .dark
        let hex = { (v: UInt32) in Color(red: Double(v >> 16 & 255) / 255, green: Double(v >> 8 & 255) / 255, blue: Double(v & 255) / 255) }
        GeometryReader { g in
            let r = max(g.size.width, g.size.height) * 0.62
            let glow = { (c: Color, x: Double, y: Double) in RadialGradient(colors: [c, c.opacity(0)], center: .init(x: x, y: y), startRadius: 0, endRadius: r) }
            ZStack {
                hex(dark ? 0x0b0b0d : 0xf5f5f7)
                glow(hex(dark ? 0x1c2a4d : 0xdfe9ff), 0.12, 0.08)
                glow(hex(dark ? 0x45291c : 0xffe9d6), 0.92, 0.18)
                glow(hex(dark ? 0x153828 : 0xdff7ea), 0.6, 1.05)
            }
        }
        .ignoresSafeArea()
    }
}

/// The studio's button: a soft pill that springs under the finger; blue when it is the thing to press.
struct Pill: ButtonStyle {
    var primary = false
    @Environment(\.isEnabled) private var enabled

    func makeBody(configuration c: Configuration) -> some View {
        c.label.font(.subheadline.weight(.medium)).lineLimit(1).fixedSize().padding(.horizontal, 15).padding(.vertical, 8)
            .foregroundStyle(primary ? .white : .primary)
            .background(primary ? AnyShapeStyle(.blue) : AnyShapeStyle(.fill.tertiary), in: .capsule)
            .opacity(enabled ? 1 : 0.35).scaleEffect(c.isPressed ? 0.92 : 1).animation(morph, value: c.isPressed)
    }
}

/// A round icon button (the play pad, the sensor dock).
struct Orb: ButtonStyle {
    var tint: Color?
    var size: CGFloat = 52
    @Environment(\.isEnabled) private var enabled

    func makeBody(configuration c: Configuration) -> some View {
        c.label.font(.system(size: size * 0.4, weight: .medium)).frame(width: size, height: size)
            .foregroundStyle(tint == nil ? AnyShapeStyle(.primary) : AnyShapeStyle(.white))
            .background(tint.map { AnyShapeStyle($0) } ?? AnyShapeStyle(.fill.tertiary), in: .circle)
            .opacity(enabled ? 1 : 0.3).scaleEffect(c.isPressed ? 0.88 : 1).animation(morph, value: c.isPressed)
    }
}

func nice(_ joint: String) -> String { joint.replacingOccurrences(of: "left_", with: "L ").replacingOccurrences(of: "right_", with: "R ").replacingOccurrences(of: "_", with: " ") }
func short(_ export: String) -> String {
    export.replacingOccurrences(of: "_hd1910", with: "").replacingOccurrences(of: ".onnx", with: "").replacingOccurrences(of: "iter", with: "").replacingOccurrences(of: "_", with: " ")
}

// ---- connections ---------------------------------------------------------------------------------------------------------
/// Four lights: the studio on the Mac, the servo bus, the duck's own computer, the engine. Tap: they open into rows
/// with the two switches.
struct LinkPill: View {
    @Environment(Duck.self) private var duck
    @Binding var open: Bool
    @Namespace private var ns

    private var joints: Int { duck.ids.filter { duck.names[$0] != nil }.count }
    private var busTint: Color { !duck.busOn ? .secondary : joints >= duck.names.count && joints > 0 ? .green : .orange }

    private func icon(_ name: String, _ id: String, _ tint: Color) -> some View {
        Image(systemName: name).font(.system(size: 16, weight: .medium)).foregroundStyle(tint).frame(width: 26, height: 24)
            .matchedGeometryEffect(id: id, in: ns)
    }

    private func row<T: View>(_ name: String, _ id: String, _ tint: Color, _ title: String, _ note: String, @ViewBuilder _ tail: () -> T) -> some View {
        HStack(spacing: 12) {
            icon(name, id, tint)
            Text(title).fontWeight(.semibold)
            Text(note).font(.footnote).foregroundStyle(.secondary).monospacedDigit().lineLimit(1)
            Spacer(minLength: 0)
            tail()
        }
        .frame(height: 38)
    }

    var body: some View {
        let on = { (v: Bool) in v ? Color.green : Color.secondary }
        Group {
            if open {
                VStack(spacing: 2) {
                    row("laptopcomputer", "studio", on(duck.studioUp), "Studio", duck.studioUp ? duck.studioHost : "studio.py --lan") { EmptyView() }
                    row("cable.connector", "bus", busTint, "Servos", duck.busOn ? "\(joints)/\(duck.names.count)" + (duck.mock ? " · mock" : "") : "") {
                        Toggle("", isOn: .init(get: { duck.busOn }, set: { _ in Task { await duck.toggleBus() } })).labelsHidden()
                            .disabled(!duck.studioUp || duck.busy).accessibilityIdentifier("bus")
                    }
                    row("cpu", "brain", on(duck.brainOn), "Duck", duck.brainOn ? "\(duck.brainHost) · \(Int(duck.brainTemp))°" : "") {
                        Toggle("", isOn: .init(get: { duck.brainOn }, set: { _ in Task { await duck.toggleBrain() } })).labelsHidden()
                            .disabled(!duck.studioUp || duck.busy).accessibilityIdentifier("brain")
                    }
                    row("gamecontroller.fill", "live", on(duck.liveUp), "Engine", duck.liveUp ? String(format: "%.2f×", duck.rate) : "server.py --lan") { EmptyView() }
                }
                .padding(.horizontal, 16).padding(.vertical, 10)
            } else {
                Button { open = true } label: {
                    HStack(spacing: 6) {
                        icon("laptopcomputer", "studio", on(duck.studioUp))
                        icon("cable.connector", "bus", busTint)
                        if duck.busOn { Text("\(joints)/\(duck.names.count)").font(.footnote.weight(.semibold)).monospacedDigit().foregroundStyle(.secondary) }
                        icon("cpu", "brain", on(duck.brainOn))
                        icon("gamecontroller.fill", "live", on(duck.liveUp))
                    }
                    .padding(.horizontal, 14).padding(.vertical, 10).contentShape(.capsule)
                }
                .buttonStyle(.plain).accessibilityIdentifier("link")
                .accessibilityValue("studio=\(duck.studioUp) bus=\(duck.busOn) servos=\(joints) brain=\(duck.brainOn) engine=\(duck.liveUp)")
            }
        }
        .glassEffect(.regular, in: .rect(cornerRadius: open ? 26 : 22))
        .padding(.top, 4)
    }
}

/// The servos on the bus, as the studio's row under its bar: tap one; one highlight slides between them.
struct Found: View {
    @Environment(Duck.self) private var duck
    let ns: Namespace.ID

    var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 2) {
                ForEach(duck.ids, id: \.self) { id in
                    let sel = duck.sel == id
                    Button { duck.open = nil; duck.sel = sel ? nil : id } label: {
                        HStack(spacing: 7) {
                            Circle().fill(sel ? .white : duck.motors[id] == nil ? .red : duck.names[id] == nil ? .orange : .green).frame(width: 8, height: 8)
                            Text(String(id)).font(.system(size: 16, weight: .semibold)).monospacedDigit()
                        }
                        .padding(.horizontal, 12).padding(.vertical, 7).foregroundStyle(sel ? .white : .primary)
                        .background { if sel { Capsule().fill(.blue).matchedGeometryEffect(id: "found", in: ns) } }
                        .contentShape(.capsule)
                    }
                    .buttonStyle(.plain).accessibilityIdentifier("servo.\(id)").accessibilityValue(sel ? "selected" : "")
                }
            }
            .padding(6)
        }
        .scrollIndicators(.hidden).clipShape(.capsule).glassEffect(.regular, in: .capsule)
    }
}

// ---- a servo: what it is doing, and its calibration ----------------------------------------------------------------------
struct ServoCard: View {
    @Environment(Duck.self) private var duck
    @State private var drag: Double?
    @State private var p = 0
    @State private var d = 0
    @State private var asking = false

    var body: some View {
        let id = duck.sel ?? 0, m = duck.motors[id]
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Text(String(id)).font(.system(size: 40, weight: .bold)).monospacedDigit().contentTransition(.numericText())
                VStack(alignment: .leading, spacing: 1) {
                    Text(duck.names[id].map(nice) ?? "unassigned").fontWeight(.medium)
                    if let m {
                        Text(verbatim: "\(Int(m.pos)) · \(String(format: "%.1f", m.volt)) V").monospacedDigit().foregroundStyle(.secondary)
                            .accessibilityIdentifier("pos").accessibilityValue(Text(verbatim: String(Int(m.torque ? m.goal : m.pos))))
                    }
                }
                .font(.subheadline).lineLimit(1)
                Spacer(minLength: 0)
                Button("Test") { Task { await duck.wiggle() } }.accessibilityIdentifier("test")
                Button("Zero") { asking = true }.accessibilityIdentifier("zero")
            }
            .buttonStyle(Pill())
            let tr = duck.tick >= 0 ? duck.traces[id] ?? Trace() : Trace()
            Grid(horizontalSpacing: 8, verticalSpacing: 8) {
                GridRow { Spark(title: "Position", unit: "°", digits: 1, n: tr.n, t: tr.t, v: tr.pos); Spark(title: "Velocity", unit: "°/s", digits: 0, t: tr.t, v: tr.vel) }
                GridRow { Spark(title: "Load", unit: "%", digits: 0, t: tr.t, v: tr.load); Spark(title: "Temp", unit: "°C", digits: 0, t: tr.t, v: tr.temp) }
            }
            if let m {
                HStack(spacing: 14) {
                    Toggle("", isOn: .init(get: { m.torque }, set: { on in Task { await duck.torque(on) } })).labelsHidden().accessibilityIdentifier("torque")
                    Slider(value: .init(get: { drag ?? (m.torque ? m.goal : m.pos) }, set: { drag = $0; duck.move($0) }), in: m.lo...max(m.lo + 1, m.hi)) { editing in
                        if !editing, let v = drag { duck.move(v, last: true); Task { try? await Task.sleep(for: .milliseconds(400)); drag = nil } }
                    }
                    .accessibilityIdentifier("slide")
                }
            }
            HStack(spacing: 8) {
                Nudge(name: "P", v: $p); Nudge(name: "D", v: $d)
                Spacer(minLength: 0)
                Button("Set") { Task { await duck.setGains(p, d) } }.buttonStyle(Pill(primary: true))
                    .disabled(duck.gains == nil || (duck.gains?.p == p && duck.gains?.d == d)).accessibilityIdentifier("gset")
            }
            .buttonStyle(Pill())
        }
        .padding(18)
        .glassEffect(.regular, in: .rect(cornerRadius: 28))
        .onChange(of: duck.gains, initial: true) { if let g = duck.gains { p = g.p; d = g.d } }
        .confirmationDialog("Make this pose read 2048?", isPresented: $asking, titleVisibility: .visible) {
            Button("Zero \(id)", role: .destructive) { Task { await duck.zero() } }
        }
        .accessibilityElement(children: .contain).accessibilityIdentifier("card.servo")
    }
}

/// A gain: its value between − and +. Hold the value to fill the replica's gain (P 5, D 20, beak P 10).
struct Nudge: View {
    @Environment(Duck.self) private var duck
    let name: String
    @Binding var v: Int

    var body: some View {
        HStack(spacing: 0) {
            Button { v = max(0, v - 1) } label: { Image(systemName: "minus").frame(width: 26, height: 34) }.accessibilityIdentifier("\(name).minus")
            Text(verbatim: "\(name) \(v)").font(.subheadline.weight(.semibold)).monospacedDigit().frame(minWidth: 40).contentTransition(.numericText())
                .onLongPressGesture { if let g = duck.gains { withAnimation { v = name == "P" ? g.duckP : g.duckD } } }
                .accessibilityIdentifier("gain.\(name)").accessibilityValue(String(v))
            Button { v = min(254, v + 1) } label: { Image(systemName: "plus").frame(width: 26, height: 34) }.accessibilityIdentifier("\(name).plus")
        }
        .buttonStyle(.plain).font(.footnote.weight(.semibold)).background(.fill.tertiary, in: .capsule)
    }
}

/// One quantity over the last ten seconds, the way the studio plots it: name, value now, a line.
struct Spark: View {
    let title: String, unit: String, digits: Int
    var n = 0                       // samples received so far
    let t: [Double], v: [Double]

    var body: some View {
        Canvas { c, size in
            guard let end = t.last, v.count == t.count else { return }
            let i0 = t.firstIndex { $0 >= end - 10 } ?? 0
            guard t.count - i0 > 1 else { return }
            var lo = v[i0...].min()!, hi = v[i0...].max()!
            let pad = max((hi - lo) * 0.12, 1); lo -= pad; hi += pad
            let top = 26.0, x = { (s: Double) in 8 + (s - (end - 10)) / 10 * (size.width - 16) }
            let y = { (s: Double) in top + (hi - s) / (hi - lo) * (size.height - top - 8) }
            var path = Path()
            let step = max(1, (t.count - i0) / Int(size.width))
            for i in stride(from: i0, to: t.count, by: step) { let pt = CGPoint(x: x(t[i]), y: y(v[i])); i == i0 ? path.move(to: pt) : path.addLine(to: pt) }
            path.addLine(to: CGPoint(x: x(end), y: y(v[v.count - 1])))
            c.stroke(path, with: .color(.blue), style: .init(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }
        .frame(height: 62)
        .overlay(alignment: .top) {
            HStack {
                Text(title).foregroundStyle(.secondary)
                Spacer()
                if let x = v.last { Text("\(x.formatted(.number.precision(.fractionLength(digits)))) \(unit)").fontWeight(.semibold).monospacedDigit() }
            }
            .font(.caption).padding(.horizontal, 10).padding(.top, 7)
        }
        .background(.fill.tertiary, in: .rect(cornerRadius: 14))
        .accessibilityElement().accessibilityIdentifier("plot.\(title)").accessibilityValue(String(n))
    }
}

// ---- the duck's sensors: round buttons down the left edge; the one that is on grows into its window -----------------------
struct Dock: View {
    @Environment(Duck.self) private var duck
    static let icons = ["camera": "video.fill", "tof": "dot.radiowaves.right", "imu": "gyroscope", "mic": "mic.fill", "speaker": "speaker.wave.2.fill"]
    static let voice = [("chirp", "Quack"), ("greet", "Greet"), ("coo", "Coo"), ("peck", "Peck"), ("inquire", "Ask"), ("alarm", "Alarm")]
    static let titles = ["camera": "Camera", "tof": "ToF", "imu": "IMU", "mic": "Mic", "speaker": "Speaker"]

    private func note(_ k: String) -> String {
        if let s = duck.sensors[k], !s.ok { return s.note }
        switch k {
        case "camera": return duck.camera.map { "\($0.width)×\($0.height)" } ?? ""
        case "tof": return duck.tof.min().map { String(format: "%.2f m", $0 / 1000) } ?? ""
        case "imu": return String(format: "%.0f° %.0f°", duck.roll, duck.pitch)
        case "mic": return duck.wave.isEmpty ? "" : String(format: "%.0f dB", duck.micLevel)
        default: return ""
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(SENSORS, id: \.self) { k in
                let on = duck.open == k, ok = duck.sensors[k]?.ok
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: 2) {
                        Image(systemName: Self.icons[k]!).font(.system(size: 17, weight: .medium)).frame(width: 46, height: 46)
                            .foregroundStyle(on ? .blue : ok == nil ? .secondary : ok! ? .green : .red)
                        if on {
                            Text(Self.titles[k]!).fontWeight(.semibold)
                            Spacer(minLength: 4)
                            Text(note(k)).font(.caption).monospacedDigit().lineLimit(1).foregroundStyle(ok == false ? .red : .secondary).padding(.trailing, 14)
                                .accessibilityIdentifier("note.\(k)")
                        }
                    }
                    if on { window(k).padding([.horizontal, .bottom], 10).transition(.blurReplace) }
                }
                .frame(width: on ? 214 : 46, alignment: .leading)
                .contentShape(.rect(cornerRadius: 23))
                .onTapGesture { if duck.brainOn { duck.open = on ? nil : k } else { duck.say("Duck offline") } }
                .glassEffect(.regular, in: .rect(cornerRadius: 23))
                .accessibilityElement(children: .contain).accessibilityAddTraits(.isButton)
                .accessibilityIdentifier("sensor.\(k)").accessibilityValue(ok == nil ? "off" : ok! ? "ok" : "bad")
            }
        }
    }

    @ViewBuilder private func window(_ k: String) -> some View {
        switch k {
        case "camera":
            ZStack {
                Rectangle().fill(.fill.tertiary)
                if let img = duck.camera { Image(decorative: img, scale: 1).resizable().accessibilityIdentifier("picture") }
            }
            .aspectRatio(4 / 3, contentMode: .fit).clipShape(.rect(cornerRadius: 12))
        case "tof":
            Canvas { c, size in                               // 8 x 8 distances, near = warm
                let s = size.height / 8, x0 = (size.width - size.height) / 2
                for (i, mm) in duck.tof.prefix(64).enumerated() {
                    let r = CGRect(x: x0 + Double(i % 8) * s + 1.5, y: Double(i / 8) * s + 1.5, width: s - 3, height: s - 3)
                    c.fill(Path(roundedRect: r, cornerRadius: 4), with: .color(Color(hue: (18 + 200 * min(1, mm / 2000)) / 360, saturation: 0.85, brightness: 0.9)))
                }
            }
            .frame(height: 150).background(.fill.tertiary, in: .rect(cornerRadius: 12))
        case "imu":
            Canvas { c, size in                               // an artificial horizon
                var h = c
                h.translateBy(x: size.width / 2, y: size.height / 2); h.rotate(by: .degrees(-duck.roll)); h.translateBy(x: 0, y: duck.pitch * 1.5)
                h.fill(Path(CGRect(x: -size.width, y: -size.height * 2, width: size.width * 2, height: size.height * 2)), with: .color(.blue.opacity(0.28)))
                h.stroke(Path { $0.move(to: .init(x: -size.width, y: 0)); $0.addLine(to: .init(x: size.width, y: 0)) }, with: .color(.blue), lineWidth: 2)
                let mark = Path { p in for sx in [-1.0, 1.0] { p.move(to: .init(x: size.width / 2 + sx * 12, y: size.height / 2)); p.addLine(to: .init(x: size.width / 2 + sx * 36, y: size.height / 2)) } }
                c.stroke(mark, with: .color(.primary), style: .init(lineWidth: 3, lineCap: .round))
            }
            .frame(height: 70).background(.fill.tertiary, in: .rect(cornerRadius: 12)).clipShape(.rect(cornerRadius: 12))
        case "mic":
            Canvas { c, size in
                guard duck.wave.count > 1 else { return }
                var p = Path()
                for (i, a) in duck.wave.enumerated() {
                    let x: Double = 6 + Double(i) * (size.width - 12) / Double(duck.wave.count - 1)
                    let amp: Double = max(-1, min(1, a * 3))
                    let pt = CGPoint(x: x, y: size.height / 2 - amp * (size.height / 2 - 6))
                    i == 0 ? p.move(to: pt) : p.addLine(to: pt)
                }
                c.stroke(p, with: .color(.blue), style: .init(lineWidth: 2, lineCap: .round, lineJoin: .round))
            }
            .frame(height: 56).background(.fill.tertiary, in: .rect(cornerRadius: 12))
        default:
            Grid(horizontalSpacing: 6, verticalSpacing: 6) {             // the duck's voice, as in the studio
                ForEach(0..<2, id: \.self) { r in
                    GridRow {
                        ForEach(Self.voice[r * 3..<r * 3 + 3], id: \.0) { tag, name in
                            Button { Task { await duck.voice(tag) } } label: { Text(name).font(.caption.weight(.medium)).frame(maxWidth: .infinity).padding(.vertical, 8).background(.fill.tertiary, in: .capsule) }
                                .buttonStyle(.plain).accessibilityIdentifier("say.\(tag)")
                        }
                    }
                }
            }
        }
    }
}

// ---- play: joystick, four moves, the emotions ------------------------------------------------------------------------------
let STATES: [String: (String, Color)] = [
    "stand": ("Standing", .green), "walk": ("Walking", .blue), "run": ("Running", .orange), "sprint": ("Sprinting", .orange),
    "stopping": ("Stopping", .teal), "sit_down": ("Sitting down", .orange), "seated": ("Seated", .orange), "stand_up": ("Standing up", .orange),
    "jump": ("Jump", .pink), "roulade": ("Front flip", .pink), "recover": ("Getting up", .pink), "fallen": ("Fell", .red), "dead": ("Playing dead", .gray)]

struct StateChip: View {
    @Environment(Duck.self) private var duck

    var body: some View {
        let s = STATES[duck.state] ?? (duck.state, .gray), e = duck.emotes.first { $0.name == duck.playing }
        HStack(spacing: 9) {
            Circle().fill(duck.down ? .red : s.1).frame(width: 9, height: 9)
            Text(e.map { "\($0.emoji) \($0.label)" } ?? s.0).fontWeight(.semibold).contentTransition(.interpolate)
            Text(String(format: "%.2f m/s", duck.speed)).font(.footnote).monospacedDigit().foregroundStyle(.secondary)
            Button { duck.press("enter") } label: { Image(systemName: "arrow.counterclockwise").font(.footnote.weight(.semibold)).frame(width: 28, height: 28) }
                .buttonStyle(.plain).foregroundStyle(.secondary).accessibilityIdentifier("reset")
        }
        .padding(.leading, 14).padding(.trailing, 6).padding(.vertical, 4)
        .glassEffect(.regular, in: .capsule)
        .animation(morph, value: duck.state).animation(morph, value: duck.playing)
        .accessibilityElement(children: .contain).accessibilityIdentifier("state")
        .accessibilityLabel(duck.state).accessibilityValue("\(duck.odometer) \(duck.playing)")
    }
}

struct Offline: View {
    let hint: String
    var body: some View {
        HStack(spacing: 10) { Image(systemName: "wifi.slash"); Text(hint).font(.footnote.monospaced()) }
            .foregroundStyle(.secondary).padding(.horizontal, 20).padding(.vertical, 14).glassEffect(.regular, in: .capsule)
            .accessibilityElement(children: .combine).accessibilityIdentifier("offline")
    }
}

struct PlayCard: View {
    @Environment(Duck.self) private var duck

    private func act(_ icon: String, _ key: String, _ id: String, _ can: Bool = true) -> some View {
        Button { duck.press(key) } label: { Image(systemName: icon) }.buttonStyle(Orb()).disabled(!can).accessibilityIdentifier("act.\(id)")
    }

    var body: some View {
        if !duck.liveUp { Offline(hint: "app/server.py --lan") } else {
            VStack(spacing: 14) {
                ScrollView(.horizontal) {
                    HStack(spacing: 8) {
                        ForEach(duck.emotes) { e in
                            let on = duck.playing == e.name
                            Button { duck.emote(e.name) } label: {
                                Text(e.emoji).font(.system(size: 23)).frame(width: 46, height: 46)
                                    .background(on ? AnyShapeStyle(.blue.opacity(0.22)) : AnyShapeStyle(.fill.tertiary), in: .circle)
                                    .overlay { if on { Circle().trim(from: 0, to: duck.progress).stroke(.blue, style: .init(lineWidth: 3, lineCap: .round)).rotationEffect(.degrees(-90)).padding(1.5) } }
                                    .scaleEffect(on ? 1.08 : 1)
                            }
                            .buttonStyle(.plain).disabled(!e.ok).opacity(e.ok ? 1 : 0.3)
                            .accessibilityIdentifier("emote.\(e.name)").accessibilityLabel(e.label)
                        }
                    }
                    .padding(.horizontal, 18).padding(.vertical, 3)
                }
                .scrollIndicators(.hidden).padding(.horizontal, -18).animation(morph, value: duck.playing)
                HStack(spacing: 22) {
                    Stick(tint: duck.run ? .orange : .blue) { x, y in duck.drive(x, y) }
                    VStack(spacing: 12) {
                        Grid(horizontalSpacing: 12, verticalSpacing: 12) {
                            GridRow {
                                act(duck.state == "seated" || duck.state == "sit_down" ? "figure.stand" : "figure.seated.side.right", "s", "sit")
                                Button { duck.run.toggle() } label: { Image(systemName: "hare.fill") }.buttonStyle(Orb(tint: duck.run ? .orange : nil))
                                    .accessibilityIdentifier("act.run").accessibilityValue(duck.run ? "on" : "off")
                            }
                            GridRow {
                                act("arrow.up.to.line", "space", "jump", duck.has["jump"] ?? false)
                                act("arrow.trianglehead.2.clockwise.rotate.90", "r", "flip", duck.has["roulade"] ?? false)
                            }
                        }
                        Level()
                    }
                }
            }
            .padding(18)
            .glassEffect(.regular, in: .rect(cornerRadius: 28))
        }
    }
}

/// The speed level, one to five: tap one.
struct Level: View {
    @Environment(Duck.self) private var duck

    var body: some View {
        let tint: Color = duck.run ? .orange : .blue
        HStack(spacing: 4) {
            ForEach(1...5, id: \.self) { i in
                Capsule().fill(i <= duck.level ? tint : Color.secondary.opacity(0.25)).frame(height: 7)
                    .frame(height: 26).contentShape(Rectangle()).onTapGesture { duck.setLevel(i) }
                    .accessibilityElement().accessibilityAddTraits(.isButton).accessibilityIdentifier("level.\(i)")
            }
        }
        .frame(width: 116).animation(morph, value: duck.level)
        .accessibilityElement(children: .contain).accessibilityIdentifier("level").accessibilityValue("\(duck.level)")
    }
}

/// The joystick. Push it where the duck should go: forward and back walk, left and right turn; let go and it stops.
struct Stick: View {
    var tint: Color
    var move: (Double, Double) -> Void
    @State private var k = CGSize.zero
    private let reach = 46.0

    var body: some View {
        ZStack {
            Circle().fill(.fill.tertiary)
            ForEach(0..<4, id: \.self) { i in
                Image(systemName: "chevron.up").font(.footnote.weight(.bold)).foregroundStyle(.tertiary).offset(y: -58).rotationEffect(.degrees(Double(i) * 90))
            }
            Circle().fill(tint.gradient).frame(width: 62, height: 62).shadow(color: tint.opacity(0.45), radius: k == .zero ? 4 : 12, y: 3)
                .scaleEffect(k == .zero ? 1 : 1.08).offset(k)
        }
        .frame(width: 152, height: 152).contentShape(.circle)
        .gesture(DragGesture(minimumDistance: 0)
            .onChanged { g in
                let d = hypot(g.translation.width, g.translation.height), s = d > reach ? reach / d : 1
                k = CGSize(width: g.translation.width * s, height: g.translation.height * s)
                move(k.width / reach, -k.height / reach)
            }
            .onEnded { _ in withAnimation(morph) { k = .zero }; move(0, 0) })
        .accessibilityElement().accessibilityIdentifier("stick")
    }
}

// ---- policies: which export each skill runs ------------------------------------------------------------------------------
struct PolicyCard: View {
    @Environment(Duck.self) private var duck
    static let rows: [(String, String, String)] = [
        ("walk", "figure.walk", "Walk"), ("run", "figure.run", "Run"), ("sprint", "hare.fill", "Sprint"), ("sitstand", "figure.seated.side.right", "Sit"),
        ("jump", "arrow.up.to.line", "Jump"), ("roulade", "arrow.trianglehead.2.clockwise.rotate.90", "Flip"), ("recovery", "figure.stand", "Get up")]

    var body: some View {
        if !duck.liveUp { Offline(hint: "app/server.py --lan") } else {
            VStack(spacing: 0) {
                ForEach(Self.rows, id: \.0) { role, icon, title in
                    let files = duck.policies[role] ?? [], now = duck.chosen[role]
                    Menu {
                        ForEach(files, id: \.self) { f in
                            Button { Task { await duck.setPolicy(role, f) } } label: { if f == now { Label(short(f), systemImage: "checkmark") } else { Text(short(f)) } }
                        }
                        if OPTIONAL.contains(role) { Divider(); Button("Off", role: .destructive) { Task { await duck.setPolicy(role, "") } } }
                    } label: {
                        HStack(spacing: 12) {
                            Image(systemName: icon).font(.system(size: 16, weight: .medium)).frame(width: 26).foregroundStyle(now == nil ? AnyShapeStyle(.secondary) : AnyShapeStyle(.blue))
                            Text(title).fontWeight(.semibold).foregroundStyle(.primary)
                            Spacer(minLength: 8)
                            Text(now.map(short) ?? "—").font(.footnote).monospacedDigit().foregroundStyle(.secondary).lineLimit(1).contentTransition(.interpolate)
                            Image(systemName: "chevron.up.chevron.down").font(.caption2.weight(.semibold)).foregroundStyle(.tertiary)
                        }
                        .frame(height: 42).contentShape(Rectangle())
                    }
                    .buttonStyle(.plain).disabled(files.isEmpty).accessibilityIdentifier("policy.\(role)").accessibilityValue(now.map(short) ?? "off")
                    if role != "recovery" { Divider().padding(.leading, 38) }
                }
            }
            .padding(.horizontal, 18).padding(.vertical, 8)
            .glassEffect(.regular, in: .rect(cornerRadius: 28))
        }
    }
}

// ---- the three modes ----------------------------------------------------------------------------------------------------
struct ModeBar: View {
    @Environment(Duck.self) private var duck
    let ns: Namespace.ID
    static let icons: [Mode: String] = [.duck: "bird.fill", .play: "gamecontroller.fill", .policy: "brain.fill"]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(Mode.allCases, id: \.self) { m in
                let on = duck.mode == m
                Button { duck.mode = m } label: {
                    Image(systemName: Self.icons[m]!).font(.system(size: 19, weight: .medium)).frame(width: 72, height: 44)
                        .foregroundStyle(on ? .white : .secondary)
                        .background { if on { Capsule().fill(.blue).matchedGeometryEffect(id: "mode", in: ns) } }
                        .contentShape(.capsule)
                }
                .buttonStyle(.plain).accessibilityIdentifier("mode.\(m.rawValue)").accessibilityValue(on ? "selected" : "")
            }
        }
        .padding(5).glassEffect(.regular, in: .capsule)
    }
}
