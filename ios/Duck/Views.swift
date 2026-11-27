// One screen: the duck, a connection pill, two modes. Same glass, colours and spring as the servo studio
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
                    if let m = UserDefaults.standard.string(forKey: "mode").flatMap(Mode.init) { duck.mode = m }     // `-mode play -target sim`
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

    private var real: Bool { duck.target == .real }
    private var lift: CGFloat {
        if real { return duck.mode == .play ? (duck.realPlay ? -105 : -24) : duck.sel != nil ? -84 : duck.busOn ? 34 : 0 }
        return duck.mode == .play ? (duck.liveUp ? -105 : -24) : 0
    }

    var body: some View {
        ZStack {
            Backdrop()
            DuckView(duck: duck).offset(y: lift).ignoresSafeArea()
            if links { Color.clear.contentShape(Rectangle()).onTapGesture { links = false }.ignoresSafeArea() }
            VStack(spacing: 10) {
                LinkPill(open: $links)
                if real {
                    if duck.mode == .duck && duck.busOn { Found(ns: ns).transition(.blurReplace) }
                    if duck.mode == .duck && duck.calOn { CalBar().transition(.blurReplace) }
                    if duck.mode == .play && duck.realPlay { if duck.runMode == "run" { RunChip().transition(.blurReplace) } else { PilotChip().transition(.blurReplace) } }
                }
                else if duck.liveUp { StateChip().transition(.blurReplace) }
                if let t = duck.toast { Text(t).font(.callout.weight(.medium)).padding(.horizontal, 16).padding(.vertical, 9)
                    .glassEffect(.regular, in: .capsule).transition(.blurReplace).accessibilityIdentifier("toast") }
                Spacer(minLength: 0)
                Group {
                    switch (duck.mode, duck.target) {
                    case (.duck, .real): if duck.sel != nil { ServoCard() }
                    case (.duck, .sim): if !duck.liveUp { Offline(hint: "Simulator not found") }
                    case (.play, .sim): PlayCard()
                    case (.play, .real): RealPlay()
                    }
                }
                .transition(.blurReplace.combined(with: .scale(0.85, anchor: .bottom)))
                ModeBar(ns: ns)
            }
            .padding(.horizontal, 14).padding(.bottom, 4)
            if real && duck.mode == .duck && duck.sel == nil && !links {
                Dock().frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading).padding(.leading, 14)
                    .transition(.blurReplace.combined(with: .move(edge: .leading)))
            }
        }
        .animation(morph, value: duck.mode).animation(morph, value: duck.target).animation(morph, value: duck.sel).animation(morph, value: duck.busOn).animation(morph, value: duck.calOn)
        .animation(morph, value: duck.toast).animation(morph, value: links).animation(morph, value: duck.liveUp).animation(morph, value: duck.open)
        .animation(morph, value: duck.rtUp).animation(morph, value: duck.runMode).animation(morph, value: duck.rtReady).animation(morph, value: duck.brainOn).animation(morph, value: duck.pilotOn)
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
// ---- connections ---------------------------------------------------------------------------------------------------------
/// The top bar. Left: what the app talks to, the real duck or the simulated one. Middle: that one's lights (real:
/// the studio on the Mac, the servo bus, the duck's computer; sim: the engine); tap them for the details and the two
/// switches. Right, while the real duck is not connected: Connect.
struct LinkPill: View {
    @Environment(Duck.self) private var duck
    @Binding var open: Bool

    private var joints: Int { duck.ids.filter { duck.names[$0] != nil }.count }
    private var busTint: Color { !duck.busOn ? .secondary : joints >= duck.names.count && joints > 0 ? .green : .orange }
    private func on(_ v: Bool) -> Color { v ? .green : .secondary }
    private func warm(_ v: Double, _ lim: (Double, Double)) -> Color { v >= lim.1 ? .red : v >= lim.0 ? .orange : .secondary }
    private var heatTint: Color { duck.heat == 2 ? .red : duck.heat == 1 ? .orange : .secondary }
    /// The temperature to show in the bar: the processor's when the computer is linked, else the hottest servo's.
    private var shown: Double? { duck.brainOn ? duck.brainTemp : duck.hottest?.temp }
    private func charge(_ b: Int) -> Color { b < 15 ? .red : b < 40 ? .orange : .secondary }
    private func cell(_ b: Int) -> String { "battery.\(b >= 88 ? 100 : b >= 63 ? 75 : b >= 38 ? 50 : b >= 13 ? 25 : 0)percent" }

    private func icon(_ name: String, _ tint: Color) -> some View {
        Image(systemName: name).font(.system(size: 15, weight: .medium)).foregroundStyle(tint).frame(width: 20, height: 24)
    }

    private func row<T: View>(_ name: String, _ tint: Color, _ title: String, _ note: String, @ViewBuilder _ tail: () -> T) -> some View {
        HStack(spacing: 12) {
            icon(name, tint)
            Text(title).fontWeight(.semibold)
            Text(note).font(.footnote).foregroundStyle(.secondary).monospacedDigit().lineLimit(1)
            Spacer(minLength: 0)
            tail()
        }
        .frame(height: 38)
    }

    var body: some View {
        @Bindable var duck = duck
        let real = duck.target == .real, full = duck.busOn && duck.brainOn      // room for the numbers once Connect is gone
        VStack(spacing: 4) {
            HStack(spacing: 8) {
                Picker("Target", selection: $duck.target) {
                    Text("Duck").tag(Target.real)
                    Text("Sim").tag(Target.sim)
                }
                .pickerStyle(.segmented).frame(width: 100).accessibilityIdentifier("target").accessibilityValue(duck.target.rawValue)
                Button { open.toggle() } label: {
                    HStack(spacing: 4) {
                        if real {
                            icon("laptopcomputer", on(duck.studioUp)); icon("cable.connector", busTint)
                            if full { Text(verbatim: "\(joints)/\(duck.names.count)").font(.footnote.weight(.semibold)).monospacedDigit().foregroundStyle(.secondary).fixedSize() }
                            icon("cpu", on(duck.brainOn))
                            if let b = duck.battery {                 // charge left; orange under 40 %, red under 15 %
                                HStack(spacing: 3) {
                                    Image(systemName: cell(b)).font(.system(size: 17, weight: .medium))
                                    if full { Text(verbatim: "\(b)%").font(.footnote.weight(.semibold)).monospacedDigit().fixedSize() }
                                }
                                .foregroundStyle(charge(b)).padding(.leading, 2)
                                .accessibilityElement().accessibilityIdentifier("battery").accessibilityValue(Text(verbatim: String(b)))
                            }
                            if let t = shown {                        // temperature; orange when warm, red when hot (processor or a servo)
                                HStack(spacing: 1) {
                                    Image(systemName: duck.heat == 2 ? "thermometer.high" : duck.heat == 1 ? "thermometer.medium" : "thermometer.low").font(.system(size: 15, weight: .medium))
                                    if full { Text(verbatim: "\(Int(t))°").font(.footnote.weight(.semibold)).monospacedDigit().fixedSize() }
                                }
                                .foregroundStyle(heatTint)
                                .accessibilityElement().accessibilityIdentifier("temp").accessibilityValue(Text(verbatim: "\(Int(t)) heat=\(duck.heat)"))
                            }
                        } else {
                            icon("gamecontroller.fill", on(duck.liveUp))
                            if duck.liveUp { Text(verbatim: String(format: "%.2f×", duck.rate)).font(.footnote.weight(.semibold)).monospacedDigit().foregroundStyle(.secondary) }
                        }
                    }
                    .padding(.horizontal, 2).frame(height: 32).contentShape(Rectangle())
                }
                .buttonStyle(.plain).accessibilityIdentifier("link")
                .accessibilityValue(Text(verbatim: "studio=\(duck.studioUp) mock=\(duck.mock) bus=\(duck.busOn) servos=\(joints) brain=\(duck.brainOn) engine=\(duck.liveUp)"))
                if real && !(duck.busOn && duck.brainOn) && !(duck.brainOn && duck.runMode == "run") {      // in Run the duck has its own servos: nothing to connect
                    Button { Task { await duck.connectReal() } } label: {
                        if duck.busy || (duck.searching && !duck.studioUp) { ProgressView().tint(.white).frame(width: 58) } else { Text("Connect") }
                    }
                    .buttonStyle(Pill(primary: true)).disabled(duck.busy)
                        .transition(.blurReplace).accessibilityIdentifier("connect")
                }
            }
            if open {
                VStack(spacing: 0) {
                    if real {
                        row("laptopcomputer", on(duck.studioUp), "Host", duck.studioUp ? duck.studioHost : duck.searching ? "Searching…" : "Not found") { EmptyView() }
                        row("cable.connector", busTint, "Servos", duck.busOn ? "\(joints)/\(duck.names.count)" + (duck.mock ? " · mock" : "") : "") {
                            Toggle("", isOn: .init(get: { duck.busOn }, set: { _ in Task { await duck.toggleBus() } })).labelsHidden()
                                .disabled(!duck.studioUp || duck.busy).accessibilityIdentifier("bus")
                        }
                        if duck.busOn {
                            row("scope", duck.calSaved == nil ? .secondary : .green, "Calibration", duck.calSaved.map { String($0.prefix(10)) } ?? "") {
                                Button("Calibrate") { open = false; Task { await duck.calStart() } }.buttonStyle(Pill())
                                    .disabled(joints < duck.names.count || duck.calOn).accessibilityIdentifier("calibrate")
                            }
                        }
                        row("cpu", on(duck.brainOn), "Computer", duck.brainOn ? "\(duck.brainHost) · load \(String(format: "%.1f", duck.brainLoad))" : "") {
                            Toggle("", isOn: .init(get: { duck.brainOn }, set: { _ in Task { await duck.toggleBrain() } })).labelsHidden()
                                .disabled(!duck.studioUp || duck.busy).accessibilityIdentifier("brain")
                        }
                        if duck.brainOn, !duck.runMode.isEmpty {       // who has the duck's servos and sensors
                            row("play.circle", duck.runMode == "run" ? .green : .secondary, "Mode", "") {
                                Picker("Mode", selection: .init(get: { duck.runMode }, set: { to in Task { await duck.setRun(to) } })) {
                                    Text("Setup").tag("setup")
                                    Text("Run").tag("run")
                                }
                                .pickerStyle(.segmented).frame(width: 132).disabled(duck.busy)
                                .accessibilityIdentifier("runmode").accessibilityValue(duck.runMode)
                            }
                        }
                        if duck.brainOn || duck.hottest != nil {       // temperatures: the processor, and the hottest servo
                            row("thermometer.medium", heatTint, "Heat", duck.hottest.map { "servo \($0.id)" } ?? "") {
                                HStack(spacing: 10) {
                                    if let h = duck.hottest {
                                        Text(verbatim: "\(Int(h.temp))°").foregroundStyle(h.temp >= Duck.servoLimits.0 ? warm(h.temp, Duck.servoLimits) : .primary)
                                            .accessibilityIdentifier("heat.servo").accessibilityValue(Text(verbatim: "\(h.id) \(Int(h.temp))"))
                                    }
                                    if duck.brainOn {
                                        Label { Text(verbatim: "\(Int(duck.brainTemp))°") } icon: { Image(systemName: "cpu") }.labelStyle(.titleAndIcon)
                                            .foregroundStyle(duck.brainTemp >= Duck.cpuLimits.0 ? warm(duck.brainTemp, Duck.cpuLimits) : .primary)
                                            .accessibilityElement().accessibilityIdentifier("heat.cpu").accessibilityValue(Text(verbatim: String(Int(duck.brainTemp))))
                                    }
                                }
                                .fontWeight(.semibold).monospacedDigit()
                            }
                        }
                        if let b = duck.battery {
                            row(cell(b), charge(b), "Battery", String(format: "%.1f V · estimate", duck.volts)) {
                                Text(verbatim: "\(b)%").fontWeight(.semibold).monospacedDigit().foregroundStyle(b < 40 ? charge(b) : .primary)
                            }
                        }
                    } else {
                        row("gamecontroller.fill", on(duck.liveUp), "Engine", duck.liveUp ? duck.liveHost : "Not found") {
                            if duck.liveUp { Text(verbatim: String(format: "%.2f× · %.1f ms", duck.rate, duck.liveStep)).font(.footnote.weight(.semibold)).monospacedDigit().foregroundStyle(.secondary) }
                        }
                    }
                }
                .padding(.horizontal, 10).padding(.bottom, 2).transition(.blurReplace)
            }
        }
        .padding(7)
        .glassEffect(.regular, in: .rect(cornerRadius: open ? 26 : 23))
        .padding(.top, 4)
        .animation(morph, value: duck.busOn).animation(morph, value: duck.brainOn)
    }
}

/// Every servo at once: one row per limb (right leg 10-14, left leg 20-24, head 30-34), then any servo that is not a
/// joint yet. The number's colour is its state, as on the studio's tags: green = answering, red = silent,
/// grey = not on the bus, orange = no joint ID; the selected one sits on the blue highlight, which slides between them.
struct Found: View {
    @Environment(Duck.self) private var duck
    let ns: Namespace.ID

    var body: some View {
        let joints = duck.names.keys.sorted(), extra = duck.ids.filter { duck.names[$0] == nil }
        let rows = Dictionary(grouping: joints) { $0 / 10 }.sorted { $0.key < $1.key }.map(\.value) + (extra.isEmpty ? [] : [extra])
        VStack(spacing: 1) {
            ForEach(rows, id: \.self) { row in
                HStack(spacing: 1) { ForEach(row, id: \.self) { cell($0) } }
            }
        }
        .padding(5).glassEffect(.regular, in: .rect(cornerRadius: 18))
    }

    private func cell(_ id: Int) -> some View {
        let sel = duck.sel == id, on = duck.ids.contains(id)
        let tint: Color = sel ? .white : !on ? .secondary.opacity(0.45) : duck.motors[id] == nil ? .red : duck.names[id] == nil ? .orange : .green
        return Button { duck.open = nil; duck.sel = sel ? nil : id } label: {
            Text(String(id)).font(.system(size: 14, weight: .semibold)).monospacedDigit().foregroundStyle(tint)
                .frame(width: 42, height: 25)
                .background { if sel { Capsule().fill(.blue).matchedGeometryEffect(id: "found", in: ns) } }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain).disabled(!on)
        .accessibilityIdentifier("servo.\(id)").accessibilityValue(sel ? "selected" : !on ? "absent" : duck.motors[id] == nil ? "silent" : "ok")
    }
}

/// Calibration under way: the duck lies on its back, every joint posed at home. Hold switches the torque of all of
/// them, so a joint can be trimmed with its card; Save records the pose; Cancel changes nothing.
struct CalBar: View {
    @Environment(Duck.self) private var duck

    var body: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 0) {
                Text("Home pose").fontWeight(.semibold)
                Text("on its back, straight").font(.caption).foregroundStyle(.secondary)
            }
            .lineLimit(1)
            Spacer(minLength: 0)
            Toggle("Hold", isOn: .init(get: { duck.allHeld }, set: { on in Task { await duck.calHold(on) } })).labelsHidden().accessibilityIdentifier("cal.hold")
            Button("Save") { Task { await duck.calSave() } }.buttonStyle(Pill(primary: true)).accessibilityIdentifier("cal.save")
            Button("Cancel") { Task { await duck.calCancel() } }.buttonStyle(Pill()).accessibilityIdentifier("cal.cancel")
        }
        .padding(.leading, 18).padding(.trailing, 8).padding(.vertical, 8)
        .glassEffect(.regular, in: .rect(cornerRadius: 24))
        .accessibilityElement(children: .contain).accessibilityIdentifier("cal.bar")
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
                        Text(verbatim: (m.deg.map { String(format: "%.1f° · ", abs($0) < 0.05 ? 0 : $0) } ?? "") + "\(Int(m.pos)) · \(String(format: "%.1f", m.volt)) V").monospacedDigit().foregroundStyle(.secondary)
                            .accessibilityIdentifier("pos").accessibilityValue(Text(verbatim: String(Int(m.torque ? m.goal : m.pos))))
                    }
                }
                .font(.subheadline).lineLimit(1)
                Spacer(minLength: 0)
                if !duck.calOn {                            // while calibrating, a joint is only trimmed
                    Button("Test") { Task { await duck.wiggle() } }.accessibilityIdentifier("test")
                    Button("Zero") { asking = true }.accessibilityIdentifier("zero")
                }
            }
            .buttonStyle(Pill())
            let tr = duck.tick >= 0 ? duck.traces[id] ?? Trace() : Trace()
            Grid(horizontalSpacing: 8, verticalSpacing: 8) {
                GridRow { Spark(title: "Position", unit: "°", digits: 1, n: tr.n, t: tr.t, v: tr.pos); Spark(title: "Velocity", unit: "°/s", digits: 0, t: tr.t, v: tr.vel) }
                GridRow { Spark(title: "Load", unit: "%", digits: 0, t: tr.t, v: tr.load); Spark(title: "Temp", unit: "°C", digits: 0, t: tr.t, v: tr.temp) }
            }
            if let m {
                HStack(spacing: 10) {
                    Toggle("", isOn: .init(get: { m.torque }, set: { on in Task { await duck.torque(on) } })).labelsHidden().accessibilityIdentifier("torque")
                    Button { duck.nudge(-2) } label: { Image(systemName: "chevron.left").frame(width: 30, height: 32).contentShape(Rectangle()) }
                        .buttonStyle(.plain).foregroundStyle(.secondary).accessibilityIdentifier("nudge.down")
                    Slider(value: .init(get: { drag ?? (m.torque ? m.goal : m.pos) }, set: { drag = $0; duck.move($0) }), in: m.lo...max(m.lo + 1, m.hi)) { editing in
                        if !editing, let v = drag { duck.move(v, last: true); Task { try? await Task.sleep(for: .milliseconds(400)); drag = nil } }
                    }
                    .accessibilityIdentifier("slide")
                    Button { duck.nudge(2) } label: { Image(systemName: "chevron.right").frame(width: 30, height: 32).contentShape(Rectangle()) }
                        .buttonStyle(.plain).foregroundStyle(.secondary).accessibilityIdentifier("nudge.up")
                }
                .font(.footnote.weight(.bold))
            }
            HStack(spacing: 8) {
                Nudge(name: "P", v: $p); Nudge(name: "D", v: $d)
                Spacer(minLength: 0)
                if m?.deg != nil && !duck.calOn {            // the joint turns the wrong way on screen: reverse its direction
                    Button("Flip") { Task { await duck.flip() } }.accessibilityIdentifier("flip")
                }
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
            Button { v = max(0, v - 1) } label: { Image(systemName: "minus").frame(width: 26, height: 34).contentShape(Rectangle()) }.accessibilityIdentifier("\(name).minus")
            Text(verbatim: "\(name) \(v)").font(.subheadline.weight(.semibold)).monospacedDigit().frame(minWidth: 36).contentTransition(.numericText())
                .onLongPressGesture { if let g = duck.gains { withAnimation { v = name == "P" ? g.duckP : g.duckD } } }
                .accessibilityIdentifier("gain.\(name)").accessibilityValue(String(v))
            Button { v = min(254, v + 1) } label: { Image(systemName: "plus").frame(width: 26, height: 34).contentShape(Rectangle()) }.accessibilityIdentifier("\(name).plus")
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
            let top = 24.0, x = { (s: Double) in 8 + (s - (end - 10)) / 10 * (size.width - 16) }
            let y = { (s: Double) in top + (hi - s) / (hi - lo) * (size.height - top - 8) }
            var path = Path()
            let step = max(1, (t.count - i0) / Int(size.width))
            for i in stride(from: i0, to: t.count, by: step) { let pt = CGPoint(x: x(t[i]), y: y(v[i])); i == i0 ? path.move(to: pt) : path.addLine(to: pt) }
            path.addLine(to: CGPoint(x: x(end), y: y(v[v.count - 1])))
            c.stroke(path, with: .color(.blue), style: .init(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }
        .frame(height: 54)
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
                    if on { VStack(spacing: 0) { window(k) }.padding([.horizontal, .bottom], 10).transition(.blurReplace) }
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
            .aspectRatio(3 / 4, contentMode: .fit).frame(height: 190).frame(maxWidth: .infinity).clipShape(.rect(cornerRadius: 12))
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
            // the IMU's orientation in the trunk, as in the studio. From two poses: upright, then on its back with the beak
            // up. Below: turn the front a quarter left, or flip about the front (X), left (Y) or up (Z) axis.
            let cap = { (t: String, on: Bool) in Text(t).font(.caption.weight(.medium)).frame(maxWidth: .infinity).padding(.vertical, 8)
                .background(.fill.tertiary, in: .capsule).overlay { if on { Capsule().strokeBorder(.green, lineWidth: 1.5) } } }
            HStack(spacing: 6) {
                Button { Task { await duck.imuCal(1) } } label: { cap("1 Upright", duck.imuStep == 1) }.accessibilityIdentifier("imu.1")
                Button { Task { await duck.imuCal(2) } } label: { cap("2 Beak up", duck.imuStep == 0 && duck.imuHow == "two poses") }
                    .disabled(duck.imuStep == 0).opacity(duck.imuStep == 0 ? 0.45 : 1).accessibilityIdentifier("imu.2")
            }
            .buttonStyle(.plain).padding(.top, 6)
            HStack(spacing: 6) {
                Button { Task { await duck.imuTurn("z", 90) } } label: { cap("↺", false) }.accessibilityIdentifier("imu.turn")
                ForEach(["x", "y", "z"], id: \.self) { ax in
                    Button { Task { await duck.imuTurn(ax) } } label: { cap(ax.uppercased(), false) }.accessibilityIdentifier("imu.\(ax)")
                }
            }
            .buttonStyle(.plain).padding(.top, 6).disabled(duck.imuHow.isEmpty).opacity(duck.imuHow.isEmpty ? 0.45 : 1)
            .accessibilityElement(children: .contain).accessibilityIdentifier("imu.how").accessibilityValue(duck.imuHow)
        case "mic":
            Canvas { c, size in
                guard duck.wave.count > 1 else { return }
                var p = Path()
                for (i, a) in duck.wave.enumerated() {
                    let x: Double = 6 + Double(i) * (size.width - 12) / Double(duck.wave.count - 1)
                    let amp: Double = max(-1, min(1, a / duck.micPeak * 0.85))
                    let pt = CGPoint(x: x, y: size.height / 2 - amp * (size.height / 2 - 6))
                    i == 0 ? p.move(to: pt) : p.addLine(to: pt)
                }
                c.stroke(p, with: .color(.blue), style: .init(lineWidth: 2, lineCap: .round, lineJoin: .round))
            }
            .frame(height: 56).background(.fill.tertiary, in: .rect(cornerRadius: 12))
            .accessibilityElement().accessibilityIdentifier("trace")       // how much of the window's height the trace spans, 0...1
            .accessibilityValue(Text(verbatim: String(format: "%.2f", ((duck.wave.max() ?? 0) - (duck.wave.min() ?? 0)) / duck.micPeak * 0.85 / 2)))
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

// ---- play: two sticks, four moves, the emotions ------------------------------------------------------------------------------
let STATES: [String: (String, Color)] = [
    "stand": ("Standing", .green), "walk": ("Walking", .blue), "run": ("Running", .orange), "sprint": ("Sprinting", .orange),
    "stopping": ("Stopping", .teal), "sit_down": ("Sitting down", .orange), "seated": ("Seated", .orange), "stand_up": ("Standing up", .orange),
    "jump": ("Jump", .pink), "roulade": ("Front flip", .pink), "flip": ("Front flip", .pink), "kick": ("Kick", .pink), "recover": ("Getting up", .pink), "fallen": ("Fell", .red), "dead": ("Playing dead", .gray)]

struct StateChip: View {
    @Environment(Duck.self) private var duck

    var body: some View {
        let s = STATES[duck.state] ?? (duck.state, .gray), e = duck.emotes.first { $0.name == duck.playing }
        HStack(spacing: 9) {
            Circle().fill(duck.down ? .red : s.1).frame(width: 9, height: 9)
            Text(e.map { "\($0.emoji) \($0.label)" } ?? s.0).fontWeight(.semibold).contentTransition(.interpolate)
            Text(String(format: "%.2f m/s", duck.speed)).font(.footnote).monospacedDigit().foregroundStyle(.secondary)
            Button { duck.press("enter") } label: { Image(systemName: "arrow.counterclockwise").font(.footnote.weight(.semibold)).frame(width: 36, height: 32).contentShape(Rectangle()) }
                .buttonStyle(.plain).foregroundStyle(.secondary).accessibilityIdentifier("reset")
        }
        .padding(.leading, 14).padding(.trailing, 4).padding(.vertical, 3)
        .glassEffect(.regular, in: .capsule)
        .animation(morph, value: duck.state).animation(morph, value: duck.playing)
        .accessibilityElement(children: .contain).accessibilityIdentifier("state")
        .accessibilityLabel(duck.state).accessibilityValue(Text(verbatim: "\(duck.odometer) \(duck.playing)|\(duck.pose)|\(duck.level)"))
    }
}

/// No duck (or simulator) to play them on: the note, above every emotion greyed out, so the list is always there to see.
struct OfflineEmotes: View {
    let hint: String
    var body: some View {
        VStack(spacing: 12) {
            Offline(hint: hint)
            Emotes(items: Emote.catalog, playing: "", enabled: false) { _ in }
        }
        .padding(18).glassEffect(.regular, in: .rect(cornerRadius: 28))
    }
}

struct Offline: View {
    let hint: String
    var body: some View {
        HStack(spacing: 10) { Image(systemName: "wifi.slash"); Text(hint).font(.footnote) }
            .foregroundStyle(.secondary).padding(.horizontal, 20).padding(.vertical, 14).glassEffect(.regular, in: .capsule)
            .accessibilityElement(children: .combine).accessibilityIdentifier("offline")
    }
}

/// The real duck's state while its runtime drives it: fallen, limp, or the policy it is running, and the speed applied.
struct RunChip: View {
    @Environment(Duck.self) private var duck
    static let names = ["stand": "Standing", "walk": "Walking", "sitstand": "Sitting", "sit": "Sitting", "roulade": "Rolling",
                        "kick_left": "Kick", "kick_right": "Kick", "ground_pick": "Picking up", "init": "Standing up"]

    var body: some View {
        // unpowered, the runtime reads "held" and, lying down, "fallen": neither is news until the joints are on
        let word = !duck.rtPower ? "Ready" : duck.rtFallen ? "Down" : Self.names[duck.rtPolicy] ?? (duck.rtPolicy == "held" || duck.rtPolicy.isEmpty ? "Standing" : nice(duck.rtPolicy).capitalized)
        HStack(spacing: 9) {
            Circle().fill(!duck.rtPower ? .gray : duck.rtFallen ? .red : duck.rtWalk ? .blue : .green).frame(width: 9, height: 9)
            Text(word).fontWeight(.semibold).contentTransition(.interpolate)
            Text(String(format: "%.2f m/s", duck.rtSpeed)).font(.footnote).monospacedDigit().foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14).padding(.vertical, 9).glassEffect(.regular, in: .capsule)
        .animation(morph, value: word)
        .accessibilityElement(children: .ignore).accessibilityIdentifier("run.state").accessibilityLabel(word)
        .accessibilityValue(Text(verbatim: "\(duck.rtPolicy) power=\(duck.rtPower) walk=\(duck.rtWalk) v=\(duck.rtSpeed)|\(duck.rtPose)|\(duck.gait)"))
    }
}

/// Play with the real duck selected. In Setup the policies trained for this duck drive it (`PilotCard`). In Run (the
/// Mode switch under the connection pill) the factory runtime does: stand it up, let it walk, steer it with the same
/// two sticks, and run its skills.
struct RealPlay: View {
    @Environment(Duck.self) private var duck
    @State private var drop = false

    private func note(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 10) { Image(systemName: icon); Text(text).font(.footnote) }.foregroundStyle(.secondary)
    }

    private func skill(_ icon: String, _ name: String, _ side: String? = nil) -> some View {
        Button { Task { await duck.rtDo(name) } } label: {
            ZStack(alignment: .bottomTrailing) {
                Image(systemName: icon)
                if let side { Text(side).font(.system(size: 10, weight: .heavy)).offset(x: 9, y: 6) }
            }
        }
        .buttonStyle(Orb(size: 46)).disabled(!duck.rtWalk || duck.rtBusy).accessibilityIdentifier("skill.\(name)")
    }

    var body: some View {
        if !duck.studioUp || !duck.brainOn {
            OfflineEmotes(hint: "Duck can’t be reached")
        } else if duck.runMode != "run" {                   // Setup: the duck's own policies, run by its pilot
            PilotCard()
        } else if !duck.rtUp {
            HStack(spacing: 10) { ProgressView(); Text("Waking the duck").font(.footnote).foregroundStyle(.secondary) }
                .padding(.horizontal, 20).padding(.vertical, 14).glassEffect(.regular, in: .capsule).accessibilityIdentifier("run.waking")
        } else if !duck.rtReady {
            note("exclamationmark.triangle.fill", "Servos aren’t answering").padding(.horizontal, 20).padding(.vertical, 14)
                .glassEffect(.regular, in: .capsule).accessibilityElement(children: .combine).accessibilityIdentifier("run.noservos")
        } else {
            VStack(spacing: 14) {
                HStack(spacing: 0) {
                    skill(duck.rtPolicy.hasPrefix("sit") ? "figure.stand" : "figure.seated.side.right", "sit_toggle")
                    Spacer(minLength: 0)
                    skill("arrow.trianglehead.2.clockwise.rotate.90", "roulade")
                    Spacer(minLength: 0)
                    skill("soccerball", "kick_left", "L")
                    Spacer(minLength: 0)
                    skill("soccerball", "kick_right", "R")
                    Spacer(minLength: 0)
                    skill("arrow.down.to.line", "ground_pick")
                }
                Slider(value: .init(get: { Double(duck.gait) }, set: { duck.gait = Int($0.rounded()) }), in: 1...5, step: 1) {
                    Text("Speed")
                } minimumValueLabel: { Image(systemName: "tortoise.fill") } maximumValueLabel: { Image(systemName: "hare.fill") }
                    .tint(.blue).foregroundStyle(.secondary).font(.footnote).accessibilityIdentifier("run.gait")
                HStack(spacing: 16) {                       // left: where to go. right: power, walk, and which way to face
                    Stick(tint: .blue) { x, y in duck.drive(x, y) }.opacity(duck.rtWalk ? 1 : 0.4).allowsHitTesting(duck.rtWalk)
                    VStack(spacing: 14) {
                        HStack(spacing: 0) {
                            // power: stand up from limp; pressed again, let go (asked first: an unsupported duck drops)
                            Button { if !duck.rtPower { Task { await duck.rtStand() } } else { drop = true } } label: { Image(systemName: "power") }
                                .buttonStyle(Orb(tint: duck.rtPower ? .green : nil, size: 46)).disabled(duck.rtBusy)
                                .accessibilityIdentifier("run.power").accessibilityValue(duck.rtPower ? "on" : "off")
                            Spacer(minLength: 0)
                            Button { Task { await duck.rtSetWalk(!duck.rtWalk) } } label: { Image(systemName: "figure.walk") }
                                .buttonStyle(Orb(tint: duck.rtWalk ? .blue : nil, size: 46)).disabled(!duck.rtPower || duck.rtBusy)
                                .accessibilityIdentifier("run.walk").accessibilityValue(duck.rtWalk ? "on" : "off")
                            Spacer(minLength: 0)
                            Button { Task { await duck.setRun("setup") } } label: { Image(systemName: "wrench.and.screwdriver") }
                                .buttonStyle(Orb(size: 46)).disabled(duck.rtPower || duck.busy).accessibilityIdentifier("run.setup")
                        }
                        Turn { w in duck.turn(w) }.opacity(duck.rtWalk ? 1 : 0.4).allowsHitTesting(duck.rtWalk)
                    }
                }
            }
            .padding(18)
            .glassEffect(.regular, in: .rect(cornerRadius: 28))
            .confirmationDialog("The duck will drop", isPresented: $drop, titleVisibility: .visible) {
                Button("Let go", role: .destructive) { Task { await duck.rtRelax() } }
            }
        }
    }
}

/// The emotions: a row to swipe, or (the chevron) every clip as a named tile, grouped as in Duck Live -- its own
/// originals, then Pollen Robotics' set. The one playing carries a ring that fills as it plays: from `progress`, or,
/// given `clock` (when it began, how long it lasts), from the time. Open or closed is remembered.
struct Emotes: View {
    let items: [Emote]
    let playing: String
    var progress = 0.0
    var clock: (start: Date, span: Double)?
    var enabled = true
    let tap: (String) -> Void
    @AppStorage("allEmotes") private var all = false

    private func ring(_ p: Double) -> some View {
        Circle().trim(from: 0, to: min(1, max(0, p))).stroke(.blue, style: .init(lineWidth: 3, lineCap: .round)).rotationEffect(.degrees(-90)).padding(1.5)
    }

    private var more: some View {
        Button { withAnimation(morph) { all.toggle() } } label: {
            Image(systemName: "chevron.down").font(.footnote.weight(.bold)).rotationEffect(.degrees(all ? 180 : 0)).frame(width: 34, height: 34)
                .background(.fill.tertiary, in: .circle)
        }
        .buttonStyle(.plain).accessibilityIdentifier("emote.all").accessibilityLabel(all ? "Fewer emotions" : "All emotions")
    }

    private var row: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(items) { e in
                    let on = playing == e.name
                    Button { tap(e.name) } label: {
                        Text(e.emoji).font(.system(size: 23)).frame(width: 46, height: 46)
                            .background(on ? AnyShapeStyle(.blue.opacity(0.22)) : AnyShapeStyle(.fill.tertiary), in: .circle)
                            .overlay {
                                if on, let clock { TimelineView(.animation(minimumInterval: 1 / 30)) { t in ring(t.date.timeIntervalSince(clock.start) / clock.span) } }
                                else if on { ring(progress) }
                            }
                            .scaleEffect(on ? 1.08 : 1)
                    }
                    .buttonStyle(.plain).disabled(!e.ok || !enabled).opacity(e.ok && enabled ? 1 : 0.3)
                    .accessibilityIdentifier("emote.\(e.name)").accessibilityLabel(e.label).accessibilityValue(on ? "playing" : "")
                }
            }
            .padding(.leading, 18).padding(.vertical, 3)
        }
        .scrollIndicators(.hidden)
    }

    private func tile(_ e: Emote) -> some View {
        let on = playing == e.name
        return Button { tap(e.name) } label: {
            VStack(spacing: 2) {
                Text(e.emoji).font(.system(size: 24))
                Text(e.label).font(.caption.weight(.semibold)).lineLimit(1).minimumScaleFactor(0.7)
            }
            .frame(maxWidth: .infinity).padding(.vertical, 9)
            .background(on ? AnyShapeStyle(.blue.opacity(0.22)) : AnyShapeStyle(.fill.tertiary), in: .rect(cornerRadius: 16))
            .overlay(alignment: .topTrailing) {
                if e.headOnly { Text("body").font(.system(size: 9)).foregroundStyle(.secondary).padding(.top, 5).padding(.trailing, 8) }
            }
            .overlay { if on { RoundedRectangle(cornerRadius: 16).stroke(.blue, lineWidth: 2) } }
        }
        .buttonStyle(.plain).disabled(!e.ok || !enabled).opacity(e.ok && enabled ? 1 : 0.3)
        .accessibilityIdentifier("emote.\(e.name)").accessibilityLabel(e.label).accessibilityValue(on ? "playing" : "")
    }

    private func section(_ title: String, _ note: String, _ list: [Emote]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack { Text(title).fontWeight(.bold); Spacer(); Text(note) }.font(.caption).foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 84), spacing: 8)], spacing: 8) { ForEach(list) { tile($0) } }
        }
    }

    var body: some View {
        if all {
            VStack(spacing: 6) {
                HStack { Spacer(); more }
                ScrollView {
                    VStack(spacing: 12) {
                        let live = items.filter { $0.group == "live" }, pollen = items.filter { $0.group == "pollen" }
                        let other = items.filter { $0.group != "live" && $0.group != "pollen" }
                        if !live.isEmpty { section("Duck Live", "head · beak · sound", live) }
                        if !pollen.isEmpty { section("Pollen Robotics", "open-source set", pollen) }
                        if !other.isEmpty { section("More", "", other) }
                    }
                }
                .frame(maxHeight: 340).scrollIndicators(.hidden)
            }
            .animation(morph, value: playing)
        } else {
            HStack(spacing: 8) { row; more.padding(.trailing, 18) }
                .padding(.horizontal, -18).animation(morph, value: playing)
        }
    }
}

/// The real duck's state under its own policies: what the pilot is doing and the speed it commands; with the joints
/// off, the animation playing, or Ready.
struct PilotChip: View {
    @Environment(Duck.self) private var duck

    var body: some View {
        let s = STATES[duck.pilotState] ?? (duck.pilotState, .gray), e = duck.clips.first { $0.name == duck.clipNow }
        let word = duck.pilotOn ? (duck.pilotDown ? "Down" : s.0) : e.map { "\($0.emoji) \($0.label)" } ?? "Ready"
        HStack(spacing: 9) {
            Circle().fill(!duck.pilotOn ? .gray : duck.pilotDown ? .red : s.1).frame(width: 9, height: 9)
            Text(word).fontWeight(.semibold).contentTransition(.interpolate)
            if duck.pilotOn { Text(String(format: "%.2f m/s", duck.pilotSpeed)).font(.footnote).monospacedDigit().foregroundStyle(.secondary) }
        }
        .padding(.horizontal, 14).padding(.vertical, 9).glassEffect(.regular, in: .capsule)
        .animation(morph, value: word)
        .accessibilityElement(children: .ignore).accessibilityIdentifier("pilot.state").accessibilityLabel(duck.pilotOn ? duck.pilotState : "off")
        .accessibilityValue(Text(verbatim: "\(duck.pilotSpeed) \(duck.clipNow)|\(duck.pilotPose)|\(duck.gait)|\(duck.pilotFalls)"))
    }
}

/// Play with the real duck in Setup: the policies trained for this duck, run on the duck. The same pad as the
/// simulated duck's (emotions, speed, two sticks, sit, run, jump, flip, kick) plus power: on, the policies take the joints;
/// off, every joint goes limp. Both are asked first. The emotions play either way: head, beak and sound, over a running
/// policy too (the clip takes the head, the policy keeps the legs).
struct PilotCard: View {
    @Environment(Duck.self) private var duck
    @State private var ask = false
    @State private var drop = false

    private func act(_ icon: String, _ key: String) -> some View {
        Button { Task { await duck.pilotPress(key) } } label: { Image(systemName: icon) }.buttonStyle(Orb(size: 44)).disabled(!duck.pilotOn)
            .accessibilityIdentifier("pilot.\(key)")
    }

    var body: some View {
        let on = duck.pilotOn, seated = duck.pilotState == "seated" || duck.pilotState == "sit_down"
        VStack(spacing: 14) {
            // the duck's own clips; until it has sent them, the built-in list, greyed out
            Emotes(items: duck.clips.isEmpty ? Emote.catalog : duck.clips, playing: duck.clipNow, clock: duck.clipClock,
                   enabled: !duck.clips.isEmpty) { name in Task { await duck.clip(name) } }
            HStack(spacing: 14) {
                Button { if on { drop = true } else { ask = true } } label: { Image(systemName: "power") }
                    .buttonStyle(Orb(tint: on ? .green : nil, size: 44)).disabled(duck.pilotBusy)
                    .accessibilityIdentifier("pilot.power").accessibilityValue(on ? "on" : "off")
                    .confirmationDialog("The legs will move", isPresented: $ask, titleVisibility: .visible) {
                        Button("Start") { Task { await duck.pilotStart() } }
                    }
                Slider(value: .init(get: { Double(duck.gait) }, set: { duck.setGait(Int($0.rounded())) }), in: 1...5, step: 1) {
                    Text("Speed")
                } minimumValueLabel: { Image(systemName: "tortoise.fill") } maximumValueLabel: { Image(systemName: "hare.fill") }
                    .tint(duck.run ? .orange : .blue).foregroundStyle(.secondary).font(.footnote).accessibilityIdentifier("pilot.gait")
            }
            HStack(spacing: 0) {                            // the five moves, a row of their own (beside the stick they do not fit)
                act(seated ? "figure.stand" : "figure.seated.side.right", "sit")
                Spacer(minLength: 0)
                Button { duck.run.toggle() } label: { Image(systemName: "figure.run") }.buttonStyle(Orb(tint: duck.run ? .orange : nil, size: 44))
                    .disabled(!on).accessibilityIdentifier("pilot.run").accessibilityValue(duck.run ? "on" : "off")
                Spacer(minLength: 0)
                act("arrow.up.to.line", "jump")
                Spacer(minLength: 0)
                act("arrow.trianglehead.2.clockwise.rotate.90", "flip")
                Spacer(minLength: 0)
                act("soccerball", "kick")
            }
            HStack(spacing: 16) {                           // left: where to go. right: which way to face
                Stick(tint: duck.run ? .orange : .blue) { x, y in duck.drive(x, y) }.opacity(on ? 1 : 0.4).allowsHitTesting(on)
                Turn { w in duck.turn(w) }.opacity(on ? 1 : 0.4).allowsHitTesting(on)
            }
        }
        .padding(18)
        .glassEffect(.regular, in: .rect(cornerRadius: 28))
        .confirmationDialog("The duck will drop", isPresented: $drop, titleVisibility: .visible) {
            Button("Let go", role: .destructive) { Task { await duck.pilotStop() } }
        }
    }
}

struct PlayCard: View {
    @Environment(Duck.self) private var duck

    private func act(_ icon: String, _ key: String, _ id: String, _ can: Bool = true) -> some View {
        Button { duck.press(key) } label: { Image(systemName: icon) }.buttonStyle(Orb(size: 44)).disabled(!can).accessibilityIdentifier("act.\(id)")
    }

    var body: some View {
        if !duck.liveUp { OfflineEmotes(hint: "Simulator not found") } else {
            VStack(spacing: 14) {
                Emotes(items: duck.emotes, playing: duck.playing, progress: duck.progress) { duck.emote($0) }
                Slider(value: .init(get: { Double(duck.level) }, set: { duck.setLevel(Int($0.rounded())) }), in: 1...5, step: 1) {
                    Text("Speed")
                } minimumValueLabel: { Image(systemName: "tortoise.fill") } maximumValueLabel: { Image(systemName: "hare.fill") }
                    .tint(duck.run ? .orange : .blue).foregroundStyle(.secondary).font(.footnote).accessibilityIdentifier("level")
                HStack(spacing: 0) {                        // the five moves, a row of their own (beside the stick they do not fit)
                    act(duck.state == "seated" || duck.state == "sit_down" ? "figure.stand" : "figure.seated.side.right", "s", "sit")
                    Spacer(minLength: 0)
                    Button { duck.run.toggle() } label: { Image(systemName: "figure.run") }.buttonStyle(Orb(tint: duck.run ? .orange : nil, size: 44))
                        .accessibilityIdentifier("act.run").accessibilityValue(duck.run ? "on" : "off")
                    Spacer(minLength: 0)
                    act("arrow.up.to.line", "space", "jump", duck.has["jump"] ?? false)
                    Spacer(minLength: 0)
                    act("arrow.trianglehead.2.clockwise.rotate.90", "r", "flip", duck.has["roulade"] ?? false)
                    Spacer(minLength: 0)
                    act("soccerball", "b", "kick", duck.has["kick"] ?? false)
                }
                HStack(spacing: 16) {                       // left: where to go. right: which way to face
                    Stick(tint: duck.run ? .orange : .blue) { x, y in duck.drive(x, y) }
                    Turn { w in duck.turn(w) }
                }
            }
            .padding(18)
            .glassEffect(.regular, in: .rect(cornerRadius: 28))
        }
    }
}

/// The left stick: push it where the duck should go. Forward and back walk, left and right sidestep, as far as it is
/// pushed; let go and it stops.
struct Stick: View {
    var tint: Color
    var move: (Double, Double) -> Void
    @State private var k = CGSize.zero
    private let reach = 40.0

    var body: some View {
        ZStack {
            Circle().fill(.fill.tertiary)
            ForEach(0..<4, id: \.self) { i in
                Image(systemName: "chevron.up").font(.footnote.weight(.bold)).foregroundStyle(.tertiary).offset(y: -51).rotationEffect(.degrees(Double(i) * 90))
            }
            Circle().fill(tint.gradient).frame(width: 54, height: 54).shadow(color: tint.opacity(0.45), radius: k == .zero ? 4 : 12, y: 3)
                .scaleEffect(k == .zero ? 1 : 1.08).offset(k)
        }
        .frame(width: 132, height: 132).contentShape(.circle)
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

/// The right stick: a slider that springs back to the middle. Slide it the way the duck should turn.
struct Turn: View {
    var move: (Double) -> Void
    @State private var k = 0.0
    private let reach = 56.0

    var body: some View {
        ZStack {
            Capsule().fill(.fill.tertiary)
            HStack {
                Image(systemName: "arrow.counterclockwise"); Spacer(); Image(systemName: "arrow.clockwise")
            }
            .font(.footnote.weight(.bold)).foregroundStyle(.tertiary).padding(.horizontal, 12)
            Circle().fill(Color.blue.gradient).frame(width: 50, height: 50).shadow(color: .blue.opacity(0.45), radius: k == 0 ? 4 : 12, y: 3)
                .scaleEffect(k == 0 ? 1 : 1.08).offset(x: k)
        }
        .frame(height: 62).frame(maxWidth: .infinity).contentShape(.capsule)
        .gesture(DragGesture(minimumDistance: 0)
            .onChanged { g in k = max(-reach, min(reach, g.translation.width)); move(k / reach) }
            .onEnded { _ in withAnimation(morph) { k = 0 }; move(0) })
        .accessibilityElement().accessibilityIdentifier("turn")
    }
}

// ---- the two modes ----------------------------------------------------------------------------------------------------
struct ModeBar: View {
    @Environment(Duck.self) private var duck
    let ns: Namespace.ID
    static let icons: [Mode: String] = [.duck: "bird.fill", .play: "gamecontroller.fill"]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(Mode.allCases, id: \.self) { m in
                let on = duck.mode == m
                Button { duck.mode = m } label: {
                    Image(systemName: Self.icons[m]!).font(.system(size: 19, weight: .medium)).frame(width: 84, height: 44)
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
