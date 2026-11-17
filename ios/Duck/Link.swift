// The phone's side of the two servers on the Mac: the servo studio (firmware/studio.py --lan: the real servos and
// sensors) and the Duck Live engine (app/server.py --lan: the policies). Both are found over Bonjour; nothing is typed.
import CoreGraphics
import Foundation
import Network
import Observation

enum Mode: String, CaseIterable { case duck, play, policy }

struct Motor: Equatable { var torque = false; var pos = 0.0, goal = 0.0, volt = 0.0, lo = 0.0, hi = 4095.0 }
struct Sensor: Equatable { var ok = false; var note = "" }
struct Emote: Identifiable, Equatable { let name, emoji, label: String; let ok: Bool; var id: String { name } }
struct Gains: Equatable { var p = 0, d = 0, duckP = 5, duckD = 20 }
struct Trace { var n = 0; var t: [Double] = [], pos: [Double] = [], vel: [Double] = [], load: [Double] = [], temp: [Double] = [] }
struct ApiError: LocalizedError { let errorDescription: String? }

let SENSORS = ["camera", "tof", "imu", "mic", "speaker"]
let ROLES = ["walk", "run", "sprint", "sitstand", "jump", "roulade", "recovery"]
let OPTIONAL: Set = ["sprint", "jump", "roulade", "recovery"]

/// Bonjour: every `_microduck._tcp` service, resolved to an IPv4 host and port.
final class Finder {
    struct Found: Equatable { var host = ""; var port = 0; var txt: [String: String] = [:] }
    var changed: ([String: Found]) -> Void = { _ in }
    private var browser: NWBrowser?
    private var found: [String: Found] = [:], pending: [String: [String: String]] = [:]

    func start() {
        let b = NWBrowser(for: .bonjourWithTXTRecord(type: "_microduck._tcp", domain: nil), using: .tcp)
        b.browseResultsChangedHandler = { [weak self] results, _ in self?.sort(results) }
        b.stateUpdateHandler = { [weak self] s in      // refused (no permission yet) or failed: try again
            if case .failed = s { b.cancel(); DispatchQueue.main.asyncAfter(deadline: .now() + 2) { self?.start() } }
        }
        b.start(queue: .main)
        browser = b
    }

    private func sort(_ results: Set<NWBrowser.Result>) {
        var seen = Set<String>()
        for r in results {
            guard case let .bonjour(rec) = r.metadata, let kind = rec["kind"] else { continue }
            let txt = rec.dictionary
            seen.insert(kind)
            if found[kind]?.txt == txt || pending[kind] == txt { continue }
            pending[kind] = txt
            resolve(r.endpoint) { [weak self] host, port in
                guard let self, self.pending[kind] == txt else { return }
                self.pending[kind] = nil
                if let host { self.found[kind] = Found(host: host, port: port, txt: txt); self.changed(self.found) }
            }
        }
        for k in found.keys where !seen.contains(k) { found[k] = nil; changed(found) }
    }

    private func resolve(_ ep: NWEndpoint, _ done: @escaping (String?, Int) -> Void) {
        let p = NWParameters.tcp
        (p.defaultProtocolStack.internetProtocol as? NWProtocolIP.Options)?.version = .v4
        let c = NWConnection(to: ep, using: p)
        c.stateUpdateHandler = { s in
            switch s {
            case .ready:
                if case let .hostPort(h, port)? = c.currentPath?.remoteEndpoint {
                    done("\(h)".components(separatedBy: "%")[0], Int(port.rawValue))
                } else { done(nil, 0) }
                c.cancel()
            case .failed, .waiting: done(nil, 0); c.cancel()
            default: break
            }
        }
        c.start(queue: .main)
    }
}

@MainActor @Observable final class Duck {
    var mode = Mode.duck { didSet { if mode != oldValue { modeChanged() } } }
    var dark = false { didSet { look() } }
    var toast: String?

    // ---- the studio: servos and sensors -----------------------------------------------------------------------------
    var studioUp = false
    var busOn = false
    var mock = false
    var brainOn = false
    var busy = false
    var studioHost = ""
    var brainHost = ""
    var brainTemp = 0.0
    var ports: [String] = []
    var ids: [Int] = []
    var names: [Int: String] = [:]
    var motors: [Int: Motor] = [:]
    var sensors: [String: Sensor] = [:]
    var sel: Int? { didSet { if sel != oldValue { gains = nil; look(); if let sel { loadGains(sel) } } } }
    var gains: Gains?
    var tick = 0                                          // bumps when the traces grow
    var open: String? { didSet { look(); if open != "imu" { scene.tilt(nil) } } }      // the sensor whose window is open
    var camera: CGImage?
    var tof: [Double] = []
    var wave: [Double] = []
    var micLevel = 0.0
    var roll = 0.0
    var pitch = 0.0
    @ObservationIgnored var traces: [Int: Trace] = [:]
    @ObservationIgnored private var seq = 0
    @ObservationIgnored private var camSize = (320, 240)
    @ObservationIgnored private var moved = Date.distantPast

    // ---- the engine: policies, driving, emotions ----------------------------------------------------------------------
    var liveUp = false
    var liveHost = ""
    var state = ""
    var policy = ""
    var level = 3
    var speed = 0.0
    var odometer = 0.0
    var rate = 0.0
    var down = false
    var has: [String: Bool] = [:]
    var playing = ""
    var progress = 0.0
    var run = false { didSet { sendKeys() } }
    var emotes: [Emote] = []
    var policies: [String: [String]] = [:]
    var chosen: [String: String] = [:]
    @ObservationIgnored private var token = ""
    @ObservationIgnored private var bodies: [String] = []
    @ObservationIgnored private var ws: URLSessionWebSocketTask?
    @ObservationIgnored private var lastKeys = ""
    @ObservationIgnored private var stick = (0.0, 0.0)

    @ObservationIgnored let scene = DuckScene()
    @ObservationIgnored private var studio: URL?
    @ObservationIgnored private var live: URL?
    @ObservationIgnored private let finder = Finder()
    @ObservationIgnored private let http: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 30
        c.waitsForConnectivity = false
        return URLSession(configuration: c)
    }()

    func start() {
        finder.changed = { [weak self] f in Task { @MainActor in self?.found(f) } }
        finder.start()
        Task { while true { await poll(); try? await Task.sleep(for: .milliseconds(busOn ? 100 : 700)) } }
        Task { while true { await sense(); try? await Task.sleep(for: .milliseconds(100)) } }
        Task { while true { if live != nil && ws == nil { await openLive() }; try? await Task.sleep(for: .milliseconds(800)) } }
        look()
    }

    private func found(_ f: [String: Finder.Found]) {
        let s = f["studio"].flatMap { URL(string: "http://\($0.host):\($0.port)") }
        if s != studio { studio = s; seq = 0; studioHost = f["studio"]?.host ?? "" }
        let l = f["live"].flatMap { URL(string: "http://\($0.host):\($0.port)") }, t = f["live"]?.txt["token"] ?? ""
        if l != live || t != token { live = l; token = t; liveHost = f["live"]?.host ?? ""; dropLive() }
    }

    private func set<T: Equatable>(_ k: ReferenceWritableKeyPath<Duck, T>, _ v: T) { if self[keyPath: k] != v { self[keyPath: k] = v } }

    func say(_ t: String) {
        toast = t
        Task { try? await Task.sleep(for: .seconds(2.4)); if toast == t { toast = nil } }
    }

    // ---- HTTP -------------------------------------------------------------------------------------------------------
    private func fetch(_ base: URL, _ path: String, _ body: [String: Any]? = nil) async throws -> Data {
        var r = URLRequest(url: URL(string: path, relativeTo: base)!)
        if let body { r.httpMethod = "POST"; r.httpBody = try JSONSerialization.data(withJSONObject: body) }
        let (d, resp) = try await http.data(for: r)
        if (resp as? HTTPURLResponse)?.statusCode != 200 {
            let msg = (try? JSONSerialization.jsonObject(with: d) as? [String: Any])?["error"] as? String
            throw ApiError(errorDescription: msg ?? "failed")
        }
        return d
    }

    private func dict(_ d: Data) -> [String: Any] { (try? JSONSerialization.jsonObject(with: d) as? [String: Any]) ?? [:] }

    /// A studio command. Its reply is the new state when it carries one. Errors surface as a toast.
    @discardableResult func call(_ name: String, _ body: [String: Any] = [:]) async -> [String: Any]? {
        guard let studio else { return nil }
        do {
            let j = dict(try await fetch(studio, "/api/" + name, body))
            if j["ids"] != nil { ingest(j) }
            return j
        } catch { say(error.localizedDescription); return nil }
    }

    // ---- studio state -------------------------------------------------------------------------------------------------
    private func poll() async {
        guard let studio else { if studioUp { studioUp = false; clearStudio() }; return }
        do { ingest(dict(try await fetch(studio, "/api/state?since=\(seq)"))); set(\.studioUp, true) }
        catch { if studioUp { studioUp = false; clearStudio() } }
    }

    private func clearStudio() {
        busOn = false; brainOn = false; ids = []; motors = [:]; sensors = [:]; sel = nil; open = nil
        look()
    }

    private func ingest(_ j: [String: Any]) {
        set(\.busOn, j["connected"] as? Bool ?? false)
        set(\.mock, j["mock"] as? Bool ?? false)
        set(\.ports, j["ports"] as? [String] ?? [])
        set(\.ids, j["ids"] as? [Int] ?? [])
        if names.isEmpty, let d = j["duck"] as? [String: String] { names = Dictionary(uniqueKeysWithValues: d.compactMap { k, v in Int(k).map { ($0, v) } }) }
        var m: [Int: Motor] = [:]
        for (k, v) in j["motors"] as? [String: [String: Any]] ?? [:] {
            guard let id = Int(k) else { continue }
            let span = v["span"] as? [Double] ?? [0, 4095], num = { (key: String) in v[key] as? Double ?? 0 }
            m[id] = Motor(torque: v["torque"] as? Bool ?? false, pos: num("pos"), goal: num("goal"), volt: num("volt"), lo: span[0], hi: span[1])
        }
        set(\.motors, m)
        let b = j["brain"] as? [String: Any] ?? [:], s = b["sensors"] as? [String: Any] ?? [:]
        set(\.brainOn, b["on"] as? Bool ?? false)
        set(\.brainHost, s["host"] as? String ?? "")
        set(\.brainTemp, s["temp"] as? Double ?? 0)
        if let z = b["size"] as? [Int], z.count == 2 { camSize = (z[0], z[1]) }
        set(\.sensors, Dictionary(uniqueKeysWithValues: SENSORS.compactMap { k in
            (s[k] as? [String: Any]).map { (k, Sensor(ok: $0["ok"] as? Bool ?? false, note: $0["note"] as? String ?? "")) } }))
        // telemetry: [seq, t, {id: [pos, vel, load, temp, volt, cur, goal]}], raw units -> deg, deg/s, %, degC
        let now = j["seq"] as? Int ?? 0, deg = 360.0 / 4096
        if now < seq { traces = [:] }
        seq = now
        let samples = j["samples"] as? [[Any]] ?? []
        for row in samples {
            guard row.count == 3, let t = row[1] as? Double, let all = row[2] as? [String: [Double]] else { continue }
            for (k, v) in all where v.count >= 4 {
                guard let id = Int(k) else { continue }
                var tr = traces[id] ?? Trace()
                tr.n += 1; tr.t.append(t); tr.pos.append((v[0] - 2048) * deg); tr.vel.append(v[1] * 50 * deg); tr.load.append(v[2] / 10); tr.temp.append(v[3])
                if tr.t.count > 600 { let n = tr.t.count - 480; tr.t.removeFirst(n); tr.pos.removeFirst(n); tr.vel.removeFirst(n); tr.load.removeFirst(n); tr.temp.removeFirst(n) }
                traces[id] = tr
            }
        }
        if !samples.isEmpty { tick += 1 }
        if let sel, !ids.contains(sel) { self.sel = nil }
        if !brainOn, open != nil { open = nil }
        look()
    }

    func look() {
        scene.apply(.init(live: mode != .duck, ids: Set(ids), answering: Set(motors.keys), sel: sel,
                          sensors: brainOn ? sensors.mapValues(\.ok) : nil, open: open, dark: dark, fov: mode == .play ? 34 : mode == .policy || sel != nil ? 41 : 25))
    }

    private func modeChanged() {
        stick = (0, 0); run = false; sendKeys()             // never leave the duck walking behind a hidden joystick
        if mode == .duck { scene.stand() }
        look()
    }

    // ---- studio commands ----------------------------------------------------------------------------------------------
    func toggleBus() async {
        busy = true; defer { busy = false }
        if busOn { await call("disconnect") }
        else if let port = ports.first(where: { $0 == "duck" }) ?? ports.first {
            if let j = await call("connect", ["port": port, "baud": 1_000_000]) { let n = (j["ids"] as? [Int])?.count ?? 0; say(n > 0 ? "\(n) found" : "No servos answered") }
        } else { say("No adapter") }
    }

    func toggleBrain() async { busy = true; defer { busy = false }; await call(brainOn ? "brain_disconnect" : "brain_connect") }

    private func loadGains(_ id: Int) {
        Task {
            guard let studio, let j = try? dict(await fetch(studio, "/api/gains", ["id": id])), sel == id,
                  let p = j["p"] as? Int, let d = j["d"] as? Int else { return }
            let k = j["duck"] as? [Int] ?? [5, 20]
            gains = Gains(p: p, d: d, duckP: k[0], duckD: k[1])
        }
    }

    func setGains(_ p: Int, _ d: Int) async {
        guard let sel, let j = await call("set_gains", ["id": sel, "p": p, "d": d]), let np = j["p"] as? Int, let nd = j["d"] as? Int else { return }
        gains?.p = np; gains?.d = nd
        say("\(sel)  P \(np)  D \(nd)")
    }

    func torque(_ on: Bool) async { if let sel { await call("torque", ["id": sel, "on": on]) } }
    func wiggle() async { if let sel { await call("wiggle", ["id": sel]) } }

    func zero() async {
        guard let sel, let j = await call("center", ["id": sel]) else { return }
        let pos = ((j["motors"] as? [String: [String: Any]])?["\(sel)"]?["pos"] as? Double).map { String(Int($0)) } ?? "?"
        say("\(sel) zeroed → \(pos)")
    }

    /// The slider: at most 25 commands a second while dragging, and always the last one.
    func move(_ pos: Double, last: Bool = false) {
        guard let sel, last || Date().timeIntervalSince(moved) > 0.04 else { return }
        moved = Date()
        Task { await call("move", ["id": sel, "pos": Int(pos)]) }
    }

    /// One of the duck's own sounds (firmware/duck_voice.py); "chirp" is the quack.
    func voice(_ tag: String) async { await call("say", ["tag": tag]) }

    // ---- sensors: a reading ten times a second for the one whose window is open -------------------------------------------
    private func sense() async {
        guard let studio, brainOn, let k = open, k != "speaker", let d = try? await fetch(studio, "/sense/" + k) else { return }
        switch k {
        case "camera":
            let (w, h) = camSize
            if d.count == w * h * 3 / 2 { camera = await Task.detached { nv12(d, w, h) }.value }
        case "tof": if let v = try? JSONSerialization.jsonObject(with: d) as? [Double] { tof = v }
        case "imu":
            let v = dict(d)
            guard let r = v["roll"] as? Double, let p = v["pitch"] as? Double else { return }
            roll = r; pitch = p
            scene.tilt(v["quat"] as? [Double] ?? [], roll: r, pitch: p)
        case "mic":
            let v = dict(d)
            wave = v["wave"] as? [Double] ?? []; micLevel = v["level"] as? Double ?? 0
        default: break
        }
    }

    // ---- the engine ------------------------------------------------------------------------------------------------------
    private func dropLive() { ws?.cancel(with: .goingAway, reason: nil); ws = nil; liveUp = false; lastKeys = "" }

    private func q(_ path: String) -> String { path + "?token=" + token }

    private func openLive() async {
        guard let live, let h = try? dict(await fetch(live, q("/api/health"))), let port = h["ws_port"] as? Int,
              let sc = try? dict(await fetch(live, q("/api/scene"))), live == self.live else { return }
        health(h)
        bodies = sc["bodies"] as? [String] ?? []
        emotes = (sc["anims"] as? [[String: Any]] ?? []).map {
            Emote(name: $0["name"] as? String ?? "", emoji: $0["emoji"] as? String ?? "", label: $0["label"] as? String ?? "", ok: $0["available"] as? Bool ?? false) }
        var c = URLComponents(); c.scheme = "ws"; c.host = live.host; c.port = port; c.path = "/"
        c.queryItems = [URLQueryItem(name: "token", value: token)]
        let t = http.webSocketTask(with: c.url!)
        ws = t; t.resume(); listen(t)
    }

    private func health(_ h: [String: Any]) {
        policies = h["policies"] as? [String: [String]] ?? [:]
        chosen = h["chosen"] as? [String: String] ?? [:]
    }

    private func listen(_ t: URLSessionWebSocketTask) {
        t.receive { [weak self] r in
            Task { @MainActor in
                guard let self, self.ws === t else { return }
                switch r {
                case .success(.string(let s)): self.snapshot(s); self.listen(t)
                case .success: self.listen(t)
                case .failure: self.ws = nil; self.liveUp = false
                }
            }
        }
    }

    private func snapshot(_ text: String) {
        guard let s = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else { return }
        if !liveUp { liveUp = true; lastKeys = ""; sendKeys() }
        set(\.state, s["state"] as? String ?? "")
        set(\.policy, s["policy"] as? String ?? "")
        set(\.level, s["level"] as? Int ?? 3)
        set(\.down, s["down"] as? Bool ?? false)
        set(\.has, s["has"] as? [String: Bool] ?? [:])
        let v = s["v"] as? [Double] ?? [0, 0]
        set(\.speed, (hypot(v[0], v[1]) * 20).rounded() / 20)
        set(\.odometer, ((s["odometer"] as? Double ?? 0) * 100).rounded() / 100)
        set(\.rate, (((s["rate"] as? Double ?? 0) / 50) * 20).rounded() / 20)
        let a = s["anim"] as? [String: Any]
        set(\.playing, a?["name"] as? String ?? "")
        set(\.progress, a.map { (((($0["t"] as? Double ?? 0) / max(0.1, $0["T"] as? Double ?? 1)) * 25).rounded() / 25) } ?? 0)
        if mode != .duck, let p = s["pos"] as? [Double], let q = s["quat"] as? [Double] { scene.pose(bodies, p, q) }
    }

    private func send(_ m: [String: Any]) {
        guard liveUp, let d = try? JSONSerialization.data(withJSONObject: m), let s = String(data: d, encoding: .utf8) else { return }
        ws?.send(.string(s)) { _ in }
    }

    /// The joystick, x right and y forward in -1...1. The engine is driven by held keys (as the Mac app is): past a
    /// third of the way out a direction is held; running is straight only, so the run switch acts on pure forward.
    func drive(_ x: Double, _ y: Double) { stick = (x, y); sendKeys() }

    private func sendKeys() {
        let (x, y) = stick
        var held: [String] = []
        if y > 0.35 { held.append("up") } else if y < -0.35 { held.append("down") }
        if x < -0.35 { held.append("left") } else if x > 0.35 { held.append("right") }
        let m: [String: Any] = ["t": "keys", "held": held, "meta": run && held == ["up"], "alt": false]
        let key = "\(held)\(run)"
        if key != lastKeys, liveUp { lastKeys = key; send(m) }
    }

    func press(_ key: String) { send(["t": "press", "key": key]) }
    func emote(_ name: String) { send(["t": "emote", "name": name]) }

    func setLevel(_ n: Int) { Task { if let live { _ = try? await fetch(live, q("/api/config"), ["level": n]) } } }

    func setPolicy(_ role: String, _ file: String) async {
        guard let live else { return }
        do {
            _ = try await fetch(live, q("/api/policies"), [role: file])
            try? await Task.sleep(for: .milliseconds(600))          // the engine swaps it inside its loop
            health(dict(try await fetch(live, q("/api/health"))))
            say(chosen[role] == file || (file.isEmpty && chosen[role] == nil) ? "Deployed" : "Not loaded")
        } catch { say(error.localizedDescription) }
    }

    /// The app left the screen: stop the duck.
    func pause() { stick = (0, 0); run = false; sendKeys() }
}

/// NV12 (the duck's camera) -> an image.
func nv12(_ d: Data, _ w: Int, _ h: Int) -> CGImage? {
    var px = [UInt8](repeating: 255, count: w * h * 4)
    d.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
        for j in 0..<h {
            for i in 0..<w {
                let y = Float(b[j * w + i]), q = w * h + (j >> 1) * w + (i & ~1)
                let u = Float(b[q]) - 128, v = Float(b[q + 1]) - 128, p = (j * w + i) * 4
                px[p] = UInt8(max(0, min(255, y + 1.402 * v)))
                px[p + 1] = UInt8(max(0, min(255, y - 0.344 * u - 0.714 * v)))
                px[p + 2] = UInt8(max(0, min(255, y + 1.772 * u)))
            }
        }
    }
    guard let prov = CGDataProvider(data: Data(px) as CFData) else { return nil }
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: prov, decode: nil,
                   shouldInterpolate: true, intent: .defaultIntent)
}
