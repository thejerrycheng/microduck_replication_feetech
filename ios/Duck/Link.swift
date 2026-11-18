// The phone's side of the two servers: the servo studio (on the duck itself, or on the Mac: the real servos and sensors)
// and the Duck Live engine (the simulated duck). Both are found by the app (Bonjour, the duck's known addresses, a sweep
// of the Wi-Fi network); nothing is typed.
import CoreGraphics
import Darwin
import Foundation
import Network
import Observation

enum Mode: String, CaseIterable { case duck, play }
/// What the app is talking to: the real duck (through the servo studio) or the simulated one (the Duck Live engine).
enum Target: String, CaseIterable { case real, sim }

struct Motor: Equatable { var torque = false; var pos = 0.0, goal = 0.0, volt = 0.0, lo = 0.0, hi = 4095.0; var deg: Double?; var temp = 0.0 }      // deg: joint angle, once calibrated
struct Sensor: Equatable { var ok = false; var note = "" }
struct Emote: Identifiable, Equatable { let name, emoji, label: String; let ok: Bool; var id: String { name } }
struct Gains: Equatable { var p = 0, d = 0, duckP = 5, duckD = 20 }
struct Trace { var n = 0; var t: [Double] = [], pos: [Double] = [], vel: [Double] = [], load: [Double] = [], temp: [Double] = [] }
struct ApiError: LocalizedError { let errorDescription: String? }

let SENSORS = ["camera", "tof", "imu", "mic", "speaker"]

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
            guard case let .bonjour(rec) = r.metadata, var kind = rec["kind"] else { continue }
            let txt = rec.dictionary
            if txt["mock"] == "1" { kind += "-mock" }            // a mock studio never stands in front of a real one
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
    var target = Target(rawValue: UserDefaults.standard.string(forKey: "target") ?? "") ?? .real {
        didSet { if target != oldValue { UserDefaults.standard.set(target.rawValue, forKey: "target"); sel = nil; open = nil; modeChanged() } }
    }
    var dark = false { didSet { look() } }
    var toast: String?

    // ---- the studio: servos and sensors -----------------------------------------------------------------------------
    var studioUp = false
    var searching = false           // a sweep of the network for the duck is under way
    var busOn = false
    var mock = false
    var brainOn = false
    var busy = false
    var studioHost = ""
    var brainHost = ""
    var brainTemp = 0.0                                   // the duck computer's processor [degC]
    var brainLoad = 0.0                                   // its 1-minute load average
    var liveStep = 0.0                                    // the engine's control step [ms]
    /// Temperature limits [degC]: (warm, hot). Processor: the RK3566 throttles from about 85 and shuts down near 115.
    /// Servo: the HD1910's own over-temperature protection is 70.
    static let cpuLimits = (75.0, 90.0), servoLimits = (55.0, 65.0)
    /// The hottest servo on the bus: (ID, degC).
    var hottest: (id: Int, temp: Double)? { motors.max { $0.value.temp < $1.value.temp }.map { ($0.key, $0.value.temp) } }
    /// 0 fine, 1 warm, 2 hot: the worse of the processor and the hottest servo.
    var heat: Int {
        let c = brainOn ? (brainTemp >= Self.cpuLimits.1 ? 2 : brainTemp >= Self.cpuLimits.0 ? 1 : 0) : 0
        let t = hottest?.temp ?? 0, v = t >= Self.servoLimits.1 ? 2 : t >= Self.servoLimits.0 ? 1 : 0
        return max(c, v)
    }
    @ObservationIgnored private var warned = 0
    var battery: Int?                                     // charge left [%], the studio's estimate from the servo supply voltage
    var volts = 0.0
    var calOn = false                                     // a calibration is under way (every servo limp or held)
    var calSaved: String?                                 // when the saved calibration was made
    var runMode = ""                                      // "setup": the studio has the duck's devices; "run": its own runtime
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
    var micPeak = 0.002
    var roll = 0.0
    var pitch = 0.0
    @ObservationIgnored var traces: [Int: Trace] = [:]
    @ObservationIgnored private var seq = 0
    @ObservationIgnored private var camSize = (320, 240)
    @ObservationIgnored private var moved = Date.distantPast

    // ---- the engine: policies, driving, emotions ----------------------------------------------------------------------
    var liveUp = false { didSet { if liveUp != oldValue { look() } } }
    var liveHost = ""
    var state = ""
    var level = 3
    var speed = 0.0
    var odometer = 0.0
    var rate = 0.0
    var down = false
    var has: [String: Bool] = [:]
    var playing = ""
    var progress = 0.0
    var run = false { didSet { sendStick(); sendPilot() } }
    var pose = ""                                         // trunk x, y [m] and heading [deg], for the tests
    var emotes: [Emote] = []
    @ObservationIgnored private var token = ""
    @ObservationIgnored private var bodies: [String] = []
    @ObservationIgnored private var ws: URLSessionWebSocketTask?
    @ObservationIgnored private var lastKeys = ""
    @ObservationIgnored private var stick = (0.0, 0.0)
    @ObservationIgnored private var spin = 0.0
    @ObservationIgnored private var sentAt = Date.distantPast

    // ---- the duck's own runtime (Run): its policies, driven from Play with the real duck selected --------------------
    var rtUp = false { didSet { if rtUp != oldValue { look() } } }      // its socket answers
    var rtPolicy = ""                                     // the policy it is running
    var rtPower = false                                   // the joints are powered. The state stream has no flag for it (before
                                                          // power it reads policy "held", limp false): known from our own
                                                          // commands, and from a policy that is running
    var rtLimp = false                                    // the runtime's own safety reaction has let the joints go
    var rtFallen = false
    var rtWalk = false                                    // the walking policy has the legs (else it holds the stance)
    var rtSpeed = 0.0                                     // the speed it is applying, after its own limits [m/s]
    var rtReady = true                                    // false: it runs but no servo answers it
    var rtBusy = false
    var rtPose = ""                                       // x, y [m] and heading [deg]; only the stand-in runtime reports it (tests)
    var gait = UserDefaults.standard.object(forKey: "gait") as? Int ?? 3 { didSet { UserDefaults.standard.set(gait, forKey: "gait"); sendTwist() } }
    /// Full stick at the top speed level: forward and sideways [m/s], turning [rad/s]. The limits of the runtime's own
    /// console. Level n of 5 scales all three by n/5.
    static let reach = (0.3, 0.3, 1.5)
    /// Where the runtime's console answers: on the duck, and beside a mock studio (ios/tools/mock_runtime.py).
    static let runtimePort = (duck: 8090, mock: 8791)
    @ObservationIgnored private var rt: URLSessionWebSocketTask?
    @ObservationIgnored private var rtId = 0
    @ObservationIgnored private var rtWait: [Int: CheckedContinuation<String?, Never>] = [:]
    @ObservationIgnored private var rtOrder: [Int] = []   // the servo ID of each entry of the runtime's joint list
    @ObservationIgnored private var rtTwist = ""
    @ObservationIgnored private var rtSent = Date.distantPast
    @ObservationIgnored private var rtAsked = Date.distantPast

    // ---- the duck's own policies (Setup): its pilot (firmware/duck_pilot.py, run by the duck's bridge), driven from Play
    // with the real duck selected; and its animations, which the duck plays by itself on its head, beak and speaker ----
    var pilotOn = false { didSet { if pilotOn != oldValue { look() } } }      // the policies have the joints
    var pilotState = ""                                   // stand, walk, run, sprint, stopping, sit_down, seated, stand_up, jump, flip, recover
    var pilotSpeed = 0.0                                  // the speed the pilot commands [m/s]
    var pilotDown = false                                 // tilted past 60 deg
    var pilotFalls = 0
    var pilotBusy = false
    var pilotPose = ""                                    // x, y [m] and heading [deg]; only the stand-in duck reports it (tests)
    var clips: [Emote] = []                               // the animations the duck holds
    var clipNow = ""                                      // the one playing
    var clipClock: (start: Date, span: Double)?           // when it began and how long it lasts [s], for the progress ring
    var imuHow = ""                                       // how the IMU was calibrated: "two poses", "level only", or not at all
    /// Play shows controls for the real duck: its pilot (Setup), or its runtime once that answers (Run).
    var realPlay: Bool { studioUp && brainOn && (runMode != "run" || (rtUp && rtReady)) }
    @ObservationIgnored private var pilotAt = Date.distantPast      // our last start or stop: the studio's word lags up to 1 s
    @ObservationIgnored private var pilotLevel = 3        // the pilot's speed level, 1...5
    @ObservationIgnored private var gaitUntil = Date.distantPast    // until then the pilot's level is brought to `gait`
    @ObservationIgnored private var pressAt = Date.distantPast

    @ObservationIgnored let scene = DuckScene()
    @ObservationIgnored private var studio: URL?
    @ObservationIgnored private var announced: URL?       // the studio Bonjour found
    @ObservationIgnored private var direct: URL?          // the duck itself, answering at one of its own addresses
    @ObservationIgnored private var misses = 0
    /// Where the duck's own studio (firmware/duck_serve.sh) answers when Bonjour does not get through: its name on the
    /// Wi-Fi, and its address on its own hotspot.
    static let ducks = ["radxa-zero3.local", "10.42.0.1"]
    @ObservationIgnored private var swept = Date.distantPast
    @ObservationIgnored private var auto = true           // connect by itself once a real duck answers
    @ObservationIgnored private var autoAt = Date.distantPast
    @ObservationIgnored private let sweeper: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 1.2
        c.waitsForConnectivity = false
        return URLSession(configuration: c)
    }()
    @ObservationIgnored private let quick: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 1.5
        c.waitsForConnectivity = false
        return URLSession(configuration: c)
    }()
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
        Task { while true { await poll(); try? await Task.sleep(for: .milliseconds(busOn || pilotOn || !clipNow.isEmpty ? 100 : 700)) } }
        Task { while true { await sense(); try? await Task.sleep(for: .milliseconds(100)) } }
        Task { while true { await probe(); try? await Task.sleep(for: .seconds(studio == nil ? 1 : 3)) } }
        Task { while true { if live != nil && ws == nil { await openLive() }; try? await Task.sleep(for: .milliseconds(800)) } }
        Task { while true { await runtime(); try? await Task.sleep(for: .milliseconds(100)) } }
        look()
    }

    /// `-find bonjour|name|sweep`: use only that way of finding the duck (the connection tests time each one alone).
    private var only: String? { UserDefaults.standard.string(forKey: "find") }
    private var mockOnly: Bool { UserDefaults.standard.bool(forKey: "mockOnly") }      // `-mockOnly 1`: the UI tests never touch a real studio

    private func found(_ f: [String: Finder.Found]) {
        // a studio on the duck itself first, then one on the Mac, then a mock
        let real = [f["studio"]].compactMap { $0 }
        let st = mockOnly ? f["studio-mock"] : (only ?? "bonjour") != "bonjour" ? nil : real.first ?? f["studio-mock"]
        announced = st.flatMap { URL(string: "http://\($0.host):\($0.port)") }
        choose()
        let l = f["live"].flatMap { URL(string: "http://\($0.host):\($0.port)") }, t = f["live"]?.txt["token"] ?? ""
        if l != live || t != token { live = l; token = t; liveHost = f["live"]?.host ?? ""; dropLive() }
    }

    /// The duck answering at its own address wins over anything announced: it is the real robot, with no Mac between.
    private func choose() {
        let s = direct ?? announced
        if s != studio { studio = s; seq = 0; misses = 0; studioHost = s?.host ?? "" }
    }

    /// A real (not mock) studio answering at this host, or nil.
    private func studioAt(_ host: String, _ session: URLSession) async -> URL? {
        guard let u = URL(string: "http://\(host):8770"), let (d, _) = try? await session.data(from: u.appendingPathComponent("api/state")),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any], j["ids"] != nil, j["mock"] as? Bool == false else { return nil }
        return u
    }

    private func take(_ u: URL) {
        direct = u; choose()
        if let h = u.host { UserDefaults.standard.set(h, forKey: "duckHost") }
    }

    /// The first of these hosts with a real studio, all asked at once.
    private func first(_ hosts: [String], _ session: URLSession) async -> URL? {
        await withTaskGroup(of: URL?.self) { g -> URL? in
            for h in hosts { g.addTask { await self.studioAt(h, session) } }
            for await u in g { if let u { g.cancelAll(); return u } }
            return nil
        }
    }

    /// Look for the duck while not already talking to it: where it answered last time, its name on the Wi-Fi, its
    /// hotspot address; and every 15 s with nothing found, every address of the network the phone is on.
    private func probe() async {
        guard !mockOnly, direct == nil, only != "bonjour" else { return }
        let last = only == nil ? UserDefaults.standard.string(forKey: "duckHost") : nil
        if only != "sweep", let u = await first([last].compactMap({ $0 }) + Self.ducks.filter({ $0 != last }), quick) { take(u); return }
        if only != "name", studio == nil, target == .real, Date().timeIntervalSince(swept) >= 15 { await sweep() }
    }

    /// Every address of the phone's own /24 networks, asked at once: finds the duck when its name does not resolve.
    private func sweep() async {
        guard !mockOnly, direct == nil, !searching else { return }
        searching = true; defer { searching = false; swept = Date() }
        if let hit = await first(Self.neighbours(), sweeper), direct == nil { take(hit) }
    }

    /// All 254 addresses of each private IPv4 network this phone has an address on (Wi-Fi, or the duck's hotspot).
    nonisolated static func neighbours() -> [String] {
        var out: [String] = [], list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0 else { return [] }
        defer { freeifaddrs(list) }
        var p = list
        while let i = p {
            defer { p = i.pointee.ifa_next }
            guard let a = i.pointee.ifa_addr, a.pointee.sa_family == UInt8(AF_INET), String(cString: i.pointee.ifa_name).hasPrefix("en") else { continue }
            let ip = UInt32(bigEndian: a.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee.sin_addr.s_addr })
            let b = [UInt8(ip >> 24), UInt8(ip >> 16 & 255), UInt8(ip >> 8 & 255), UInt8(ip & 255)]
            guard b[0] == 10 || (b[0] == 172 && (16...31).contains(b[1])) || (b[0] == 192 && b[1] == 168) else { continue }
            for n in 1...254 { out.append("\(b[0]).\(b[1]).\(b[2]).\(n)") }
        }
        return out
    }

    /// Search now (the Connect button with no duck in reach). True once a studio answers.
    private func search() async -> Bool {
        if !mockOnly, direct == nil, (only ?? "name") == "name", let u = await first(Self.ducks, quick) { take(u) }
        if (only ?? "sweep") == "sweep" { await sweep() }
        for _ in 0..<15 where !studioUp { try? await Task.sleep(for: .milliseconds(200)) }
        return studioUp
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
    /// `short`: the duck's own reason, cut at its colon ("calibrate first: joints and IMU" -> "Calibrate first").
    @discardableResult func call(_ name: String, _ body: [String: Any] = [:], quiet: Bool = false, short: Bool = false) async -> [String: Any]? {
        guard let studio else { return nil }
        do {
            let j = dict(try await fetch(studio, "/api/" + name, body))
            if j["ids"] != nil { ingest(j) }
            return j
        } catch { if !quiet { say(error is URLError ? "Duck can’t be reached" : short ? plain(error.localizedDescription) : error.localizedDescription) }; return nil }
    }

    private func plain(_ why: String) -> String {
        let t = why.split(separator: ":").first.map(String.init) ?? why
        return t.prefix(1).uppercased() + t.dropFirst()
    }

    // ---- studio state -------------------------------------------------------------------------------------------------
    private func poll() async {
        guard let studio else { if studioUp { studioUp = false; clearStudio() }; return }
        do {
            ingest(dict(try await fetch(studio, "/api/state?since=\(seq)"))); set(\.studioUp, true); misses = 0
            // a real duck just found: bring its computer and servo bus up without being asked. Tried every 2 s until
            // both are up (or the duck is in Run, which keeps the bus); a switch turned off by hand ends it
            if busOn && brainOn { auto = false }
            else if auto, !mock, target == .real, !busy, Date().timeIntervalSince(autoAt) >= 2 {
                await connectReal(quiet: true); autoAt = Date()
                if brainOn && runMode == "run" { auto = false }
            }
        }
        catch {
            if studioUp { studioUp = false; clearStudio() }
            auto = true
            misses += 1
            if misses >= 3, direct != nil { direct = nil; choose() }       // the duck left: fall back to what is announced, keep looking
        }
    }

    private func clearStudio() {
        busOn = false; brainOn = false; ids = []; motors = [:]; sensors = [:]; sel = nil; open = nil; battery = nil; calOn = false; runMode = ""
        pilotOn = false; pilotState = ""; clips = []; clipNow = ""; clipClock = nil
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
            m[id] = Motor(torque: v["torque"] as? Bool ?? false, pos: num("pos"), goal: num("goal"), volt: num("volt"), lo: span[0], hi: span[1], deg: v["deg"] as? Double, temp: num("temp"))
        }
        set(\.motors, m)
        if target == .real { scene.mouth((m[34]?.deg ?? 0) * .pi / 180) }       // the real beak, once the duck is calibrated
        let b = j["brain"] as? [String: Any] ?? [:], s = b["sensors"] as? [String: Any] ?? [:]
        set(\.brainOn, b["on"] as? Bool ?? false)
        set(\.brainHost, s["host"] as? String ?? "")
        set(\.brainTemp, ((s["temp"] as? Double ?? 0) * 2).rounded() / 2)
        set(\.brainLoad, ((s["load"] as? Double ?? 0) * 10).rounded() / 10)
        if heat == 2, warned < 2 {                              // once per excursion: say which, and how hot
            if brainOn, brainTemp >= Self.cpuLimits.1 { say("Computer \(Int(brainTemp))°: let it cool") }
            else if let h = hottest { say("Servo \(h.id) at \(Int(h.temp))°: let it rest") }
        }
        warned = heat
        let c = j["cal"] as? [String: Any] ?? [:]
        set(\.calOn, c["on"] as? Bool ?? false)
        set(\.calSaved, c["saved"] as? String)
        let step = c["imu_step"] as? Int ?? 0, how = c["imu_how"] as? String ?? ""
        if step == 1, imuStep == 1, Date().timeIntervalSince(imuNag) > 20 { imuNag = Date(); say("IMU half done: on its back, beak up, then 2") }
        set(\.imuStep, step); set(\.imuOld, brainOn && !how.isEmpty && how != "two poses")
        set(\.runMode, s["mode"] as? String ?? "")
        set(\.imuStep, c["imu_step"] as? Int ?? 0)
        set(\.imuHow, c["imu"] as? Bool ?? false ? c["imu_how"] as? String ?? "level only" : "")
        // the pilot: the studio reads the duck's word once a second, so right after our own start or stop ours counts
        let p = s["pilot"] as? [String: Any], on = p != nil && !(p?["ended"] as? Bool ?? false) && (p?["live"] as? Bool ?? false)
        if Date().timeIntervalSince(pilotAt) > 1.5 {
            if pilotOn, !on, let why = p?["error"] as? String { say(plain(why)) }      // it stopped by itself: say why
            set(\.pilotOn, on)
        }
        if pilotOn, on, let p {
            set(\.pilotState, p["state"] as? String ?? "")
            let v = p["cmd"] as? [Double] ?? [0, 0]
            set(\.pilotSpeed, (hypot(v[0], v[1]) * 20).rounded() / 20)
            set(\.pilotDown, (p["tilt"] as? Double ?? 0) > 60)
            set(\.pilotFalls, p["falls"] as? Int ?? 0)
            if let o = p["odom"] as? [Double], o.count >= 3 { set(\.pilotPose, String(format: "%.2f %.2f %.0f", o[0], o[1], o[2] * 180 / .pi)) }
            if Date().timeIntervalSince(pressAt) > 1.3 { pilotLevel = p["level"] as? Int ?? pilotLevel }
        } else if !pilotOn { set(\.pilotState, ""); set(\.pilotSpeed, 0); set(\.pilotDown, false) }
        set(\.clips, brainOn ? (j["anims"] as? [[String: Any]] ?? []).map {
            Emote(name: $0["name"] as? String ?? "", emoji: $0["emoji"] as? String ?? "", label: $0["label"] as? String ?? "", ok: true) } : [])
        if let a = s["anim"] as? [String: Any], let name = a["name"] as? String, let t = a["t"] as? Double, let T = a["T"] as? Double {
            if clipNow != name { clipNow = name; clipClock = (Date().addingTimeInterval(-t), max(0.1, T)) }
        } else if !clipNow.isEmpty, clipClock.map({ Date().timeIntervalSince($0.start) > 1.2 }) ?? true { clipNow = ""; clipClock = nil }
        let bt = j["battery"] as? [String: Any], was = battery
        if bt != nil || !rtUp {                                  // in Run the reading is the runtime's (health())
            set(\.battery, bt?["percent"] as? Int)
            set(\.volts, bt?["volts"] as? Double ?? 0)
        }
        if let b = battery, b < 15, (was ?? 100) >= 15 { say("Battery \(b)%: charge it") }
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
        // the 3-D duck takes the real joints' angles once calibrated; while calibrating it stays at home, as the target
        if target == .real {
            let live = calSaved != nil && !calOn
            scene.joints(live ? m.compactMapValues(\.deg) : [:])
        }
        if !brainOn, open != nil { open = nil }
        look()
    }

    func look() {
        scene.apply(.init(live: target == .sim, ids: Set(ids), answering: Set(motors.keys), sel: sel,
                          sensors: brainOn ? sensors.mapValues(\.ok) : nil, open: open, dark: dark, fov: sel != nil ? 52 : mode == .play && (target == .sim ? liveUp : realPlay) ? 41 : target == .real && busOn && mode == .duck ? 29 : 26))
    }

    private func modeChanged() {
        pause()                                             // never leave the duck walking behind a hidden joystick
        if target == .real { scene.stand() }
        look()
    }

    // ---- studio commands ----------------------------------------------------------------------------------------------
    func toggleBus(quiet: Bool = false) async {
        busy = true; defer { busy = false }
        if busOn { auto = false; await call("disconnect") }
        else if let port = ports.first(where: { $0 == "duck" }) ?? ports.first {
            if let j = await call("connect", ["port": port, "baud": 1_000_000], quiet: quiet) { let n = (j["ids"] as? [Int])?.count ?? 0; say(n > 0 ? "\(n) found" : "No servos answered") }
        } else if !quiet { say("No adapter") }
    }

    /// The Connect button: the duck's computer first (its servo port then shows up), then the servo bus.
    func connectReal(quiet: Bool = false) async {
        busy = true
        if !studioUp, !(await search()) { busy = false; say("Duck can’t be reached"); return }
        if !brainOn { await call("brain_connect", quiet: quiet) }
        busy = false
        if !busOn {
            if runMode == "run" { if !quiet { say("Duck is in Run. Switch to Setup") } } else { await toggleBus(quiet: quiet) }
        }
    }

    func toggleBrain() async { busy = true; defer { busy = false }; if brainOn { auto = false }; await call(brainOn ? "brain_disconnect" : "brain_connect") }

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

    /// A step of the slider: 2 ticks (0.2 deg), for trimming a joint.
    func nudge(_ ticks: Double) { if let sel, let m = motors[sel] { Task { await call("move", ["id": sel, "pos": Int((m.torque ? m.goal : m.pos) + ticks)]) } } }

    // ---- calibration: the studio's own flow (firmware/README.md). Only Save and Level write the calibration. ----------
    var allHeld: Bool { !ids.isEmpty && ids.allSatisfy { motors[$0]?.torque == true } }
    func calStart() async { if await call("cal_start") != nil { say("Lay the duck on its back and pose it at home") } }
    func calHold(_ on: Bool) async { await call("cal_hold", ["on": on]) }
    func calCancel() async { await call("cal_cancel") }
    func calSave() async {
        guard let j = await call("cal_save") else { return }
        let imu = (j["cal"] as? [String: Any])?["imu"] as? Bool ?? false
        say("Calibration saved" + (brainOn ? " on the duck" : "") + (imu ? "" : " (no IMU)"))
    }
    /// Reverse a calibrated joint's direction (its angle changes sign), when it turns the wrong way on screen.
    func flip() async { if let sel, await call("flip", ["id": sel]) != nil { say("\(sel) reversed") } }
    /// The IMU's orientation from gravity in two poses (firmware/studio.py `imu_cal`): 1 = upright, trunk level;
    /// 2 = on its back, beak straight up. Step 2 writes the calibration.
    var imuStep = 0
    var imuOld = false                                    // calibrated the one-pose way: the front may point the wrong way
    @ObservationIgnored private var imuNag = Date()
    func imuCal(_ step: Int) async {
        guard let j = await call("imu_cal", ["step": step]) else { return }
        imuStep = ((j["cal"] as? [String: Any])?["imu_step"] as? Int) ?? 0
        say(step == 1 ? "Upright taken. Now on its back, beak up, then 2" : "IMU calibrated from two poses")
    }
    /// Turn the calibrated IMU's body frame (`imu_turn`): a quarter turn of the front to the left (z, 90), or a half
    /// turn about the front (x), left (y) or up (z) axis, for a front that points the wrong way or a reading upside down.
    func imuTurn(_ axis: String, _ deg: Int = 180) async {
        if await call("imu_turn", ["axis": axis, "deg": deg], short: true) != nil { say(deg == 90 ? "Front turned a quarter left" : "Flipped about \(axis.uppercased())") }
    }

    /// Setup: the studio has the servos and sensors and the duck's runtime is paused. Run: the runtime drives the duck.
    func setRun(_ to: String) async {
        guard to != runMode else { return }
        busy = true; defer { busy = false }
        if to == "run" { auto = false }
        if await call("mode", ["to": to]) != nil, mode == .duck { say(to == "run" ? "Run: the duck drives itself" : "Setup: servos and sensors are yours") }
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
            // the trace is drawn against the loudest recent sample (falling 10 % a reading), so a quiet room still shows
            // its signal and a shout does not clip; never below 0.002 of full scale, so silence stays a flat line
            micPeak = max(wave.map(abs).max() ?? 0, micPeak * 0.9, 0.002)
        default: break
        }
    }

    // ---- the engine ------------------------------------------------------------------------------------------------------
    private func dropLive() { ws?.cancel(with: .goingAway, reason: nil); ws = nil; liveUp = false; lastKeys = "" }

    private func q(_ path: String) -> String { path + "?token=" + token }

    private func openLive() async {
        guard let live, let h = try? dict(await fetch(live, q("/api/health"))), let port = h["ws_port"] as? Int,
              let sc = try? dict(await fetch(live, q("/api/scene"))), live == self.live else { return }
        _ = h
        bodies = sc["bodies"] as? [String] ?? []
        emotes = (sc["anims"] as? [[String: Any]] ?? []).map {
            Emote(name: $0["name"] as? String ?? "", emoji: $0["emoji"] as? String ?? "", label: $0["label"] as? String ?? "", ok: $0["available"] as? Bool ?? false) }
        var c = URLComponents(); c.scheme = "ws"; c.host = live.host; c.port = port; c.path = "/"
        c.queryItems = [URLQueryItem(name: "token", value: token)]
        let t = http.webSocketTask(with: c.url!)
        ws = t; t.resume(); listen(t)
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
        if !liveUp { liveUp = true; lastKeys = ""; sendStick() }
        set(\.state, s["state"] as? String ?? "")
        set(\.level, s["level"] as? Int ?? 3)
        set(\.down, s["down"] as? Bool ?? false)
        set(\.has, s["has"] as? [String: Bool] ?? [:])
        let v = s["v"] as? [Double] ?? [0, 0]
        set(\.speed, (hypot(v[0], v[1]) * 20).rounded() / 20)
        set(\.odometer, ((s["odometer"] as? Double ?? 0) * 100).rounded() / 100)
        set(\.rate, (((s["rate"] as? Double ?? 0) / 50) * 20).rounded() / 20)
        set(\.liveStep, ((s["step_ms"] as? Double ?? 0) * 10).rounded() / 10)
        let a = s["anim"] as? [String: Any]
        set(\.playing, a?["name"] as? String ?? "")
        set(\.progress, a.map { (((($0["t"] as? Double ?? 0) / max(0.1, $0["T"] as? Double ?? 1)) * 25).rounded() / 25) } ?? 0)
        guard let p = s["pos"] as? [Double], let q = s["quat"] as? [Double], p.count >= 3, q.count >= 4 else { return }
        set(\.pose, String(format: "%.2f %.2f %.0f", p[0], p[1], atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3])) * 180 / .pi))
        if target == .sim { scene.pose(bodies, p, q); scene.mouth((s["mouth"] as? Double ?? 0) * scene.jawOpen) }
    }

    private func send(_ m: [String: Any]) {
        guard liveUp, let d = try? JSONSerialization.data(withJSONObject: m), let s = String(data: d, encoding: .utf8) else { return }
        ws?.send(.string(s)) { _ in }
    }

    /// The two sticks. Left: x = sidestep right, y = forward. Right: w = turn clockwise. Each -1...1, a fraction of the
    /// speed level's walk, sidestep and turn speed (engine.Duck.set_stick). Running is straight only, so the run switch
    /// acts on a forward push with nothing else deflected.
    func drive(_ x: Double, _ y: Double) { stick = (x, y); sendStick(); sendTwist(); sendPilot() }
    func turn(_ w: Double) { spin = w; sendStick(); sendTwist(); sendPilot() }

    private func sendStick() {
        let r = { (v: Double) in (v * 50).rounded() / 50 }
        let x = r(stick.0), y = r(stick.1), w = r(spin), rest = x == 0 && y == 0 && w == 0
        let fast = run && y > 0.35 && abs(x) < 0.12 && abs(w) < 0.12
        let key = "\(x) \(y) \(w) \(fast)"
        guard liveUp, key != lastKeys, rest || Date().timeIntervalSince(sentAt) > 0.03 else { return }      // 30 a second, and always the release
        lastKeys = key; sentAt = Date()
        send(["t": "stick", "x": x, "y": y, "w": w, "run": fast])
    }

    func press(_ key: String) { send(["t": "press", "key": key]) }
    func emote(_ name: String) { send(["t": "emote", "name": name]) }

    func setLevel(_ n: Int) { guard n != level else { return }; level = n; Task { if let live { _ = try? await fetch(live, q("/api/config"), ["level": n]) } } }

    /// The app left the screen: stop the duck.
    func pause() { stick = (0, 0); spin = 0; sendTwist(); sendPilot(); if run { run = false } else { sendStick() } }

    // ---- the duck's runtime ------------------------------------------------------------------------------------------------
    // In Run the duck's own control daemon has the servos and runs the policies. Its console (port 8090 on the duck)
    // relays JSON-RPC 2.0 over a WebSocket at /ws: requests `hello`, `robot.subscribe`, `robot.health`, `robot.init`
    // (power the joints, go to the stance), `robot.enable` (walk / hold), `robot.relax`, `robot.do` (a skill), the
    // notification `robot.move` (a velocity), and a stream of `robot.state` back.
    private var wantRt: Bool { target == .real && mode == .play && brainOn && runMode == "run" }

    /// Ten times a second: keep the socket while Play shows a duck in Run, and repeat a held stick (the runtime stops
    /// the duck by itself when the commands cease).
    private func runtime() async {
        guard wantRt else {
            if rt != nil { dropRt() }
            if pilotOn, target == .real, mode == .play { sendPilot(keep: true); matchLevel() }      // a held stick is repeated: the pilot lets go after 0.5 s of silence
            return
        }
        if rt == nil { await openRt(); return }
        sendTwist(keep: true)
        if rtUp, Date().timeIntervalSince(rtAsked) > 2 { rtAsked = Date(); Task { await health() } }
    }

    private func dropRt() {
        if rtUp { tell("robot.move", ["vx": 0, "vy": 0, "vyaw": 0]) }
        rt?.cancel(with: .goingAway, reason: nil); rt = nil; rtTwist = ""
        for (_, c) in rtWait { c.resume(returning: nil) }
        rtWait = [:]
        rtUp = false; rtWalk = false; rtPower = false; rtLimp = false; rtFallen = false; rtPolicy = ""; rtSpeed = 0; rtReady = true; rtPose = ""
        if target == .real { scene.stand() }
    }

    private func openRt() async {
        guard let host = studio?.host else { return }
        let port = mock ? Self.runtimePort.mock : Self.runtimePort.duck
        // the duck is where its own studio is; through a studio on a Mac, it is at its name on the Wi-Fi
        let hosts = [host] + (mock || brainHost.isEmpty ? [] : [brainHost + ".local"])
        for h in hosts {
            guard let u = URL(string: "http://\(h):\(port)/api/meta"), let d = try? await quick.data(from: u).0, wantRt, rt == nil else { continue }
            let meta = dict(d)
            guard let list = meta["joints"] as? [[String: Any]], let ws = URL(string: "ws://\(h):\(port)/ws") else { continue }
            rtOrder = list.map { $0["id"] as? Int ?? 0 }
            let t = http.webSocketTask(with: ws)
            rt = t; t.resume(); hear(t)
            guard await rpc("hello", ["api_version": meta["api_version"] as? Int ?? 17]) != nil, rt === t else { if rt === t { dropRt() }; continue }
            _ = await rpc("robot.subscribe", ["hz": 10])
            guard rt === t else { return }
            rtUp = true; rtAsked = Date()
            await health()
            return
        }
    }

    private func hear(_ t: URLSessionWebSocketTask) {
        t.receive { [weak self] r in
            Task { @MainActor in
                guard let self, self.rt === t else { return }
                switch r {
                case .success(.string(let s)): self.heard(s); self.hear(t)
                case .success: self.hear(t)
                case .failure: self.dropRt()
                }
            }
        }
    }

    private func heard(_ text: String) {
        guard let m = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else { return }
        if m["method"] as? String == "robot.state", let p = m["params"] as? [String: Any] { rtState(p) }
        else if let id = m["id"] as? Int, let c = rtWait.removeValue(forKey: id) { c.resume(returning: text) }
    }

    private func rtState(_ p: [String: Any]) {
        let safe = p["safety"] as? [String: Any] ?? [:]
        set(\.rtPolicy, p["policy"] as? String ?? "")
        set(\.rtLimp, safe["limp"] as? Bool ?? false)
        set(\.rtFallen, safe["fallen"] as? Bool ?? false)
        if rtLimp { if rtPower { rtPower = false; rtWalk = false } }
        else if Self.active.contains(rtPolicy) {             // found already up, or walking
            if !rtPower { rtPower = true }
            if rtPolicy == "walk", !rtWalk { rtWalk = true }
        }
        let v = (p["move"] as? [String: Any])?["applied"] as? [Double] ?? []
        set(\.rtSpeed, v.count >= 2 ? (hypot(v[0], v[1]) * 100).rounded() / 100 : 0)
        if let o = p["odom"] as? [Double], o.count >= 3 { set(\.rtPose, String(format: "%.2f %.2f %.0f", o[0], o[1], o[2] * 180 / .pi)) }
    }

    /// Battery and whether the servos answer: in Run only the runtime can tell.
    private func health() async {
        guard let text = await rpc("robot.health"), let h = dict(Data(text.utf8))["result"] as? [String: Any] else { return }
        set(\.rtReady, !((h["reason"] as? String ?? "").contains("motor bus")))
        if let b = h["battery"] as? [String: Any], let pc = b["percent"] as? Double {
            set(\.battery, Int(pc.rounded())); set(\.volts, b["volts"] as? Double ?? 0)
        }
    }

    /// A request: the reply's text, or nil after 8 s without one.
    private func rpc(_ method: String, _ params: [String: Any] = [:]) async -> String? {
        guard let rt else { return nil }
        rtId += 1
        let id = rtId
        guard let d = try? JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": id, "method": method, "params": params]),
              let s = String(data: d, encoding: .utf8) else { return nil }
        return await withCheckedContinuation { c in
            rtWait[id] = c
            rt.send(.string(s)) { _ in }
            Task { try? await Task.sleep(for: .seconds(8)); self.rtWait.removeValue(forKey: id)?.resume(returning: nil) }
        }
    }

    /// A notification: no reply.
    private func tell(_ method: String, _ params: [String: Any]) {
        guard let rt, let d = try? JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "method": method, "params": params]),
              let s = String(data: d, encoding: .utf8) else { return }
        rt.send(.string(s)) { _ in }
    }

    /// A command the runtime may refuse; a refusal is shown in its own words when they are plain text.
    @discardableResult private func ask(_ method: String, _ params: [String: Any] = [:]) async -> Bool {
        guard rtUp else { return false }
        rtBusy = true; defer { rtBusy = false }
        guard let text = await rpc(method, params) else { say("Duck didn’t answer"); return false }
        let j = dict(Data(text.utf8)), r = j["result"] as? [String: Any] ?? [:]
        let why = (j["error"] as? [String: Any])?["message"] as? String ?? (r["accepted"] as? Bool == false ? r["reason"] as? String ?? "" : nil)
        guard let why else { return true }
        say(why.isEmpty || why.count > 60 || !why.allSatisfy(\.isASCII) ? "Not now" : why.prefix(1).uppercased() + why.dropFirst())
        return false
    }

    /// Power the joints and go to the stance (about two seconds): hold the duck, or have it on its feet.
    func rtStand() async {
        if await ask("robot.init") { rtPower = true }
    }
    /// Policies that only run on powered joints.
    static let active: Set<String> = ["walk", "stand", "sitstand", "roulade", "kick_left", "kick_right", "ground_pick"]
    /// Cut the power: an unsupported duck drops.
    func rtRelax() async {
        pause()
        if await ask("robot.relax") { rtWalk = false; rtPower = false }
    }
    func rtSetWalk(_ on: Bool) async {
        if !on { pause() }; if await ask("robot.enable", ["on": on, "toggle": false]) { rtWalk = on }
    }
    /// `sit_toggle`, `roulade`, `kick_left`, `kick_right` or `ground_pick`.
    func rtDo(_ skill: String) async {
        await ask("robot.do", ["skill": skill])
    }

    /// The sticks as a velocity: forward, to the left [m/s] and anticlockwise [rad/s], the runtime's axes. Sent when
    /// it changes (20 a second at most, the release always) and, with `keep`, again every 0.1 s while it is not zero.
    private func sendTwist(keep: Bool = false) {
        guard rt != nil, rtUp else { return }
        let k = Double(gait) / 5, r = { (v: Double) in (v * 100).rounded() / 100 }
        let vx = r(stick.1 * Self.reach.0 * k), vy = r(-stick.0 * Self.reach.1 * k), w = r(-spin * Self.reach.2 * k)
        let rest = vx == 0 && vy == 0 && w == 0, key = "\(vx) \(vy) \(w)", age = Date().timeIntervalSince(rtSent)
        if keep { guard !rest, age >= 0.09 else { return } }
        else { guard key != rtTwist, rest || age > 0.05 else { return } }
        guard rtWalk || (rest && !rtTwist.isEmpty) else { return }
        rtTwist = rest ? "" : key; rtSent = Date()
        tell("robot.move", ["vx": vx, "vy": vy, "vyaw": w])
    }

    // ---- the duck's pilot and animations (Setup) ----------------------------------------------------------------------------
    // The policies trained for this duck run on the duck (firmware/duck_agent.Drive). The studio only passes the driver's
    // inputs on: `pilot_start {live}`, `pilot_stop`, `pilot_drive {x, y, w, run}` (the two sticks, -1...1, the same axes as
    // the simulated duck's), `pilot_press {key}` (sit, jump, flip, faster, slower); `anim_play {name}` and `anim_stop`.

    /// Power: the policies take the joints where they are and blend to their stance over 2 s. Refused with the duck's
    /// own reason (not calibrated, no policies on it, servo power off).
    func pilotStart() async {
        pilotBusy = true; defer { pilotBusy = false }
        guard await call("pilot_start", ["live": true], short: true) != nil else { return }
        pilotAt = Date(); pilotOn = true; pilotState = "stand"; clipNow = ""; clipClock = nil
        pilotLevel = 3; pressAt = .distantPast; gaitUntil = Date().addingTimeInterval(4)       // a new pilot starts at level 3
    }

    /// Every joint goes limp: an unsupported duck drops.
    func pilotStop() async {
        pause()
        pilotBusy = true; defer { pilotBusy = false }
        if await call("pilot_stop", short: true) != nil { pilotAt = Date(); pilotOn = false; pilotState = ""; pilotSpeed = 0 }
    }

    /// `sit` (down, or up again), `jump` or `flip` (both from a stand).
    func pilotPress(_ key: String) async { await call("pilot_press", ["key": key], short: true) }

    func setGait(_ n: Int) { guard n != gait else { return }; gait = n; gaitUntil = Date().addingTimeInterval(4) }

    /// The pilot's speed level follows the slider: one `faster` or `slower` at a time until it reads the same. Outside
    /// the 4 s after the slider moved (or the pilot started), a level changed elsewhere moves the slider instead.
    private func matchLevel() {
        guard pilotLevel != gait else { return }
        guard Date() < gaitUntil else { if Date().timeIntervalSince(pressAt) > 1.3 { gait = pilotLevel }; return }
        guard Date().timeIntervalSince(pressAt) > 0.3 else { return }
        let up = gait > pilotLevel
        pressAt = Date(); pilotLevel += up ? 1 : -1
        Task { await call("pilot_press", ["key": up ? "faster" : "slower"], quiet: true) }
    }

    /// The sticks to the pilot: when they change (20 a second at most, the release always) and, with `keep`, again
    /// every 0.1 s while one is held. The run switch rides along; the pilot runs only straight ahead.
    private func sendPilot(keep: Bool = false) {
        guard pilotOn, target == .real, let studio else { return }
        let r = { (v: Double) in (v * 50).rounded() / 50 }
        let x = r(stick.0), y = r(stick.1), w = r(spin), rest = x == 0 && y == 0 && w == 0
        let key = "\(x) \(y) \(w) \(run)", age = Date().timeIntervalSince(rtSent)
        if keep { guard !rest, age >= 0.09 else { return } }
        else { guard key != rtTwist, rest || age > 0.05, !rest || !rtTwist.isEmpty else { return } }
        rtTwist = rest ? "" : key; rtSent = Date()
        Task { _ = try? await fetch(studio, "/api/pilot_drive", ["x": x, "y": y, "w": w, "run": run]) }
    }

    /// Play one of the duck's animations (head, beak and sound; the legs are not moved); the one playing: stop it and
    /// let the head go limp.
    func clip(_ name: String) async {
        if clipNow == name { if await call("anim_stop", short: true) != nil { clipNow = ""; clipClock = nil }; return }
        guard let j = await call("anim_play", ["name": name], short: true), let a = j["anim"] as? [String: Any] else { return }
        clipNow = name; clipClock = (Date().addingTimeInterval(0.5), max(0.1, a["T"] as? Double ?? 1))      // the duck eases in for 0.5 s first
    }
}
