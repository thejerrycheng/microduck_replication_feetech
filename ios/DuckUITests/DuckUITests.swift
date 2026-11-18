// Drives the app in the simulator against the mock studio and the real engine (see tools/test.sh, which starts both).
// Every test asserts on what the servers did, read back through the app, and saves a screenshot. Lines that start with
// "METRIC" are the numbers in docs/duck_ios_app.tex (tools/plot_tests.py reads them from the log).
import XCTest

final class DuckUITests: XCTestCase {
    let app = XCUIApplication()
    let env = ProcessInfo.processInfo.environment

    override func setUp() { continueAfterFailure = false }

    func launch(_ mode: String, _ target: String = "real") {
        app.launchArguments = ["-mode", mode, "-target", target, "-mockOnly", "1"]      // these tests switch torque and write zeros: mock only
        app.launch()
    }

    func el(_ id: String) -> XCUIElement { app.descendants(matching: .any).matching(identifier: id).firstMatch }
    func value(_ id: String) -> String { el(id).value as? String ?? "" }

    @discardableResult func wait(_ what: String, _ timeout: TimeInterval = 12, file: StaticString = #filePath, line: UInt = #line, _ ok: () -> Bool) -> TimeInterval {
        let t0 = Date()
        while Date().timeIntervalSince(t0) < timeout { if ok() { return Date().timeIntervalSince(t0) }; usleep(100_000) }
        XCTFail("timed out: \(what)", file: file, line: line)
        return timeout
    }

    var m: [String: Double] = [:]
    func metric(_ name: String, _ v: Double) { m[name] = v; print(String(format: "METRIC %@ %.3f", name, v)) }

    func shot(_ name: String) {
        let png = app.screenshot().pngRepresentation
        if let dir = env["SHOTS"] { try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name + (env["DARK"] == nil ? "" : "_dark") + ".png")) }
        let a = XCTAttachment(data: png, uniformTypeIdentifier: "public.png"); a.name = name; a.lifetime = .keepAlways; add(a)
    }

    func link(_ key: String) -> Bool { value("link").contains(key) }

    /// Open the connection pill, flip one switch, close it again.
    func flip(_ id: String) {
        el("link").tap()
        let sw = el(id)
        XCTAssertTrue(sw.waitForExistence(timeout: 4))
        sw.switches.firstMatch.exists ? sw.switches.firstMatch.tap() : sw.tap()
        usleep(1_500_000)
        app.coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.62)).tap()            // anywhere outside closes it
        XCTAssertTrue(el("link").waitForExistence(timeout: 4))
    }

    /// The studio the app found must be the mock: nothing below may ever run against real servos.
    func requireMock() {
        wait("studio found", 25) { link("studio=true") }
        XCTAssertTrue(link("mock=true"), "refusing to test against a real studio")
    }

    /// A call straight to the mock studio (tools/test.sh runs it on 8780), for what the app has no control for.
    func mockStudio(_ name: String, _ body: [String: Any]) {
        var r = URLRequest(url: URL(string: "http://127.0.0.1:8780/api/" + name)!)
        r.httpMethod = "POST"; r.httpBody = try? JSONSerialization.data(withJSONObject: body)
        let done = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: r) { _, _, _ in done.signal() }.resume()
        _ = done.wait(timeout: .now() + 5)
    }

    func connectBus() {
        requireMock()
        if !link("bus=true") { flip("bus") }
        wait("bus on") { link("bus=true") }
    }

    // ---- 1. connections --------------------------------------------------------------------------------------------
    func testLinks() {
        let t0 = Date()
        launch("duck")
        wait("studio and engine found over Bonjour", 25) { link("studio=true") && link("engine=true") }
        metric("discover_s", Date().timeIntervalSince(t0))
        requireMock()
        XCTAssertTrue(link("bus=false") || link("bus=true"))
        if link("bus=true") { flip("bus"); wait("bus off") { link("bus=false") } }
        XCTAssertFalse(el("servo.10").exists)
        shot("01_idle")
        flip("bus")
        wait("14 of 15 joints on the mock bus") { link("servos=14") }
        XCTAssertEqual(value("servo.10"), "ok")
        XCTAssertEqual(value("servo.24"), "absent", "the mock has no ID 24")
        XCTAssertEqual(value("servo.1"), "ok", "the mock's factory-ID servo is listed")
        for id in [10, 14, 20, 24, 30, 34] { XCTAssertTrue(el("servo.\(id)").isHittable, "\(id) is on screen without scrolling") }
        if !link("brain=true") { flip("brain") }
        wait("duck computer on") { link("brain=true") }
        wait("sensors report") { value("sensor.camera") == "ok" }
        XCTAssertFalse(el("connect").exists, "nothing left to connect")
        // Setup / Run: who has the duck's servos, the studio or the duck's own runtime
        el("link").tap()
        XCTAssertTrue(el("runmode").waitForExistence(timeout: 4))
        XCTAssertEqual(value("runmode"), "setup")
        XCTAssertFalse(el("calibrate").isEnabled, "Calibrate needs all 15 servos; the mock has 14")
        app.segmentedControls["runmode"].buttons["Run"].tap()
        wait("run mode") { value("runmode") == "run" }
        app.segmentedControls["runmode"].buttons["Setup"].tap()
        wait("setup mode") { value("runmode") == "setup" }
        app.coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.72)).tap()
        XCTAssertTrue(el("link").waitForExistence(timeout: 4))
        // battery: the studio's estimate from the servo supply (the mock's 7.9 V), in the bar and in the panel
        wait("battery shown") { (Int(value("battery")) ?? -1) > 0 && (Int(value("battery")) ?? 101) <= 100 }
        metric("battery_pct", el("battery").exists ? Double(value("battery")) ?? -1 : -1)
        // temperature: the mock's processor is at 41 degC and its servos at 31-35, so nothing is warm
        wait("temperature shown: \(value("temp"))") { value("temp") == "41 heat=0" }
        el("link").tap()
        XCTAssertTrue(app.staticTexts["Battery"].waitForExistence(timeout: 4))
        XCTAssertEqual(value("heat.cpu"), "41")
        let hot = value("heat.servo").split(separator: " ").compactMap { Int($0) }
        XCTAssertEqual(hot.count, 2); XCTAssertTrue((31...35).contains(hot.last ?? 0), "hottest servo: \(hot)")
        metric("servo_hottest_c", Double(hot.last ?? -1))
        shot("02_links")
        app.coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.62)).tap()
        shot("03_connected")
        // Connect: with the bus and the computer both off, one tap brings both back
        flip("bus"); flip("brain")
        wait("both off") { link("bus=false") && link("brain=false") }
        XCTAssertTrue(el("connect").waitForExistence(timeout: 4))
        shot("03b_disconnected")
        el("connect").tap()
        wait("Connect brought the computer and the bus up", 20) { link("bus=true") && link("brain=true") && link("servos=14") }
        wait("Connect button gone") { !el("connect").exists }
        // the target switch: real is the default; Sim shows the simulated duck and its state, Play then drives it
        XCTAssertEqual(value("target"), "real")
        app.segmentedControls["target"].buttons["Sim"].tap()
        wait("sim selected") { value("target") == "sim" && el("state").exists && !el("servo.10").exists }
        shot("03c_sim")
        el("mode.play").tap()
        XCTAssertTrue(el("stick").waitForExistence(timeout: 4))
        app.segmentedControls["target"].buttons["Duck"].tap()
        wait("real selected: Play has the duck's own pad, power off") { value("target") == "real" && value("pilot.power") == "off" && !el("state").exists }
        shot("03d_real_play")
        el("mode.duck").tap()
        wait("back on the real duck") { value("target") == "real" && el("servo.10").exists }
        flip("brain")
        wait("duck computer off") { link("brain=false") }
        XCTAssertEqual(value("sensor.camera"), "off")
        flip("brain")
        wait("duck computer on again") { link("brain=true") }
    }

    // ---- 1b. calibration: needs all 15 joints, so the mock's factory-ID servo is first given the missing ID 24 ----
    func testCalibrate() {
        launch("duck")
        connectBus()
        if link("servos=14") { mockStudio("set_id", ["id": 1, "to": 24]) }
        wait("15 of 15 joints") { link("servos=15") }
        el("link").tap()
        XCTAssertTrue(el("calibrate").waitForExistence(timeout: 4))
        wait("Calibrate enabled") { el("calibrate").isEnabled }
        el("calibrate").tap()
        XCTAssertTrue(el("cal.bar").waitForExistence(timeout: 5))
        wait("every servo limp") { (el("cal.hold").value as? String) == "0" }
        shot("05b_cal_start")
        el("cal.hold").switches.firstMatch.tap()
        wait("every servo holds") { (el("cal.hold").value as? String) == "1" }
        // trim one joint: its card has the slider and the nudges, but no Test or Zero while calibrating
        el("servo.22").tap()
        XCTAssertTrue(el("card.servo").waitForExistence(timeout: 4))
        XCTAssertFalse(el("test").exists); XCTAssertFalse(el("zero").exists)
        let p0 = Int(value("pos")) ?? 0
        for _ in 0..<3 { el("nudge.up").tap(); usleep(400_000) }
        wait("trimmed 6 ticks from \(p0), reads \(value("pos"))") { Int(value("pos")) == p0 + 6 }
        shot("05c_cal_trim")
        el("cal.save").tap()
        wait("saved") { el("toast").exists && el("toast").label.contains("Calibration saved") }
        wait("bar gone, torque released") { !el("cal.bar").exists && (el("torque").value as? String) == "0" }
        wait("the card now shows the joint angle, zero at the saved pose: \(el("pos").label)") { el("pos").label.hasPrefix("0.0°") }
        XCTAssertTrue(el("test").exists)
        shot("05d_calibrated")
        // Flip: the joint's direction reverses, so a nudge now changes the angle the other way
        el("torque").switches.firstMatch.tap()
        wait("torque on") { (el("torque").value as? String) == "1" }
        for _ in 0..<5 { el("nudge.up").tap(); usleep(350_000) }
        wait("angle after 10 ticks up: \(el("pos").label)") { !el("pos").label.hasPrefix("0.0°") }
        let before = el("pos").label.hasPrefix("-")
        el("flip").tap()
        wait("reversed") { el("toast").exists && el("toast").label.contains("reversed") }
        wait("the angle changed sign: \(el("pos").label)") { el("pos").label.hasPrefix("-") != before && !el("pos").label.hasPrefix("0.0°") }
        el("flip").tap()                                            // and back
        wait("sign restored") { el("pos").label.hasPrefix("-") == before }
        // again, and Cancel: nothing changes
        el("servo.22").tap()
        el("link").tap()
        XCTAssertTrue(el("calibrate").waitForExistence(timeout: 4))
        el("calibrate").tap()
        XCTAssertTrue(el("cal.bar").waitForExistence(timeout: 5))
        el("cal.cancel").tap()
        wait("cancelled") { !el("cal.bar").exists }
        mockStudio("set_id", ["id": 24, "to": 1])                  // the mock as the other tests expect it
        wait("14 of 15 again") { link("servos=14") }
    }

    // ---- 2. a servo: telemetry and calibration ---------------------------------------------------------------------
    func testServo() {
        launch("duck")
        connectBus()
        el("servo.22").tap()
        XCTAssertTrue(el("card.servo").waitForExistence(timeout: 4))
        XCTAssertEqual(value("servo.22"), "selected")
        // telemetry reaches the plots: count the samples that arrive in 3 s (the studio samples at 40 Hz)
        let n0 = Int(value("plot.Position")) ?? 0, t0 = Date()
        sleep(3)
        let n1 = Int(value("plot.Position")) ?? 0
        metric("telemetry_hz", Double(n1 - n0) / Date().timeIntervalSince(t0))
        XCTAssertGreaterThan(n1 - n0, 60)
        // torque, then the slider: the goal lands where the slider was put (joint 22 spans 1024-3072 ticks)
        XCTAssertNotNil(Int(value("pos")), "the position is a plain number of ticks")
        if (el("torque").value as? String) == "1" { el("torque").switches.firstMatch.tap(); wait("torque off") { (el("torque").value as? String) == "0" } }
        el("torque").switches.firstMatch.tap()
        wait("torque on") { (el("torque").value as? String) == "1" }
        // the slider drives the goal: three settings land in order, well apart, inside the joint's travel.
        // (XCUITest sets a slider only to within a few percent, and the travel moves with the calibration.)
        var at: [Int] = []
        for f in [0.8, 0.2, 0.5] {
            let before = Int(value("pos")) ?? 0
            app.sliders["slide"].adjust(toNormalizedSliderPosition: f)
            wait("slider at \(f) moved the servo from \(before)") { abs((Int(value("pos")) ?? before) - before) > 200 }
            usleep(600_000)
            at.append(Int(value("pos")) ?? 0)
        }
        XCTAssertGreaterThan(at[0] - at[2], 300, "0.8 above 0.5: \(at)"); XCTAssertGreaterThan(at[2] - at[1], 300, "0.5 above 0.2: \(at)")
        XCTAssertTrue(at.allSatisfy { (900...3200).contains($0) }, "inside the travel: \(at)")
        shot("04_servo")
        // the nudges: 2 ticks (0.2 deg) a tap
        let tick0 = Int(value("pos")) ?? 0
        el("nudge.up").tap(); usleep(400_000); el("nudge.up").tap()
        wait("nudged up 4 ticks from \(tick0), reads \(value("pos"))") { Int(value("pos")) == tick0 + 4 }
        el("nudge.down").tap()
        wait("nudged back 2") { Int(value("pos")) == tick0 + 2 }
        // Test: +-10 deg and back, then torque off
        let before = Int(value("pos")) ?? 0
        el("test").tap()
        var seen = Set<Int>()
        let t1 = Date()
        while Date().timeIntervalSince(t1) < 4 { seen.insert(Int(value("pos")) ?? 0); usleep(150_000) }
        XCTAssertGreaterThan((seen.max() ?? 0) - before, 60, "Test swung up")
        wait("Test released the servo", 8) { (el("torque").value as? String) == "0" }
        // gains: the mock reads the factory 32 / 40; hold a value for the replica's, then Set
        wait("gains read") { value("gain.P") == "32" && value("gain.D") == "40" }
        XCTAssertFalse(el("gset").isEnabled)
        el("gain.P").press(forDuration: 0.8); el("gain.D").press(forDuration: 0.8)
        wait("replica gains filled") { value("gain.P") == "5" && value("gain.D") == "20" }
        el("P.plus").tap()
        XCTAssertEqual(value("gain.P"), "6")
        el("P.minus").tap()
        XCTAssertTrue(el("gset").isEnabled)
        el("gset").tap()
        wait("gains written and read back") { !el("gset").isEnabled && value("gain.P") == "5" }
        // Zero asks first; then the pose reads 2048
        el("zero").tap()
        let yes = app.buttons["Zero 22"]
        XCTAssertTrue(yes.waitForExistence(timeout: 4))
        shot("05_zero_ask")
        yes.tap()
        wait("zeroed") { value("pos") == "2048" }
        // another servo, then none
        el("servo.10").tap()
        wait("card follows the selection") { value("servo.10") == "selected" && value("gain.P") == "32" }
        el("servo.10").tap()
        wait("card closed") { !el("card.servo").exists }
        // put ID 22 back the way the next run expects it
        el("servo.22").tap()
        wait("gains read") { value("gain.P") == "5" }
    }

    // ---- 3. sensors ------------------------------------------------------------------------------------------------
    func testSensors() {
        launch("duck")
        requireMock()
        if !link("brain=true") { flip("brain") }
        wait("sensors report") { value("sensor.camera") == "ok" }
        for (k, want) in [("camera", "240×320"), ("tof", " m"), ("imu", "°"), ("mic", " dB")] {
            el("sensor.\(k)").coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.5)).tap()
            wait("\(k) reading", 8) { (el("note.\(k)").label).contains(want) }
            if k == "camera" { XCTAssertTrue(el("picture").waitForExistence(timeout: 4)) }
            if k == "imu" {
                // the IMU from two poses. Step 2 waits for step 1; the mock duck is "laid on its back" after step 1
                // (tools/mock_studio.py), as the hands would
                XCTAssertFalse(el("imu.2").isEnabled, "step 2 before step 1")
                el("imu.1").tap()
                wait("upright taken") { el("toast").exists && el("toast").label.contains("Upright taken") }
                wait("step 2 offered") { el("imu.2").isEnabled }
                shot("06_sensor_imu_step")
                el("imu.2").tap()
                wait("calibrated from two poses") { value("imu.how") == "two poses" }
                // the turns: a quarter turn of the front, and a flip about one axis (twice: back as it was)
                el("imu.turn").tap()
                wait("front turned") { el("toast").exists && el("toast").label.contains("quarter") }
                for _ in 0..<2 {
                    usleep(600_000); el("imu.x").tap()
                    wait("flipped") { el("toast").exists && el("toast").label.contains("Flipped about X") }
                }
                for _ in 0..<3 { usleep(400_000); el("imu.turn").tap() }       // four quarter turns: the front is back
                usleep(600_000)
            }
            shot("06_sensor_\(k)")
        }
        el("sensor.speaker").coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.5)).tap()
        for tag in ["chirp", "alarm"] {                    // the mock renders a sound and names the file it played
            XCTAssertTrue(el("say.\(tag)").waitForExistence(timeout: 4))
            el("say.\(tag)").tap()
            usleep(900_000)
        }
        shot("06_sensor_speaker")
        XCTAssertFalse(el("toast").exists, "a sound raised an error: \(el("toast").label)")
        // only one window at a time
        XCTAssertFalse(el("note.mic").exists)
    }

    // ---- 4. play: two sticks, moves, emotions -----------------------------------------------------------------------
    var state: String { el("state").label }
    var parts: [String] { value("state").components(separatedBy: "|") }
    var odometer: Double { Double(parts[0].split(separator: " ").first ?? "") ?? -1 }
    /// Trunk x, y [m] and heading [deg] in the world; the reset puts the duck at 0, 0 facing +x.
    var pose: (x: Double, y: Double, yaw: Double) {
        let v = (parts.count > 1 ? parts[1] : "").split(separator: " ").compactMap { Double($0) }
        return v.count == 3 ? (v[0], v[1], v[2]) : (0, 0, 0)
    }
    var level: Int { Int(parts.last ?? "") ?? 0 }

    /// Hold the left stick at (dx right, dy forward), each -1...1 of its travel.
    func hold(_ dx: Double, _ dy: Double, _ secs: TimeInterval) {
        let s = el("stick")
        s.coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.5))
            .press(forDuration: 0.05, thenDragTo: s.coordinate(withNormalizedOffset: .init(dx: 0.5 + dx * 0.32, dy: 0.5 - dy * 0.32)), withVelocity: .fast, thenHoldForDuration: secs)
    }

    /// Hold the right slider at w (-1 anticlockwise ... 1 clockwise).
    func spin(_ w: Double, _ secs: TimeInterval) {
        let s = el("turn")
        s.coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.5))
            .press(forDuration: 0.05, thenDragTo: s.coordinate(withNormalizedOffset: .init(dx: 0.5 + w * 0.32, dy: 0.5)), withVelocity: .fast, thenHoldForDuration: secs)
    }

    func settle() { wait("standing still", 15) { state == "stand" } ; usleep(400_000) }
    func home() {
        el("reset").tap(); usleep(900_000)
        wait("reset: \(state) \(value("state"))") { state == "stand" && abs(pose.x) < 0.1 && abs(pose.y) < 0.1 }
    }
    func setLevel(_ n: Int) {
        app.sliders["level"].adjust(toNormalizedSliderPosition: Double(n - 1) / 4)
        wait("speed slider at \(n), reads \(level)") { level == n }
    }

    func testPlay() {
        launch("play", "sim")
        wait("engine found", 25) { el("state").exists && state == "stand" }
        setLevel(3)
        home()
        shot("07_play")
        // left stick forward: the duck walks ahead (+x) while it is held, and stops when it is let go
        hold(0, 1, 3); settle()
        var p = pose
        metric("fwd_x_m", p.x); metric("fwd_y_m", p.y); metric("fwd_yaw_deg", p.yaw)
        XCTAssertGreaterThan(p.x, 0.5, "walked forward"); XCTAssertLessThan(abs(p.y), p.x, "mostly ahead")   // the heading drifts: recorded, see the doc
        let d = odometer; usleep(1_500_000)
        XCTAssertLessThan(odometer - d, 0.03, "stopped")
        // left stick back
        home(); hold(0, -1, 3); settle(); p = pose
        metric("back_x_m", p.x)
        XCTAssertLessThan(p.x, -0.3, "walked backward")
        // left stick right, then left: a sidestep (the duck's left is +y), heading kept
        home(); hold(1, 0, 3); settle(); p = pose
        metric("right_y_m", p.y); metric("right_yaw_deg", p.yaw)
        XCTAssertLessThan(p.y, -0.2, "sidestepped right")        // the walk policy also yaws under a pure sidestep command: recorded, not asserted
        home(); hold(-1, 0, 3); settle(); p = pose
        metric("left_y_m", p.y); metric("left_yaw_deg", p.yaw)
        XCTAssertGreaterThan(p.y, 0.2, "sidestepped left")
        // right slider: turns on the spot, clockwise then anticlockwise
        home(); spin(1, 1.5); settle(); p = pose          // 1.5 s: a 3 s turn can pass 180 deg and wrap
        metric("cw_yaw_deg", p.yaw); metric("cw_drift_m", hypot(p.x, p.y))
        XCTAssertLessThan(p.yaw, -30, "turned clockwise")       // how far varies a lot run to run (53-163 deg seen): recorded; XCTAssertLessThan(hypot(p.x, p.y), 0.3, "on the spot")
        home(); spin(-1, 1.5); settle(); p = pose
        metric("ccw_yaw_deg", p.yaw)
        XCTAssertGreaterThan(p.yaw, 30, "turned anticlockwise")
        // proportional: half a push covers clearly less ground than a full one
        home(); hold(0, 0.5, 3); settle()
        metric("half_x_m", pose.x)
        XCTAssertGreaterThan(pose.x, 0.15); XCTAssertLessThan(pose.x, 0.8 * (m["fwd_x_m"] ?? 9))
        shot("07b_walked")
        // sit, stand
        home()
        var t = Date()
        el("act.sit").tap()
        wait("seated", 6) { state == "seated" }
        metric("sit_s", Date().timeIntervalSince(t))
        shot("08_seated")
        t = Date()
        el("act.sit").tap()
        wait("standing", 6) { state == "stand" }
        metric("stand_s", Date().timeIntervalSince(t))
        // jump, flip: the state passes through the move and comes back
        el("act.jump").tap()
        wait("jump", 3) { state == "jump" }
        wait("landed", 8) { state == "stand" }
        el("act.flip").tap()
        wait("flip", 3) { state == "roulade" }
        shot("09_flip")
        wait("on its feet", 10) { state == "stand" }
        // an emotion plays and ends
        el("emote.hello").tap()
        wait("hello plays", 5) { parts[0].contains("hello") }
        shot("10_emote")
        wait("hello ends", 10) { !parts[0].contains("hello") }
        settle()
        // leaving Play stops the duck: the mode bar goes to Duck and back
        el("mode.duck").tap()
        XCTAssertEqual(value("mode.duck"), "selected")
        el("mode.play").tap()
        wait("back in play") { el("stick").exists && state == "stand" }
    }

    /// The speed slider and the run switch: distance covered in a 3 s full forward push, levels 1-5, walk then run.
    func testSpeeds() {
        launch("play", "sim")
        wait("engine found", 25) { el("state").exists && state == "stand" }
        for run in [false, true] {
            if (value("act.run") == "on") != run { el("act.run").tap() }
            for level in 1...5 {
                setLevel(level)
                home()
                hold(0, 1, 3); settle()
                let d = pose.x
                metric("\(run ? "run" : "walk")_m_level\(level)", d)
                XCTAssertGreaterThan(d, 0.15)
            }
            if run { shot("11_run") }
        }
        el("act.run").tap()
        setLevel(3)
        XCTAssertEqual(value("act.run"), "off")
    }

    // ---- 5. Play with the real duck selected: the duck's own runtime drives it. Here the runtime is the stand-in
    // (tools/mock_runtime.py, beside the mock studio), so no servo is involved; the app speaks to it as to the real one.
    var run: [String] { value("run.state").components(separatedBy: "|") }
    var runPose: (x: Double, y: Double, yaw: Double) {
        let v = (run.count > 1 ? run[1] : "").split(separator: " ").compactMap { Double($0) }
        return v.count == 3 ? (v[0], v[1], v[2]) : (0, 0, 0)
    }
    var runSpeed: Double { Double(run[0].components(separatedBy: "v=").last ?? "") ?? -1 }
    func still() { wait("stopped", 6) { runSpeed == 0 }; usleep(300_000) }
    /// How far the duck went since `p0`, in the frame it had there: (forward, to its left) [m]. The stand-in keeps its
    /// pose from one test run to the next, so nothing here may assume it starts at the origin.
    func went(_ p0: (x: Double, y: Double, yaw: Double)) -> (f: Double, l: Double) {
        let p = runPose, a = p0.yaw * .pi / 180, dx = p.x - p0.x, dy = p.y - p0.y
        return (dx * cos(a) + dy * sin(a), -dx * sin(a) + dy * cos(a))
    }

    func testRunPlay() {
        launch("play", "real")
        requireMock()
        wait("duck computer on", 15) { link("brain=true") }
        // the duck is in Setup: Play shows its own policies' pad. Run is the Mode switch under the connection pill
        XCTAssertTrue(el("pilot.power").waitForExistence(timeout: 8))
        var t = runMode("Run")
        wait("runtime reached", 15) { el("run.state").exists }
        metric("rt_ready_s", Date().timeIntervalSince(t))
        XCTAssertTrue(run[0].contains("power=false"), "starts unpowered: \(run[0])")
        XCTAssertFalse(el("run.walk").isEnabled, "no walking while limp")
        XCTAssertFalse(el("skill.roulade").isEnabled)
        shot("12b_run_limp")
        // power: the joints come on and the duck goes to its stance
        t = Date()
        el("run.power").tap()
        wait("standing", 8) { el("run.state").label == "Standing" }
        metric("rt_stand_s", Date().timeIntervalSince(t))
        XCTAssertEqual(value("run.power"), "on")
        // walk: the legs go to the walk policy, and the sticks now steer
        el("run.walk").tap()
        wait("walking") { run[0].contains("walk=true") }
        app.sliders["run.gait"].adjust(toNormalizedSliderPosition: 0.5)
        wait("speed level 3") { run.last == "3" }
        var p0 = runPose
        hold(0, 1, 3); still()
        var p = runPose, d = went(p0)
        metric("rt_fwd_m", d.f); metric("rt_fwd_side_m", d.l)
        XCTAssertEqual(d.f, 0.3 * 0.6 * 3, accuracy: 0.12, "forward at level 3: 0.18 m/s for 3 s")
        XCTAssertLessThan(abs(d.l), 0.03)
        shot("13_run_walked")
        p0 = p; hold(1, 0, 2); still(); p = runPose; d = went(p0)
        metric("rt_right_m", -d.l)
        XCTAssertEqual(-d.l, 0.3 * 0.6 * 2, accuracy: 0.1, "sidestep right is -y")
        p0 = p; spin(1, 1); still(); p = runPose
        metric("rt_cw_deg", p0.yaw - p.yaw)
        XCTAssertEqual(p0.yaw - p.yaw, 1.5 * 0.6 * 1 * 180 / .pi, accuracy: 20, "clockwise is -yaw")
        // the slider scales the speed: level 5 covers more ground than level 1 in the same push
        var far: [Int: Double] = [:]
        for n in [1, 5] {
            app.sliders["run.gait"].adjust(toNormalizedSliderPosition: Double(n - 1) / 4)
            wait("speed level \(n)") { run.last == String(n) }
            p0 = runPose; hold(0, 1, 2); still(); p = runPose
            far[n] = hypot(p.x - p0.x, p.y - p0.y)
            metric("rt_m_level\(n)", far[n]!)
        }
        XCTAssertGreaterThan(far[5]!, 3 * far[1]!)
        app.sliders["run.gait"].adjust(toNormalizedSliderPosition: 0.5)
        // a skill: the policy changes and comes back
        el("skill.roulade").tap()
        wait("rolling", 4) { el("run.state").label == "Rolling" }
        shot("13b_run_skill")
        wait("walking again", 6) { el("run.state").label == "Walking" }
        // leaving Play while pushing the stick stops the duck: the socket closes and the commands cease
        el("mode.duck").tap()
        XCTAssertEqual(value("mode.duck"), "selected")
        el("mode.play").tap()
        wait("back in play", 10) { el("run.state").exists && runSpeed == 0 }
        // power again asks first, then lets go; limp, the wrench returns the duck to Setup
        el("run.power").tap()
        let letGo = app.buttons["Let go"]
        XCTAssertTrue(letGo.waitForExistence(timeout: 4))
        shot("13c_run_drop")
        letGo.tap()
        wait("let go") { run[0].contains("power=false") }
        XCTAssertEqual(value("run.walk"), "off")
        el("run.setup").tap()
        wait("back in Setup", 10) { el("pilot.power").exists }
    }

    /// The Mode switch under the connection pill: "Setup" or "Run". Returns when it was tapped.
    func runMode(_ to: String) -> Date {
        el("link").tap()
        let pick = app.segmentedControls["runmode"]
        XCTAssertTrue(pick.waitForExistence(timeout: 4))
        pick.buttons[to].tap()
        let t = Date()
        usleep(900_000)
        app.coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.62)).tap()            // anywhere outside closes it
        return t
    }

    // ---- 5b. Play with the real duck in Setup: the duck's own policies (its pilot) and its animations. The duck is the
    // stand-in of tools/mock_studio.py: the real pilot code and policy pack, a still upright IMU, no servo.
    var pilot: [String] { value("pilot.state").components(separatedBy: "|") }
    var pilotWord: String { el("pilot.state").label }
    var pilotV: Double { Double(pilot[0].split(separator: " ").first ?? "") ?? -1 }
    var pilotPose: (x: Double, y: Double, yaw: Double) {
        let v = (pilot.count > 1 ? pilot[1] : "").split(separator: " ").compactMap { Double($0) }
        return v.count == 3 ? (v[0], v[1], v[2]) : (0, 0, 0)
    }
    /// The studio reads the duck's state once a second, so every wait here allows for that.
    func rest() { wait("standing still: \(pilotWord) \(value("pilot.state"))", 10) { pilotWord == "stand" && pilotV == 0 }; usleep(1_200_000) }
    func pilotWent(_ p0: (x: Double, y: Double, yaw: Double)) -> (f: Double, l: Double) {
        let p = pilotPose, a = p0.yaw * .pi / 180, dx = p.x - p0.x, dy = p.y - p0.y
        return (dx * cos(a) + dy * sin(a), -dx * sin(a) + dy * cos(a))
    }
    func gait(_ n: Int) {
        app.sliders["pilot.gait"].adjust(toNormalizedSliderPosition: Double(n - 1) / 4)
        wait("speed slider at \(n)") { pilot.count > 2 && pilot[2] == String(n) }
        usleep(2_500_000)                                   // the pilot's level follows, a step at a time
    }
    /// What the stand-in duck was told, from its log (tools/test.sh passes the path): the lines holding `text`.
    func told(_ text: String) -> Int? {
        guard let path = env["PILOTLOG"], let log = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        return log.components(separatedBy: "\n").filter { $0.contains(text) }.count
    }

    func testPilot() {
        launch("play", "real")
        requireMock()
        wait("duck computer on", 15) { link("brain=true") }
        XCTAssertTrue(el("pilot.power").waitForExistence(timeout: 8))
        wait("the duck's animations are listed", 15) { el("emote.hello").exists }
        XCTAssertEqual(value("pilot.power"), "off")
        XCTAssertFalse(el("pilot.jump").isEnabled, "no moves with the power off")
        XCTAssertEqual(pilotWord, "off")
        // power, on a duck that is not calibrated: refused in the duck's words, nothing starts
        mockStudio("connect", ["port": "mock"])
        let start = app.buttons["Start"], letGo = app.buttons["Let go"]
        func power(_ confirm: XCUIElement) {
            el("pilot.power").tap()
            XCTAssertTrue(confirm.waitForExistence(timeout: 4), "power asks first")
        }
        power(start)
        shot("15b_pilot_ask")
        start.tap()
        let t0 = Date()
        var refused = false
        while Date().timeIntervalSince(t0) < 4, value("pilot.power") != "on" {
            if el("toast").exists, el("toast").label.contains("Calibrate first") { refused = true; break }
            usleep(100_000)
        }
        print("METRIC pilot_refused \(refused ? 1 : 0)")
        if refused {
            XCTAssertEqual(value("pilot.power"), "off")
            shot("15c_pilot_refused")
            mockStudio("set_id", ["id": 1, "to": 24]); usleep(1_500_000)        // joints, then the IMU's two poses
            mockStudio("cal_start", [:]); mockStudio("cal_save", [:])
            mockStudio("set_id", ["id": 24, "to": 1])
            mockStudio("imu_cal", ["step": 1]); mockStudio("imu_cal", ["step": 2])
            usleep(1_500_000)
        } else {                                            // calibrated by an earlier test: it started. Off again
            wait("on") { value("pilot.power") == "on" }
            power(letGo); letGo.tap()
            wait("off") { value("pilot.power") == "off" }
            usleep(1_500_000)
        }
        // an animation: the duck plays it by itself; a second tap on it stops it
        var t = Date()
        el("emote.hello").tap()
        wait("hello playing") { value("emote.hello") == "playing" }
        metric("clip_start_s", Date().timeIntervalSince(t))
        wait("the chip names it") { value("pilot.state").contains("hello") }
        shot("15_pilot_clip")
        wait("over by itself", 8) { value("emote.hello") == "" }
        metric("clip_s", Date().timeIntervalSince(t))
        el("emote.sad").tap()
        wait("sad playing") { value("emote.sad") == "playing" }
        usleep(700_000); el("emote.sad").tap()
        wait("stopped by the second tap", 3) { value("emote.sad") == "" }
        // power: the policies take the joints
        power(start)
        t = Date()
        start.tap()
        wait("standing", 8) { value("pilot.power") == "on" && pilotWord == "stand" }
        metric("pilot_start_s", Date().timeIntervalSince(t))
        XCTAssertFalse(el("emote.hello").isEnabled, "no animation while the policies drive")
        XCTAssertTrue(el("pilot.jump").isEnabled)
        shot("16_pilot_on")
        // the sticks, as in the simulator: forward, sidestep right, turn clockwise. Level 3: 0.40 m/s, 0.28 m/s, 0.70 rad/s
        gait(3)
        var p0 = pilotPose
        hold(0, 1, 3); rest()
        var d = pilotWent(p0)
        metric("pilot_fwd_m", d.f); metric("pilot_fwd_side_m", d.l)
        XCTAssertEqual(d.f, 0.40 * 3, accuracy: 0.3, "forward at level 3: 0.40 m/s for 3 s")
        XCTAssertLessThan(abs(d.l), 0.05)
        shot("16b_pilot_walked")
        p0 = pilotPose; hold(1, 0, 2); rest(); d = pilotWent(p0)
        metric("pilot_right_m", -d.l)
        XCTAssertEqual(-d.l, 0.28 * 2, accuracy: 0.2, "the stick's right is the duck's right")
        p0 = pilotPose; spin(1, 1); rest()
        metric("pilot_cw_deg", p0.yaw - pilotPose.yaw)
        XCTAssertEqual(p0.yaw - pilotPose.yaw, 0.70 * 180 / .pi, accuracy: 20, "clockwise")
        // the slider is the pilot's speed level: level 5 walks 3 times as fast as level 1
        var far: [Int: Double] = [:]
        for n in [1, 5] {
            gait(n)
            p0 = pilotPose; hold(0, 1, 2); rest()
            far[n] = pilotWent(p0).f
            metric("pilot_m_level\(n)", far[n]!)
        }
        XCTAssertGreaterThan(far[5]!, 2.2 * far[1]!)
        gait(3)
        // run: straight ahead it runs (level 3: 0.70 m/s), and covers more ground than the walk
        el("pilot.run").tap()
        XCTAssertEqual(value("pilot.run"), "on")
        p0 = pilotPose; hold(0, 1, 2); rest()
        metric("pilot_run_m", pilotWent(p0).f)
        XCTAssertGreaterThan(pilotWent(p0).f, 0.40 * 2 + 0.15)
        el("pilot.run").tap()
        // sit and stand again
        el("pilot.sit").tap()
        wait("seated", 6) { pilotWord == "seated" }
        shot("16c_pilot_seated")
        el("pilot.sit").tap()
        wait("standing again", 6) { pilotWord == "stand" }
        // jump and flip: shorter than the studio's one reading a second can be sure to show, so the duck's own log is asked
        let had = (told("\"state\": \"jump\""), told("\"state\": \"flip\""))
        el("pilot.jump").tap(); usleep(2_000_000)
        el("pilot.flip").tap(); usleep(2_500_000)
        if let j0 = had.0, let f0 = had.1 {
            XCTAssertGreaterThan(told("\"state\": \"jump\"")!, j0, "the pilot jumped")
            XCTAssertGreaterThan(told("\"state\": \"flip\"")!, f0, "the pilot flipped")
        }
        rest()
        // leaving Play with the stick pushed: the sticks are released, the policies keep the duck standing
        el("mode.duck").tap()
        XCTAssertEqual(value("mode.duck"), "selected")
        el("mode.play").tap()
        wait("still on, standing", 6) { value("pilot.power") == "on" && pilotWord == "stand" }
        // power again asks first, then every joint goes limp
        power(letGo)
        shot("16d_pilot_drop")
        letGo.tap()
        wait("off") { value("pilot.power") == "off" && pilotWord == "off" }
        usleep(2_000_000)
        XCTAssertEqual(value("pilot.power"), "off", "and stays off once the studio's word catches up")
        XCTAssertTrue(el("emote.hello").isEnabled)
    }

    // ---- 5a. the real duck, read-only: found and connected with no tap (tools/test_connect.sh; FIND = one way alone) --
    // Nothing here sends a servo command: the app only opens the duck's servo bus (a scan) and reads.
    func testRealDuck() throws {
        try XCTSkipUnless(env["REAL"] != nil, "run by tools/test_connect.sh against the real duck")
        app.launchArguments = ["-mode", "duck", "-target", "real"] + (env["FIND"].map { ["-find", $0] } ?? [])
        let t0 = Date()
        app.launch()
        wait("the real duck found", 60) { link("studio=true") && link("mock=false") }
        metric("found_s", Date().timeIntervalSince(t0))
        wait("computer and servo bus up with no tap", 40) { link("brain=true") && link("bus=true") }
        metric("ready_s", Date().timeIntervalSince(t0))
        XCTAssertFalse(el("connect").exists, "Connect still offered")
        wait("temperature shown", 15) { el("temp").exists }
        sleep(2)
        metric("servos", Double(value("link").components(separatedBy: "servos=").last?.prefix { $0.isNumber } ?? "") ?? -1)
        metric("battery_pct", el("battery").exists ? Double(value("battery")) ?? -1 : -1)
        metric("temp_c", Double(value("temp").prefix { $0.isNumber }) ?? -1)
        if env["SENSE"] != nil {                   // each sensor window shows a live reading from the real duck
            wait("sensors report") { value("sensor.camera") != "off" }
            for (k, want) in [("camera", "240×320"), ("tof", " m"), ("imu", "°"), ("mic", " dB")] {
                print("SENSOR \(k) \(value("sensor.\(k)"))")
                el("sensor.\(k)").coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.5)).tap()
                let t = Date(); var got = false
                while Date().timeIntervalSince(t) < 10 && !got { got = el("note.\(k)").label.contains(want); usleep(100_000) }
                metric("sense_\(k)_s", got ? Date().timeIntervalSince(t) : -1)
                print("NOTE \(k) \(el("note.\(k)").label)")
                if k == "mic", got {                 // the trace must be visibly moving, not a flat line
                    var span: [Double] = []
                    for _ in 0..<20 { span.append(Double(value("trace")) ?? 0); usleep(150_000) }
                    metric("mic_span_min", span.min()!); metric("mic_span_mean", span.reduce(0, +) / 20)
                    XCTAssertGreaterThan(span.reduce(0, +) / 20, 0.15, "the microphone's trace is flat")
                }
                if k == "camera" { XCTAssertTrue(got, "no camera picture"); XCTAssertTrue(el("picture").exists) }
                sleep(1); shot("19_real_\(k)")
            }
            el("sensor.mic").coordinate(withNormalizedOffset: .init(dx: 0.1, dy: 0.1)).tap()
        }
        if env["SHOT"] != nil { shot("17_real_duck"); el("link").tap(); sleep(1); shot("18_real_duck_panel") }
        if env["DROP"] != nil {                    // the script restarts the duck's server now: lost, then back, with no tap
            print("READY_FOR_DROP"); let t = Date()
            wait("the duck lost", 60) { link("studio=false") }
            metric("lost_s", Date().timeIntervalSince(t)); let u = Date()
            wait("the duck back and connected", 90) { link("studio=true") && link("brain=true") && link("bus=true") }
            metric("back_s", Date().timeIntervalSince(u))
        }
    }

    // ---- 5b. no duck in reach: Connect searches, then says so in plain words (run with the servers stopped) ----------
    func testNoDuck() throws {
        try XCTSkipUnless(env["OFFLINE"] != nil, "run by tools/test.sh with the servers stopped")
        launch("duck", "real")
        XCTAssertTrue(el("connect").waitForExistence(timeout: 6))
        let t = Date()
        el("connect").tap()
        wait("told the duck is out of reach", 15) { el("toast").exists && el("toast").label.contains("reached") }
        metric("unreachable_s", Date().timeIntervalSince(t))
        usleep(700_000)
        shot("15_unreachable")
        el("link").tap()
        sleep(1)
        shot("16_unreachable_panel")
        for t in app.staticTexts.allElementsBoundByIndex.map(\.label) {
            XCTAssertFalse(t.contains(".py") || t.contains("--") || t.contains(".sh") || t.contains("/"), "command-line wording on screen: \(t)")
        }
    }

    // ---- 6. nothing on the network, then the servers appear (tools/test.sh stops and starts them) --------------------
    func testOffline() throws {
        try XCTSkipUnless(env["OFFLINE"] != nil, "run by tools/test.sh with the servers stopped")
        launch("play", "sim")
        sleep(4)
        XCTAssertTrue(link("studio=false") && link("engine=false"))
        XCTAssertTrue(el("offline").exists)
        shot("14_offline")
        let t = Date()
        wait("servers found once they start", 90) { link("studio=true") && link("engine=true") && el("stick").exists }
        metric("recover_s", Date().timeIntervalSince(t))
        wait("standing") { state == "stand" }
    }
}
