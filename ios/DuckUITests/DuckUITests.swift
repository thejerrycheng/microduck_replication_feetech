// Drives the app in the simulator against the mock studio and the real engine (see tools/test.sh, which starts both).
// Every test asserts on what the servers did, read back through the app, and saves a screenshot. Lines that start with
// "METRIC" are the numbers in docs/duck_ios_app.tex (tools/plot_tests.py reads them from the log).
import XCTest

final class DuckUITests: XCTestCase {
    let app = XCUIApplication()
    let env = ProcessInfo.processInfo.environment

    override func setUp() { continueAfterFailure = false }

    func launch(_ mode: String) {
        app.launchArguments = ["-mode", mode]
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

    func metric(_ name: String, _ v: Double) { print(String(format: "METRIC %@ %.3f", name, v)) }

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

    func connectBus() {
        wait("studio found", 25) { link("studio=true") }
        if !link("bus=true") { flip("bus") }
        wait("bus on") { link("bus=true") }
    }

    // ---- 1. connections --------------------------------------------------------------------------------------------
    func testLinks() {
        let t0 = Date()
        launch("duck")
        wait("studio and engine found over Bonjour", 25) { link("studio=true") && link("engine=true") }
        metric("discover_s", Date().timeIntervalSince(t0))
        XCTAssertTrue(link("bus=false") || link("bus=true"))
        if link("bus=true") { flip("bus"); wait("bus off") { link("bus=false") } }
        XCTAssertFalse(el("servo.10").exists)
        shot("01_idle")
        el("link").tap()
        XCTAssertTrue(el("bus").waitForExistence(timeout: 4))
        shot("02_links")
        app.coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.62)).tap()
        flip("bus")
        wait("14 of 15 joints on the mock bus") { link("servos=14") }
        XCTAssertTrue(el("servo.10").exists)
        XCTAssertTrue(el("servo.1").exists, "the mock's factory-ID servo is listed")
        if !link("brain=true") { flip("brain") }
        wait("duck computer on") { link("brain=true") }
        wait("sensors report") { value("sensor.camera") == "ok" }
        shot("03_connected")
        flip("brain")
        wait("duck computer off") { link("brain=false") }
        XCTAssertEqual(value("sensor.camera"), "off")
        flip("brain")
        wait("duck computer on again") { link("brain=true") }
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
        for (f, lo, hi) in [(0.8, 2450, 2900), (0.2, 1200, 1650), (0.5, 1800, 2300)] {       // XCUITest lands within ~7 % of the asked fraction
            app.sliders["slide"].adjust(toNormalizedSliderPosition: f)
            wait("slider at \(f) drove the servo into \(lo)-\(hi), reads \(value("pos"))") { (lo...hi).contains(Int(value("pos")) ?? 0) }
        }
        shot("04_servo")
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
        wait("studio found", 25) { link("studio=true") }
        if !link("brain=true") { flip("brain") }
        wait("sensors report") { value("sensor.camera") == "ok" }
        for (k, want) in [("camera", "320×240"), ("tof", " m"), ("imu", "°"), ("mic", " dB")] {
            el("sensor.\(k)").coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.5)).tap()
            wait("\(k) reading", 8) { (el("note.\(k)").label).contains(want) }
            if k == "camera" { XCTAssertTrue(el("picture").waitForExistence(timeout: 4)) }
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

    // ---- 4. play: joystick, moves, emotions ------------------------------------------------------------------------
    var state: String { el("state").label }
    var odometer: Double { Double(value("state").split(separator: " ").first ?? "") ?? -1 }

    func hold(_ dx: Double, _ dy: Double, _ secs: TimeInterval) {
        let s = el("stick")
        s.coordinate(withNormalizedOffset: .init(dx: 0.5, dy: 0.5))
            .press(forDuration: 0.05, thenDragTo: s.coordinate(withNormalizedOffset: .init(dx: 0.5 + dx * 0.3, dy: 0.5 - dy * 0.3)), withVelocity: .fast, thenHoldForDuration: secs)
    }

    func settle() { wait("standing still", 15) { state == "stand" } ; usleep(400_000) }

    func testPlay() {
        launch("play")
        wait("engine found", 25) { el("state").exists && state == "stand" }
        el("reset").tap(); usleep(900_000)
        wait("reset") { state == "stand" }
        shot("07_play")
        // forward: the duck walks while the stick is held and stops when it is let go
        let d0 = odometer
        hold(0, 1, 3)
        let d = odometer - d0
        XCTAssertGreaterThan(d, 0.3, "walked forward")
        settle()
        let d2 = odometer; usleep(1_500_000)
        XCTAssertLessThan(odometer - d2, 0.03, "stopped")
        // back and turn do not fall
        hold(0, -1, 2); settle()
        hold(1, 0, 2); settle()
        hold(-0.75, 0.75, 2); settle()
        // sit, stand
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
        wait("hello plays", 5) { value("state").contains("hello") }
        shot("10_emote")
        wait("hello ends", 10) { !value("state").contains("hello") }
        // the app going to the background stops the duck (a held stick cannot be tested across it; the keys are cleared)
        settle()
    }

    /// The speed level and the run switch: distance covered in a 3 s push, levels 1-5, walk then run.
    func testSpeeds() {
        launch("play")
        wait("engine found", 25) { el("state").exists && state == "stand" }
        for run in [false, true] {
            if (value("act.run") == "on") != run { el("act.run").tap() }
            for level in 1...5 {
                el("level.\(level)").tap()
                wait("level \(level)") { value("level") == "\(level)" }
                el("reset").tap(); usleep(900_000)
                wait("reset") { state == "stand" }
                let d0 = odometer
                hold(0, 1, 3)
                usleep(300_000)
                let d = odometer - d0
                metric("\(run ? "run" : "walk")_m_level\(level)", d)
                XCTAssertGreaterThan(d, 0.15)
                settle()
            }
            if run { shot("11_run") }
        }
        el("act.run").tap()
        el("level.3").tap()
        wait("level back to 3") { value("level") == "3" && value("act.run") == "off" }
    }

    // ---- 5. policies -----------------------------------------------------------------------------------------------
    func pick(_ role: String, _ title: String) {
        el("policy.\(role)").tap()
        let b = app.buttons[title]
        XCTAssertTrue(b.waitForExistence(timeout: 4), "menu shows \(title)")
        b.tap()
    }

    func testPolicy() {
        launch("policy")
        wait("engine found", 25) { el("policy.walk").exists }
        XCTAssertEqual(value("policy.walk"), "velstand allcoll 21495")
        XCTAssertEqual(value("policy.roulade"), "roulade head 7000")
        shot("12_policy")
        el("policy.walk").tap()
        XCTAssertTrue(app.buttons["walk allcoll 16498"].waitForExistence(timeout: 4))
        shot("13_policy_menu")
        let t = Date()
        app.buttons["walk allcoll 16498"].tap()
        wait("walk policy swapped") { value("policy.walk") == "walk allcoll 16498" }
        metric("deploy_s", Date().timeIntervalSince(t))
        XCTAssertTrue(el("toast").waitForExistence(timeout: 3))
        pick("walk", "velstand allcoll 21495")
        wait("walk policy restored") { value("policy.walk") == "velstand allcoll 21495" }
        // an optional skill switched off greys its button in Play; back on, it works again
        pick("jump", "Off")
        wait("jump off") { value("policy.jump") == "off" }
        el("mode.play").tap()
        XCTAssertTrue(el("act.jump").waitForExistence(timeout: 4))
        wait("jump button disabled") { !el("act.jump").isEnabled }
        el("mode.policy").tap()
        pick("jump", "jump 1500")
        wait("jump back") { value("policy.jump") == "jump 1500" }
        el("mode.play").tap()
        wait("jump button enabled") { el("act.jump").isEnabled }
        el("mode.duck").tap()
        XCTAssertEqual(value("mode.duck"), "selected")
    }

    // ---- 6. nothing on the network, then the servers appear (tools/test.sh stops and starts them) --------------------
    func testOffline() throws {
        try XCTSkipUnless(env["OFFLINE"] != nil, "run by tools/test.sh with the servers stopped")
        launch("play")
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
