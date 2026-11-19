// Duck Live: the macOS shell of the Mac app. It starts the local server (app/server.py in the Mac's Python env),
// waits for its "DUCK_LIVE_READY <port> <token>" line, shows the page in a WKWebView and stops the server on quit.
// Keys are read natively (NSEvent monitor) and forwarded to the page: WebKit drops key-up events while ⌘ is held,
// which would leave the duck running after ⌘+arrow is released. Built by app/build_app.sh (swiftc, no Xcode project).
import Cocoa
import WebKit

let defaults: [String: String] = [
  "DuckRepo": NSString(string: "~/Desktop/microduck_replication_feetech").expandingTildeInPath,
  "DuckPython": NSString(string: "~/.venvs/microduck_live/bin/python").expandingTildeInPath,
]

func setting(_ key: String) -> String {
  if let v = ProcessInfo.processInfo.environment[key == "DuckRepo" ? "DUCK_REPO" : "DUCK_PYTHON"], !v.isEmpty { return v }
  if let v = Bundle.main.object(forInfoDictionaryKey: key) as? String, !v.isEmpty { return NSString(string: v).expandingTildeInPath }
  return defaults[key]!
}

final class WebView: WKWebView {
  override var acceptsFirstResponder: Bool { true }
  override func willOpenMenu(_ menu: NSMenu, with event: NSEvent) { menu.removeAllItems() }
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate {
  var window: NSWindow!
  var web: WebView!
  var server: Process?
  var status: NSTextField!
  var buffer = ""
  var loaded = false
  var monitor: Any?

  // keys the page owns; everything else (⌘Q, ⌘R, ⌘L …) stays with the menu
  static let keyNames: [UInt16: String] = [126: "up", 125: "down", 123: "left", 124: "right", 49: "space", 36: "enter",
                                           76: "enter", 1: "s", 0: "a", 2: "d", 15: "r", 53: "escape",
                                           // emotions (the page maps these names to clips: app/animations/manifest.json)
                                           18: "1", 19: "2", 20: "3", 21: "4", 23: "5", 22: "6", 26: "7", 28: "8", 25: "9", 29: "0",
                                           12: "q", 13: "w", 14: "e", 17: "t", 16: "y", 32: "u", 34: "i", 31: "o", 35: "p",
                                           3: "f", 5: "g", 4: "h", 38: "j", 40: "k", 37: "l", 6: "z", 7: "x", 46: "m"]

  func applicationDidFinishLaunching(_ note: Notification) {
    buildMenu()
    let frame = NSRect(x: 0, y: 0, width: 1440, height: 900)
    window = NSWindow(contentRect: frame, styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                      backing: .buffered, defer: false)
    window.title = "Duck Live"
    window.titlebarAppearsTransparent = true
    window.titleVisibility = .hidden
    window.minSize = NSSize(width: 980, height: 640)
    window.backgroundColor = NSColor(calibratedRed: 0.11, green: 0.12, blue: 0.14, alpha: 1)
    window.center()
    window.setFrameAutosaveName("DuckLiveMain")

    let cfg = WKWebViewConfiguration()
    cfg.preferences.setValue(true, forKey: "developerExtrasEnabled")
    cfg.mediaTypesRequiringUserActionForPlayback = []     // the duck's voice plays without a click first
    web = WebView(frame: window.contentView!.bounds, configuration: cfg)
    web.autoresizingMask = [.width, .height]
    web.navigationDelegate = self
    web.uiDelegate = self
    web.setValue(false, forKey: "drawsBackground")
    web.isHidden = true
    window.contentView!.addSubview(web)

    status = NSTextField(labelWithString: "Starting the duck…")
    status.font = NSFont.systemFont(ofSize: 15, weight: .semibold)
    status.textColor = .white
    status.alignment = .center
    status.frame = NSRect(x: 0, y: frame.height / 2 - 40, width: frame.width, height: 80)
    status.autoresizingMask = [.width, .minYMargin, .maxYMargin]
    status.maximumNumberOfLines = 4
    window.contentView!.addSubview(status)
    window.makeKeyAndOrderFront(nil)
    NSApp.activate(ignoringOtherApps: true)
    installKeys()
    startServer()
  }

  // ---- native keys -> page ---------------------------------------------------------------------------------------------
  func installKeys() {
    monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp, .flagsChanged]) { [weak self] ev in
      guard let self = self, self.loaded else { return ev }
      let f = ev.modifierFlags
      let meta = f.contains(.command), alt = f.contains(.option), shift = f.contains(.shift), ctrl = f.contains(.control)
      if ev.type == .flagsChanged {
        self.js("duckKey('flags', '', \(meta), \(alt), \(shift))")
        return ev
      }
      guard let name = AppDelegate.keyNames[ev.keyCode] else { return ev }
      if ctrl { return ev }
      // ⌘ + arrows is "run"; ⌘ + any other key belongs to the menu (⌘Q, ⌘⇧R, ⌘L …)
      if meta && !["up", "down", "left", "right"].contains(name) { return ev }
      if ev.type == .keyDown && ev.isARepeat { return nil }
      self.js("duckKey('\(ev.type == .keyDown ? "down" : "up")', '\(name)', \(meta), \(alt), \(shift))")
      return nil
    }
  }

  func js(_ s: String) { web.evaluateJavaScript(s, completionHandler: nil) }

  func buildMenu() {
    let main = NSMenu()
    let appItem = NSMenuItem(); main.addItem(appItem)
    let appMenu = NSMenu()
    appMenu.addItem(withTitle: "Reload the page", action: #selector(reload), keyEquivalent: "r").keyEquivalentModifierMask = [.command, .shift]
    appMenu.addItem(withTitle: "Restart the duck server", action: #selector(restartServer), keyEquivalent: "R").keyEquivalentModifierMask = [.command, .option]
    appMenu.addItem(withTitle: "Open the server log", action: #selector(openLog), keyEquivalent: "l")
    appMenu.addItem(NSMenuItem.separator())
    appMenu.addItem(withTitle: "Quit Duck Live", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    appItem.submenu = appMenu
    let winItem = NSMenuItem(); main.addItem(winItem)
    let win = NSMenu(title: "Window")
    win.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
    win.addItem(withTitle: "Toggle Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
    winItem.submenu = win
    NSApp.mainMenu = main
  }

  var logPath: String { setting("DuckRepo") + "/logs/app/app_stdout.log" }

  func startServer() {
    let repo = setting("DuckRepo"), py = setting("DuckPython")
    let script = repo + "/app/server.py"
    guard FileManager.default.fileExists(atPath: py) else {
      return fail("The Python environment is missing:\n\(py)\nRun app/setup.sh once.")
    }
    guard FileManager.default.fileExists(atPath: script) else { return fail("Cannot find \(script)") }
    let p = Process()
    p.executableURL = URL(fileURLWithPath: py)
    p.arguments = [script]
    p.currentDirectoryURL = URL(fileURLWithPath: repo)
    var env = ProcessInfo.processInfo.environment
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
    p.environment = env
    let out = Pipe()
    p.standardOutput = out
    try? FileManager.default.createDirectory(atPath: (logPath as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: logPath, contents: nil)
    p.standardError = FileHandle(forWritingAtPath: logPath)
    out.fileHandleForReading.readabilityHandler = { [weak self] h in
      let d = h.availableData
      guard !d.isEmpty, let s = String(data: d, encoding: .utf8) else { return }
      DispatchQueue.main.async { self?.onOutput(s) }
    }
    p.terminationHandler = { [weak self] proc in
      DispatchQueue.main.async {
        guard let self = self, self.server === proc else { return }
        self.fail("The duck server stopped (exit \(proc.terminationStatus)).\nSee \(self.logPath) · ⌘⌥R restarts it.")
      }
    }
    do { try p.run(); server = p } catch { fail("Could not start the server: \(error.localizedDescription)") }
  }

  func onOutput(_ s: String) {
    buffer += s
    for line in buffer.components(separatedBy: "\n") where line.hasPrefix("DUCK_LIVE_READY ") {
      let f = line.split(separator: " ")
      if f.count >= 3 && !loaded {
        status.stringValue = "Loading…"
        web.load(URLRequest(url: URL(string: "http://127.0.0.1:\(f[1])/?token=\(f[2])&native=1")!))
      }
    }
    if let last = buffer.components(separatedBy: "\n").last { buffer = last }
  }

  func fail(_ msg: String) {
    status.stringValue = msg
    status.isHidden = false
    web.isHidden = true
    loaded = false
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    web.isHidden = false
    status.isHidden = true
    loaded = true
    window.makeFirstResponder(web)
  }

  func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
    if let u = action.request.url, u.host != "127.0.0.1", action.navigationType == .linkActivated {
      NSWorkspace.shared.open(u); decisionHandler(.cancel); return
    }
    decisionHandler(.allow)
  }

  @objc func reload() { web.reload() }
  @objc func openLog() { NSWorkspace.shared.open(URL(fileURLWithPath: logPath)) }
  @objc func restartServer() {
    stopServer()
    loaded = false; buffer = ""
    status.stringValue = "Restarting the duck server…"; status.isHidden = false; web.isHidden = true
    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { self.startServer() }
  }

  func stopServer() {
    guard let p = server else { return }
    server = nil
    if p.isRunning {
      p.terminate()
      let deadline = Date().addingTimeInterval(3)
      while p.isRunning && Date() < deadline { usleep(50_000) }
      if p.isRunning { kill(p.processIdentifier, SIGKILL) }
    }
  }

  func applicationWillTerminate(_ note: Notification) { stopServer() }
  func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
// a SIGTERM (kill, pkill, logout) must stop the server too, not only ⌘Q
signal(SIGTERM) { _ in DispatchQueue.main.async { NSApp.terminate(nil) } }
app.run()
