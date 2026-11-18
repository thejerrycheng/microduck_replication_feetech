// The 3-D duck. One model (firmware/web/duck.bin, baked by firmware/bake_duck.py) drawn two ways:
//   studio look  - see-through shell, solid servo housings coloured by status (as the servo studio draws it);
//   live look    - the real colours, every body posed from the engine's 50 Hz snapshots, on a grid floor.
import SceneKit
import SwiftUI
import UIKit

final class DuckScene {
    struct Look: Equatable {
        var live = false, ids = Set<Int>(), answering = Set<Int>(), sel: Int?
        var sensors: [String: Bool]?, open: String?, dark = false, fov = 25.0      // a card is up: a wider view, so the duck sits smaller above it
    }

    let scene = SCNScene(), camera = SCNNode()
    private let world = SCNNode(), robot = SCNNode(), floor = SCNNode(), shadow = SCNNode()
    private var bodies: [String: SCNNode] = [:], standing: [String: (SIMD3<Float>, simd_quatf)] = [:]
    private var parts: [(node: SCNNode, rgba: [CGFloat], servo: Int, sensor: String)] = []
    private var chips: [SCNNode] = []
    private var chains: [String: [Int]] = [:], pivots: [Int: (at: SIMD3<Float>, axis: SIMD3<Float>)] = [:]     // per body: the joints that carry it, trunk first
    private let jaw = SCNNode()                 // the lower beak hangs from this, at the beak servo's axis (the model has no jaw joint)
    private var jawAxis: SIMD3<Float> = [0, 1, 0]
    private(set) var jawOpen = 0.42             // radians at fully open, for the live engine's 0...1 "mouth"
    private var look = Look(), trunk = "trunk_base", painted = false, lastJoints: [Int: Double] = [:]
    static let tile: Float = 0.2

    init() {
        scene.background.contents = UIColor.clear
        world.simdOrientation = simd_quatf(angle: -.pi / 2, axis: [1, 0, 0])       // the model is z up, SceneKit y up
        scene.rootNode.addChildNode(world)
        world.addChildNode(robot)
        build()
        lights()
        ground()
        let cam = SCNCamera()
        cam.fieldOfView = 25; cam.zNear = 0.01; cam.zFar = 30
        camera.camera = cam
        camera.simdPosition = [0.74, 0.375, 0.84]
        camera.simdLook(at: [0, 0.125, 0])
        scene.rootNode.addChildNode(camera)
    }

    private static func quat(_ q: [Double]) -> simd_quatf { simd_quatf(ix: Float(q[1]), iy: Float(q[2]), iz: Float(q[3]), r: Float(q[0])) }
    private static func vec(_ v: [Double]) -> SIMD3<Float> { [Float(v[0]), Float(v[1]), Float(v[2])] }

    // ---- the model: [uint32 n][n bytes JSON][per mesh: int16 xyz, int8 normals, uint32 indices, each padded to 4] --------
    private func build() {
        guard let url = Bundle.main.url(forResource: "duck", withExtension: "bin"), let d = try? Data(contentsOf: url) else { return }
        let n = Int(d.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        guard let meta = try? JSONSerialization.jsonObject(with: d.subdata(in: 4..<4 + n)) as? [String: Any] else { return }
        var o = 4 + n, geos: [SCNGeometry] = []
        for m in meta["meshes"] as? [[String: Int]] ?? [] {
            let nv = m["nv"] ?? 0, ni = m["ni"] ?? 0
            var pos = [Float](repeating: 0, count: nv * 3), nor = pos
            d.withUnsafeBytes { b in
                for i in 0..<nv * 3 { pos[i] = Float(b.loadUnaligned(fromByteOffset: o + 2 * i, as: Int16.self)) * 1e-5 }
                o += nv * 6 + nv * 6 % 4
                for i in 0..<nv * 3 { nor[i] = Float(b.load(fromByteOffset: o + i, as: Int8.self)) / 127 }
                o += nv * 3 + (4 - nv * 3 % 4) % 4
            }
            let src = { (v: [Float], s: SCNGeometrySource.Semantic) in
                SCNGeometrySource(data: v.withUnsafeBufferPointer { Data(buffer: $0) }, semantic: s, vectorCount: nv, usesFloatComponents: true,
                                  componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12) }
            let el = SCNGeometryElement(data: d.subdata(in: o..<o + ni * 4), primitiveType: .triangles, primitiveCount: ni / 3, bytesPerIndex: 4)
            o += ni * 4
            geos.append(SCNGeometry(sources: [src(pos, .vertex), src(nor, .normal)], elements: [el]))
        }
        for b in meta["bodies"] as? [[String: Any]] ?? [] {
            guard let name = b["name"] as? String, let p = b["pos"] as? [Double], let q = b["quat"] as? [Double] else { continue }
            let node = SCNNode()
            standing[name] = (Self.vec(p), Self.quat(q))
            bodies[name] = node
            robot.addChildNode(node)
        }
        var pivot = SIMD3<Float>.zero
        if let j = meta["jaw"] as? [String: Any], let head = bodies[j["body"] as? String ?? ""], let pv = j["pivot"] as? [Double],
           let ax = j["axis"] as? [Double] {
            pivot = Self.vec(pv); jawAxis = Self.vec(ax); jawOpen = j["open_rad"] as? Double ?? jawOpen
            jaw.simdPosition = pivot
            head.addChildNode(jaw)
        }
        for p in meta["parts"] as? [[String: Any]] ?? [] {
            guard let mi = p["mesh"] as? Int, let body = bodies[p["body"] as? String ?? ""], let lp = p["lpos"] as? [Double],
                  let lq = p["lquat"] as? [Double], let rgba = p["rgba"] as? [Double] else { continue }
            let g = geos[mi].copy() as! SCNGeometry, mat = SCNMaterial(), node = SCNNode(geometry: g)
            mat.lightingModel = .blinn
            mat.specular.contents = UIColor(white: 1, alpha: 0.25)
            g.materials = [mat]
            node.simdPosition = Self.vec(lp); node.simdOrientation = Self.quat(lq)
            let servo = p["servo"] as? Int ?? 0, sensor = p["sensor"] as? String ?? ""
            if let c = p["chain"] as? [Int], let name = p["body"] as? String { chains[name] = c }
            if servo > 0 { node.name = "servo:\(servo)" } else if !sensor.isEmpty { node.name = "sensor:\(sensor)" }
            if p["jaw"] as? Bool == true, jaw.parent != nil {      // a lower-beak mesh: carried by the jaw, not the head
                node.simdPosition -= pivot
                jaw.addChildNode(node)
            } else { body.addChildNode(node) }
            parts.append((node, rgba.map { CGFloat($0) }, servo, sensor))
        }
        for sv in meta["servos"] as? [[String: Any]] ?? [] {
            guard let id = sv["id"] as? Int, let at = sv["at"] as? [Double], let ax = sv["axis"] as? [Double] else { continue }
            pivots[id] = (Self.vec(at), simd_normalize(Self.vec(ax)))
        }
        // the two sensors with no part of their own in the model: a small chip where each sits
        let have = Set(parts.map(\.sensor))
        for s in meta["sensors"] as? [[String: Any]] ?? [] {
            guard let id = s["id"] as? String, !have.contains(id), let at = s["at"] as? [Double] else { continue }
            let g = SCNBox(width: 0.011, height: 0.011, length: 0.0025, chamferRadius: 0.0008), mat = SCNMaterial(), node = SCNNode(geometry: g)
            mat.lightingModel = .blinn; g.materials = [mat]
            node.name = "sensor:\(id)"; node.simdPosition = Self.vec(at)
            if s["dir"] != nil { node.simdOrientation = simd_quatf(angle: .pi / 2, axis: [0, 1, 0]) }      // faces forward
            robot.addChildNode(node)
            chips.append(node)
            parts.append((node, [0.5, 0.5, 0.5, 1], 0, id))
        }
        stand(animated: false)
    }

    private func lights() {
        let amb = SCNNode(); amb.light = SCNLight(); amb.light!.type = .ambient; amb.light!.intensity = 520
        let sun = SCNNode(); sun.light = SCNLight(); sun.light!.type = .directional; sun.light!.intensity = 760
        sun.simdPosition = [0.6, 1.4, 0.8]; sun.simdLook(at: [0, 0.1, 0])
        let fill = SCNNode(); fill.light = SCNLight(); fill.light!.type = .directional; fill.light!.intensity = 260
        fill.simdPosition = [-0.8, 0.5, -0.9]; fill.simdLook(at: [0, 0.1, 0])
        [amb, sun, fill].forEach(scene.rootNode.addChildNode)
    }

    /// Live look only: a grid that slides under the duck (the duck stays in the middle), fading out with distance, and
    /// a soft shadow.
    private func ground() {
        let fade = { (inner: CGFloat, color: UIColor) -> UIImage in
            UIGraphicsImageRenderer(size: .init(width: 256, height: 256)).image { c in
                let g = CGGradient(colorsSpace: nil, colors: [color.cgColor, color.withAlphaComponent(0).cgColor] as CFArray, locations: [inner, 1])!
                c.cgContext.drawRadialGradient(g, startCenter: .init(x: 128, y: 128), startRadius: 0, endCenter: .init(x: 128, y: 128), endRadius: 128, options: [])
            }
        }
        let plane = SCNPlane(width: 8, height: 8), m = SCNMaterial()
        m.lightingModel = .constant; m.writesToDepthBuffer = false; m.isDoubleSided = true
        m.diffuse.wrapS = .repeat; m.diffuse.wrapT = .repeat
        m.diffuse.contentsTransform = SCNMatrix4MakeScale(8 / Self.tile, 8 / Self.tile, 1)
        m.diffuse.mipFilter = .linear; m.diffuse.maxAnisotropy = 8
        m.transparent.contents = fade(0.12, .white)
        plane.materials = [m]
        floor.geometry = plane; floor.renderingOrder = -2; floor.opacity = 0
        let blob = SCNPlane(width: 0.3, height: 0.3), bm = SCNMaterial()
        bm.lightingModel = .constant; bm.writesToDepthBuffer = false; bm.diffuse.contents = fade(0.05, UIColor(white: 0, alpha: 0.42))
        blob.materials = [bm]
        shadow.geometry = blob; shadow.simdPosition = [0, 0, 0.0005]; shadow.renderingOrder = -1; shadow.opacity = 0
        world.addChildNode(floor); world.addChildNode(shadow)
    }

    private func grid(dark: Bool) -> UIImage {
        UIGraphicsImageRenderer(size: .init(width: 128, height: 128)).image { c in
            (dark ? UIColor(white: 1, alpha: 0.2) : UIColor(white: 0, alpha: 0.16)).setStroke()
            c.cgContext.setLineWidth(3)
            c.cgContext.stroke(CGRect(x: 0, y: 0, width: 128, height: 128))
        }
    }

    // ---- poses ------------------------------------------------------------------------------------------------------
    func stand(animated: Bool = true) {
        SCNTransaction.begin(); SCNTransaction.animationDuration = animated ? 0.45 : 0
        for (name, node) in bodies { if let s = standing[name] { node.simdPosition = s.0; node.simdOrientation = s.1 } }
        robot.simdPosition = .zero; floor.simdPosition = .zero
        SCNTransaction.commit()
        lastJoints = [:]
    }

    /// The real joints' angles [deg] by servo ID (calibrated joints only; none = the home pose). Each body is moved by
    /// the joints that carry it: the product, trunk first, of each joint's rotation about its baked axis.
    func joints(_ deg: [Int: Double]) {
        guard deg != lastJoints else { return }
        lastJoints = deg
        var turn: [Int: simd_float4x4] = [:]
        for (id, a) in deg where a != 0 {
            guard let pv = pivots[id] else { continue }
            var to = matrix_identity_float4x4, back = matrix_identity_float4x4
            to.columns.3 = SIMD4(pv.at, 1); back.columns.3 = SIMD4(-pv.at, 1)
            turn[id] = to * simd_float4x4(simd_quatf(angle: Float(a * .pi / 180), axis: pv.axis)) * back
        }
        SCNTransaction.begin(); SCNTransaction.animationDuration = 0.12
        for (name, node) in bodies {
            guard let s = standing[name] else { continue }
            var m = matrix_identity_float4x4
            for id in chains[name] ?? [] { if let t = turn[id] { m = m * t } }
            var home = simd_float4x4(s.1); home.columns.3 = SIMD4(s.0, 1)
            node.simdTransform = m * home
        }
        SCNTransaction.commit()
    }

    /// The engine's snapshot: world position and quaternion (w x y z) of every body after `world`, in `names` order.
    func pose(_ names: [String], _ pos: [Double], _ quat: [Double]) {
        guard pos.count >= 3 * (names.count - 1), quat.count >= 4 * (names.count - 1) else { return }
        for i in 1..<max(1, names.count) {
            guard let node = bodies[names[i]] else { continue }
            let k = i - 1
            node.simdPosition = [Float(pos[3 * k]), Float(pos[3 * k + 1]), Float(pos[3 * k + 2])]
            node.simdOrientation = simd_quatf(ix: Float(quat[4 * k + 1]), iy: Float(quat[4 * k + 2]), iz: Float(quat[4 * k + 3]), r: Float(quat[4 * k]))
            if names[i] == trunk {                 // keep the duck over the origin; the grid slides the other way
                let x = node.simdPosition.x, y = node.simdPosition.y
                robot.simdPosition = [-x, -y, 0]
                floor.simdPosition = [-x.truncatingRemainder(dividingBy: Self.tile), -y.truncatingRemainder(dividingBy: Self.tile), 0]
            }
        }
    }

    /// The beak: `rad` about the beak servo's axis, positive = open.
    func mouth(_ rad: Double) {
        SCNTransaction.begin(); SCNTransaction.animationDuration = 0.1
        jaw.simdOrientation = simd_quatf(angle: Float(rad), axis: jawAxis)
        SCNTransaction.commit()
    }

    /// The IMU turns the whole duck (studio look). nil = upright.
    func tilt(_ quat: [Double]?, roll: Double = 0, pitch: Double = 0) {
        var q = simd_quatf(angle: 0, axis: [0, 0, 1])
        if let quat, quat.count == 4 { q = Self.quat(quat) }
        else if quat != nil { q = simd_quatf(angle: Float(pitch * .pi / 180), axis: [0, 1, 0]) * simd_quatf(angle: Float(roll * .pi / 180), axis: [1, 0, 0]) }
        SCNTransaction.begin(); SCNTransaction.animationDuration = 0.12
        robot.simdOrientation = look.live ? simd_quatf(angle: 0, axis: [0, 0, 1]) : q
        SCNTransaction.commit()
    }

    // ---- colours ----------------------------------------------------------------------------------------------------
    func apply(_ l: Look) {
        guard l != look || !painted else { return }
        let modeChanged = l.live != look.live || l.dark != look.dark || !painted
        look = l; painted = true
        let trait = UITraitCollection(userInterfaceStyle: l.dark ? .dark : .light)
        let c = { (u: UIColor) in u.resolvedColor(with: trait) }
        let green = c(.systemGreen), red = c(.systemRed), blue = c(.systemBlue), off = UIColor(white: l.dark ? 0.55 : 0.78, alpha: 1)
        SCNTransaction.begin(); SCNTransaction.animationDuration = 0.35
        if modeChanged { (floor.geometry as? SCNPlane)?.firstMaterial?.diffuse.contents = grid(dark: l.dark) }
        camera.camera?.fieldOfView = l.fov
        floor.opacity = l.live ? 1 : 0; shadow.opacity = l.live ? 1 : 0
        if l.live { robot.simdOrientation = simd_quatf(angle: 0, axis: [0, 0, 1]) }
        chips.forEach { $0.isHidden = l.live }
        for p in parts {
            guard let m = p.node.geometry?.firstMaterial else { continue }
            var color = UIColor(red: p.rgba[0], green: p.rgba[1], blue: p.rgba[2], alpha: 1), ghost = false, glow = UIColor.black
            if !l.live {
                if p.servo > 0 {                   // green = answering, red = found but silent, see-through = absent, blue = selected
                    if p.servo == l.sel { color = blue; glow = blue.withAlphaComponent(0.55) }
                    else if !l.ids.contains(p.servo) { color = off; ghost = true }
                    else { color = l.answering.contains(p.servo) ? green : red }
                } else if !p.sensor.isEmpty {
                    if p.sensor == l.open { color = blue; glow = blue.withAlphaComponent(0.5) }
                    else if let s = l.sensors { color = s[p.sensor] == true ? green : red }
                    else { color = off; ghost = true }
                } else { ghost = true }
            }
            m.diffuse.contents = color
            m.emission.contents = glow
            m.transparency = ghost ? 0.22 : 1
            m.writesToDepthBuffer = !ghost
            p.node.renderingOrder = ghost ? 1 : 0
        }
        SCNTransaction.commit()
    }
}

/// The view: orbit and zoom with the fingers; a tap on a servo or a sensor picks it.
struct DuckView: UIViewRepresentable {
    let duck: Duck

    func makeCoordinator() -> Pick { Pick(duck) }

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = duck.scene.scene
        v.pointOfView = duck.scene.camera
        v.backgroundColor = .clear; v.isOpaque = false
        v.antialiasingMode = .multisampling4X
        v.allowsCameraControl = true
        let c = v.defaultCameraController
        c.interactionMode = .orbitTurntable; c.target = SCNVector3(0, 0.125, 0); c.inertiaEnabled = true
        c.maximumVerticalAngle = 80; c.minimumVerticalAngle = 2
        v.accessibilityIdentifier = "duck3d"
        v.addGestureRecognizer(UITapGestureRecognizer(target: context.coordinator, action: #selector(Pick.tap(_:))))
        return v
    }

    func updateUIView(_ v: SCNView, context: Context) {}

    @MainActor final class Pick: NSObject {
        let duck: Duck
        init(_ d: Duck) { duck = d }

        @objc func tap(_ g: UITapGestureRecognizer) {
            guard duck.mode == .duck, let v = g.view as? SCNView else { return }
            let hits = v.hitTest(g.location(in: v), options: [.searchMode: SCNHitTestSearchMode.all.rawValue, .ignoreHiddenNodes: true])
            let name = hits.lazy.compactMap { $0.node.name }.first ?? ""
            withAnimation(.spring(duration: 0.5, bounce: 0.25)) {
                if name.hasPrefix("servo:"), let id = Int(name.dropFirst(6)), duck.ids.contains(id) { duck.sel = duck.sel == id ? nil : id }
                else if name.hasPrefix("sensor:"), duck.brainOn { let k = String(name.dropFirst(7)); duck.sel = nil; duck.open = duck.open == k ? nil : k }
                else { duck.sel = nil }
            }
        }
    }
}
