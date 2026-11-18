// Runs the app's own NV12 decoder (Duck/Pixels.swift) on the Mac:  check_codec in.nv12 W H lift(0|1) out.rgba
import Foundation
let a = CommandLine.arguments
let d = try! Data(contentsOf: URL(fileURLWithPath: a[1]))
try! Data(nv12Pixels(d, Int(a[2])!, Int(a[3])!, lift: a[4] == "1")).write(to: URL(fileURLWithPath: a[5]))
