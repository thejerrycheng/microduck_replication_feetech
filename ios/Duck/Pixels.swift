// The camera's frames: NV12 (full range, BT.601, as the duck's ISP delivers them) to RGBA, turned a quarter turn
// clockwise because the sensor sits on its side in the head. On its own so tools/check_codec.py can run the very same
// code on the Mac against a reference decoder.
import CoreGraphics
import Foundation

/// The exponent that lifts a dark frame: mean luma `mean` (0...255) is brought to about 35 % grey; never darkens,
/// never below 0.45 (a black frame stays black instead of turning into noise).
func liftGamma(_ mean: Double) -> Double {
    guard mean > 1, mean < 89 else { return 1 }
    return max(0.45, min(1, log(0.35) / log(mean / 255)))
}

/// RGBA bytes, h wide and w tall (rotated). `lift`: brighten dark frames (see liftGamma); off gives the plain decode.
func nv12Pixels(_ d: Data, _ w: Int, _ h: Int, lift: Bool = true) -> [UInt8] {
    var px = [UInt8](repeating: 255, count: w * h * 4)
    guard d.count == w * h * 3 / 2 else { return px }
    d.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
        var sum = 0
        for k in 0..<(w * h) { sum += Int(b[k]) }
        let g = lift ? liftGamma(Double(sum) / Double(w * h)) : 1
        var lut = [Float](repeating: 0, count: 256), sat = [Float](repeating: 1, count: 256)
        for k in 0..<256 {
            lut[k] = g == 1 ? Float(k) : Float(255 * pow(Double(k) / 255, g))
            sat[k] = g == 1 ? 1 : min(2, (lut[k] / Float(max(k, 1))).squareRoot())      // colour follows the lift, gently
        }
        for j in 0..<h {
            for i in 0..<w {
                let k = Int(b[j * w + i]), y = lut[k], q = w * h + (j >> 1) * w + (i & ~1)
                let u = (Float(b[q]) - 128) * sat[k], v = (Float(b[q + 1]) - 128) * sat[k]
                let p = (i * h + (h - 1 - j)) * 4                 // source (i, j) -> column h-1-j, row i
                px[p] = UInt8(max(0, min(255, (y + 1.402 * v).rounded())))
                px[p + 1] = UInt8(max(0, min(255, (y - 0.344 * u - 0.714 * v).rounded())))
                px[p + 2] = UInt8(max(0, min(255, (y + 1.772 * u).rounded())))
            }
        }
    }
    return px
}

func nv12(_ d: Data, _ w: Int, _ h: Int) -> CGImage? {
    guard d.count == w * h * 3 / 2, let prov = CGDataProvider(data: Data(nv12Pixels(d, w, h)) as CFData) else { return nil }
    return CGImage(width: h, height: w, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: h * 4, space: CGColorSpaceCreateDeviceRGB(),
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: prov, decode: nil,
                   shouldInterpolate: true, intent: .defaultIntent)
}
