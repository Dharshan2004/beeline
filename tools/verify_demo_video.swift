import AppKit
import AVFoundation
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    fputs("usage: swift tools/verify_demo_video.swift <video.mp4> <frames-dir>\n", stderr)
    exit(2)
}

let videoURL = URL(fileURLWithPath: arguments[1])
let outputDirectory = URL(fileURLWithPath: arguments[2], isDirectory: true)
try FileManager.default.createDirectory(
    at: outputDirectory,
    withIntermediateDirectories: true
)
let asset = AVURLAsset(url: videoURL)
let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero

let times: [Double] = [1, 15, 32, 47, 67, 82, 106, 124, 144, 164, 174]
var failures = 0
for seconds in times {
    let image = try generator.copyCGImage(
        at: CMTime(seconds: seconds, preferredTimescale: 600),
        actualTime: nil
    )
    let bitmap = NSBitmapImageRep(cgImage: image)
    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        fputs("could not encode verification frame\n", stderr)
        exit(3)
    }
    let name = String(format: "frame-%03d.png", Int(seconds))
    try png.write(to: outputDirectory.appendingPathComponent(name))

    var colors = Set<UInt32>()
    let stepX = max(1, image.width / 64)
    let stepY = max(1, image.height / 36)
    guard let data = bitmap.bitmapData else { exit(3) }
    for y in stride(from: 0, to: image.height, by: stepY) {
        for x in stride(from: 0, to: image.width, by: stepX) {
            let offset = y * bitmap.bytesPerRow + x * 4
            let red = UInt32(data[offset]) >> 4
            let green = UInt32(data[offset + 1]) >> 4
            let blue = UInt32(data[offset + 2]) >> 4
            colors.insert((red << 8) | (green << 4) | blue)
        }
    }
    // Flat-color compositor failures produce exactly one quantized color bin.
    // The cinematic storyboard intentionally uses a narrow dark palette, so
    // six bins is enough to distinguish rendered content without rejecting it.
    let passed = colors.count >= 6
    if !passed { failures += 1 }
    print("t=\(Int(seconds))s color_bins=\(colors.count) \(passed ? "PASS" : "BLANK")")
}

let duration = CMTimeGetSeconds(asset.duration)
let videoTracks = asset.tracks(withMediaType: .video).count
let audioTracks = asset.tracks(withMediaType: .audio).count
print("duration=\(duration) video_tracks=\(videoTracks) audio_tracks=\(audioTracks)")
if abs(duration - 180.0) > 0.05 || videoTracks != 1 || audioTracks != 0 {
    failures += 1
}
if failures > 0 { exit(1) }
