import AVFoundation
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 5 else {
    fputs("usage: swift tools/mux_demo_voiceover.swift <silent.mp4> <voiceover-dir> <output.mp4> <durations-csv>\n", stderr)
    exit(2)
}

let videoURL = URL(fileURLWithPath: arguments[1])
let voiceoverDirectory = URL(fileURLWithPath: arguments[2], isDirectory: true)
let outputURL = URL(fileURLWithPath: arguments[3])
let durations = arguments[4].split(separator: ",").compactMap { Double($0) }
let composition = AVMutableComposition()
let videoAsset = AVURLAsset(url: videoURL)

guard let sourceVideo = videoAsset.tracks(withMediaType: .video).first,
      let videoTrack = composition.addMutableTrack(
        withMediaType: .video,
        preferredTrackID: kCMPersistentTrackID_Invalid
      ) else {
    fputs("could not read source video\n", stderr)
    exit(3)
}
try videoTrack.insertTimeRange(
    CMTimeRange(start: .zero, duration: videoAsset.duration),
    of: sourceVideo,
    at: .zero
)
videoTrack.preferredTransform = sourceVideo.preferredTransform

guard let audioTrack = composition.addMutableTrack(
    withMediaType: .audio,
    preferredTrackID: kCMPersistentTrackID_Invalid
) else {
    fputs("could not create narration track\n", stderr)
    exit(4)
}

var sceneStart = CMTime.zero
for index in durations.indices {
    let audioURL = voiceoverDirectory.appendingPathComponent(
        String(format: "scene-%02d.aiff", index + 1)
    )
    let asset = AVURLAsset(url: audioURL)
    guard let sourceAudio = asset.tracks(withMediaType: .audio).first else {
        fputs("could not read \(audioURL.path)\n", stderr)
        exit(4)
    }
    let sceneDuration = CMTime(seconds: durations[index], preferredTimescale: 600)
    let padding = CMTime(seconds: 0.45, preferredTimescale: 600)
    let maximumAudio = CMTimeSubtract(sceneDuration, padding)
    let clipDuration = CMTimeMinimum(asset.duration, maximumAudio)
    try audioTrack.insertTimeRange(
        CMTimeRange(start: .zero, duration: clipDuration),
        of: sourceAudio,
        at: sceneStart
    )
    sceneStart = CMTimeAdd(sceneStart, sceneDuration)
}

try? FileManager.default.removeItem(at: outputURL)
guard let exporter = AVAssetExportSession(
    asset: composition,
    presetName: AVAssetExportPreset1920x1080
) else {
    fputs("could not create media exporter\n", stderr)
    exit(5)
}
exporter.outputURL = outputURL
exporter.outputFileType = .mp4
let semaphore = DispatchSemaphore(value: 0)
exporter.exportAsynchronously { semaphore.signal() }
semaphore.wait()
guard exporter.status == .completed else {
    fputs("voiceover export failed: \(exporter.error?.localizedDescription ?? "unknown error")\n", stderr)
    exit(6)
}
print(outputURL.path)
