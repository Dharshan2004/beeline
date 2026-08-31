import AppKit
import AVFoundation
import CoreVideo
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 3 || arguments.count == 4 else {
    fputs("usage: swift tools/render_demo_video.swift <slides-dir> <output> [codec-fourcc]\n", stderr)
    exit(2)
}

let slidesDirectory = URL(fileURLWithPath: arguments[1], isDirectory: true)
let outputURL = URL(fileURLWithPath: arguments[2])
let codec = AVVideoCodecType(rawValue: arguments.count == 4 ? arguments[3] : "avc1")
let slideDurations: [Double] = [12, 16, 24, 28, 20, 32, 28, 15, 5]
let width = 1920
let height = 1080
let framesPerSecond: Int32 = 15
let transitionSeconds = 0.8

let slideURLs = (1...slideDurations.count).map {
    slidesDirectory.appendingPathComponent(String(format: "slide-%02d.png", $0))
}

func loadImage(_ url: URL) -> CGImage {
    guard let image = NSImage(contentsOf: url) else {
        fputs("could not load \(url.path)\n", stderr)
        exit(3)
    }
    var rect = CGRect(origin: .zero, size: image.size)
    guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
        fputs("could not decode \(url.path)\n", stderr)
        exit(3)
    }
    return cgImage
}

let images = slideURLs.map(loadImage)
try? FileManager.default.removeItem(at: outputURL)

let fileType: AVFileType = outputURL.pathExtension.lowercased() == "mov" ? .mov : .mp4
let writer = try AVAssetWriter(outputURL: outputURL, fileType: fileType)
var settings: [String: Any] = [
    AVVideoCodecKey: codec,
    AVVideoWidthKey: width,
    AVVideoHeightKey: height,
]
if !codec.rawValue.hasPrefix("ap") {
    settings[AVVideoCompressionPropertiesKey] = [
        AVVideoAverageBitRateKey: 3_500_000,
    ]
}
let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
input.expectsMediaDataInRealTime = false
let attributes: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
    kCVPixelBufferWidthKey as String: width,
    kCVPixelBufferHeightKey as String: height,
]
let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: input,
    sourcePixelBufferAttributes: attributes
)
guard writer.canAdd(input) else {
    fputs("AVAssetWriter rejected codec \(codec.rawValue) for \(fileType.rawValue)\n", stderr)
    exit(4)
}
writer.add(input)
guard writer.startWriting() else {
    fputs("could not start writer: \(writer.error?.localizedDescription ?? "unknown error")\n", stderr)
    exit(4)
}
writer.startSession(atSourceTime: .zero)

func draw(_ image: CGImage, in context: CGContext, zoom: CGFloat, alpha: CGFloat) {
    let drawWidth = CGFloat(width) * zoom
    let drawHeight = CGFloat(height) * zoom
    let rect = CGRect(
        x: (CGFloat(width) - drawWidth) / 2,
        y: (CGFloat(height) - drawHeight) / 2,
        width: drawWidth,
        height: drawHeight
    )
    context.saveGState()
    context.setAlpha(alpha)
    context.interpolationQuality = .high
    context.draw(image, in: rect)
    context.restoreGState()
}

var frameNumber: Int64 = 0
for slideIndex in images.indices {
    let duration = slideDurations[slideIndex]
    let frameCount = Int(duration * Double(framesPerSecond))
    let transitionFrames = slideIndex < images.count - 1
        ? Int(transitionSeconds * Double(framesPerSecond))
        : 0
    for localFrame in 0..<frameCount {
        while !input.isReadyForMoreMediaData {
            usleep(1_000)
        }
        var maybeBuffer: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attributes as CFDictionary,
            &maybeBuffer
        )
        guard let buffer = maybeBuffer else {
            fputs("could not allocate video frame\n", stderr)
            exit(5)
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer),
              let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
              ) else {
            fputs("could not create frame context\n", stderr)
            exit(5)
        }
        context.setFillColor(NSColor(calibratedRed: 0.027, green: 0.063, blue: 0.051, alpha: 1).cgColor)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let progress = CGFloat(localFrame) / CGFloat(max(frameCount - 1, 1))
        let zoom = 1.0 + 0.014 * progress
        if transitionFrames > 0 && localFrame >= frameCount - transitionFrames {
            let transition = CGFloat(localFrame - (frameCount - transitionFrames)) / CGFloat(transitionFrames)
            draw(images[slideIndex], in: context, zoom: zoom, alpha: 1.0 - transition)
            draw(images[slideIndex + 1], in: context, zoom: 1.0, alpha: transition)
        } else {
            draw(images[slideIndex], in: context, zoom: zoom, alpha: 1.0)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let presentationTime = CMTime(value: frameNumber, timescale: framesPerSecond)
        guard adaptor.append(buffer, withPresentationTime: presentationTime) else {
            fputs("could not append frame: \(writer.error?.localizedDescription ?? "unknown error")\n", stderr)
            exit(6)
        }
        frameNumber += 1
    }
    print("encoded slide \(slideIndex + 1)/\(images.count)")
}

input.markAsFinished()
let semaphore = DispatchSemaphore(value: 0)
writer.finishWriting {
    semaphore.signal()
}
semaphore.wait()

guard writer.status == .completed else {
    fputs("video export failed: \(writer.error?.localizedDescription ?? "unknown error")\n", stderr)
    exit(7)
}
print(outputURL.path)
