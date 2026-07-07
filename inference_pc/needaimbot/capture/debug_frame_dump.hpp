#pragma once

// Debug aid: periodically dumps the latest received RGB frame to a BMP file
// (with a crosshair marker burned in) so capture/alignment issues can be
// eyeballed without a display attached to the inference box.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

inline void writeU16LE(std::ostream& out, uint16_t value) {
    const char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

inline void writeU32LE(std::ostream& out, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

inline void writeI32LE(std::ostream& out, int32_t value) {
    writeU32LE(out, static_cast<uint32_t>(value));
}

// Writes a 24bpp top-down BMP with a small red crosshair marker at the image
// center (a green dot at the very center pixel) so the dump also shows where
// the aim point lands relative to the frame.
inline bool writeRgbBmp(
    const std::filesystem::path& path,
    const uint8_t* rgbData,
    unsigned int width,
    unsigned int height) {
    if (!rgbData || width == 0 || height == 0) return false;

    const uint64_t rawRowBytes = static_cast<uint64_t>(width) * 3u;
    const uint64_t rowStride = (rawRowBytes + 3u) & ~uint64_t{3u};
    const uint64_t pixelBytes = rowStride * static_cast<uint64_t>(height);
    constexpr uint32_t kHeaderBytes = 14u + 40u;
    if (pixelBytes > std::numeric_limits<uint32_t>::max() - kHeaderBytes ||
        width > static_cast<unsigned int>(std::numeric_limits<int32_t>::max()) ||
        height > static_cast<unsigned int>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) return false;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    const uint32_t fileSize = kHeaderBytes + static_cast<uint32_t>(pixelBytes);

    out.write("BM", 2);
    writeU32LE(out, fileSize);
    writeU16LE(out, 0);
    writeU16LE(out, 0);
    writeU32LE(out, kHeaderBytes);

    writeU32LE(out, 40);  // BITMAPINFOHEADER
    writeI32LE(out, static_cast<int32_t>(width));
    writeI32LE(out, -static_cast<int32_t>(height));  // top-down BMP
    writeU16LE(out, 1);
    writeU16LE(out, 24);
    writeU32LE(out, 0);
    writeU32LE(out, static_cast<uint32_t>(pixelBytes));
    writeI32LE(out, 0);
    writeI32LE(out, 0);
    writeU32LE(out, 0);
    writeU32LE(out, 0);

    std::vector<uint8_t> row(static_cast<size_t>(rowStride), 0);
    const unsigned int centerX = width / 2u;
    const unsigned int centerY = height / 2u;
    const unsigned int markerRadius = std::max(8u, std::min(width, height) / 16u);
    for (unsigned int y = 0; y < height; ++y) {
        const uint8_t* src = rgbData + static_cast<size_t>(y) * static_cast<size_t>(width) * 3u;
        std::fill(row.begin(), row.end(), 0);
        for (unsigned int x = 0; x < width; ++x) {
            uint8_t r = src[static_cast<size_t>(x) * 3u + 0u];
            uint8_t g = src[static_cast<size_t>(x) * 3u + 1u];
            uint8_t b = src[static_cast<size_t>(x) * 3u + 2u];
            const unsigned int dx = (x > centerX) ? (x - centerX) : (centerX - x);
            const unsigned int dy = (y > centerY) ? (y - centerY) : (centerY - y);
            const bool onCenterDot = dx <= 1u && dy <= 1u;
            const bool onVerticalMarker = dx <= 1u && dy <= markerRadius;
            const bool onHorizontalMarker = dy <= 1u && dx <= markerRadius;
            if (onVerticalMarker || onHorizontalMarker) {
                r = 255;
                g = onCenterDot ? 255 : 0;
                b = 0;
            }
            row[static_cast<size_t>(x) * 3u + 0u] = b;
            row[static_cast<size_t>(x) * 3u + 1u] = g;
            row[static_cast<size_t>(x) * 3u + 2u] = r;
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
        if (!out) return false;
    }

    return true;
}

// Schedules a 1Hz BMP dump of the latest frame (with a retry backoff when a
// frame isn't available yet).
struct DebugFrameDumper {
    using Clock = std::chrono::steady_clock;

    bool enabled = false;
    std::filesystem::path outputPath;
    Clock::time_point nextCaptureTime{};
    uint64_t savedFrames = 0;
    bool reportedSaveError = false;

    bool due(Clock::time_point now) const {
        return enabled && now >= nextCaptureTime;
    }

    void scheduleNext(Clock::time_point now) {
        nextCaptureTime = now + std::chrono::seconds(1);
    }

    void scheduleRetry(Clock::time_point now) {
        nextCaptureTime = now + std::chrono::milliseconds(100);
    }

    void save(const void* rgbData, unsigned int width, unsigned int height, uint64_t frameId) {
        if (!enabled) return;

        if (writeRgbBmp(outputPath, static_cast<const uint8_t*>(rgbData), width, height)) {
            ++savedFrames;
            reportedSaveError = false;
            (void)frameId;
        } else if (!reportedSaveError) {
            std::cerr << "\n[Debug] Failed to save received frame to "
                      << outputPath.lexically_normal().string() << std::endl;
            reportedSaveError = true;
        }
    }
};
