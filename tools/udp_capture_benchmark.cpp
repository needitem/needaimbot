#include "capture/udp_capture.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int kBytesPerPixel = 3;

struct Args {
    int port = 5097;
    int frames = 1000;
    double fps = 144.0;
    int width = 256;
    int height = 256;
    int payloadBytes = 60000;
    int acquireTimeoutMs = 16;
    int dropEvery = 0;
    int dropChunk = -1;
    bool creditMode = false;
    int creditTimeoutMs = 50;
    std::vector<int> payloadSweep;
};

struct Stats {
    int acquired = 0;
    int duplicateOrOld = 0;
    uint64_t receivedCounter = 0;
    uint64_t droppedCounter = 0;
    double durationMs = 0.0;
    double acquiredFps = 0.0;
    double ageAvgMs = 0.0;
    double ageP50Ms = 0.0;
    double ageP95Ms = 0.0;
    double ageMaxMs = 0.0;
};

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch())
        .count();
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = (values.size() - 1) * p / 100.0;
    const auto lo = static_cast<size_t>(std::floor(pos));
    const auto hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return values[lo];
    return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

bool parseIntArg(const char* text, int& out) {
    if (!text) return false;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    out = static_cast<int>(value);
    return true;
}

bool parseDoubleArg(const char* text, double& out) {
    if (!text) return false;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0') return false;
    out = value;
    return true;
}

std::vector<int> parsePayloadSweep(const char* text) {
    std::vector<int> values;
    if (!text) return values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        int value = 0;
        if (!parseIntArg(item.c_str(), value)) {
            std::cerr << "Invalid --payload-sweep value: " << item << "\n";
            std::exit(2);
        }
        values.push_back(value);
    }
    return values;
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto needValue = [&](int& target) {
            if (i + 1 >= argc || !parseIntArg(argv[++i], target)) {
                std::cerr << "Invalid value for " << key << "\n";
                std::exit(2);
            }
        };
        if (key == "--port") {
            needValue(args.port);
        } else if (key == "--frames") {
            needValue(args.frames);
        } else if (key == "--fps") {
            if (i + 1 >= argc || !parseDoubleArg(argv[++i], args.fps)) {
                std::cerr << "Invalid value for --fps\n";
                std::exit(2);
            }
        } else if (key == "--width") {
            needValue(args.width);
        } else if (key == "--height") {
            needValue(args.height);
        } else if (key == "--payload-bytes") {
            needValue(args.payloadBytes);
        } else if (key == "--payload-sweep") {
            if (i + 1 >= argc) {
                std::cerr << "Invalid value for --payload-sweep\n";
                std::exit(2);
            }
            args.payloadSweep = parsePayloadSweep(argv[++i]);
        } else if (key == "--acquire-timeout-ms") {
            needValue(args.acquireTimeoutMs);
        } else if (key == "--drop-every") {
            needValue(args.dropEvery);
        } else if (key == "--drop-chunk") {
            needValue(args.dropChunk);
        } else if (key == "--credit-mode") {
            args.creditMode = true;
        } else if (key == "--credit-timeout-ms") {
            needValue(args.creditTimeoutMs);
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: udp_capture_benchmark [options]\n"
                << "  --frames N              Frames to send (default 1000)\n"
                << "  --fps N                 Sender FPS (default 144)\n"
                << "  --width N --height N    Frame shape (default 256x256)\n"
                << "  --payload-bytes N       UDP payload bytes (default 60000)\n"
                << "  --payload-sweep A,B,C   Run multiple payload sizes\n"
                << "  --drop-every N          Drop one chunk every N frames\n"
                << "  --drop-chunk N          Chunk index to drop, -1 means last\n"
                << "  --credit-mode           Wait for one-frame credits between sends\n"
                << "  --credit-timeout-ms N   Recovery send if credit is lost\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << key << "\n";
            std::exit(2);
        }
    }
    return args;
}

bool pollCredit(int sock) {
    UDPCreditPacket packet{};
    sockaddr_in fromAddr{};
    socklen_t fromLen = sizeof(fromAddr);
    const ssize_t ret = recvfrom(sock, &packet, sizeof(packet), 0,
                                 reinterpret_cast<sockaddr*>(&fromAddr), &fromLen);
    return ret == static_cast<ssize_t>(sizeof(packet)) &&
           packet.magic == UDP_CREDIT_MAGIC &&
           packet.size == sizeof(UDPCreditPacket) &&
           packet.credits > 0;
}

void senderThread(const Args& args, const int totalChunks,
                  std::vector<std::atomic<int64_t>>& sendTimesNs,
                  std::atomic<bool>& senderDone) {
    const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        std::cerr << "sender socket failed\n";
        senderDone.store(true);
        return;
    }
    if (args.creditMode) {
        const int flags = fcntl(sock, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(sock, F_SETFL, flags | O_NONBLOCK);
        }
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(args.port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    const int frameBytes = args.width * args.height * kBytesPerPixel;
    std::vector<uint8_t> payload(static_cast<size_t>(args.payloadBytes), 0x5a);

    const auto interval = std::chrono::duration<double>(1.0 / args.fps);
    auto nextFrameAt = Clock::now() + std::chrono::milliseconds(50);
    int credits = 1;
    auto lastSendAt = Clock::now() - std::chrono::milliseconds(args.creditTimeoutMs);

    for (int frameId = 0; frameId < args.frames; ++frameId) {
        std::this_thread::sleep_until(nextFrameAt);
        nextFrameAt += std::chrono::duration_cast<Clock::duration>(interval);

        if (args.creditMode) {
            while (credits <= 0) {
                if (pollCredit(sock)) {
                    credits = 1;
                    break;
                }
                if (Clock::now() - lastSendAt >= std::chrono::milliseconds(args.creditTimeoutMs)) {
                    credits = 1;
                    break;
                }
                if (senderDone.load(std::memory_order_acquire)) break;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            --credits;
        }

        sendTimesNs[static_cast<size_t>(frameId)].store(nowNs(), std::memory_order_release);
        int chunkToDrop = args.dropChunk;
        if (chunkToDrop < 0) chunkToDrop = totalChunks - 1;
        const bool dropThisFrame =
            args.dropEvery > 0 && frameId > 0 && (frameId % args.dropEvery) == 0;

        for (int chunk = 0; chunk < totalChunks; ++chunk) {
            if (dropThisFrame && chunk == chunkToDrop) continue;

            const int offset = chunk * args.payloadBytes;
            const int remaining = frameBytes - offset;
            const int chunkSize = std::min(remaining, args.payloadBytes);

            UDPPacketHeaderV2 header{};
            header.magic = UDP_PACKET_V2_MAGIC;
            header.headerSize = static_cast<uint16_t>(sizeof(UDPPacketHeaderV2));
            header.flags = 0;
            header.frameId = static_cast<uint32_t>(frameId);
            header.payloadOffset = static_cast<uint32_t>(offset);
            header.frameBytes = static_cast<uint32_t>(frameBytes);
            header.chunkIndex = static_cast<uint16_t>(chunk);
            header.totalChunks = static_cast<uint16_t>(totalChunks);
            header.chunkSize = static_cast<uint32_t>(chunkSize);
            header.frameWidth = static_cast<uint16_t>(args.width);
            header.frameHeight = static_cast<uint16_t>(args.height);
            header.pixelFormat = UDP_PIXEL_FORMAT_RGB;
            header.bytesPerPixel = static_cast<uint8_t>(kBytesPerPixel);
            header.reserved = 0;

            iovec iov[2]{};
            iov[0].iov_base = &header;
            iov[0].iov_len = sizeof(header);
            iov[1].iov_base = payload.data();
            iov[1].iov_len = static_cast<size_t>(chunkSize);

            msghdr msg{};
            msg.msg_name = &addr;
            msg.msg_namelen = sizeof(addr);
            msg.msg_iov = iov;
            msg.msg_iovlen = 2;
            sendmsg(sock, &msg, 0);
        }
        lastSendAt = Clock::now();
    }

    close(sock);
    senderDone.store(true, std::memory_order_release);
}

Stats runBenchmark(const Args& args) {
    const int frameBytes = args.width * args.height * kBytesPerPixel;
    const int totalChunks = (frameBytes + args.payloadBytes - 1) / args.payloadBytes;
    std::vector<std::atomic<int64_t>> sendTimesNs(static_cast<size_t>(args.frames));
    for (auto& item : sendTimesNs) {
        item.store(0, std::memory_order_relaxed);
    }

    UDPCapture capture;
    if (!capture.Initialize(static_cast<unsigned short>(args.port))) {
        std::cerr << "UDPCapture initialize failed\n";
        std::exit(1);
    }
    if (!capture.StartCapture()) {
        std::cerr << "UDPCapture start failed\n";
        std::exit(1);
    }

    std::atomic<bool> senderDone{false};
    std::thread sender(senderThread, std::cref(args), totalChunks, std::ref(sendTimesNs),
                       std::ref(senderDone));

    std::vector<double> agesMs;
    agesMs.reserve(static_cast<size_t>(args.frames));
    uint64_t lastFrameId = UINT64_MAX;
    int duplicateOrOld = 0;
    int idleAfterDone = 0;
    const auto startedAt = Clock::now();

    while (true) {
        void* data = nullptr;
        unsigned int width = 0;
        unsigned int height = 0;
        uint64_t frameId = 0;
        int bufferIndex = -1;
        uint8_t bytesPerPixel = 0;
        uint8_t format = 0;
        const bool got = capture.AcquireFramePinned(
            &data, &width, &height, &frameId, &bufferIndex,
            static_cast<uint32_t>(args.acquireTimeoutMs), &bytesPerPixel, &format);
        if (got) {
            if (lastFrameId != UINT64_MAX && frameId <= lastFrameId) {
                ++duplicateOrOld;
            } else if (frameId < static_cast<uint64_t>(args.frames)) {
                const int64_t sentNs =
                    sendTimesNs[static_cast<size_t>(frameId)].load(std::memory_order_acquire);
                if (sentNs != 0) {
                    agesMs.push_back(static_cast<double>(nowNs() - sentNs) / 1'000'000.0);
                }
                lastFrameId = frameId;
                if (args.creditMode) {
                    capture.SendFrameCredit(static_cast<uint32_t>(frameId + 1), 1);
                }
            }
            if (bufferIndex >= 0) capture.ReleaseFrame(bufferIndex);
        } else if (senderDone.load(std::memory_order_acquire)) {
            ++idleAfterDone;
        }

        if (senderDone.load(std::memory_order_acquire) && idleAfterDone >= 8) {
            break;
        }
    }

    if (sender.joinable()) sender.join();
    const auto endedAt = Clock::now();
    Stats stats;
    stats.acquired = static_cast<int>(agesMs.size());
    stats.duplicateOrOld = duplicateOrOld;
    stats.receivedCounter = capture.GetReceivedFrameCount();
    stats.droppedCounter = capture.GetDroppedFrameCount();
    stats.durationMs = std::chrono::duration<double, std::milli>(endedAt - startedAt).count();
    stats.acquiredFps = stats.acquired * 1000.0 / stats.durationMs;
    if (!agesMs.empty()) {
        double sum = 0.0;
        for (double value : agesMs) sum += value;
        stats.ageAvgMs = sum / agesMs.size();
        stats.ageP50Ms = percentile(agesMs, 50);
        stats.ageP95Ms = percentile(agesMs, 95);
        stats.ageMaxMs = *std::max_element(agesMs.begin(), agesMs.end());
    }

    capture.StopCapture();
    return stats;
}
}  // namespace

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);
    std::vector<int> payloadRuns = args.payloadSweep;
    if (payloadRuns.empty()) {
        payloadRuns.push_back(args.payloadBytes);
    }

    for (size_t run = 0; run < payloadRuns.size(); ++run) {
    args.payloadBytes = payloadRuns[run];
    const int frameBytes = args.width * args.height * kBytesPerPixel;
    const int chunks = (frameBytes + args.payloadBytes - 1) / args.payloadBytes;
    std::cout << "bench frames=" << args.frames
              << " fps=" << args.fps
              << " shape=" << args.width << "x" << args.height << "x" << kBytesPerPixel
              << " frameBytes=" << frameBytes
              << " chunks=" << chunks
              << " dropEvery=" << args.dropEvery
              << " credit=" << (args.creditMode ? "on" : "off")
              << "\n";

    const Stats stats = runBenchmark(args);
    std::cout << std::fixed << std::setprecision(3)
              << "result acquired=" << stats.acquired
              << " duplicateOrOld=" << stats.duplicateOrOld
              << " receivedCounter=" << stats.receivedCounter
              << " droppedCounter=" << stats.droppedCounter
              << " durationMs=" << stats.durationMs
              << " acquiredFps=" << stats.acquiredFps
              << " ageAvgMs=" << stats.ageAvgMs
              << " ageP50Ms=" << stats.ageP50Ms
              << " ageP95Ms=" << stats.ageP95Ms
              << " ageMaxMs=" << stats.ageMaxMs
              << "\n";
    }
    return 0;
}
