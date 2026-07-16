#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>

typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket close
#define WSAGetLastError() errno
#define WSAEWOULDBLOCK EWOULDBLOCK
#endif

#include <cuda_runtime.h>

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma pack(push, 1)
// v3 wire format: same layout as v2 plus a trailing capture timestamp. The magic
// is bumped so a v2 and v3 peer reject each other cleanly instead of misparsing;
// rebuild BOTH game PC and inference PC together when this changes.
static constexpr uint32_t UDP_PACKET_V3_MAGIC = 0x33415047u;  // "GPA3" little-endian
static constexpr uint8_t UDP_PIXEL_FORMAT_RGB = 2;

struct UDPPacketHeaderV2 {
    uint32_t magic;
    uint16_t headerSize;
    uint16_t flags;
    uint32_t frameId;
    uint32_t payloadOffset;
    uint32_t frameBytes;
    uint16_t chunkIndex;
    uint16_t totalChunks;
    uint32_t chunkSize;
    uint16_t frameWidth;
    uint16_t frameHeight;
    uint8_t pixelFormat;
    uint8_t bytesPerPixel;
    uint16_t reserved;
    // Game-PC std::chrono::system_clock epoch microseconds stamped at capture,
    // same in every chunk of a frame. 0 = unknown. Used for capture->inference
    // end-to-end latency; assumes the two PCs' wall clocks are NTP-synced.
    uint64_t captureUnixMicros;
};
static_assert(sizeof(UDPPacketHeaderV2) == 44, "UDPPacketHeaderV2 must stay wire-compatible");

static constexpr uint32_t UDP_CREDIT_MAGIC = 0x43504147u;  // "GPAC" little-endian
struct UDPCreditPacket {
    uint32_t magic;
    uint16_t size;
    uint16_t flags;
    uint32_t minFrameId;
    uint32_t credits;
    uint64_t sequence;
};
static_assert(sizeof(UDPCreditPacket) == 24, "UDPCreditPacket must stay wire-compatible");

// Clock-sync round trip (NTP-style) so capture->inference E2E works without the
// two PCs' wall clocks being tightly synced. Inference PC sends a ping with its
// send time t1; game PC replies with a pong echoing t1 plus its own receive (t2)
// and send (t3) times; inference stamps arrival t4 and derives the offset.
static constexpr uint32_t UDP_SYNC_PING_MAGIC = 0x50415047u;  // "GPAP"
static constexpr uint32_t UDP_SYNC_PONG_MAGIC = 0x4F415047u;  // "GPAO"
struct UDPSyncPing {
    uint32_t magic;   // GPAP
    uint32_t seq;
    uint64_t t1;      // inference send time (inference clock, unix us)
};
static_assert(sizeof(UDPSyncPing) == 16, "UDPSyncPing must stay wire-compatible");
struct UDPSyncPong {
    uint32_t magic;   // GPAO
    uint32_t seq;     // echoed
    uint64_t t1;      // echoed inference send time
    uint64_t t2;      // game receive time (game clock)
    uint64_t t3;      // game send time (game clock)
};
static_assert(sizeof(UDPSyncPong) == 32, "UDPSyncPong must stay wire-compatible");
#pragma pack(pop)

class UDPCapture {
public:
    UDPCapture();
    ~UDPCapture();

    bool Initialize(unsigned short listenPort = 5007);
    void Shutdown();

    bool StartCapture();
    void StopCapture();

    bool AcquireFramePinned(void** pinnedRgbData, unsigned int* width, unsigned int* height,
                            uint64_t* outFrameId = nullptr, int* bufferIndex = nullptr,
                            uint32_t timeoutMs = 16, uint8_t* bytesPerPixel = nullptr,
                            uint8_t* pixelFormat = nullptr,
                            uint64_t* outCaptureUnixMicros = nullptr);
    void ReleaseFrame(int bufferIndex);

    uint64_t GetReceivedFrameCount() const { return m_receivedFrames.load(std::memory_order_relaxed); }
    uint64_t GetDroppedFrameCount() const { return m_droppedFrames.load(std::memory_order_relaxed); }
    bool IsPinnedMemoryEnabled() const { return m_usePinnedMemory; }
    bool SendFrameCredit(uint32_t minFrameId = 0, uint32_t credits = 1);

    // Send a clock-sync ping to the credit target (self-throttled to ~10/s, so
    // it is safe to call every main-loop iteration). Pongs are handled on the
    // receive thread and update the offset estimate.
    void SendClockSyncPing();
    // Estimated game_clock - inference_clock in microseconds. Sets *valid=false
    // until the first pong round trip completes; add this to an inference-clock
    // timestamp to compare it against a game-clock (capture) timestamp.
    int64_t GetClockOffsetMicros(bool* valid = nullptr) const;

    // Called on the receive thread immediately after a frame is published
    // (newest-wins buffer swapped in), so a consumer can react without polling.
    // No locks are held at the call site, but the callback still runs ON the
    // receive thread - keep it fast (microseconds) and non-blocking, or the
    // socket stops being drained for as long as the callback takes.
    // Must be set before StartCapture(); not safe to change while running.
    using FrameReadyCallback = std::function<void()>;
    void SetFrameReadyCallback(FrameReadyCallback cb) { m_frameReadyCallback = std::move(cb); }

    // Override the receive thread's CPU core. >=0 pins to that core, <0 leaves
    // it unpinned. Call before StartCapture(). When never set, the thread keeps
    // its built-in default of pinning to the last core.
    static constexpr int kAffinityUnset = -1000;
    void SetReceiveAffinity(int core) { m_receiveAffinityCore = core; }

    // Busy-poll receive (Linux only). Instead of sleeping in recvmmsg and paying
    // the IRQ -> scheduler wakeup latency (~5-15us, worse with DVFS/idle states)
    // on the first packet of every frame, spin on the (dedicated) receive core
    // with MSG_DONTWAIT. Costs 100% of that core while running. Call before
    // StartCapture().
    void SetBusySpin(bool enabled) { m_busySpin = enabled; }

private:
    enum BufferState : int {
        BUFFER_FREE = 0,
        BUFFER_ASSEMBLING = 1,
        BUFFER_READY = 2,
        BUFFER_IN_USE = 3
    };

    struct FrameFragments {
        std::vector<uint8_t> received;
        uint64_t receivedMask = 0;
        bool useReceivedMask = false;
        bool active = false;
        uint32_t frameId = 0;
        int slotIndex = -1;
        int activeListIndex = -1;
        int nextInBucket = -1;
        uint16_t totalPackets = 0;
        uint16_t receivedCount = 0;
        uint16_t width = 0;
        uint16_t height = 0;
        uint8_t bytesPerPixel = 3;
        uint8_t pixelFormat = UDP_PIXEL_FORMAT_RGB;
        size_t frameBytes = 0;
        int bufferIndex = -1;
        bool dropped = false;
        uint64_t captureUnixMicros = 0;
        std::chrono::steady_clock::time_point lastUpdate;
    };

    void receiveThread();
    bool allocatePinnedBuffers(size_t size);
    bool ensurePinnedCapacity(size_t size);
    void freePinnedBuffers();

    FrameFragments* acquireFragment();
    void releaseFragment(FrameFragments* frag);
    FrameFragments* findFragment(uint32_t frameId);
    void linkFragment(FrameFragments* frag, uint32_t frameId);
    void unlinkFragment(FrameFragments* frag);
    int reserveAssemblingBuffer();
    void releaseAssemblingBuffer(int bufferIndex);
    bool publishAssembledBuffer(int bufferIndex, uint16_t width, uint16_t height,
                                uint32_t frameId, uint8_t bytesPerPixel,
                                uint8_t pixelFormat, uint64_t captureUnixMicros,
                                std::chrono::steady_clock::time_point publishTime);
    void clearFragmentState();
    void rememberCreditTarget(const sockaddr_in& addr);
    void handleSyncPong(const uint8_t* data, int len);  // receive-thread only

    static constexpr size_t MAX_FRAGMENT_SLOTS = 256;
    static constexpr size_t FRAGMENT_BUCKETS = 512;
    static_assert((FRAGMENT_BUCKETS & (FRAGMENT_BUCKETS - 1)) == 0,
                  "FRAGMENT_BUCKETS must be power-of-two");
    std::array<FrameFragments, MAX_FRAGMENT_SLOTS> m_fragmentStorage{};
    std::array<int, MAX_FRAGMENT_SLOTS> m_freeFragmentStack{};
    std::array<int, MAX_FRAGMENT_SLOTS> m_activeFragmentSlots{};
    std::array<int, FRAGMENT_BUCKETS> m_bucketHeads{};
    size_t m_freeFragmentCount = 0;
    size_t m_activeFragmentCount = 0;

    SOCKET m_recvSocket = INVALID_SOCKET;

    std::thread m_recvThread;
    int m_receiveAffinityCore = kAffinityUnset;  // see SetReceiveAffinity()
    bool m_busySpin = false;                     // see SetBusySpin()
    std::atomic<bool> m_running{false};

    // 5 buffers: with newest-wins + one in-use by the consumer, leaves room for
    // two frames assembling concurrently (fragments interleave across frames),
    // reducing reservation failures / dropped frames. ~1.6MB each at 640^3.
    static constexpr int NUM_BUFFERS = 5;
    uint8_t* m_pinnedFrameBuffer[NUM_BUFFERS] = {};
    size_t m_pinnedBufferSize = 0;
    bool m_usePinnedMemory = false;

    std::atomic<int> m_bufferState[NUM_BUFFERS] = {
        BUFFER_FREE, BUFFER_FREE, BUFFER_FREE, BUFFER_FREE, BUFFER_FREE};
    std::atomic<unsigned int> m_bufferWidth[NUM_BUFFERS] = {};
    std::atomic<unsigned int> m_bufferHeight[NUM_BUFFERS] = {};
    std::atomic<unsigned int> m_bufferBytesPerPixel[NUM_BUFFERS] = {3, 3, 3, 3, 3};
    std::atomic<unsigned int> m_bufferPixelFormat[NUM_BUFFERS] = {
        UDP_PIXEL_FORMAT_RGB, UDP_PIXEL_FORMAT_RGB, UDP_PIXEL_FORMAT_RGB,
        UDP_PIXEL_FORMAT_RGB, UDP_PIXEL_FORMAT_RGB};
    std::atomic<uint64_t> m_bufferFrameId[NUM_BUFFERS] = {};
    std::atomic<uint64_t> m_bufferCaptureUnixMicros[NUM_BUFFERS] = {};

    std::atomic<int> m_latestBufferIndex{-1};
    std::condition_variable m_publishCv;
    std::mutex m_publishCvMutex;
    std::atomic<uint64_t> m_publishSeq{0};
    uint64_t m_consumedSeq = 0;
    int m_reserveCursor = 0;
    bool m_hasLatestPublishedFrameId = false;
    uint32_t m_latestPublishedFrameId = 0;
    std::chrono::steady_clock::time_point m_latestPublishTime{};
    std::mutex m_creditTargetMutex;
    sockaddr_in m_creditTargetAddr{};
    bool m_hasCreditTarget = false;
    // Lock-free fast-path identity of the learned credit target. rememberCreditTarget
    // runs for every valid chunk (~10^5/s) on the RT receive thread; the sender's
    // address is effectively constant for a session, so these let the per-packet
    // path skip m_creditTargetMutex entirely once learned and unchanged. Updated
    // only under the mutex (learn / addr change); read relaxed on the hot path.
    std::atomic<bool> m_creditTargetLearned{false};
    std::atomic<uint32_t> m_creditTargetAddrId{0};  // raw sin_addr.s_addr snapshot
    std::atomic<uint16_t> m_creditTargetPortId{0};  // raw sin_port snapshot
    std::atomic<uint64_t> m_creditSeq{0};

    std::atomic<uint64_t> m_receivedFrames{0};
    std::atomic<uint64_t> m_droppedFrames{0};

    // Clock-offset estimation (NTP-style) over the ping/pong round trip.
    std::atomic<int64_t> m_clockOffsetUs{0};    // game_clock - inference_clock
    std::atomic<bool> m_clockOffsetValid{false};
    int64_t m_syncBestRttUs = INT64_MAX;        // receive-thread only (clock filter)
    std::chrono::steady_clock::time_point m_syncBestResetTime{};  // receive-thread only
    uint32_t m_pingSeq = 0;                     // main-thread only (SendClockSyncPing)
    std::chrono::steady_clock::time_point m_lastPingTime{};  // main-thread only

    FrameReadyCallback m_frameReadyCallback;
};
