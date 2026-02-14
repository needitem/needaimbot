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
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#pragma pack(push, 1)
struct UDPPacketHeader {
    uint32_t frameId;
    uint16_t chunkIndex;
    uint16_t totalChunks;
    uint32_t chunkSize;
    uint16_t frameWidth;
    uint16_t frameHeight;
};
#pragma pack(pop)

class UDPCapture {
public:
    UDPCapture();
    ~UDPCapture();

    bool Initialize(unsigned short listenPort = 5007);
    void Shutdown();

    bool StartCapture();
    void StopCapture();
    bool IsCapturing() const { return m_isCapturing.load(std::memory_order_relaxed); }

    bool AcquireFramePinned(void** pinnedRgbData, unsigned int* width, unsigned int* height,
                            uint64_t* outFrameId = nullptr, int* bufferIndex = nullptr,
                            uint32_t timeoutMs = 16);
    void ReleaseFrame(int bufferIndex);

    uint64_t GetDroppedFrameCount() const { return m_droppedFrames.load(std::memory_order_relaxed); }
    bool IsPinnedMemoryEnabled() const { return m_usePinnedMemory; }

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
        int nextInBucket = -1;
        uint16_t totalPackets = 0;
        uint16_t receivedCount = 0;
        uint16_t width = 0;
        uint16_t height = 0;
        int bufferIndex = -1;
        bool dropped = false;
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
    void publishAssembledBuffer(int bufferIndex, uint16_t width, uint16_t height, uint32_t frameId);
    void clearFragmentState();

    static constexpr size_t MAX_FRAGMENT_SLOTS = 256;
    static constexpr size_t FRAGMENT_BUCKETS = 512;
    static_assert((FRAGMENT_BUCKETS & (FRAGMENT_BUCKETS - 1)) == 0,
                  "FRAGMENT_BUCKETS must be power-of-two");
    std::array<FrameFragments, MAX_FRAGMENT_SLOTS> m_fragmentStorage{};
    std::array<int, MAX_FRAGMENT_SLOTS> m_freeFragmentStack{};
    std::array<int, FRAGMENT_BUCKETS> m_bucketHeads{};
    size_t m_freeFragmentCount = 0;
    size_t m_activeFragmentCount = 0;

    SOCKET m_recvSocket = INVALID_SOCKET;
    unsigned short m_listenPort = 5007;

    std::thread m_recvThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_isCapturing{false};

    static constexpr int NUM_BUFFERS = 3;
    uint8_t* m_pinnedFrameBuffer[NUM_BUFFERS] = {nullptr, nullptr, nullptr};
    size_t m_pinnedBufferSize = 0;
    bool m_usePinnedMemory = false;

    std::atomic<int> m_bufferState[NUM_BUFFERS] = {BUFFER_FREE, BUFFER_FREE, BUFFER_FREE};
    std::atomic<unsigned int> m_bufferWidth[NUM_BUFFERS] = {0, 0, 0};
    std::atomic<unsigned int> m_bufferHeight[NUM_BUFFERS] = {0, 0, 0};
    std::atomic<uint64_t> m_bufferFrameId[NUM_BUFFERS] = {0, 0, 0};

    std::atomic<int> m_latestBufferIndex{-1};
    std::atomic<uint64_t> m_publishSeq{0};
    uint64_t m_consumedSeq = 0;
    int m_reserveCursor = 0;

    std::atomic<uint64_t> m_droppedFrames{0};
};
