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
// Linux socket headers
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>

// Windows compatibility
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket close
#define WSAGetLastError() errno
#define WSAEWOULDBLOCK EWOULDBLOCK
#endif

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Packet header matching game_pc sender (chunked BGRA)
#pragma pack(push, 1)
struct UDPPacketHeader {
    uint32_t frameId;         // 4 bytes - frame number
    uint16_t chunkIndex;      // 2 bytes - chunk index (0-based)
    uint16_t totalChunks;     // 2 bytes - total chunks
    uint32_t chunkSize;       // 4 bytes - this chunk's data size
    uint16_t frameWidth;      // 2 bytes - frame width
    uint16_t frameHeight;     // 2 bytes - frame height
};  // Total: 16 bytes
#pragma pack(pop)

class UDPCapture {
public:
    UDPCapture();
    ~UDPCapture();

    // Initialize with network settings
    bool Initialize(unsigned short listenPort = 5007);
    void Shutdown();

    bool StartCapture();
    void StopCapture();
    bool IsCapturing() const { return m_isCapturing.load(); }

    // Get latest frame as RGB data (returns PINNED memory - zero-copy ready)
    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size);

    // Synchronous API - waits for next frame with timeout
    // Returns RGB data in CUDA pinned memory (zero-copy to GPU)
    bool AcquireFrameSync(void** rgbData, unsigned int* width, unsigned int* height,
                          uint64_t* outFrameId = nullptr, uint32_t timeoutMs = 16);

    // Zero-copy API - returns pinned memory pointer for direct GPU access
    // No memcpy needed - data is already in pinned memory
    // Returns the buffer index for double-buffering (0 or 1)
    bool AcquireFramePinned(void** pinnedRgbData, unsigned int* width, unsigned int* height,
                            uint64_t* outFrameId = nullptr, int* bufferIndex = nullptr,
                            uint32_t timeoutMs = 16);

    // Release the pinned buffer after GPU is done (for double-buffering)
    void ReleaseFrame(int bufferIndex);

    // CUDA API - uploads frame to device memory
    bool AcquireFrameToCuda(void* d_rgbBuffer, size_t bufferSize,
                            unsigned int* width, unsigned int* height,
                            cudaStream_t stream = nullptr,
                            uint32_t timeoutMs = 16);

    // Frame info
    uint64_t GetLastFrameId() const { return m_lastFrameId.load(std::memory_order_acquire); }
    uint64_t GetFrameCounter() const { return m_frameCounter.load(std::memory_order_acquire); }

    int GetWidth() const { return m_frameWidth; }
    int GetHeight() const { return m_frameHeight; }
    int GetScreenWidth() const { return m_frameWidth; }   // For compatibility
    int GetScreenHeight() const { return m_frameHeight; }

    // Capture region is determined by game_pc, not configurable here
    bool SetCaptureRegion(int x, int y, int width, int height) { return false; }
    void GetCaptureRegion(int* x, int* y, int* width, int* height) const {
        if (x) *x = 0;
        if (y) *y = 0;
        if (width) *width = m_frameWidth;
        if (height) *height = m_frameHeight;
    }

    // Performance stats
    uint64_t GetReceivedFrameCount() const { return m_receivedFrames.load(); }
    uint64_t GetDroppedFrameCount() const { return m_droppedFrames.load(); }
    double GetReceiveFps() const;

    // Check if using pinned memory
    bool IsPinnedMemoryEnabled() const { return m_usePinnedMemory; }

private:
    void receiveThread();
    bool allocatePinnedBuffers(size_t size);
    void freePinnedBuffers();

    // Fragment reassembly (uses regular memory, converted to pinned on completion)
    struct FrameFragments {
        std::vector<uint8_t> data;
        std::vector<bool> received;  // Track which packets received
        uint16_t totalPackets;
        uint16_t receivedCount;
        uint16_t width;
        uint16_t height;
        std::chrono::steady_clock::time_point lastUpdate;
    };
    std::map<uint32_t, FrameFragments> m_fragmentMap;
    std::mutex m_fragmentMutex;

    // Network
    SOCKET m_recvSocket = INVALID_SOCKET;
    unsigned short m_listenPort = 5007;

    // Receive thread
    std::thread m_recvThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_isCapturing{false};

    // Double-buffered PINNED memory for zero-copy GPU transfer
    // Using CUDA pinned (page-locked) memory eliminates one memcpy
    static constexpr int NUM_BUFFERS = 2;
    uint8_t* m_pinnedFrameBuffer[NUM_BUFFERS] = {nullptr, nullptr};
    size_t m_pinnedBufferSize = 0;
    bool m_usePinnedMemory = false;
    
    // Buffer state for double-buffering
    std::atomic<int> m_writeBuffer{0};
    std::atomic<int> m_readBuffer{1};
    std::atomic<bool> m_bufferInUse[NUM_BUFFERS] = {false, false};
    
    std::mutex m_bufferMutex;
    std::condition_variable m_frameReady;
    bool m_newFrameAvailable = false;

    // Frame info
    std::atomic<int> m_frameWidth{0};
    std::atomic<int> m_frameHeight{0};
    std::atomic<uint64_t> m_lastFrameId{0};
    std::atomic<uint64_t> m_frameCounter{0};

    // Stats
    std::atomic<uint64_t> m_receivedFrames{0};
    std::atomic<uint64_t> m_droppedFrames{0};
    std::chrono::steady_clock::time_point m_startTime;
};
