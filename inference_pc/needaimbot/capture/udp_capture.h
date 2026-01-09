#pragma once

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

// Frame header matching game_pc sender
#pragma pack(push, 1)
struct UDPFrameHeader {
    uint32_t magic;      // 0x46524D45 = "FRME"
    uint32_t frameId;
    uint16_t width;
    uint16_t height;
    uint32_t dataSize;
    uint8_t compressed;  // 1 = RLE compressed
};
#pragma pack(pop)

class UDPCapture {
public:
    UDPCapture();
    ~UDPCapture();

    // Initialize with network settings
    bool Initialize(unsigned short listenPort = 5007, 
                   const std::string& gamePcIp = "", 
                   unsigned short mouseStatePort = 5006);
    void Shutdown();

    bool StartCapture();
    void StopCapture();
    bool IsCapturing() const { return m_isCapturing.load(); }

    // Get latest frame as RGB data
    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size);

    // Synchronous API - waits for next frame with timeout
    // Returns RGB data in CUDA pinned memory
    bool AcquireFrameSync(void** rgbData, unsigned int* width, unsigned int* height,
                          uint64_t* outFrameId = nullptr, uint32_t timeoutMs = 16);

    // CUDA API - uploads frame to device memory
    bool AcquireFrameToCuda(void* d_rgbBuffer, size_t bufferSize,
                            unsigned int* width, unsigned int* height,
                            cudaStream_t stream = nullptr,
                            uint32_t timeoutMs = 16);

    // Broadcast mouse state to game PC
    void SendMouseState(bool aimActive, bool shootActive);

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

private:
    void receiveThread();
    bool decompressRLE(const uint8_t* compressed, size_t compressedSize,
                       std::vector<uint8_t>& output, size_t expectedSize);

    // Network
    SOCKET m_recvSocket = INVALID_SOCKET;
    SOCKET m_sendSocket = INVALID_SOCKET;
    sockaddr_in m_gamePcAddr{};
    bool m_hasGamePcAddr = false;
    unsigned short m_listenPort = 5007;
    unsigned short m_mouseStatePort = 5006;

    // Receive thread
    std::thread m_recvThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_isCapturing{false};

    // Frame buffers (double-buffered)
    std::vector<uint8_t> m_frameBuffer[2];
    int m_writeBuffer = 0;
    int m_readBuffer = 1;
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

    // CUDA pinned memory for zero-copy to GPU
    void* m_pinnedBuffer = nullptr;
    size_t m_pinnedBufferSize = 0;

    // Previous aim/shoot state to detect changes
    bool m_lastAimState = false;
    bool m_lastShootState = false;
};
