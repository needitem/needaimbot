#pragma once

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

// CUDA interop
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "../utils/cuda_utils.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

class DDACapture {
public:
    DDACapture();
    ~DDACapture();

    bool Initialize(HWND targetWindow = nullptr);
    void Shutdown();

    bool StartCapture();
    void StopCapture();

    bool IsCapturing() const { return m_isCapturing.load(); }

    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size);
    void SetFrameCallback(std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback);

    // GPU-direct API - returns CUDA array pointer (zero-copy)
    bool GetLatestFrameGPU(cudaArray_t* cudaArray, unsigned int* width, unsigned int* height);

    // Synchronization helpers
    // Blocks until a frame with DXGI LastPresentTime >= minPresentQpc is captured
    // or timeoutMs elapses. Returns true if condition met, false on timeout or stop.
    bool WaitForNewFrameSince(uint64_t minPresentQpc, uint32_t timeoutMs);
    uint64_t GetLastPresentQpc() const { return m_lastPresentQpc.load(std::memory_order_acquire); }
    uint64_t GetFrameCounter() const { return m_frameCounter.load(std::memory_order_acquire); }

    // Performance monitoring getters
    uint64_t GetFrameDropCount() const { return m_frameDropCount.load(std::memory_order_relaxed); }
    uint64_t GetCaptureErrorCount() const { return m_captureErrorCount.load(std::memory_order_relaxed); }
    uint64_t GetAccessLostCount() const { return m_accessLostCount.load(std::memory_order_relaxed); }

    int GetWidth() const { return m_screenWidth; }
    int GetHeight() const { return m_screenHeight; }
    int GetScreenWidth() const { return m_screenWidth; }
    int GetScreenHeight() const { return m_screenHeight; }

    bool SetCaptureRegion(int x, int y, int width, int height);
    void GetCaptureRegion(int* x, int* y, int* width, int* height) const;
    void ResetToFullScreen();

    static bool IsDDACaptureAvailable();

private:
    enum class FrameAcquireResult {
        kFrameCaptured,
        kNoFrame,
        kError,
    };

    // Capture mode state machine
    enum class CaptureMode {
        kGpuDirect,      // Zero-copy CUDA interop (fast path)
        kCpuFallback,    // CPU staging + pinned memory (slow path)
        kProbing         // Testing GPU direct availability
    };

    bool InitializeDeviceAndOutput(HWND targetWindow);
    bool CreateDuplicationInterface();
    void ReleaseDuplication();

    bool EnsureStagingTexture(UINT width, UINT height, DXGI_FORMAT format);
    bool EnsureFrameBuffer(size_t requiredSize);
    FrameAcquireResult AcquireFrame();
    void CaptureThreadProc();

    // State machine helpers
    void TransitionToCaptureMode(CaptureMode newMode);
    bool ProbeGpuDirectCapability();
    FrameAcquireResult AcquireFrameGpuDirect();
    FrameAcquireResult AcquireFrameCpuFallback();

    Microsoft::WRL::ComPtr<ID3D11Device> m_device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
    Microsoft::WRL::ComPtr<IDXGIOutput> m_output;
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> m_duplication;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_stagingTexture;

    DXGI_OUTPUT_DESC m_outputDesc{};
    D3D11_TEXTURE2D_DESC m_frameDesc{};

    // Pinned host buffer to accelerate async cudaMemcpy2DAsync
    std::unique_ptr<CudaPinnedMemory<unsigned char>> m_frameBuffer;
    size_t m_bufferSize = 0;

    // CUDA interop - shared texture for zero-copy
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_sharedTexture;
    cudaGraphicsResource* m_cudaGraphicsResource = nullptr;
    cudaArray_t m_cudaMappedArray = nullptr;
    bool m_cudaInteropEnabled = false;

    // State machine for capture path selection
    std::atomic<CaptureMode> m_currentMode{CaptureMode::kProbing};
    std::atomic<bool> m_cpuFallbackNeeded{false};  // External signal to force CPU path
    bool m_cpuBufferValid = false;

    // Probing state
    std::chrono::steady_clock::time_point m_lastProbeTime;
    int m_consecutiveGpuFailures = 0;
    static constexpr int kMaxGpuFailuresBeforeFallback = 3;
    static constexpr std::chrono::milliseconds kProbeInterval{5000};

    // Adaptive backoff state
    enum class BackoffLevel {
        kNone,           // 0: Immediate retry (success)
        kMinimal,        // 1: light retry, no explicit yield
        kShort,          // 2: additional backoff stage, no yield
        kMedium,         // 3: Sleep(1ms) (65-128 failures)
        kLong            // 4: Sleep(2ms) (128+ failures)
    };
    int m_consecutiveNoFrames = 0;
    BackoffLevel m_currentBackoffLevel = BackoffLevel::kNone;

    mutable std::mutex m_frameMutex;
    std::function<void(void*, unsigned int, unsigned int, unsigned int)> m_frameCallback;

    std::thread m_captureThread;
    std::atomic<bool> m_isCapturing{false};
    bool m_isInitialized = false;

    int m_screenWidth = 0;
    int m_screenHeight = 0;

    DXGI_FORMAT m_duplicationFormat = DXGI_FORMAT_UNKNOWN;

    RECT m_captureRegion{0, 0, 0, 0};
    unsigned int m_captureWidth = 0;
    unsigned int m_captureHeight = 0;

    // Frame timing/state for synchronization with consumers
    std::atomic<uint64_t> m_lastPresentQpc{0};
    std::atomic<uint64_t> m_frameCounter{0};
    mutable std::mutex m_presentMutex;
    std::condition_variable m_presentCv;

    // Performance monitoring
    std::atomic<uint64_t> m_frameDropCount{0};
    std::atomic<uint64_t> m_captureErrorCount{0};
    std::atomic<uint64_t> m_accessLostCount{0};

    // Adaptive timing state (to reduce excessive yields without adding jitter)
    uint64_t m_prevPresentQpc{0};
    double m_qpcToMs{0.0};
    double m_estimatedIntervalMs{6.9}; // start near 144Hz

    // Computes an appropriate AcquireNextFrame timeout based on recent frame interval.
    // Clamped to [2, 8] ms: 2ms supports up to ~500Hz; 8ms keeps latency low at 60-144Hz.
    UINT AcquireTimeoutMs() const {
        double base = m_estimatedIntervalMs * 0.60; // wake earlier to reduce jitter
        if (base < 2.0) base = 2.0;
        if (base > 8.0) base = 8.0;
        return static_cast<UINT>(base + 0.5);
    }
};
