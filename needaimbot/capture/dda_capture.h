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

    // Initialize capture resources (no background thread in sync mode)
    bool StartCapture();
    void StopCapture();

    bool IsCapturing() const { return m_isCapturing.load(); }

    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size);
    void SetFrameCallback(std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback);

    // Synchronous capture API - acquires next frame directly without background thread
    // Returns: true if new frame captured, false if timeout or error
    // This is the preferred API for single-threaded pipeline usage
    bool AcquireFrameSync(cudaArray_t* cudaArray, unsigned int* width, unsigned int* height,
                          uint64_t* outPresentQpc = nullptr, uint32_t timeoutMs = 8);

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
    bool InitializeDeviceAndOutput(HWND targetWindow);
    bool CreateDuplicationInterface();
    void ReleaseDuplication();

    bool EnsureStagingTexture(UINT width, UINT height, DXGI_FORMAT format);
    bool EnsureFrameBuffer(size_t requiredSize);

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

    mutable std::mutex m_frameMutex;
    std::function<void(void*, unsigned int, unsigned int, unsigned int)> m_frameCallback;

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
    // Uses AppContext.config.capture_timeout_scale; clamps to [1, 8] ms.
    UINT AcquireTimeoutMs() const;
};
