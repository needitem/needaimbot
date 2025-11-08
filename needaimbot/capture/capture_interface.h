#pragma once

#include <windows.h>
#include <cstdint>
#include <cuda_runtime.h>

class ICaptureProvider {
public:
    virtual ~ICaptureProvider() = default;

    virtual bool Initialize(HWND targetWindow = nullptr) = 0;
    virtual void Shutdown() = 0;

    virtual bool StartCapture() = 0;
    virtual void StopCapture() = 0;
    virtual bool IsCapturing() const = 0;

    // GPU-direct frame path; return false if unsupported.
    virtual bool GetLatestFrameGPU(cudaArray_t* cudaArray, unsigned int* width, unsigned int* height) = 0;

    // CPU fallback path; frameData points to BGRA8 buffer of size bytes.
    virtual bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size) = 0;

    // Synchronize to a frame presented after given QPC; return true on success/timeout handled.
    virtual bool WaitForNewFrameSince(uint64_t minPresentQpc, uint32_t timeoutMs) = 0;
    // Optional: last DXGI LastPresentTime (QPC). Return 0 if unsupported.
    virtual uint64_t GetLastPresentQpc() const { return 0; }

    // Screen and region management
    virtual int GetScreenWidth() const = 0;
    virtual int GetScreenHeight() const = 0;
    virtual bool SetCaptureRegion(int x, int y, int width, int height) = 0;
    virtual void GetCaptureRegion(int* x, int* y, int* width, int* height) const = 0;
};
