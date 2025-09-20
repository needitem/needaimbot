#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <memory>
#include <functional>
#include <mutex>

// Forward declaration for NVFBC
class NVFBCCapture;

class GraphicsCapture {
public:
    GraphicsCapture();
    ~GraphicsCapture();

    bool Initialize(HWND targetWindow = nullptr);
    void Shutdown();

    bool StartCapture();
    void StopCapture();

    bool IsCapturing() const { return m_isCapturing; }

    // Get the latest captured frame data (raw NVFBC buffer)
    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size);

    // Set callback for new frames (raw NVFBC buffer)
    void SetFrameCallback(std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback);

    // Get capture dimensions
    int GetWidth() const { return m_captureWidth; }
    int GetHeight() const { return m_captureHeight; }

    // Check if NVFBC is available
    static bool IsNVFBCAvailable();

private:
    // NVFBC capture (only capture method)
    std::unique_ptr<NVFBCCapture> m_nvfbcCapture;

    // Frame callback
    std::function<void(void*, unsigned int, unsigned int, unsigned int)> m_frameCallback;

    // Capture state
    bool m_isInitialized = false;
    bool m_isCapturing = false;
    int m_captureWidth = 0;
    int m_captureHeight = 0;
};