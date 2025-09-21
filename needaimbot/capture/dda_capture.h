#pragma once

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

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
    bool AcquireFrame();
    void CaptureThreadProc();

    Microsoft::WRL::ComPtr<ID3D11Device> m_device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
    Microsoft::WRL::ComPtr<IDXGIOutput> m_output;
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> m_duplication;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_stagingTexture;

    DXGI_OUTPUT_DESC m_outputDesc{};
    D3D11_TEXTURE2D_DESC m_frameDesc{};

    std::unique_ptr<unsigned char[]> m_frameBuffer;
    size_t m_bufferSize = 0;

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
};
