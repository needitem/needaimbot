#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wrl/client.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <memory>
#include <functional>
#include <mutex>

using Microsoft::WRL::ComPtr;

// Forward declarations to avoid WinRT header conflicts with CUDA
struct IGraphicsCaptureItemInterop;
struct IGraphicsCaptureItem;

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

    // Get the latest captured frame
    ID3D11Texture2D* GetLatestFrame();

    // Set callback for new frames
    void SetFrameCallback(std::function<void(ID3D11Texture2D*)> callback);

    // Get capture dimensions
    int GetWidth() const { return m_captureWidth; }
    int GetHeight() const { return m_captureHeight; }

    // Check if NVFBC is available and preferred
    static bool IsNVFBCAvailable();
    bool IsUsingNVFBC() const { return m_useNVFBC; }

private:
    bool CreateD3DDevice();
    bool CreateCaptureSession(HWND targetWindow);
    void ConvertNVFBCFrameToD3D11(void* nvfbcData, unsigned int width, unsigned int height, unsigned int size);

    // D3D11 resources
    ComPtr<ID3D11Device> m_d3dDevice;
    ComPtr<ID3D11DeviceContext> m_d3dContext;
    ComPtr<IDXGIDevice> m_dxgiDevice;

    // Graphics Capture resources (opaque pointers to avoid WinRT header conflicts)
    void* m_captureItem = nullptr;
    void* m_framePool = nullptr;
    void* m_captureSession = nullptr;
    void* m_winrtDevice = nullptr;

    // NVFBC capture
    std::unique_ptr<NVFBCCapture> m_nvfbcCapture;
    bool m_useNVFBC = false;

    // Frame management
    ComPtr<ID3D11Texture2D> m_latestFrame;
    std::mutex m_frameMutex;
    std::function<void(ID3D11Texture2D*)> m_frameCallback;

    // Capture state
    bool m_isInitialized = false;
    bool m_isCapturing = false;
    int m_captureWidth = 0;
    int m_captureHeight = 0;

    // Event tokens (opaque to avoid WinRT dependencies)
    uint64_t m_frameArrivedToken = 0;
};