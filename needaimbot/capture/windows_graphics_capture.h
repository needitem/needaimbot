#ifndef WINDOWS_GRAPHICS_CAPTURE_H
#define WINDOWS_GRAPHICS_CAPTURE_H

#include "../cuda/simple_cuda_mat.h"
#include "capture.h"
#include <memory>
#include <atomic>
#include <d3d11.h>
#include <winrt/base.h>

// Forward declarations to avoid including heavy WinRT headers
namespace winrt::Windows::Graphics::Capture {
    struct GraphicsCaptureItem;
    struct Direct3D11CaptureFramePool;
    struct GraphicsCaptureSession;
}

namespace winrt::Windows::Graphics::DirectX::Direct3D11 {
    struct IDirect3DDevice;
}

class WindowsGraphicsCapture : public IScreenCapture
{
public:
    // Constructor takes desired capture region size (not full screen)
    WindowsGraphicsCapture(int captureWidth, int captureHeight);
    ~WindowsGraphicsCapture();
    
    // IScreenCapture interface implementation
    SimpleCudaMat GetNextFrameGpu() override;
    cudaEvent_t GetCaptureDoneEvent() const override { return m_captureDoneEvent; }
    bool IsInitialized() const override { return m_initialized; }
    
    // Get capture method description
    static const char* GetDescription() {
        return "Windows Graphics Capture API - Optimized for specific region capture with minimal GPU overhead";
    }
    
private:
    bool InitializeCapture();
    bool CreateCaptureItemForMonitor(int monitorIndex);
    void OnFrameArrived();
    
    // D3D11 resources
    ID3D11Device* m_d3dDevice = nullptr;
    ID3D11DeviceContext* m_d3dContext = nullptr;
    ID3D11Texture2D* m_stagingTexture = nullptr;
    
    // WinRT capture objects (using void* to avoid heavy includes)
    void* m_captureItem = nullptr;      // GraphicsCaptureItem
    void* m_framePool = nullptr;        // Direct3D11CaptureFramePool  
    void* m_captureSession = nullptr;   // GraphicsCaptureSession
    void* m_d3dDevice_rt = nullptr;     // IDirect3DDevice
    
    // CUDA resources
    cudaGraphicsResource* m_cudaResource = nullptr;
    cudaStream_t m_cudaStream = nullptr;
    cudaEvent_t m_captureDoneEvent = nullptr;
    
    // Capture parameters
    int m_captureWidth;
    int m_captureHeight;
    int m_screenWidth;
    int m_screenHeight;
    RECT m_captureRegion;
    
    // State
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_newFrameAvailable{false};
    
    // Frame buffers
    SimpleCudaMat m_gpuFrame;
    SimpleMat m_cpuFrame;
    
    // Performance optimization
    void* m_pinnedMemory = nullptr;
    size_t m_pinnedMemorySize = 0;
};

#endif // WINDOWS_GRAPHICS_CAPTURE_H