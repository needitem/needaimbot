#pragma once

#include "../core/windows_headers.h"
#include <d3d11.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <atomic>

// WinRT headers
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

class RegionCapture {
private:
    // Capture session
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession m_session;
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool m_framePool;
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem m_item;
    
    // D3D11 resources
    Microsoft::WRL::ComPtr<ID3D11Device> m_d3dDevice;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_d3dContext;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_stagingTexture;
    Microsoft::WRL::ComPtr<IDXGIDevice> m_dxgiDevice;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_winrtDevice;
    
    // CUDA interop
    cudaGraphicsResource_t m_cudaResource;
    cudaStream_t m_captureStream;
    
    // Capture region
    int m_regionX, m_regionY;
    int m_width, m_height;
    
    // State
    std::atomic<bool> m_isCapturing;
    std::atomic<bool> m_frameAvailable;
    
    // Frame callback
    winrt::event_token m_frameArrivedToken;
    
public:
    RegionCapture(int width, int height);
    ~RegionCapture();
    
    bool Initialize();
    bool StartCapture();
    void StopCapture();
    bool WaitForNextFrame();
    void UpdateRegion(int offsetX, int offsetY);
    
    cudaGraphicsResource_t GetCudaResource() const;
    bool IsCapturing() const;
    
private:
    void Cleanup();
};