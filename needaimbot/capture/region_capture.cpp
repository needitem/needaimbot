// Region-based GPU Capture using Windows.Graphics.Capture
// This captures only the specified region, not the entire screen
// Optimized for minimal GPU memory usage and maximum performance

#include "../core/windows_headers.h"
#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <iostream>
#include <atomic>
#include <algorithm>
#include "../AppContext.h"

// WinRT headers - must be after windows.h
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.System.h>
#include <windows.graphics.capture.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

// Forward declare IDirect3DDxgiInterfaceAccess if not defined
struct __declspec(uuid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1"))
IDirect3DDxgiInterfaceAccess : public IUnknown
{
    virtual HRESULT __stdcall GetInterface(GUID const& id, void** object) = 0;
};

#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

#include "region_capture.h"

using namespace winrt;
namespace WGC = winrt::Windows::Graphics::Capture;
namespace WGD = winrt::Windows::Graphics::DirectX;
namespace WGD3D = winrt::Windows::Graphics::DirectX::Direct3D11;
using Microsoft::WRL::ComPtr;

// Constructor
RegionCapture::RegionCapture(int width, int height) 
    : m_session(nullptr), m_framePool(nullptr), m_item(nullptr),
      m_winrtDevice(nullptr),
      m_width(width), m_height(height),
      m_cudaResource(nullptr), m_captureStream(nullptr),
      m_isCapturing(false), m_frameAvailable(false) {
}

// Destructor
RegionCapture::~RegionCapture() {
    StopCapture();
    Cleanup();
}

// Initialize method
bool RegionCapture::Initialize() {
        try {
            // Initialize WinRT
            init_apartment(apartment_type::single_threaded);
            
            // 1. Create D3D11 device with BGRA support
            D3D_FEATURE_LEVEL featureLevels[] = {
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0
            };
            
            D3D_FEATURE_LEVEL featureLevel;
            UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
            
            HRESULT hr = D3D11CreateDevice(
                nullptr, 
                D3D_DRIVER_TYPE_HARDWARE, 
                nullptr,
                createFlags,
                featureLevels, 
                ARRAYSIZE(featureLevels), 
                D3D11_SDK_VERSION,
                &m_d3dDevice, 
                &featureLevel, 
                &m_d3dContext
            );
            
            if (FAILED(hr)) {
                std::cerr << "[RegionCapture] Failed to create D3D11 device: " << std::hex << hr << std::endl;
                return false;
            }
            
            // Get DXGI device
            hr = m_d3dDevice.As(&m_dxgiDevice);
            if (FAILED(hr)) {
                std::cerr << "[RegionCapture] Failed to get DXGI device" << std::endl;
                return false;
            }
            
            // 2. Create WinRT Direct3D device
            com_ptr<::IInspectable> inspectable;
            hr = CreateDirect3D11DeviceFromDXGIDevice(m_dxgiDevice.Get(), inspectable.put());
            if (FAILED(hr)) {
                std::cerr << "[RegionCapture] Failed to create WinRT device" << std::endl;
                return false;
            }
            m_winrtDevice = inspectable.as<WGD3D::IDirect3DDevice>();
            
            // 3. Create staging texture for CUDA (exact capture size)
            D3D11_TEXTURE2D_DESC desc = {};
            desc.Width = m_width;
            desc.Height = m_height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
            desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
            
            hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, &m_stagingTexture);
            if (FAILED(hr)) {
                std::cerr << "[RegionCapture] Failed to create staging texture" << std::endl;
                return false;
            }
            
            // 4. Initialize CUDA interop
            cudaError_t cudaErr = cudaStreamCreate(&m_captureStream);
            if (cudaErr != cudaSuccess) {
                std::cerr << "[RegionCapture] Failed to create CUDA stream" << std::endl;
                return false;
            }
            
            cudaErr = cudaGraphicsD3D11RegisterResource(
                &m_cudaResource,
                m_stagingTexture.Get(),
                cudaGraphicsRegisterFlagsNone
            );
            
            if (cudaErr != cudaSuccess) {
                std::cerr << "[RegionCapture] Failed to register D3D11 resource with CUDA" << std::endl;
                return false;
            }
            
            std::cout << "[RegionCapture] Initialized successfully" << std::endl;
            return true;
            
        } catch (std::exception& e) {
            std::cerr << "[RegionCapture] Exception during initialization: " << e.what() << std::endl;
            return false;
        }
    }

bool RegionCapture::StartCapture() {
        std::cout << "[RegionCapture::StartCapture] Starting capture initialization..." << std::endl;
        try {
            // Calculate capture region centered on screen with offset
            auto& ctx = AppContext::getInstance();
            int screenWidth = GetSystemMetrics(SM_CXSCREEN);
            int screenHeight = GetSystemMetrics(SM_CYSCREEN);
            
            int centerX = screenWidth / 2;
            int centerY = screenHeight / 2;
            
            // Apply offset from config
            bool useAimShootOffset = ctx.aiming && ctx.shooting;
            int offsetX = useAimShootOffset ? 
                          static_cast<int>(ctx.config.aim_shoot_offset_x) : 
                          static_cast<int>(ctx.config.crosshair_offset_x);
            int offsetY = useAimShootOffset ? 
                          static_cast<int>(ctx.config.aim_shoot_offset_y) : 
                          static_cast<int>(ctx.config.crosshair_offset_y);
            
            // Calculate region bounds
            m_regionX = centerX + offsetX - (m_width / 2);
            m_regionY = centerY + offsetY - (m_height / 2);
            
            // Clamp to screen bounds
            m_regionX = (std::max)(0, (std::min)(m_regionX, screenWidth - m_width));
            m_regionY = (std::max)(0, (std::min)(m_regionY, screenHeight - m_height));
            
            // Create virtual monitor capture item for the specific region
            auto factory = get_activation_factory<WGC::GraphicsCaptureItem>();
            auto interop = factory.as<IGraphicsCaptureItemInterop>();
            
            // Create a monitor handle for the region
            RECT captureRect;
            captureRect.left = m_regionX;
            captureRect.top = m_regionY;
            captureRect.right = m_regionX + m_width;
            captureRect.bottom = m_regionY + m_height;
            
            HMONITOR hmon = MonitorFromRect(&captureRect, MONITOR_DEFAULTTOPRIMARY);
            
            // Create capture item from monitor
            HRESULT hr = interop->CreateForMonitor(
                hmon,
                guid_of<WGC::IGraphicsCaptureItem>(),
                put_abi(m_item)
            );
            
            if (FAILED(hr)) {
                std::cerr << "[RegionCapture] Failed to create capture item: " << std::hex << hr << std::endl;
                std::cerr << "[RegionCapture] Note: Windows Graphics Capture can only capture full monitors or windows, not arbitrary regions." << std::endl;
                std::cerr << "[RegionCapture] Consider using Desktop Duplication mode (capture_method=0) for better performance." << std::endl;
                return false;
            }
            
            // Get actual capture size (will be full monitor)
            auto captureSize = m_item.Size();
            std::cout << "[RegionCapture] WARNING: Capturing full monitor (" << captureSize.Width << "x" << captureSize.Height 
                      << ") then cropping to " << m_width << "x" << m_height << std::endl;
            std::cout << "[RegionCapture] This is inefficient! Consider using Desktop Duplication mode instead." << std::endl;
            
            // Create frame pool with FULL MONITOR SIZE (not region size)
            m_framePool = WGC::Direct3D11CaptureFramePool::CreateFreeThreaded(
                m_winrtDevice,
                WGD::DirectXPixelFormat::B8G8R8A8UIntNormalized,
                2,  // Buffer count
                winrt::Windows::Graphics::SizeInt32{ captureSize.Width, captureSize.Height }
            );
            
            // Set up frame arrived callback
            m_frameArrivedToken = m_framePool.FrameArrived([this](auto&, auto&) {
                static int frameCallbackCount = 0;
                frameCallbackCount++;
                if (frameCallbackCount <= 10 || frameCallbackCount % 100 == 0) {
                    std::cout << "[RegionCapture] Frame arrived callback #" << frameCallbackCount << std::endl;
                }
                m_frameAvailable.store(true);
            });
            
            // Create and configure capture session
            m_session = m_framePool.CreateCaptureSession(m_item);
            m_session.IsBorderRequired(false);
            m_session.IsCursorCaptureEnabled(false);
            
            // Start capturing
            m_session.StartCapture();
            m_isCapturing = true;
            
            std::cout << "[RegionCapture] Started capturing region: " 
                      << m_width << "x" << m_height 
                      << " at (" << m_regionX << ", " << m_regionY << ")" 
                      << std::endl;
            
            return true;
            
        } catch (std::exception& e) {
            std::cerr << "[RegionCapture] Exception starting capture: " << e.what() << std::endl;
            return false;
        }
    }

void RegionCapture::StopCapture() {
        m_isCapturing = false;
        
        if (m_session) {
            try {
                m_session.Close();
                m_session = nullptr;
            } catch (...) {}
        }
        
        if (m_framePool) {
            try {
                m_framePool.FrameArrived(m_frameArrivedToken);
                m_framePool.Close();
                m_framePool = nullptr;
            } catch (...) {}
        }
        
        if (m_item) {
            try {
                m_item = nullptr;
            } catch (...) {}
        }
    }
bool RegionCapture::WaitForNextFrame() {
        if (!m_isCapturing || !m_framePool) {
            return false;
        }
        
        // Check if frame is available
        if (!m_frameAvailable.exchange(false)) {
            return false;
        }
        
        try {
            auto frame = m_framePool.TryGetNextFrame();
            if (!frame) {
                return false;
            }
            
            // Get the Direct3D surface
            auto surface = frame.Surface();
            auto access = surface.as<IDirect3DDxgiInterfaceAccess>();
            
            ComPtr<ID3D11Texture2D> frameTexture;
            HRESULT hr = access->GetInterface(IID_PPV_ARGS(&frameTexture));
            if (FAILED(hr)) {
                frame.Close();
                return false;
            }
            
            // Get texture description
            D3D11_TEXTURE2D_DESC frameDesc;
            frameTexture->GetDesc(&frameDesc);
            
            // If capture is exactly our size, direct copy
            if (frameDesc.Width == m_width && frameDesc.Height == m_height) {
                m_d3dContext->CopyResource(m_stagingTexture.Get(), frameTexture.Get());
            } else {
                // If not exact size, copy the region we need
                D3D11_BOX sourceBox = {};
                sourceBox.left = m_regionX;
                sourceBox.right = m_regionX + m_width;
                sourceBox.top = m_regionY;
                sourceBox.bottom = m_regionY + m_height;
                sourceBox.front = 0;
                sourceBox.back = 1;
                
                m_d3dContext->CopySubresourceRegion(
                    m_stagingTexture.Get(), 0, 0, 0, 0,
                    frameTexture.Get(), 0, &sourceBox
                );
            }
            
            // Close the frame
            frame.Close();
            return true;
            
        } catch (std::exception& e) {
            std::cerr << "[RegionCapture] Exception capturing frame: " << e.what() << std::endl;
            return false;
        }
    }
void RegionCapture::UpdateRegion(int offsetX, int offsetY) {
        // Update capture region dynamically
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);
        
        int centerX = screenWidth / 2;
        int centerY = screenHeight / 2;
        
        m_regionX = centerX + offsetX - (m_width / 2);
        m_regionY = centerY + offsetY - (m_height / 2);
        
        // Clamp to screen bounds
        m_regionX = (std::max)(0, (std::min)(m_regionX, screenWidth - m_width));
        m_regionY = (std::max)(0, (std::min)(m_regionY, screenHeight - m_height));
        
        // Note: To apply new region, need to restart capture
        // This would be called when offset changes significantly
    }
cudaGraphicsResource_t RegionCapture::GetCudaResource() const { 
    return m_cudaResource; 
}

bool RegionCapture::IsCapturing() const {
    return m_isCapturing;
}

void RegionCapture::Cleanup() {
        if (m_cudaResource) {
            cudaGraphicsUnregisterResource(m_cudaResource);
            m_cudaResource = nullptr;
        }
        
        if (m_captureStream) {
            cudaStreamDestroy(m_captureStream);
            m_captureStream = nullptr;
        }
        
        m_stagingTexture.Reset();
        m_d3dContext.Reset();
        m_d3dDevice.Reset();
        m_dxgiDevice.Reset();
        m_winrtDevice = nullptr;
    }