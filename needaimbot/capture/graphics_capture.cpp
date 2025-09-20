#include "graphics_capture.h"
#include "nvfbc_capture.h"
#include <iostream>

GraphicsCapture::GraphicsCapture() = default;

GraphicsCapture::~GraphicsCapture() {
    Shutdown();
}

bool GraphicsCapture::Initialize(HWND targetWindow) {
    // Create D3D11 device for texture management
    if (!CreateD3DDevice()) {
        std::cerr << "[GraphicsCapture] Failed to create D3D device" << std::endl;
        return false;
    }

    // Try to use NVFBC first if available
    if (IsNVFBCAvailable()) {
        m_nvfbcCapture = std::make_unique<NVFBCCapture>();
        if (m_nvfbcCapture->Initialize(targetWindow)) {
            m_useNVFBC = true;
            m_captureWidth = m_nvfbcCapture->GetWidth();
            m_captureHeight = m_nvfbcCapture->GetHeight();

            // Set callback to convert NVFBC frames to D3D11 textures
            m_nvfbcCapture->SetFrameCallback([this](void* data, unsigned int width, unsigned int height, unsigned int size) {
                ConvertNVFBCFrameToD3D11(data, width, height, size);
            });

            m_isInitialized = true;
            std::cout << "[GraphicsCapture] Initialized with NVFBC (" << m_captureWidth << "x" << m_captureHeight << ")" << std::endl;
            return true;
        } else {
            std::cout << "[GraphicsCapture] NVFBC initialization failed, falling back to stub" << std::endl;
            m_nvfbcCapture.reset();
        }
    }

    // Fallback to stub implementation
    m_useNVFBC = false;
    m_captureWidth = 1920;
    m_captureHeight = 1080;

    m_isInitialized = true;
    std::cout << "[GraphicsCapture] Initialized with stub implementation (" << m_captureWidth << "x" << m_captureHeight << ")" << std::endl;
    return true;
}

void GraphicsCapture::Shutdown() {
    StopCapture();

    if (m_nvfbcCapture) {
        m_nvfbcCapture->Shutdown();
        m_nvfbcCapture.reset();
    }

    m_captureItem = nullptr;
    m_framePool = nullptr;
    m_captureSession = nullptr;
    m_winrtDevice = nullptr;

    m_dxgiDevice = nullptr;
    m_d3dContext = nullptr;
    m_d3dDevice = nullptr;

    m_useNVFBC = false;
    m_isInitialized = false;
    std::cout << "[GraphicsCapture] Shutdown complete" << std::endl;
}

bool GraphicsCapture::StartCapture() {
    if (!m_isInitialized) {
        std::cerr << "[GraphicsCapture] Not initialized" << std::endl;
        return false;
    }

    if (m_useNVFBC && m_nvfbcCapture) {
        if (m_nvfbcCapture->StartCapture()) {
            m_isCapturing = true;
            std::cout << "[GraphicsCapture] NVFBC capture started" << std::endl;
            return true;
        } else {
            std::cerr << "[GraphicsCapture] Failed to start NVFBC capture" << std::endl;
            return false;
        }
    }

    // Fallback to stub implementation
    m_isCapturing = true;
    std::cout << "[GraphicsCapture] Capture started (stub implementation)" << std::endl;
    return true;
}

void GraphicsCapture::StopCapture() {
    if (m_isCapturing) {
        if (m_useNVFBC && m_nvfbcCapture) {
            m_nvfbcCapture->StopCapture();
        }
        m_isCapturing = false;
        std::cout << "[GraphicsCapture] Capture stopped" << std::endl;
    }
}

ID3D11Texture2D* GraphicsCapture::GetLatestFrame() {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    // Return dummy frame for now
    return nullptr;
}

void GraphicsCapture::SetFrameCallback(std::function<void(ID3D11Texture2D*)> callback) {
    m_frameCallback = callback;
}

bool GraphicsCapture::CreateD3DDevice() {
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &m_d3dDevice,
        nullptr,
        &m_d3dContext
    );

    if (FAILED(hr)) {
        std::cerr << "[GraphicsCapture] Failed to create D3D11 device" << std::endl;
        return false;
    }

    hr = m_d3dDevice.As(&m_dxgiDevice);
    if (FAILED(hr)) {
        std::cerr << "[GraphicsCapture] Failed to get DXGI device" << std::endl;
        return false;
    }

    return true;
}

bool GraphicsCapture::CreateCaptureSession(HWND targetWindow) {
    // Stub implementation - actual Graphics Capture API integration pending
    std::cout << "[GraphicsCapture] Capture session creation (stub)" << std::endl;
    return true;
}

bool GraphicsCapture::IsNVFBCAvailable() {
    return NVFBCCapture::IsNVFBCAvailable();
}

void GraphicsCapture::ConvertNVFBCFrameToD3D11(void* nvfbcData, unsigned int width, unsigned int height, unsigned int size) {
    if (!m_d3dDevice || !m_d3dContext) return;

    // Create or update D3D11 texture
    if (!m_latestFrame || m_captureWidth != width || m_captureHeight != height) {
        m_captureWidth = width;
        m_captureHeight = height;

        D3D11_TEXTURE2D_DESC textureDesc = {};
        textureDesc.Width = width;
        textureDesc.Height = height;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; // BGRA format from NVFBC
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Usage = D3D11_USAGE_DYNAMIC;
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        HRESULT hr = m_d3dDevice->CreateTexture2D(&textureDesc, nullptr, &m_latestFrame);
        if (FAILED(hr)) {
            std::cerr << "[GraphicsCapture] Failed to create D3D11 texture" << std::endl;
            return;
        }
    }

    // Copy NVFBC data to D3D11 texture
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    HRESULT hr = m_d3dContext->Map(m_latestFrame.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    if (SUCCEEDED(hr)) {
        // Copy row by row to handle potential stride differences
        const unsigned char* srcData = static_cast<const unsigned char*>(nvfbcData);
        unsigned char* dstData = static_cast<unsigned char*>(mappedResource.pData);

        unsigned int srcRowPitch = width * 4; // BGRA = 4 bytes per pixel
        for (unsigned int row = 0; row < height; ++row) {
            memcpy(dstData + row * mappedResource.RowPitch, srcData + row * srcRowPitch, srcRowPitch);
        }

        m_d3dContext->Unmap(m_latestFrame.Get(), 0);

        // Trigger frame callback if set
        if (m_frameCallback) {
            m_frameCallback(m_latestFrame.Get());
        }
    } else {
        std::cerr << "[GraphicsCapture] Failed to map D3D11 texture" << std::endl;
    }
}