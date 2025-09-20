#include "dda_capture.h"

#include <algorithm>
#include <iostream>

namespace {
constexpr UINT kFrameTimeoutMs = 1;  // Poll for new frames without blocking indefinitely
constexpr UINT kBytesPerPixel = 4;   // BGRA
}

DDACapture::DDACapture() = default;
DDACapture::~DDACapture() { Shutdown(); }

bool DDACapture::IsDDACaptureAvailable() {
    DDACapture capture;
    if (!capture.Initialize()) {
        return false;
    }

    if (!capture.StartCapture()) {
        return false;
    }

    capture.StopCapture();
    return true;
}

bool DDACapture::Initialize(HWND targetWindow) {
    if (m_isInitialized) {
        return true;
    }

    if (!InitializeDeviceAndOutput(targetWindow)) {
        std::cerr << "[DDACapture] Failed to initialize D3D11 device or output" << std::endl;
        return false;
    }

    if (!CreateDuplicationInterface()) {
        std::cerr << "[DDACapture] Failed to create duplication interface" << std::endl;
        return false;
    }

    m_isInitialized = true;
    ResetToFullScreen();
    return true;
}

void DDACapture::Shutdown() {
    StopCapture();

    std::lock_guard<std::mutex> lock(m_frameMutex);
    m_frameBuffer.reset();
    m_bufferSize = 0;
    m_captureWidth = 0;
    m_captureHeight = 0;

    ReleaseDuplication();
    m_stagingTexture.Reset();
    m_context.Reset();
    m_device.Reset();
    m_output.Reset();

    m_isInitialized = false;
    m_screenWidth = 0;
    m_screenHeight = 0;
    m_captureRegion = {0, 0, 0, 0};
}

bool DDACapture::StartCapture() {
    if (!m_isInitialized) {
        std::cerr << "[DDACapture] Cannot start capture before initialization" << std::endl;
        return false;
    }

    if (m_isCapturing.load()) {
        return true;
    }

    if (!m_duplication) {
        if (!CreateDuplicationInterface()) {
            return false;
        }
    }

    m_isCapturing = true;
    m_captureThread = std::thread(&DDACapture::CaptureThreadProc, this);
    return true;
}

void DDACapture::StopCapture() {
    if (!m_isCapturing.exchange(false)) {
        return;
    }

    if (m_captureThread.joinable()) {
        m_captureThread.join();
    }
}

bool DDACapture::GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size) {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    if (!m_frameBuffer || m_bufferSize == 0 || m_captureWidth == 0 || m_captureHeight == 0) {
        return false;
    }

    if (frameData) {
        *frameData = m_frameBuffer.get();
    }
    if (width) {
        *width = m_captureWidth;
    }
    if (height) {
        *height = m_captureHeight;
    }
    if (size) {
        *size = static_cast<unsigned int>(m_bufferSize);
    }

    return true;
}

void DDACapture::SetFrameCallback(std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback) {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    m_frameCallback = std::move(callback);
}

bool DDACapture::SetCaptureRegion(int x, int y, int width, int height) {
    if (!m_isInitialized || width <= 0 || height <= 0) {
        return false;
    }

    if (x < 0 || y < 0 || x + width > m_screenWidth || y + height > m_screenHeight) {
        return false;
    }

    RECT newRegion{ x, y, x + width, y + height };

    if (newRegion.left == m_captureRegion.left &&
        newRegion.top == m_captureRegion.top &&
        newRegion.right == m_captureRegion.right &&
        newRegion.bottom == m_captureRegion.bottom) {
        return true;
    }

    m_captureRegion = newRegion;
    m_captureWidth = static_cast<unsigned int>(width);
    m_captureHeight = static_cast<unsigned int>(height);
    return true;
}

void DDACapture::GetCaptureRegion(int* x, int* y, int* width, int* height) const {
    if (x) {
        *x = m_captureRegion.left;
    }
    if (y) {
        *y = m_captureRegion.top;
    }
    if (width) {
        *width = m_captureRegion.right - m_captureRegion.left;
    }
    if (height) {
        *height = m_captureRegion.bottom - m_captureRegion.top;
    }
}

void DDACapture::ResetToFullScreen() {
    if (!m_isInitialized) {
        return;
    }

    SetCaptureRegion(0, 0, m_screenWidth, m_screenHeight);
}

bool DDACapture::InitializeDeviceAndOutput(HWND targetWindow) {
    UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
    creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };

    D3D_FEATURE_LEVEL obtainedLevel = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        creationFlags,
        featureLevels,
        static_cast<UINT>(sizeof(featureLevels) / sizeof(featureLevels[0])),
        D3D11_SDK_VERSION,
        &m_device,
        &obtainedLevel,
        &m_context
    );

    if (FAILED(hr)) {
        std::cerr << "[DDACapture] D3D11CreateDevice failed: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    hr = m_device.As(&dxgiDevice);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] Failed to query IDXGIDevice: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] Failed to get adapter from device: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    Microsoft::WRL::ComPtr<IDXGIOutput> chosenOutput;
    HMONITOR targetMonitor = targetWindow ? MonitorFromWindow(targetWindow, MONITOR_DEFAULTTOPRIMARY) : nullptr;

    for (UINT outputIndex = 0; ; ++outputIndex) {
        Microsoft::WRL::ComPtr<IDXGIOutput> outputCandidate;
        hr = adapter->EnumOutputs(outputIndex, &outputCandidate);
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr)) {
            std::cerr << "[DDACapture] EnumOutputs failed: 0x" << std::hex << hr << std::dec << std::endl;
            return false;
        }

        DXGI_OUTPUT_DESC desc;
        if (SUCCEEDED(outputCandidate->GetDesc(&desc))) {
            if (!targetMonitor || desc.Monitor == targetMonitor) {
                chosenOutput = outputCandidate;
                break;
            }
        }
    }

    if (!chosenOutput) {
        hr = adapter->EnumOutputs(0, &chosenOutput);
        if (FAILED(hr)) {
            std::cerr << "[DDACapture] Failed to enumerate primary output: 0x" << std::hex << hr << std::dec << std::endl;
            return false;
        }
    }

    m_output = chosenOutput;
    if (!m_output) {
        return false;
    }

    hr = m_output->GetDesc(&m_outputDesc);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] Failed to query output description: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    m_screenWidth = m_outputDesc.DesktopCoordinates.right - m_outputDesc.DesktopCoordinates.left;
    m_screenHeight = m_outputDesc.DesktopCoordinates.bottom - m_outputDesc.DesktopCoordinates.top;

    if (m_screenWidth <= 0 || m_screenHeight <= 0) {
        std::cerr << "[DDACapture] Invalid output dimensions" << std::endl;
        return false;
    }

    return true;
}

bool DDACapture::CreateDuplicationInterface() {
    if (!m_output || !m_device) {
        return false;
    }

    ReleaseDuplication();

    Microsoft::WRL::ComPtr<IDXGIOutput1> output1;
    HRESULT hr = m_output.As(&output1);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] Failed to query IDXGIOutput1: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    hr = output1->DuplicateOutput(m_device.Get(), &m_duplication);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] DuplicateOutput failed: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    DXGI_OUTDUPL_DESC dupDesc{};
    m_duplication->GetDesc(&dupDesc);

    return EnsureStagingTexture(dupDesc.ModeDesc.Width, dupDesc.ModeDesc.Height, dupDesc.ModeDesc.Format);
}

void DDACapture::ReleaseDuplication() {
    m_duplication.Reset();
}

bool DDACapture::EnsureStagingTexture(UINT width, UINT height, DXGI_FORMAT format) {
    if (m_stagingTexture) {
        D3D11_TEXTURE2D_DESC currentDesc;
        m_stagingTexture->GetDesc(&currentDesc);
        if (currentDesc.Width == width && currentDesc.Height == height && currentDesc.Format == format) {
            return true;
        }
    }

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = m_device->CreateTexture2D(&desc, nullptr, &texture);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] Failed to create staging texture: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    m_stagingTexture = texture;
    m_frameDesc = desc;
    return true;
}

bool DDACapture::EnsureFrameBuffer(size_t requiredSize) {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    if (requiredSize <= m_bufferSize && m_frameBuffer) {
        return true;
    }

    try {
        auto newBuffer = std::make_unique<unsigned char[]>(requiredSize);
        m_frameBuffer = std::move(newBuffer);
        m_bufferSize = requiredSize;
        return true;
    } catch (const std::bad_alloc&) {
        std::cerr << "[DDACapture] Failed to allocate frame buffer" << std::endl;
        m_frameBuffer.reset();
        m_bufferSize = 0;
        return false;
    }
}

bool DDACapture::AcquireFrame() {
    if (!m_duplication || !m_context) {
        return false;
    }

    Microsoft::WRL::ComPtr<IDXGIResource> desktopResource;
    DXGI_OUTDUPL_FRAME_INFO frameInfo{};
    bool frameAcquired = false;

    HRESULT hr = m_duplication->AcquireNextFrame(kFrameTimeoutMs, &frameInfo, &desktopResource);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return true; // No new frame yet
    }

    if (hr == DXGI_ERROR_ACCESS_LOST) {
        CreateDuplicationInterface();
        return false;
    }

    if (FAILED(hr)) {
        std::cerr << "[DDACapture] AcquireNextFrame failed: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    frameAcquired = true;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> frameTexture;
    hr = desktopResource.As(&frameTexture);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] Failed to query frame texture: 0x" << std::hex << hr << std::dec << std::endl;
        if (frameAcquired) {
            m_duplication->ReleaseFrame();
        }
        return false;
    }

    D3D11_TEXTURE2D_DESC textureDesc{};
    frameTexture->GetDesc(&textureDesc);
    if (!EnsureStagingTexture(textureDesc.Width, textureDesc.Height, textureDesc.Format)) {
        if (frameAcquired) {
            m_duplication->ReleaseFrame();
        }
        return false;
    }

    m_context->CopyResource(m_stagingTexture.Get(), frameTexture.Get());

    hr = m_duplication->ReleaseFrame();
    frameAcquired = false;
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] ReleaseFrame failed: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = m_context->Map(m_stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] Failed to map staging texture: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    unsigned int captureWidth = m_captureRegion.right - m_captureRegion.left;
    unsigned int captureHeight = m_captureRegion.bottom - m_captureRegion.top;

    if (captureWidth == 0 || captureHeight == 0) {
        captureWidth = static_cast<unsigned int>(m_screenWidth);
        captureHeight = static_cast<unsigned int>(m_screenHeight);
    }

    size_t requiredSize = static_cast<size_t>(captureWidth) * captureHeight * kBytesPerPixel;
    if (!EnsureFrameBuffer(requiredSize)) {
        m_context->Unmap(m_stagingTexture.Get(), 0);
        return false;
    }

    std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback;
    void* callbackData = nullptr;
    unsigned int callbackWidth = captureWidth;
    unsigned int callbackHeight = captureHeight;
    unsigned int callbackSize = static_cast<unsigned int>(requiredSize);

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (!m_frameBuffer) {
            m_context->Unmap(m_stagingTexture.Get(), 0);
            return false;
        }

        unsigned char* dst = m_frameBuffer.get();
        const unsigned char* srcBase = static_cast<const unsigned char*>(mapped.pData);
        const UINT srcPitch = mapped.RowPitch;
        const UINT bytesPerRow = captureWidth * kBytesPerPixel;

        const UINT startX = static_cast<UINT>(m_captureRegion.left);
        const UINT startY = static_cast<UINT>(m_captureRegion.top);

        for (UINT row = 0; row < captureHeight; ++row) {
            const unsigned char* srcRow = srcBase + (startY + row) * srcPitch + startX * kBytesPerPixel;
            std::copy_n(srcRow, bytesPerRow, dst + row * bytesPerRow);
        }

        m_captureWidth = captureWidth;
        m_captureHeight = captureHeight;
        m_bufferSize = requiredSize;
        callback = m_frameCallback;
        callbackData = dst;
        callbackWidth = m_captureWidth;
        callbackHeight = m_captureHeight;
        callbackSize = static_cast<unsigned int>(m_bufferSize);
    }

    m_context->Unmap(m_stagingTexture.Get(), 0);

    if (callback) {
        callback(callbackData, callbackWidth, callbackHeight, callbackSize);
    }

    return true;
}

void DDACapture::CaptureThreadProc() {
    while (m_isCapturing.load()) {
        if (!AcquireFrame()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
}
