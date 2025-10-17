#include "dda_capture.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include "AppContext.h"

namespace {
// OPTIMIZATION: Short timeout (2ms) for 360Hz+ displays; aggressive backoff prevents busyspin
// Desktop Duplication returns immediately (<1ms) when new frame is available
// Backoff strategy handles prolonged no-frame periods to eliminate CPU waste
constexpr UINT kFrameTimeoutMs = 2;  // 2ms timeout supports up to 500Hz displays
constexpr UINT kBytesPerPixel = 4;   // BGRA
}

DDACapture::DDACapture()
    : m_lastProbeTime(std::chrono::steady_clock::now()) {
}

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

    // Cleanup CUDA interop
    if (m_cudaGraphicsResource) {
        cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
        m_cudaGraphicsResource = nullptr;
    }
    m_cudaMappedArray = nullptr;
    m_sharedTexture.Reset();
    m_cudaInteropEnabled = false;

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
    // Signal that CPU fallback is needed
    m_cpuFallbackNeeded.store(true, std::memory_order_release);

    // Hint CPU fallback; capture thread decides mode. Avoid immediate mode switch.
    (void)m_currentMode; // keep variable referenced to avoid warnings

    std::lock_guard<std::mutex> lock(m_frameMutex);

    // If CPU buffer is not valid, we can't provide frame data
    if (!m_cpuBufferValid || !m_frameBuffer || m_bufferSize == 0 || m_captureWidth == 0 || m_captureHeight == 0) {
        return false;
    }

    if (frameData) {
        *frameData = m_frameBuffer ? static_cast<void*>(m_frameBuffer->get()) : nullptr;
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

bool DDACapture::GetLatestFrameGPU(cudaArray_t* cudaArray, unsigned int* width, unsigned int* height) {
    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (!m_cudaInteropEnabled || !m_cudaMappedArray) {
        return false;
    }

    if (cudaArray) {
        *cudaArray = m_cudaMappedArray;
    }
    if (width) {
        *width = m_captureWidth;
    }
    if (height) {
        *height = m_captureHeight;
    }

    return true;
}

bool DDACapture::WaitForNewFrameSince(uint64_t minPresentQpc, uint32_t timeoutMs) {
    if (!m_isCapturing.load() || minPresentQpc == 0) {
        return true;
    }

    // Fast-path: already satisfied
    if (m_lastPresentQpc.load(std::memory_order_acquire) >= minPresentQpc) {
        return true;
    }

    std::unique_lock<std::mutex> lk(m_presentMutex);
    const auto timeout = std::chrono::milliseconds(timeoutMs);
    const auto ok = m_presentCv.wait_for(lk, timeout, [&]() {
        return !m_isCapturing.load() || m_lastPresentQpc.load(std::memory_order_acquire) >= minPresentQpc;
    });
    return ok && m_isCapturing.load();
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
    m_duplicationFormat = dupDesc.ModeDesc.Format;
    m_stagingTexture.Reset();
    m_frameDesc = {};

    return true;
}

void DDACapture::ReleaseDuplication() {
    m_duplication.Reset();
}

bool DDACapture::EnsureStagingTexture(UINT width, UINT height, DXGI_FORMAT format) {
    // Check if we need to recreate
    bool needRecreate = false;
    if (m_sharedTexture) {
        D3D11_TEXTURE2D_DESC currentDesc;
        m_sharedTexture->GetDesc(&currentDesc);
        if (currentDesc.Width != width || currentDesc.Height != height || currentDesc.Format != format) {
            needRecreate = true;
        }
    } else {
        needRecreate = true;
    }

    if (needRecreate) {
        // Cleanup old CUDA resource
        if (m_cudaGraphicsResource) {
            cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
            m_cudaGraphicsResource = nullptr;
        }
        m_cudaMappedArray = nullptr;
        m_sharedTexture.Reset();
        m_stagingTexture.Reset();
        m_cudaInteropEnabled = false;

        // Create shared texture for CUDA (D3D11_USAGE_DEFAULT)
        D3D11_TEXTURE2D_DESC sharedDesc{};
        sharedDesc.Width = width;
        sharedDesc.Height = height;
        sharedDesc.MipLevels = 1;
        sharedDesc.ArraySize = 1;
        sharedDesc.Format = format;
        sharedDesc.SampleDesc.Count = 1;
        sharedDesc.SampleDesc.Quality = 0;
        sharedDesc.Usage = D3D11_USAGE_DEFAULT;  // CUDA requires DEFAULT
        sharedDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        sharedDesc.CPUAccessFlags = 0;
        sharedDesc.MiscFlags = 0;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> sharedTex;
        HRESULT hr = m_device->CreateTexture2D(&sharedDesc, nullptr, &sharedTex);
        if (FAILED(hr)) {
            std::cerr << "[DDACapture] Failed to create shared texture: 0x" << std::hex << hr << std::dec << std::endl;
            return false;
        }

        m_sharedTexture = sharedTex;

        // Try to register with CUDA
        cudaError_t cudaErr = cudaGraphicsD3D11RegisterResource(
            &m_cudaGraphicsResource,
            m_sharedTexture.Get(),
            cudaGraphicsRegisterFlagsNone
        );

        if (cudaErr == cudaSuccess) {
            // Map immediately and keep mapped
            cudaErr = cudaGraphicsMapResources(1, &m_cudaGraphicsResource, 0);
            if (cudaErr == cudaSuccess) {
                cudaErr = cudaGraphicsSubResourceGetMappedArray(&m_cudaMappedArray, m_cudaGraphicsResource, 0, 0);
                if (cudaErr == cudaSuccess) {
                    m_cudaInteropEnabled = true;
                    std::cout << "[DDACapture] CUDA interop enabled successfully!" << std::endl;
                } else {
                    std::cerr << "[DDACapture] cudaGraphicsSubResourceGetMappedArray failed: "
                              << cudaGetErrorString(cudaErr) << std::endl;
                    cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0);
                    cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
                    m_cudaGraphicsResource = nullptr;
                }
            } else {
                std::cerr << "[DDACapture] cudaGraphicsMapResources failed: "
                          << cudaGetErrorString(cudaErr) << std::endl;
                cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
                m_cudaGraphicsResource = nullptr;
            }
        } else {
            std::cerr << "[DDACapture] CUDA interop not available: " << cudaGetErrorString(cudaErr)
                      << " - using CPU fallback" << std::endl;
        }

        // Create CPU staging texture for fallback only if needed now
        bool needCpuNow = (m_currentMode.load(std::memory_order_acquire) == CaptureMode::kCpuFallback) ||
                          m_cpuFallbackNeeded.load(std::memory_order_acquire);
        if (needCpuNow) {
            D3D11_TEXTURE2D_DESC stagingDesc{};
            stagingDesc.Width = width;
            stagingDesc.Height = height;
            stagingDesc.MipLevels = 1;
            stagingDesc.ArraySize = 1;
            stagingDesc.Format = format;
            stagingDesc.SampleDesc.Count = 1;
            stagingDesc.SampleDesc.Quality = 0;
            stagingDesc.Usage = D3D11_USAGE_STAGING;
            stagingDesc.BindFlags = 0;
            stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            stagingDesc.MiscFlags = 0;

            Microsoft::WRL::ComPtr<ID3D11Texture2D> stagingTex;
            hr = m_device->CreateTexture2D(&stagingDesc, nullptr, &stagingTex);
            if (FAILED(hr)) {
                std::cerr << "[DDACapture] Failed to create CPU staging texture: 0x" << std::hex << hr << std::dec << std::endl;
                return false;
            }

            m_stagingTexture = stagingTex;
        }
        m_frameDesc = sharedDesc;
    }

    return true;
}

bool DDACapture::EnsureFrameBuffer(size_t requiredSize) {
    if (requiredSize <= m_bufferSize && m_frameBuffer) {
        return true;
    }

    try {
        // Prefer write-combined for CPU write → GPU read pattern
        // Use Portable to survive context changes
        auto newBuffer = std::make_unique<CudaPinnedMemory<unsigned char>>(
            requiredSize, cudaHostAllocWriteCombined | cudaHostAllocPortable);
        m_frameBuffer = std::move(newBuffer);
        m_bufferSize = requiredSize;
        return true;
    } catch (const std::exception&) {
        std::cerr << "[DDACapture] Failed to allocate pinned frame buffer" << std::endl;
        m_frameBuffer.reset();
        m_bufferSize = 0;
        return false;
    }
}

DDACapture::FrameAcquireResult DDACapture::AcquireFrame() {
    if (!m_duplication || !m_context) {
        return FrameAcquireResult::kError;
    }

    Microsoft::WRL::ComPtr<IDXGIResource> desktopResource;
    DXGI_OUTDUPL_FRAME_INFO frameInfo{};
    bool frameAcquired = false;

    HRESULT hr = m_duplication->AcquireNextFrame(kFrameTimeoutMs, &frameInfo, &desktopResource);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return FrameAcquireResult::kNoFrame;
    }

    if (hr == DXGI_ERROR_ACCESS_LOST) {
        // Desktop duplication interface lost - typically due to mode change, desktop switch, etc.
        m_accessLostCount.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "[DDACapture] DXGI_ERROR_ACCESS_LOST - recreating duplication interface" << std::endl;

        if (CreateDuplicationInterface()) {
            // Successfully recreated - immediately retry acquisition
            hr = m_duplication->AcquireNextFrame(kFrameTimeoutMs, &frameInfo, &desktopResource);
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                return FrameAcquireResult::kNoFrame;
            }
            if (SUCCEEDED(hr)) {
                frameAcquired = true;
                // Continue with normal frame processing below
            } else {
                std::cerr << "[DDACapture] Retry after ACCESS_LOST failed: 0x" << std::hex << hr << std::dec << std::endl;
                return FrameAcquireResult::kError;
            }
        } else {
            std::cerr << "[DDACapture] Failed to recreate duplication interface after ACCESS_LOST" << std::endl;
            return FrameAcquireResult::kError;
        }
    }

    if (FAILED(hr)) {
        m_captureErrorCount.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "[DDACapture] AcquireNextFrame failed: 0x" << std::hex << hr << std::dec << std::endl;
        return FrameAcquireResult::kError;
    }

    if (!frameAcquired) {
        frameAcquired = true;
    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> frameTexture;
    hr = desktopResource.As(&frameTexture);
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] Failed to query frame texture: 0x" << std::hex << hr << std::dec << std::endl;
        if (frameAcquired) {
            m_duplication->ReleaseFrame();
        }
        return FrameAcquireResult::kError;
    }

    D3D11_TEXTURE2D_DESC textureDesc{};
    frameTexture->GetDesc(&textureDesc);

    UINT captureWidth = m_captureWidth;
    UINT captureHeight = m_captureHeight;
    UINT startX = static_cast<UINT>(m_captureRegion.left);
    UINT startY = static_cast<UINT>(m_captureRegion.top);

    if (captureWidth == 0 || captureHeight == 0 ||
        startX + captureWidth > textureDesc.Width ||
        startY + captureHeight > textureDesc.Height) {
        captureWidth = textureDesc.Width;
        captureHeight = textureDesc.Height;
        startX = 0;
        startY = 0;
    }

    DXGI_FORMAT captureFormat = textureDesc.Format != DXGI_FORMAT_UNKNOWN
        ? textureDesc.Format
        : m_duplicationFormat;

    if (!EnsureStagingTexture(captureWidth, captureHeight, captureFormat)) {
        if (frameAcquired) {
            m_duplication->ReleaseFrame();
        }
        return FrameAcquireResult::kError;
    }

    D3D11_BOX captureBox{};
    captureBox.left = startX;
    captureBox.top = startY;
    captureBox.front = 0;
    captureBox.right = startX + captureWidth;
    captureBox.bottom = startY + captureHeight;
    captureBox.back = 1;

    const bool useBox = !(startX == 0 && startY == 0 &&
                          captureWidth == textureDesc.Width &&
                          captureHeight == textureDesc.Height);

    // Copy to shared texture for CUDA interop if enabled
    if (m_cudaInteropEnabled && m_sharedTexture) {
        if (useBox) {
            m_context->CopySubresourceRegion(
                m_sharedTexture.Get(),
                0,
                0,
                0,
                0,
                frameTexture.Get(),
                0,
                &captureBox
            );
        } else {
            m_context->CopyResource(m_sharedTexture.Get(), frameTexture.Get());
        }
    }

    // Only copy to staging texture if CPU fallback is truly needed
    // Skip redundant CPU path when GPU-direct is active and working
    const bool needCpuCopy = (!m_cudaInteropEnabled) && m_cpuFallbackNeeded.load(std::memory_order_acquire);

    hr = m_duplication->ReleaseFrame();
    frameAcquired = false;
    if (FAILED(hr)) {
        std::cerr << "[DDACapture] ReleaseFrame failed: 0x" << std::hex << hr << std::dec << std::endl;
        return FrameAcquireResult::kError;
    }

    std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback;
    void* callbackData = nullptr;
    unsigned int callbackWidth = captureWidth;
    unsigned int callbackHeight = captureHeight;
    size_t requiredSize = static_cast<size_t>(captureWidth) * captureHeight * kBytesPerPixel;
    unsigned int callbackSize = static_cast<unsigned int>(requiredSize);

    // Only perform CPU staging copy and map if needed
    if (needCpuCopy) {
        // Need to re-acquire frame for CPU path (frameTexture already released)
        // Instead, copy from sharedTexture to staging
        if (m_sharedTexture) {
            m_context->CopyResource(m_stagingTexture.Get(), m_sharedTexture.Get());
        }

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = m_context->Map(m_stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            std::cerr << "[DDACapture] Failed to map staging texture: 0x" << std::hex << hr << std::dec << std::endl;
            m_cpuBufferValid = false;
            // Don't fail - GPU path may still work
        } else {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            if (!EnsureFrameBuffer(requiredSize)) {
                m_context->Unmap(m_stagingTexture.Get(), 0);
                m_cpuBufferValid = false;
            } else if (m_frameBuffer) {
                unsigned char* dst = m_frameBuffer->get();
                const unsigned char* srcBase = static_cast<const unsigned char*>(mapped.pData);
                const UINT srcPitch = mapped.RowPitch;
                const UINT bytesPerRow = captureWidth * kBytesPerPixel;

                if (srcPitch == bytesPerRow) {
                    std::memcpy(dst, srcBase, bytesPerRow * captureHeight);
                } else {
                    for (UINT row = 0; row < captureHeight; ++row) {
                        const unsigned char* srcRow = srcBase + row * srcPitch;
                        std::memcpy(dst + row * bytesPerRow, srcRow, bytesPerRow);
                    }
                }

                m_captureWidth = captureWidth;
                m_captureHeight = captureHeight;
                m_bufferSize = requiredSize;
                m_cpuBufferValid = true;
                callback = m_frameCallback;
                callbackData = dst;
                callbackWidth = m_captureWidth;
                callbackHeight = m_captureHeight;
                callbackSize = static_cast<unsigned int>(m_bufferSize);
                m_context->Unmap(m_stagingTexture.Get(), 0);
            } else {
                m_context->Unmap(m_stagingTexture.Get(), 0);
                m_cpuBufferValid = false;
            }
        }
    } else {
        // GPU-direct path - just update dimensions
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_captureWidth = captureWidth;
        m_captureHeight = captureHeight;
        m_cpuBufferValid = false;  // CPU buffer not updated
    }

    // Update present time/frame counter and notify after CPU buffer is valid
    {
        std::lock_guard<std::mutex> lk(m_presentMutex);
        m_lastPresentQpc.store(frameInfo.LastPresentTime.QuadPart, std::memory_order_release);
        m_frameCounter.fetch_add(1, std::memory_order_acq_rel);
    }
    m_presentCv.notify_all();

    if (callback) {
        callback(callbackData, callbackWidth, callbackHeight, callbackSize);
    }

    return FrameAcquireResult::kFrameCaptured;
}

void DDACapture::TransitionToCaptureMode(CaptureMode newMode) {
    CaptureMode oldMode = m_currentMode.exchange(newMode, std::memory_order_release);
    if (oldMode != newMode) {
        const char* modeNames[] = {"GpuDirect", "CpuFallback", "Probing"};
        std::cout << "[DDACapture] Mode transition: " << modeNames[static_cast<int>(oldMode)]
                  << " -> " << modeNames[static_cast<int>(newMode)] << std::endl;

        if (newMode == CaptureMode::kGpuDirect) {
            m_consecutiveGpuFailures = 0;
        } else if (newMode == CaptureMode::kCpuFallback) {
            m_cpuFallbackNeeded.store(true, std::memory_order_release);
        }
    }
}

bool DDACapture::ProbeGpuDirectCapability() {
    // Only probe if CUDA interop is supposedly available
    if (!m_cudaInteropEnabled || !m_cudaMappedArray) {
        return false;
    }

    // Simple test: verify CUDA resource is still valid
    cudaError_t err = cudaGetLastError();  // Clear previous errors

    // Try to query the array properties
    cudaChannelFormatDesc desc;
    cudaExtent extent;
    unsigned int flags;
    err = cudaArrayGetInfo(&desc, &extent, &flags, m_cudaMappedArray);

    return (err == cudaSuccess);
}

DDACapture::FrameAcquireResult DDACapture::AcquireFrameGpuDirect() {
    // GPU-only path - no CPU staging or mapping
    if (!m_duplication || !m_context) {
        return FrameAcquireResult::kError;
    }

    Microsoft::WRL::ComPtr<IDXGIResource> desktopResource;
    DXGI_OUTDUPL_FRAME_INFO frameInfo{};
    HRESULT hr = m_duplication->AcquireNextFrame(kFrameTimeoutMs, &frameInfo, &desktopResource);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return FrameAcquireResult::kNoFrame;
    }

    if (hr == DXGI_ERROR_ACCESS_LOST) {
        m_accessLostCount.fetch_add(1, std::memory_order_relaxed);
        if (CreateDuplicationInterface()) {
            hr = m_duplication->AcquireNextFrame(kFrameTimeoutMs, &frameInfo, &desktopResource);
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                return FrameAcquireResult::kNoFrame;
            }
        } else {
            return FrameAcquireResult::kError;
        }
    }

    if (FAILED(hr)) {
        m_captureErrorCount.fetch_add(1, std::memory_order_relaxed);
        m_consecutiveGpuFailures++;
        if (m_consecutiveGpuFailures >= kMaxGpuFailuresBeforeFallback) {
            TransitionToCaptureMode(CaptureMode::kCpuFallback);
        }
        return FrameAcquireResult::kError;
    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> frameTexture;
    hr = desktopResource.As(&frameTexture);
    if (FAILED(hr)) {
        m_duplication->ReleaseFrame();
        m_consecutiveGpuFailures++;
        return FrameAcquireResult::kError;
    }

    D3D11_TEXTURE2D_DESC textureDesc{};
    frameTexture->GetDesc(&textureDesc);

    UINT captureWidth = m_captureWidth;
    UINT captureHeight = m_captureHeight;
    UINT startX = static_cast<UINT>(m_captureRegion.left);
    UINT startY = static_cast<UINT>(m_captureRegion.top);

    if (captureWidth == 0 || captureHeight == 0 ||
        startX + captureWidth > textureDesc.Width ||
        startY + captureHeight > textureDesc.Height) {
        captureWidth = textureDesc.Width;
        captureHeight = textureDesc.Height;
        startX = 0;
        startY = 0;
    }

    DXGI_FORMAT captureFormat = textureDesc.Format != DXGI_FORMAT_UNKNOWN
        ? textureDesc.Format : m_duplicationFormat;

    if (!EnsureStagingTexture(captureWidth, captureHeight, captureFormat)) {
        m_duplication->ReleaseFrame();
        return FrameAcquireResult::kError;
    }

    D3D11_BOX captureBox{};
    captureBox.left = startX;
    captureBox.top = startY;
    captureBox.front = 0;
    captureBox.right = startX + captureWidth;
    captureBox.bottom = startY + captureHeight;
    captureBox.back = 1;

    const bool useBox = !(startX == 0 && startY == 0 &&
                          captureWidth == textureDesc.Width &&
                          captureHeight == textureDesc.Height);

    // GPU direct path: copy directly to shared texture
    if (useBox) {
        m_context->CopySubresourceRegion(m_sharedTexture.Get(), 0, 0, 0, 0,
                                         frameTexture.Get(), 0, &captureBox);
    } else {
        m_context->CopyResource(m_sharedTexture.Get(), frameTexture.Get());
    }

    m_duplication->ReleaseFrame();

    // Update dimensions without CPU buffer
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_captureWidth = captureWidth;
        m_captureHeight = captureHeight;
        m_cpuBufferValid = false;
    }

    // Update frame timing
    {
        std::lock_guard<std::mutex> lk(m_presentMutex);
        m_lastPresentQpc.store(frameInfo.LastPresentTime.QuadPart, std::memory_order_release);
        m_frameCounter.fetch_add(1, std::memory_order_acq_rel);
    }
    m_presentCv.notify_all();

    // Reset failure counter on success
    m_consecutiveGpuFailures = 0;
    return FrameAcquireResult::kFrameCaptured;
}

DDACapture::FrameAcquireResult DDACapture::AcquireFrameCpuFallback() {
    // Full CPU path with staging texture and memory copy
    // This is the same as the original AcquireFrame, but always does CPU copy
    if (!m_duplication || !m_context) {
        return FrameAcquireResult::kError;
    }

    Microsoft::WRL::ComPtr<IDXGIResource> desktopResource;
    DXGI_OUTDUPL_FRAME_INFO frameInfo{};
    HRESULT hr = m_duplication->AcquireNextFrame(kFrameTimeoutMs, &frameInfo, &desktopResource);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return FrameAcquireResult::kNoFrame;
    }

    if (hr == DXGI_ERROR_ACCESS_LOST) {
        m_accessLostCount.fetch_add(1, std::memory_order_relaxed);
        if (CreateDuplicationInterface()) {
            hr = m_duplication->AcquireNextFrame(kFrameTimeoutMs, &frameInfo, &desktopResource);
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                return FrameAcquireResult::kNoFrame;
            }
        } else {
            return FrameAcquireResult::kError;
        }
    }

    if (FAILED(hr)) {
        m_captureErrorCount.fetch_add(1, std::memory_order_relaxed);
        return FrameAcquireResult::kError;
    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> frameTexture;
    hr = desktopResource.As(&frameTexture);
    if (FAILED(hr)) {
        m_duplication->ReleaseFrame();
        return FrameAcquireResult::kError;
    }

    D3D11_TEXTURE2D_DESC textureDesc{};
    frameTexture->GetDesc(&textureDesc);

    UINT captureWidth = m_captureWidth;
    UINT captureHeight = m_captureHeight;
    UINT startX = static_cast<UINT>(m_captureRegion.left);
    UINT startY = static_cast<UINT>(m_captureRegion.top);

    if (captureWidth == 0 || captureHeight == 0 ||
        startX + captureWidth > textureDesc.Width ||
        startY + captureHeight > textureDesc.Height) {
        captureWidth = textureDesc.Width;
        captureHeight = textureDesc.Height;
        startX = 0;
        startY = 0;
    }

    DXGI_FORMAT captureFormat = textureDesc.Format != DXGI_FORMAT_UNKNOWN
        ? textureDesc.Format : m_duplicationFormat;

    if (!EnsureStagingTexture(captureWidth, captureHeight, captureFormat)) {
        m_duplication->ReleaseFrame();
        return FrameAcquireResult::kError;
    }

    D3D11_BOX captureBox{};
    captureBox.left = startX;
    captureBox.top = startY;
    captureBox.front = 0;
    captureBox.right = startX + captureWidth;
    captureBox.bottom = startY + captureHeight;
    captureBox.back = 1;

    const bool useBox = !(startX == 0 && startY == 0 &&
                          captureWidth == textureDesc.Width &&
                          captureHeight == textureDesc.Height);

    // Copy to staging texture for CPU access
    if (m_sharedTexture) {
        if (useBox) {
            m_context->CopySubresourceRegion(m_sharedTexture.Get(), 0, 0, 0, 0,
                                             frameTexture.Get(), 0, &captureBox);
        } else {
            m_context->CopyResource(m_sharedTexture.Get(), frameTexture.Get());
        }
        m_context->CopyResource(m_stagingTexture.Get(), m_sharedTexture.Get());
    }

    m_duplication->ReleaseFrame();

    // Map and copy to CPU buffer
    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = m_context->Map(m_stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_cpuBufferValid = false;
        return FrameAcquireResult::kError;
    }

    size_t requiredSize = static_cast<size_t>(captureWidth) * captureHeight * kBytesPerPixel;
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (!EnsureFrameBuffer(requiredSize)) {
            m_context->Unmap(m_stagingTexture.Get(), 0);
            m_cpuBufferValid = false;
            return FrameAcquireResult::kError;
        }

        std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback;
        void* callbackData = nullptr;

        if (m_frameBuffer) {
            unsigned char* dst = m_frameBuffer->get();
            const unsigned char* srcBase = static_cast<const unsigned char*>(mapped.pData);
            const UINT srcPitch = mapped.RowPitch;
            const UINT bytesPerRow = captureWidth * kBytesPerPixel;

            if (srcPitch == bytesPerRow) {
                std::memcpy(dst, srcBase, bytesPerRow * captureHeight);
            } else {
                for (UINT row = 0; row < captureHeight; ++row) {
                    const unsigned char* srcRow = srcBase + row * srcPitch;
                    std::memcpy(dst + row * bytesPerRow, srcRow, bytesPerRow);
                }
            }

            m_captureWidth = captureWidth;
            m_captureHeight = captureHeight;
            m_bufferSize = requiredSize;
            m_cpuBufferValid = true;
            callback = m_frameCallback;
            callbackData = dst;
        }

        m_context->Unmap(m_stagingTexture.Get(), 0);

        // Update frame timing
        {
            std::lock_guard<std::mutex> lk(m_presentMutex);
            m_lastPresentQpc.store(frameInfo.LastPresentTime.QuadPart, std::memory_order_release);
            m_frameCounter.fetch_add(1, std::memory_order_acq_rel);
        }
        m_presentCv.notify_all();

        if (callback && callbackData) {
            callback(callbackData, m_captureWidth, m_captureHeight, static_cast<unsigned int>(m_bufferSize));
        }
    }

    return FrameAcquireResult::kFrameCaptured;
}

void DDACapture::CaptureThreadProc() {
    // Set thread to high priority for low-latency capture
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

    while (m_isCapturing.load()) {
        // Check if we should probe for GPU direct capability
        CaptureMode currentMode = m_currentMode.load(std::memory_order_acquire);

        if (currentMode == CaptureMode::kProbing) {
            bool gpuAvailable = ProbeGpuDirectCapability();
            if (gpuAvailable) {
                TransitionToCaptureMode(CaptureMode::kGpuDirect);
            } else {
                TransitionToCaptureMode(CaptureMode::kCpuFallback);
            }
            currentMode = m_currentMode.load(std::memory_order_acquire);
        }

        // Periodically re-probe if in CPU fallback mode
        if (currentMode == CaptureMode::kCpuFallback &&
            !m_cpuFallbackNeeded.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            if (now - m_lastProbeTime >= kProbeInterval) {
                m_lastProbeTime = now;
                if (ProbeGpuDirectCapability()) {
                    TransitionToCaptureMode(CaptureMode::kGpuDirect);
                    currentMode = CaptureMode::kGpuDirect;
                }
            }
        }

        // Select acquisition path based on current mode
        FrameAcquireResult result;
        if (currentMode == CaptureMode::kGpuDirect) {
            result = AcquireFrameGpuDirect();
        } else {
            result = AcquireFrameCpuFallback();
        }

        // Adaptive backoff based on result
        if (result == FrameAcquireResult::kFrameCaptured) {
            // Success - reset backoff state
            m_consecutiveNoFrames = 0;
            m_currentBackoffLevel = BackoffLevel::kNone;
        }
        else if (result == FrameAcquireResult::kError) {
            // Error - immediate medium backoff to avoid error spam
            m_consecutiveNoFrames = 0;
            m_currentBackoffLevel = BackoffLevel::kMedium;
            Sleep(1);  // 1ms on error
        }
        else if (result == FrameAcquireResult::kNoFrame) {
            // No frame available - adaptive backoff
            m_consecutiveNoFrames++;

            // OPTIMIZATION: Ultra-aggressive backoff to eliminate busyspin
            // Transition to thread yield/sleep very quickly after initial spin attempts
            BackoffLevel newLevel;
            if (m_consecutiveNoFrames <= 1) {
                newLevel = BackoffLevel::kNone;        // Single immediate retry for 360Hz+
            } else if (m_consecutiveNoFrames <= 3) {
                newLevel = BackoffLevel::kMinimal;     // SwitchToThread (240Hz)
            } else if (m_consecutiveNoFrames <= 6) {
                newLevel = BackoffLevel::kShort;       // Sleep(0) (144Hz)
            } else if (m_consecutiveNoFrames <= 12) {
                newLevel = BackoffLevel::kMedium;      // Sleep(1ms) (60Hz/idle)
            } else {
                newLevel = BackoffLevel::kLong;        // Sleep(2ms) (minimized)
            }

            // Apply backoff only if level increased (hysteresis)
            if (static_cast<int>(newLevel) > static_cast<int>(m_currentBackoffLevel)) {
                m_currentBackoffLevel = newLevel;
            }

            // Execute backoff strategy
            switch (m_currentBackoffLevel) {
                case BackoffLevel::kNone:
                    // No backoff - immediate retry for high refresh rate (240Hz+)
                    // AcquireNextFrame already waited kFrameTimeoutMs (1ms)
                    break;

                case BackoffLevel::kMinimal:
                    // Yield to other threads without sleeping
                    // Windows: gives up remainder of time slice but stays scheduled
                    SwitchToThread();
                    break;

                case BackoffLevel::kShort:
                    // Yield and possibly context switch
                    // Sleep(0) only yields to equal/higher priority threads
                    Sleep(0);
                    break;

                case BackoffLevel::kMedium:
                    // Definite context switch - good for 60-144Hz displays
                    Sleep(1);  // Actual sleep time ~1-2ms on Windows
                    break;

                case BackoffLevel::kLong:
                    // Longer backoff for sustained no-frame periods (idle/minimized)
                    Sleep(2);  // Actual sleep time ~2-3ms
                    break;
            }
        }
    }
}
