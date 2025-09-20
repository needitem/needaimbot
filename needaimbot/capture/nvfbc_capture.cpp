#include "nvfbc_capture.h"
#include <iostream>
#include <thread>

#define NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER 1
#define NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER 1
#define NVFBC_TOBUFFER_GRAB_FRAME_PARAMS_VER 1

NVFBCCapture::NVFBCCapture()
    : m_nvfbcLib(nullptr)
    , m_pfnNvFBCCreate(nullptr)
    , m_pfnNvFBCGetStatus(nullptr)
    , m_captureSession(nullptr)
    , m_bufferSize(0)
    , m_isInitialized(false)
    , m_isCapturing(false)
    , m_captureWidth(0)
    , m_captureHeight(0)
    , m_screenWidth(0)
    , m_screenHeight(0)
    , m_useCustomRegion(false)
    , m_shouldStop(false)
{
    memset(&m_nvfbcAPI, 0, sizeof(m_nvfbcAPI));
    memset(&m_captureRegion, 0, sizeof(m_captureRegion));
}

NVFBCCapture::~NVFBCCapture() {
    Shutdown();
}

bool NVFBCCapture::IsNVFBCAvailable() {
    HMODULE hMod = LoadLibraryA("NvFBC64.dll");
    if (!hMod) {
        hMod = LoadLibraryA("NvFBC.dll");
    }

    if (hMod) {
        FreeLibrary(hMod);
        return true;
    }
    return false;
}

bool NVFBCCapture::Initialize(HWND targetWindow) {
    if (m_isInitialized) {
        std::cout << "[NVFBCCapture] Already initialized" << std::endl;
        return true;
    }

    if (!LoadNVFBCLibrary()) {
        std::cerr << "[NVFBCCapture] Failed to load NVFBC library" << std::endl;
        return false;
    }

    if (!CreateCaptureSession(targetWindow)) {
        std::cerr << "[NVFBCCapture] Failed to create capture session" << std::endl;
        UnloadNVFBCLibrary();
        return false;
    }

    m_isInitialized = true;
    std::cout << "[NVFBCCapture] Initialized successfully" << std::endl;
    return true;
}

void NVFBCCapture::Shutdown() {
    StopCapture();
    DestroyCaptureSession();
    UnloadNVFBCLibrary();
    m_isInitialized = false;
    std::cout << "[NVFBCCapture] Shutdown complete" << std::endl;
}

bool NVFBCCapture::StartCapture() {
    if (!m_isInitialized) {
        std::cerr << "[NVFBCCapture] Not initialized" << std::endl;
        return false;
    }

    if (m_isCapturing) {
        std::cout << "[NVFBCCapture] Already capturing" << std::endl;
        return true;
    }

    m_shouldStop = false;
    m_isCapturing = true;
    m_captureThread = std::thread(&NVFBCCapture::CaptureThreadProc, this);

    std::cout << "[NVFBCCapture] Capture started" << std::endl;
    return true;
}

void NVFBCCapture::StopCapture() {
    if (!m_isCapturing) return;

    m_shouldStop = true;
    if (m_captureThread.joinable()) {
        m_captureThread.join();
    }
    m_isCapturing = false;
    std::cout << "[NVFBCCapture] Capture stopped" << std::endl;
}

bool NVFBCCapture::GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size) {
    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (!m_frameBuffer || m_bufferSize == 0) {
        return false;
    }

    *frameData = m_frameBuffer.get();
    *width = m_captureWidth;
    *height = m_captureHeight;
    *size = m_bufferSize;
    return true;
}

void NVFBCCapture::SetFrameCallback(std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback) {
    m_frameCallback = callback;
}

bool NVFBCCapture::LoadNVFBCLibrary() {
    // Try loading 64-bit version first
    m_nvfbcLib = LoadLibraryA("NvFBC64.dll");
    if (!m_nvfbcLib) {
        // Fall back to 32-bit version
        m_nvfbcLib = LoadLibraryA("NvFBC.dll");
    }

    if (!m_nvfbcLib) {
        std::cerr << "[NVFBCCapture] Failed to load NVFBC library" << std::endl;
        return false;
    }

    // Get the main NVFBC creation function
    m_pfnNvFBCCreate = (PFN_NvFBC_Create)GetProcAddress(m_nvfbcLib, "NvFBC_Create");
    m_pfnNvFBCGetStatus = (PFN_NvFBC_GetStatus)GetProcAddress(m_nvfbcLib, "NvFBC_GetStatus");

    if (!m_pfnNvFBCCreate) {
        std::cerr << "[NVFBCCapture] Failed to get NvFBC_Create function pointer" << std::endl;
        UnloadNVFBCLibrary();
        return false;
    }

    // Initialize the API function list
    m_nvfbcAPI.version = 1;  // NVFBC API version
    m_nvfbcAPI.reserved = 0;

    // Call NvFBC_Create to fill the function pointers
    NVFBC_RESULT result = m_pfnNvFBCCreate(&m_nvfbcAPI);
    if (result != NVFBC_SUCCESS) {
        std::cerr << "[NVFBCCapture] NvFBC_Create failed with error: " << result << std::endl;
        UnloadNVFBCLibrary();
        return false;
    }

    std::cout << "[NVFBCCapture] NVFBC library loaded successfully" << std::endl;
    return true;
}

void NVFBCCapture::UnloadNVFBCLibrary() {
    if (m_nvfbcLib) {
        FreeLibrary(m_nvfbcLib);
        m_nvfbcLib = nullptr;
    }
    m_pfnNvFBCCreate = nullptr;
    m_pfnNvFBCGetStatus = nullptr;
    memset(&m_nvfbcAPI, 0, sizeof(m_nvfbcAPI));
}

bool NVFBCCapture::CreateCaptureSession(HWND targetWindow) {
    // Get full screen dimensions
    if (targetWindow) {
        RECT rect;
        GetClientRect(targetWindow, &rect);
        m_screenWidth = rect.right - rect.left;
        m_screenHeight = rect.bottom - rect.top;
    } else {
        m_screenWidth = GetSystemMetrics(SM_CXSCREEN);
        m_screenHeight = GetSystemMetrics(SM_CYSCREEN);
    }

    // Initialize capture region to full screen by default
    if (!m_useCustomRegion) {
        m_captureRegion.left = 0;
        m_captureRegion.top = 0;
        m_captureRegion.right = m_screenWidth;
        m_captureRegion.bottom = m_screenHeight;
    }

    // Set capture dimensions based on region
    m_captureWidth = m_captureRegion.right - m_captureRegion.left;
    m_captureHeight = m_captureRegion.bottom - m_captureRegion.top;

    // Validate capture region
    if (m_captureWidth <= 0 || m_captureHeight <= 0 ||
        m_captureRegion.right > m_screenWidth || m_captureRegion.bottom > m_screenHeight) {
        std::cerr << "[NVFBCCapture] Invalid capture region: "
                  << m_captureRegion.left << "," << m_captureRegion.top
                  << " to " << m_captureRegion.right << "," << m_captureRegion.bottom << std::endl;
        return false;
    }

    // Allocate frame buffer (BGRA = 4 bytes per pixel)
    m_bufferSize = m_captureWidth * m_captureHeight * 4;
    m_frameBuffer = std::make_unique<unsigned char[]>(m_bufferSize);

    if (m_useCustomRegion) {
        std::cout << "[NVFBCCapture] Partial capture setup completed" << std::endl;
        std::cout << "  - Screen: " << m_screenWidth << "x" << m_screenHeight << std::endl;
        std::cout << "  - Region: (" << m_captureRegion.left << "," << m_captureRegion.top
                  << ") to (" << m_captureRegion.right << "," << m_captureRegion.bottom << ")" << std::endl;
        std::cout << "  - Capture Size: " << m_captureWidth << "x" << m_captureHeight << std::endl;
    } else {
        std::cout << "[NVFBCCapture] Full screen capture setup completed. Resolution: " << m_captureWidth << "x" << m_captureHeight << std::endl;
    }

    return true;
}

void NVFBCCapture::DestroyCaptureSession() {
    // Simple cleanup
    m_captureSession = nullptr;
    m_frameBuffer.reset();
    m_bufferSize = 0;
}

void NVFBCCapture::CaptureThreadProc() {
    std::cout << "[NVFBCCapture] Capture thread started (demo mode)" << std::endl;

    while (!m_shouldStop) {
        // For demonstration, just simulate frame capture
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS

        std::lock_guard<std::mutex> lock(m_frameMutex);

        // Fill buffer with test pattern (optional)
        if (m_frameBuffer && m_bufferSize > 0) {
            // Simple test pattern - alternating colors
            static unsigned char colorValue = 0;
            memset(m_frameBuffer.get(), colorValue++, m_bufferSize);

            // Call frame callback if set
            if (m_frameCallback) {
                m_frameCallback(m_frameBuffer.get(), m_captureWidth, m_captureHeight, m_bufferSize);
            }
        }
    }

    std::cout << "[NVFBCCapture] Capture thread stopped" << std::endl;
}

bool NVFBCCapture::SetCaptureRegion(int x, int y, int width, int height) {
    // Validate input parameters
    if (x < 0 || y < 0 || width <= 0 || height <= 0) {
        std::cerr << "[NVFBCCapture] Invalid capture region parameters" << std::endl;
        return false;
    }

    if (x + width > m_screenWidth || y + height > m_screenHeight) {
        std::cerr << "[NVFBCCapture] Capture region exceeds screen bounds" << std::endl;
        return false;
    }

    m_captureRegion.left = x;
    m_captureRegion.top = y;
    m_captureRegion.right = x + width;
    m_captureRegion.bottom = y + height;
    m_useCustomRegion = true;

    std::cout << "[NVFBCCapture] Capture region set to: (" << x << "," << y
              << ") size " << width << "x" << height << std::endl;

    // If already initialized, update capture session
    if (m_isInitialized) {
        bool wasCapturing = m_isCapturing;
        if (wasCapturing) {
            StopCapture();
        }

        // Update capture dimensions and buffer
        m_captureWidth = width;
        m_captureHeight = height;
        m_bufferSize = width * height * 4;
        m_frameBuffer = std::make_unique<unsigned char[]>(m_bufferSize);

        if (wasCapturing) {
            StartCapture();
        }
    }

    return true;
}

void NVFBCCapture::GetCaptureRegion(int* x, int* y, int* width, int* height) const {
    if (x) *x = m_captureRegion.left;
    if (y) *y = m_captureRegion.top;
    if (width) *width = m_captureWidth;
    if (height) *height = m_captureHeight;
}

void NVFBCCapture::ResetToFullScreen() {
    m_useCustomRegion = false;

    if (m_isInitialized) {
        // Reset to screen dimensions
        m_captureRegion.left = 0;
        m_captureRegion.top = 0;
        m_captureRegion.right = m_screenWidth;
        m_captureRegion.bottom = m_screenHeight;

        bool wasCapturing = m_isCapturing;
        if (wasCapturing) {
            StopCapture();
        }

        // Update capture dimensions and buffer
        m_captureWidth = m_screenWidth;
        m_captureHeight = m_screenHeight;
        m_bufferSize = m_captureWidth * m_captureHeight * 4;
        m_frameBuffer = std::make_unique<unsigned char[]>(m_bufferSize);

        if (wasCapturing) {
            StartCapture();
        }

        std::cout << "[NVFBCCapture] Reset to full screen capture: " << m_captureWidth << "x" << m_captureHeight << std::endl;
    }
}