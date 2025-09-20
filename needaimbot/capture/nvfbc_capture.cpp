#include "nvfbc_capture.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>

NVFBCCapture::NVFBCCapture()
    : m_nvfbcLib(nullptr)
    , m_pfnNvFBCCreate(nullptr)
    , m_pfnNvFBCGetStatus(nullptr)
    , m_pfnCreateCaptureSession(nullptr)
    , m_pfnDestroyCaptureSession(nullptr)
    , m_pfnToSysSetUp(nullptr)
    , m_pfnToSysGrabFrame(nullptr)
    , m_captureSession(nullptr)
    , m_useNvFBC(false)
    , m_nvFBCConfigured(false)
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
    m_nvFBCConfigured = false;
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

    m_pfnCreateCaptureSession = reinterpret_cast<PFN_NvFBCCreateCaptureSession>(m_nvfbcAPI.nvFBCCreateCaptureSession);
    m_pfnDestroyCaptureSession = reinterpret_cast<PFN_NvFBCDestroyCaptureSession>(m_nvfbcAPI.nvFBCDestroyCaptureSession);
    m_pfnToSysSetUp = nullptr;
    m_pfnToSysGrabFrame = nullptr;

    if (!m_pfnCreateCaptureSession) {
        m_pfnCreateCaptureSession = reinterpret_cast<PFN_NvFBCCreateCaptureSession>(
            GetProcAddress(m_nvfbcLib, "NvFBCCreateCaptureSession"));
    }
    if (!m_pfnDestroyCaptureSession) {
        m_pfnDestroyCaptureSession = reinterpret_cast<PFN_NvFBCDestroyCaptureSession>(
            GetProcAddress(m_nvfbcLib, "NvFBCDestroyCaptureSession"));
    }
    if (!m_pfnToSysSetUp) {
        m_pfnToSysSetUp = reinterpret_cast<PFN_NvFBCToSysSetUp>(
            GetProcAddress(m_nvfbcLib, "NvFBCToSysSetUp"));
    }
    if (!m_pfnToSysGrabFrame) {
        m_pfnToSysGrabFrame = reinterpret_cast<PFN_NvFBCToSysGrabFrame>(
            GetProcAddress(m_nvfbcLib, "NvFBCToSysGrabFrame"));
    }

    if (!m_pfnCreateCaptureSession || !m_pfnDestroyCaptureSession || !m_pfnToSysSetUp || !m_pfnToSysGrabFrame) {
        std::cerr << "[NVFBCCapture] Failed to resolve required NVFBC entry points" << std::endl;
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
    m_pfnCreateCaptureSession = nullptr;
    m_pfnDestroyCaptureSession = nullptr;
    m_pfnToSysSetUp = nullptr;
    m_pfnToSysGrabFrame = nullptr;
    m_useNvFBC = false;
    m_nvFBCConfigured = false;
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
        static_cast<int>(m_captureRegion.right) > m_screenWidth ||
        static_cast<int>(m_captureRegion.bottom) > m_screenHeight) {
        std::cerr << "[NVFBCCapture] Invalid capture region: "
                  << m_captureRegion.left << "," << m_captureRegion.top
                  << " to " << m_captureRegion.right << "," << m_captureRegion.bottom << std::endl;
        return false;
    }

    // Allocate frame buffer (BGRA = 4 bytes per pixel)
    m_bufferSize = m_captureWidth * m_captureHeight * 4;
    m_frameBuffer = std::make_unique<unsigned char[]>(m_bufferSize);

    m_useNvFBC = m_pfnCreateCaptureSession && m_pfnDestroyCaptureSession && m_pfnToSysSetUp && m_pfnToSysGrabFrame;
    m_nvFBCConfigured = false;
    m_captureSession = nullptr;

    if (m_useNvFBC) {
        NVFBC_CREATE_CAPTURE_SESSION_PARAMS createParams{};
        createParams.dwVersion = NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER;
        createParams.eCaptureType = NVFBC_CAPTURE_TO_SYS;
        createParams.bWithCursor = NVFBC_TRUE;
        createParams.bDisableHotKeyReset = NVFBC_TRUE;
        createParams.hWnd = targetWindow;
        createParams.bStereoGrab = NVFBC_FALSE;
        createParams.bEnableDirectCapture = NVFBC_FALSE;
        createParams.dwReserved = 0;
        createParams.captureBox = m_captureRegion;

        NVFBC_RESULT createResult = m_pfnCreateCaptureSession(&createParams);
        if (createResult != NVFBC_SUCCESS || !createParams.hCaptureSession) {
            std::cerr << "[NVFBCCapture] NvFBC session creation failed: " << createResult
                      << ". Falling back to GDI capture." << std::endl;
            m_useNvFBC = false;
        } else {
            m_captureSession = createParams.hCaptureSession;
            if (!ConfigureNvFBCToSys()) {
                std::cerr << "[NVFBCCapture] NvFBC ToSys setup failed. Falling back to GDI capture." << std::endl;
                NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroyParams{};
                destroyParams.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER;
                destroyParams.hCaptureSession = m_captureSession;
                m_pfnDestroyCaptureSession(&destroyParams);
                m_captureSession = nullptr;
                m_useNvFBC = false;
            }
        }
    }

    if (m_useCustomRegion) {
        std::cout << "[NVFBCCapture] Partial capture setup completed" << std::endl;
        std::cout << "  - Screen: " << m_screenWidth << "x" << m_screenHeight << std::endl;
        std::cout << "  - Region: (" << m_captureRegion.left << "," << m_captureRegion.top
                  << ") to (" << m_captureRegion.right << "," << m_captureRegion.bottom << ")" << std::endl;
        std::cout << "  - Capture Size: " << m_captureWidth << "x" << m_captureHeight << std::endl;
    } else {
        std::cout << "[NVFBCCapture] Full screen capture setup completed. Resolution: " << m_captureWidth << "x" << m_captureHeight << std::endl;
    }

    if (!m_useNvFBC) {
        std::cout << "[NVFBCCapture] Using GDI BitBlt fallback path" << std::endl;
    } else {
        std::cout << "[NVFBCCapture] NVFBC ToSys capture path configured" << std::endl;
    }

    return true;
}

void NVFBCCapture::DestroyCaptureSession() {
    if (m_useNvFBC && m_captureSession && m_pfnDestroyCaptureSession) {
        NVFBC_DESTROY_CAPTURE_SESSION_PARAMS params{};
        params.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER;
        params.hCaptureSession = m_captureSession;
        m_pfnDestroyCaptureSession(&params);
    }

    m_captureSession = nullptr;
    m_useNvFBC = false;
    m_nvFBCConfigured = false;
    m_frameBuffer.reset();
    m_bufferSize = 0;
}

bool NVFBCCapture::ConfigureNvFBCToSys() {
    if (!m_useNvFBC || !m_captureSession || !m_pfnToSysSetUp) {
        m_nvFBCConfigured = false;
        return false;
    }

    NVFBC_TOSYS_SETUP_PARAMS setupParams{};
    setupParams.dwVersion = NVFBC_TOSYS_SETUP_PARAMS_VER;
    setupParams.hCaptureSession = m_captureSession;
    setupParams.bUseKVMFrameLock = NVFBC_FALSE;
    setupParams.bEnableCursor = NVFBC_TRUE;
    setupParams.bStereoGrab = NVFBC_FALSE;
    setupParams.dwFlags = 0;
    setupParams.eMode = NVFBC_TOSYS_GRAB_MODE_CROP;
    setupParams.eBufferFormat = NVFBC_BUFFER_FORMAT_ARGB;
    setupParams.dwTargetWidth = static_cast<unsigned int>(m_captureWidth);
    setupParams.dwTargetHeight = static_cast<unsigned int>(m_captureHeight);
    setupParams.dwNumBuffers = 1;
    setupParams.dwReserved = 0;
    setupParams.captureBox = m_captureRegion;

    NVFBC_RESULT setupResult = m_pfnToSysSetUp(&setupParams);
    if (setupResult != NVFBC_SUCCESS) {
        std::cerr << "[NVFBCCapture] NvFBCToSysSetUp failed with error: " << setupResult << std::endl;
        m_nvFBCConfigured = false;
        return false;
    }

    m_nvFBCConfigured = true;
    return true;
}

void NVFBCCapture::CaptureThreadProcNvFBC() {
    std::cout << "[NVFBCCapture] NVFBC capture path active" << std::endl;

    while (!m_shouldStop) {
        if (!m_nvFBCConfigured) {
            if (!ConfigureNvFBCToSys()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }

        unsigned int width = 0;
        unsigned int height = 0;
        unsigned int bufferSize = 0;
        unsigned char* bufferPtr = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            width = static_cast<unsigned int>(m_captureWidth);
            height = static_cast<unsigned int>(m_captureHeight);
            bufferSize = width * height * 4;
            if (!m_frameBuffer || m_bufferSize != bufferSize) {
                m_frameBuffer = std::make_unique<unsigned char[]>(bufferSize);
                m_bufferSize = bufferSize;
            }
            bufferPtr = m_frameBuffer.get();
        }

        if (!bufferPtr || width == 0 || height == 0) {
            std::this_thread::yield();
            continue;
        }

        NVFBC_TOSYS_GRAB_FRAME_PARAMS grabParams{};
        grabParams.dwVersion = NVFBC_TOSYS_GRAB_FRAME_PARAMS_VER;
        grabParams.hCaptureSession = m_captureSession;
        grabParams.dwFlags = NVFBC_TOSYS_GRAB_FLAGS_NOWAIT;
        grabParams.pSysmemBuffer = bufferPtr;
        grabParams.dwBufferWidth = width;
        grabParams.dwBufferHeight = height;
        grabParams.dwBufferPitch = width * 4;
        grabParams.dwBufferSize = bufferSize;
        NVFBC_FRAME_GRAB_INFO frameInfo{};
        grabParams.pFrameGrabInfo = &frameInfo;

        NVFBC_RESULT grabResult = m_pfnToSysGrabFrame(&grabParams);
        if (grabResult == NVFBC_ERROR_INVALIDATED_SESSION) {
            m_nvFBCConfigured = false;
            std::this_thread::yield();
            continue;
        }
        if (grabResult != NVFBC_SUCCESS) {
            std::cerr << "[NVFBCCapture] NvFBCToSysGrabFrame failed: " << grabResult << std::endl;
            std::this_thread::yield();
            continue;
        }

        std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback;
        void* framePtr = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            framePtr = m_frameBuffer.get();
            callback = m_frameCallback;
        }

        if (callback && framePtr) {
            callback(framePtr, width, height, bufferSize);
        }
    }

    std::cout << "[NVFBCCapture] NVFBC capture thread stopping" << std::endl;
}

void NVFBCCapture::CaptureThreadProc() {
    std::cout << "[NVFBCCapture] Capture thread started" << std::endl;

    if (m_useNvFBC && m_pfnToSysGrabFrame && m_captureSession) {
        CaptureThreadProcNvFBC();
        std::cout << "[NVFBCCapture] Capture thread stopped" << std::endl;
        return;
    }

    HDC screenDC = GetDC(nullptr);
    if (!screenDC) {
        std::cerr << "[NVFBCCapture] Failed to get screen DC" << std::endl;
        return;
    }

    HDC memoryDC = CreateCompatibleDC(screenDC);
    if (!memoryDC) {
        std::cerr << "[NVFBCCapture] Failed to create memory DC" << std::endl;
        ReleaseDC(nullptr, screenDC);
        return;
    }

    HBITMAP hBitmap = nullptr;
    HGDIOBJ oldBitmap = nullptr;
    void* dibData = nullptr;
    int currentWidth = 0;
    int currentHeight = 0;

    auto cleanupBitmap = [&]() {
        if (oldBitmap) {
            SelectObject(memoryDC, oldBitmap);
            oldBitmap = nullptr;
        }
        if (hBitmap) {
            DeleteObject(hBitmap);
            hBitmap = nullptr;
        }
        dibData = nullptr;
        currentWidth = 0;
        currentHeight = 0;
    };

    auto recreateBitmapIfNeeded = [&](int width, int height) {
        if (width <= 0 || height <= 0) {
            cleanupBitmap();
            return false;
        }

        if (width == currentWidth && height == currentHeight && hBitmap && dibData) {
            return true;
        }

        cleanupBitmap();

        BITMAPINFO bmi;
        std::memset(&bmi, 0, sizeof(bmi));
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = -height; // top-down DIB
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        hBitmap = CreateDIBSection(screenDC, &bmi, DIB_RGB_COLORS, &dibData, nullptr, 0);
        if (!hBitmap || !dibData) {
            std::cerr << "[NVFBCCapture] Failed to create capture bitmap" << std::endl;
            cleanupBitmap();
            return false;
        }

        oldBitmap = SelectObject(memoryDC, hBitmap);
        currentWidth = width;
        currentHeight = height;
        return true;
    };

    bool bitbltErrorLogged = false;

    while (!m_shouldStop) {
        int left = 0;
        int top = 0;
        int width = 0;
        int height = 0;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            left = static_cast<int>(m_captureRegion.left);
            top = static_cast<int>(m_captureRegion.top);
            width = m_captureWidth;
            height = m_captureHeight;
        }

        if (!recreateBitmapIfNeeded(width, height)) {
            std::this_thread::yield();
            continue;
        }

        if (!BitBlt(memoryDC, 0, 0, width, height, screenDC, left, top, SRCCOPY | CAPTUREBLT)) {
            if (!bitbltErrorLogged) {
                std::cerr << "[NVFBCCapture] BitBlt failed" << std::endl;
                bitbltErrorLogged = true;
            }
            std::this_thread::yield();
            continue;
        }
        bitbltErrorLogged = false;

        std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback;
        void* framePtr = nullptr;
        size_t copySize = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            if (!m_frameBuffer || m_bufferSize != copySize) {
                m_frameBuffer = std::make_unique<unsigned char[]>(copySize);
                m_bufferSize = static_cast<unsigned int>(copySize);
            }

            std::memcpy(m_frameBuffer.get(), dibData, copySize);
            framePtr = m_frameBuffer.get();
            callback = m_frameCallback;
        }

        if (callback && framePtr) {
            callback(framePtr, static_cast<unsigned int>(width), static_cast<unsigned int>(height),
                     static_cast<unsigned int>(copySize));
        }

        std::this_thread::yield();
    }

    cleanupBitmap();
    DeleteDC(memoryDC);
    ReleaseDC(nullptr, screenDC);
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
        m_bufferSize = static_cast<unsigned int>(width * height * 4);
        m_frameBuffer = std::make_unique<unsigned char[]>(m_bufferSize);
        m_nvFBCConfigured = false;
        if (m_useNvFBC && m_captureSession) {
            ConfigureNvFBCToSys();
        }

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
        m_bufferSize = static_cast<unsigned int>(m_captureWidth * m_captureHeight * 4);
        m_frameBuffer = std::make_unique<unsigned char[]>(m_bufferSize);
        m_nvFBCConfigured = false;
        if (m_useNvFBC && m_captureSession) {
            ConfigureNvFBCToSys();
        }

        if (wasCapturing) {
            StartCapture();
        }

        std::cout << "[NVFBCCapture] Reset to full screen capture: " << m_captureWidth << "x" << m_captureHeight << std::endl;
    }
}