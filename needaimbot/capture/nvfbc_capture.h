#pragma once

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <memory>
#include <functional>
#include <mutex>
#include <thread>
#include <chrono>

// NVFBC API definitions - using correct NVFBC SDK structure
typedef void* NVFBC_SESSION_HANDLE;

typedef enum {
    NVFBC_SUCCESS = 0,
    NVFBC_ERROR_GENERIC = 1,
    NVFBC_ERROR_INVALID_PARAM = 2,
    NVFBC_ERROR_INVALIDATED_SESSION = 3,
    NVFBC_ERROR_PROTECTED_CONTENT = 4,
    NVFBC_ERROR_DRIVER_FAILURE = 5,
    NVFBC_ERROR_CUDA_FAILURE = 6,
    NVFBC_ERROR_UNSUPPORTED = 7,
    NVFBC_ERROR_HW_ENC_FAILURE = 8,
    NVFBC_ERROR_INCOMPATIBLE_DRIVER = 9,
    NVFBC_ERROR_UNSUPPORTED_PLATFORM = 10,
    NVFBC_ERROR_OUT_OF_MEMORY = 11,
    NVFBC_ERROR_INVALID_PTR = 12,
    NVFBC_ERROR_INCOMPATIBLE_VERSION = 13,
    NVFBC_ERROR_OOM = 14,
    NVFBC_ERROR_INVALID_CALL = 15,
    NVFBC_ERROR_SYSTEM_ERROR = 16,
    NVFBC_ERROR_INVALID_TARGET = 17
} NVFBC_RESULT;

// NVFBC API interface - simplified for actual usage
typedef struct NVFBC_API_FUNCTION_LIST {
    unsigned int version;
    unsigned int reserved;

    // Function pointers will be filled by NvFBC_Create
    void* nvFBCCreateCaptureSession;
    void* nvFBCDestroyCaptureSession;
    void* nvFBCNextFrame;
    void* nvFBCSetUpHWCursor;
    void* nvFBCGetStatus;
    void* nvFBCGetLastErrorStr;
} NVFBC_API_FUNCTION_LIST;

// Main NVFBC creation function
typedef NVFBC_RESULT(*PFN_NvFBC_Create)(NVFBC_API_FUNCTION_LIST* api);
typedef NVFBC_RESULT(*PFN_NvFBC_GetStatus)(void);

// Capture region rectangle
typedef struct {
    unsigned int left;
    unsigned int top;
    unsigned int right;
    unsigned int bottom;
} NVFBC_RECT;

// For the session management we'll use a simplified approach
typedef struct {
    unsigned int dwVersion;
    unsigned int dwOutputId;
    unsigned int dwTargetWidth;
    unsigned int dwTargetHeight;
    void* pBuffer;
    unsigned int dwBufSize;
    unsigned int dwFlags;
    NVFBC_RECT captureBox;  // Region to capture
} NVFBC_TOSYS_SETUP_PARAMS;

class NVFBCCapture {
public:
    NVFBCCapture();
    ~NVFBCCapture();

    bool Initialize(HWND targetWindow = nullptr);
    void Shutdown();

    bool StartCapture();
    void StopCapture();

    bool IsCapturing() const { return m_isCapturing; }

    // Get the latest captured frame data
    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size);

    // Set callback for new frames
    void SetFrameCallback(std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback);

    // Get capture dimensions
    int GetWidth() const { return m_captureWidth; }
    int GetHeight() const { return m_captureHeight; }
    int GetScreenWidth() const { return m_screenWidth; }
    int GetScreenHeight() const { return m_screenHeight; }

    // Set capture region (x, y, width, height)
    bool SetCaptureRegion(int x, int y, int width, int height);

    // Get current capture region
    void GetCaptureRegion(int* x, int* y, int* width, int* height) const;

    // Reset to full screen capture
    void ResetToFullScreen();

    // Check if NVFBC is available
    static bool IsNVFBCAvailable();

private:
    bool LoadNVFBCLibrary();
    void UnloadNVFBCLibrary();
    bool CreateCaptureSession(HWND targetWindow);
    void DestroyCaptureSession();

    // NVFBC library and function pointers
    HMODULE m_nvfbcLib;
    PFN_NvFBC_Create m_pfnNvFBCCreate;
    PFN_NvFBC_GetStatus m_pfnNvFBCGetStatus;
    NVFBC_API_FUNCTION_LIST m_nvfbcAPI;

    // NVFBC session
    NVFBC_SESSION_HANDLE m_captureSession;

    // Frame buffer
    std::unique_ptr<unsigned char[]> m_frameBuffer;
    unsigned int m_bufferSize;
    std::mutex m_frameMutex;
    std::function<void(void*, unsigned int, unsigned int, unsigned int)> m_frameCallback;

    // Capture state
    bool m_isInitialized;
    bool m_isCapturing;
    int m_captureWidth;
    int m_captureHeight;

    // Screen dimensions (full screen)
    int m_screenWidth;
    int m_screenHeight;

    // Capture region
    NVFBC_RECT m_captureRegion;
    bool m_useCustomRegion;

    // Capture thread
    std::thread m_captureThread;
    bool m_shouldStop;
    void CaptureThreadProc();
};