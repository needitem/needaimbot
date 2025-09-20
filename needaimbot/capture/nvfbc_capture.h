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
#ifndef NVFBCCALL
#define NVFBCCALL __stdcall
#endif

#ifndef NVFBC_BOOL
#define NVFBC_BOOL unsigned int
#endif

#define NVFBC_TRUE 1U
#define NVFBC_FALSE 0U

#ifndef NVFBC_STRUCT_VERSION
#define NVFBC_STRUCT_VERSION(type, ver) ((unsigned int)(sizeof(type) | ((ver) << 16)))
#endif

// Capture region rectangle is referenced throughout NVFBC params
typedef struct {
    unsigned int left;
    unsigned int top;
    unsigned int right;
    unsigned int bottom;
} NVFBC_RECT;

typedef enum {
    NVFBC_BUFFER_FORMAT_ARGB = 0,
    NVFBC_BUFFER_FORMAT_XRGB = 1,
    NVFBC_BUFFER_FORMAT_NV12 = 2,
    NVFBC_BUFFER_FORMAT_YUV444P = 3,
    NVFBC_BUFFER_FORMAT_BGRA = 4
} NVFBC_BUFFER_FORMAT;

typedef enum {
    NVFBC_TOSYS_GRAB_MODE_SCALE = 0,
    NVFBC_TOSYS_GRAB_MODE_PASS = 1,
    NVFBC_TOSYS_GRAB_MODE_CROP = 2
} NVFBC_TOSYS_GRAB_MODE;

typedef enum {
    NVFBC_TOSYS_GRAB_FLAGS_NONE = 0,
    NVFBC_TOSYS_GRAB_FLAGS_NOWAIT = 1 << 0,
    NVFBC_TOSYS_GRAB_FLAGS_FORCE_REFRESH = 1 << 1
} NVFBC_TOSYS_GRAB_FLAGS;

typedef enum {
    NVFBC_CAPTURE_TO_SYS = 0,
    NVFBC_CAPTURE_TO_CUDA = 1,
    NVFBC_CAPTURE_TO_HWENC = 2,
    NVFBC_CAPTURE_TO_DX9 = 3,
    NVFBC_CAPTURE_TO_DXGI = 4
} NVFBC_CAPTURE_TYPE;

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

typedef struct {
    unsigned int dwVersion;
    unsigned int eCaptureType;
    NVFBC_BOOL bWithCursor;
    NVFBC_BOOL bDisableHotKeyReset;
    HWND hWnd;
    NVFBC_BOOL bStereoGrab;
    NVFBC_BOOL bEnableDirectCapture;
    unsigned int dwReserved;
    NVFBC_RECT captureBox;
    NVFBC_SESSION_HANDLE hCaptureSession;
    unsigned int reserved[54];
} NVFBC_CREATE_CAPTURE_SESSION_PARAMS;

#define NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER \
    NVFBC_STRUCT_VERSION(NVFBC_CREATE_CAPTURE_SESSION_PARAMS, 1)

typedef struct {
    unsigned int dwVersion;
    NVFBC_SESSION_HANDLE hCaptureSession;
    unsigned int reserved[62];
} NVFBC_DESTROY_CAPTURE_SESSION_PARAMS;

#define NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER \
    NVFBC_STRUCT_VERSION(NVFBC_DESTROY_CAPTURE_SESSION_PARAMS, 1)

typedef struct {
    unsigned int dwVersion;
    NVFBC_SESSION_HANDLE hCaptureSession;
    NVFBC_BOOL bUseKVMFrameLock;
    NVFBC_BOOL bEnableCursor;
    NVFBC_BOOL bStereoGrab;
    unsigned int dwFlags;
    NVFBC_TOSYS_GRAB_MODE eMode;
    NVFBC_BUFFER_FORMAT eBufferFormat;
    unsigned int dwTargetWidth;
    unsigned int dwTargetHeight;
    unsigned int dwNumBuffers;
    unsigned int dwReserved;
    NVFBC_RECT captureBox;
    unsigned int reserved[48];
} NVFBC_TOSYS_SETUP_PARAMS;

#define NVFBC_TOSYS_SETUP_PARAMS_VER NVFBC_STRUCT_VERSION(NVFBC_TOSYS_SETUP_PARAMS, 1)

typedef struct {
    unsigned int dwVersion;
    NVFBC_SESSION_HANDLE hCaptureSession;
    NVFBC_TOSYS_GRAB_FLAGS dwFlags;
    void* pSysmemBuffer;
    unsigned int dwBufferWidth;
    unsigned int dwBufferHeight;
    unsigned int dwBufferPitch;
    unsigned int dwBufferSize;
    void* pFrameGrabInfo;
    unsigned int reserved[56];
} NVFBC_TOSYS_GRAB_FRAME_PARAMS;

#define NVFBC_TOSYS_GRAB_FRAME_PARAMS_VER \
    NVFBC_STRUCT_VERSION(NVFBC_TOSYS_GRAB_FRAME_PARAMS, 1)

typedef struct {
    unsigned int dwWidth;
    unsigned int dwHeight;
    unsigned int dwBufferWidth;
    unsigned int dwBufferHeight;
    unsigned int dwFrameId;
    unsigned int dwFlags;
    unsigned long long qwTimestamp;
    unsigned int reserved[54];
} NVFBC_FRAME_GRAB_INFO;

// Main NVFBC creation function
typedef NVFBC_RESULT(NVFBCCALL* PFN_NvFBC_Create)(NVFBC_API_FUNCTION_LIST* api);
typedef NVFBC_RESULT(NVFBCCALL* PFN_NvFBC_GetStatus)(void);
typedef NVFBC_RESULT(NVFBCCALL* PFN_NvFBCCreateCaptureSession)(NVFBC_CREATE_CAPTURE_SESSION_PARAMS* params);
typedef NVFBC_RESULT(NVFBCCALL* PFN_NvFBCDestroyCaptureSession)(NVFBC_DESTROY_CAPTURE_SESSION_PARAMS* params);
typedef NVFBC_RESULT(NVFBCCALL* PFN_NvFBCToSysSetUp)(NVFBC_TOSYS_SETUP_PARAMS* params);
typedef NVFBC_RESULT(NVFBCCALL* PFN_NvFBCToSysGrabFrame)(NVFBC_TOSYS_GRAB_FRAME_PARAMS* params);

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
    bool ConfigureNvFBCToSys();

    // NVFBC library and function pointers
    HMODULE m_nvfbcLib;
    PFN_NvFBC_Create m_pfnNvFBCCreate;
    PFN_NvFBC_GetStatus m_pfnNvFBCGetStatus;
    PFN_NvFBCCreateCaptureSession m_pfnCreateCaptureSession;
    PFN_NvFBCDestroyCaptureSession m_pfnDestroyCaptureSession;
    PFN_NvFBCToSysSetUp m_pfnToSysSetUp;
    PFN_NvFBCToSysGrabFrame m_pfnToSysGrabFrame;
    NVFBC_API_FUNCTION_LIST m_nvfbcAPI;

    // NVFBC session
    NVFBC_SESSION_HANDLE m_captureSession;
    bool m_useNvFBC;
    bool m_nvFBCConfigured;

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
    void CaptureThreadProcNvFBC();
};
