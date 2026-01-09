#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <cstring>
#include <vector>

class DisplayWindow {
public:
    HWND hWnd = nullptr;
    unsigned int frameWidth = 320;
    unsigned int frameHeight = 320;
    unsigned char* currentFrame = nullptr;
    unsigned int frameBufferSize = 0;
    BITMAPINFO bmi;

    static inline void my_memcpy(void* dst, const void* src, size_t size) {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        for (size_t i = 0; i < size; i++) {
            d[i] = s[i];
        }
    }

    static inline void my_memset(void* dst, int value, size_t size) {
        unsigned char* d = (unsigned char*)dst;
        for (size_t i = 0; i < size; i++) {
            d[i] = (unsigned char)value;
        }
    }

    static DisplayWindow* instance;

    DisplayWindow() {
        instance = this;
        my_memset(&bmi, 0, sizeof(BITMAPINFO));
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biPlanes =1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;
    }

    static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        switch (message) {
            case WM_DESTROY:
                PostQuitMessage(0);
                break;
            case WM_PAINT: {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hWnd, &ps);
                if (instance) instance->render(hdc);
                EndPaint(hWnd, &ps);
                break;
            }
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
        }
        return 0;
    }

    bool initialize(unsigned int width, unsigned int height, const char* title = "Inference Viewer") {
        frameWidth = width;
        frameHeight = height;
        frameBufferSize = width * height * 3;
        currentFrame = (unsigned char*)GlobalAlloc(GPTR, frameBufferSize);
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = -height;

        WNDCLASSA wc = {};
        wc.lpfnWndProc = WndProc;
        wc.hInstance = GetModuleHandle(NULL);
        wc.lpszClassName = "InferenceViewer";
        wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

        RegisterClassA(&wc);

        RECT rect = {0, 0, (LONG)width, (LONG)height};
        AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

        hWnd = CreateWindowA(
            "InferenceViewer",
            title,
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            NULL, NULL, GetModuleHandle(NULL), NULL
        );

        if (!hWnd) return false;

        ShowWindow(hWnd, SW_SHOW);
        UpdateWindow(hWnd);

        return true;
    }

    void updateFrame(const unsigned char* data, unsigned int width, unsigned int height) {
        if (width != frameWidth || height != frameHeight) {
            frameWidth = width;
            frameHeight = height;
            unsigned int newSize = width * height * 3;
            if (newSize > frameBufferSize) {
                GlobalFree((HGLOBAL)currentFrame);
                frameBufferSize = newSize;
                currentFrame = (unsigned char*)GlobalAlloc(GPTR, frameBufferSize);
            }
            bmi.bmiHeader.biWidth = width;
            bmi.bmiHeader.biHeight = -height;
        }
        my_memcpy(currentFrame, data, width * height * 3);
        InvalidateRect(hWnd, NULL, FALSE);
    }

    void render(HDC hdc) {
        if (!currentFrame) return;

        HDC hdcMem = CreateCompatibleDC(hdc);
        int pitch = ((frameWidth * 24 + 31) & ~31) / 8;
        HBITMAP hBitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, NULL, NULL, 0);

        if (hBitmap) {
            SelectObject(hdcMem, hBitmap);
            SetDIBitsToDevice(hdc, 0, 0, frameWidth, frameHeight,
                            0, 0, 0, frameHeight,
                            currentFrame, &bmi,
                            DIB_RGB_COLORS);
            DeleteObject(hBitmap);
        }
        DeleteDC(hdcMem);
    }

    bool processMessages() {
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) return false;
        }
        return true;
    }

    void shutdown() {
        if (hWnd) DestroyWindow(hWnd);
    }
};

DisplayWindow* DisplayWindow::instance = nullptr;
