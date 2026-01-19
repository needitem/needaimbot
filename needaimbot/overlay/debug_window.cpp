#include "../core/windows_headers.h"

#include "debug_window.h"
#include "AppContext.h"
#include "../core/constants.h"
#include "../cuda/detection/postProcess.h"

#include <d3d11.h>
#include <dxgi.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <cstdio>

// D2D for drawing
#include <d2d1.h>
#include <dwrite.h>
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")

namespace DebugOverlay {

// State
static std::atomic<bool> g_running{false};
static std::atomic<bool> g_visible{false};
static std::thread g_thread;
static std::mutex g_mutex;

// Window handles
static HWND g_hwnd = nullptr;

// D2D resources
static ID2D1Factory* g_d2dFactory = nullptr;
static ID2D1HwndRenderTarget* g_renderTarget = nullptr;
static IDWriteFactory* g_dwriteFactory = nullptr;
static IDWriteTextFormat* g_textFormat = nullptr;
static ID2D1SolidColorBrush* g_brushGreen = nullptr;
static ID2D1SolidColorBrush* g_brushYellow = nullptr;
static ID2D1SolidColorBrush* g_brushRed = nullptr;
static ID2D1SolidColorBrush* g_brushWhite = nullptr;
static ID2D1SolidColorBrush* g_brushBackground = nullptr;

// Screen info
static int g_screenWidth = 0;
static int g_screenHeight = 0;
static int g_captureX = 0;
static int g_captureY = 0;
static int g_captureSize = 0;

// Forward declarations
static LRESULT CALLBACK DebugWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
static bool CreateDebugWindow();
static bool InitD2D();
static void CleanupD2D();
static void RenderFrame();
static void DebugThreadFunc();

bool IsVisible() {
    return g_visible.load();
}

bool IsRunning() {
    return g_running.load();
}

void SetVisible(bool visible) {
    g_visible.store(visible);
    if (g_hwnd) {
        ShowWindow(g_hwnd, visible ? SW_SHOWNA : SW_HIDE);
    }
}

void Toggle() {
    SetVisible(!IsVisible());
}

void Start() {
    if (g_running.load()) return;
    
    g_running.store(true);
    g_visible.store(false);
    
    g_thread = std::thread(DebugThreadFunc);
    
    std::cout << "[DebugOverlay] Started" << std::endl;
}

void Stop() {
    if (!g_running.load()) return;
    
    g_running.store(false);
    g_visible.store(false);
    
    // Signal window to close
    if (g_hwnd) {
        PostMessage(g_hwnd, WM_CLOSE, 0, 0);
    }
    
    if (g_thread.joinable()) {
        g_thread.join();
    }
    
    std::cout << "[DebugOverlay] Stopped" << std::endl;
}

static void DebugThreadFunc() {
    // Get screen dimensions
    g_screenWidth = GetSystemMetrics(SM_CXSCREEN);
    g_screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    if (!CreateDebugWindow()) {
        std::cerr << "[DebugOverlay] Failed to create window" << std::endl;
        g_running.store(false);
        return;
    }
    
    if (!InitD2D()) {
        std::cerr << "[DebugOverlay] Failed to init D2D" << std::endl;
        DestroyWindow(g_hwnd);
        g_running.store(false);
        return;
    }
    
    MSG msg = {};
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    
    while (g_running.load()) {
        // Handle messages
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                g_running.store(false);
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        
        if (!g_running.load()) break;
        
        // Frame limiting - 60 FPS
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrameTime).count();
        
        if (elapsed < 16) {  // ~60 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(16 - elapsed));
        }
        lastFrameTime = std::chrono::high_resolution_clock::now();
        
        if (g_visible.load()) {
            RenderFrame();
        }
    }
    
    CleanupD2D();
    
    if (g_hwnd) {
        DestroyWindow(g_hwnd);
        g_hwnd = nullptr;
    }
    
    UnregisterClassW(L"DebugOverlayClass", GetModuleHandle(nullptr));
}

static bool CreateDebugWindow() {
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = DebugWndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"DebugOverlayClass";
    
    if (!RegisterClassExW(&wc)) {
        DWORD err = GetLastError();
        if (err != ERROR_CLASS_ALREADY_EXISTS) {
            return false;
        }
    }
    
    // Create a layered, topmost, transparent window
    g_hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        L"DebugOverlayClass",
        L"Debug Overlay",
        WS_POPUP,
        0, 0,
        g_screenWidth, g_screenHeight,
        nullptr, nullptr,
        GetModuleHandle(nullptr),
        nullptr
    );
    
    if (!g_hwnd) {
        return false;
    }
    
    // Make window click-through with per-pixel alpha
    SetLayeredWindowAttributes(g_hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    
    // Initially hidden
    ShowWindow(g_hwnd, SW_HIDE);
    
    return true;
}

static bool InitD2D() {
    HRESULT hr;
    
    // Create D2D factory
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &g_d2dFactory);
    if (FAILED(hr)) return false;
    
    // Create DWrite factory for text
    hr = DWriteCreateFactory(
        DWRITE_FACTORY_TYPE_SHARED,
        __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(&g_dwriteFactory)
    );
    if (FAILED(hr)) return false;
    
    // Create text format with larger, clearer font
    hr = g_dwriteFactory->CreateTextFormat(
        L"Segoe UI",
        nullptr,
        DWRITE_FONT_WEIGHT_SEMI_BOLD,
        DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL,
        16.0f,
        L"en-us",
        &g_textFormat
    );
    if (FAILED(hr)) return false;
    
    // Set text rendering for clarity
    g_textFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    g_textFormat->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);
    
    // Create render target with better text rendering
    D2D1_SIZE_U size = D2D1::SizeU(g_screenWidth, g_screenHeight);
    D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
        0.0f, 0.0f,
        D2D1_RENDER_TARGET_USAGE_NONE,
        D2D1_FEATURE_LEVEL_DEFAULT
    );
    D2D1_HWND_RENDER_TARGET_PROPERTIES hwndProps = D2D1::HwndRenderTargetProperties(g_hwnd, size);
    
    hr = g_d2dFactory->CreateHwndRenderTarget(rtProps, hwndProps, &g_renderTarget);
    if (FAILED(hr)) return false;
    
    // Set text antialiasing mode for clearer text
    g_renderTarget->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_CLEARTYPE);
    
    // Create brushes
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 1.0f, 0.0f, 1.0f), &g_brushGreen);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 0.0f, 1.0f), &g_brushYellow);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 0.0f, 0.0f, 1.0f), &g_brushRed);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 1.0f, 1.0f), &g_brushWhite);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.8f), &g_brushBackground);
    
    return true;
}

static void CleanupD2D() {
    if (g_brushBackground) { g_brushBackground->Release(); g_brushBackground = nullptr; }
    if (g_brushWhite) { g_brushWhite->Release(); g_brushWhite = nullptr; }
    if (g_brushRed) { g_brushRed->Release(); g_brushRed = nullptr; }
    if (g_brushYellow) { g_brushYellow->Release(); g_brushYellow = nullptr; }
    if (g_brushGreen) { g_brushGreen->Release(); g_brushGreen = nullptr; }
    if (g_textFormat) { g_textFormat->Release(); g_textFormat = nullptr; }
    if (g_renderTarget) { g_renderTarget->Release(); g_renderTarget = nullptr; }
    if (g_dwriteFactory) { g_dwriteFactory->Release(); g_dwriteFactory = nullptr; }
    if (g_d2dFactory) { g_d2dFactory->Release(); g_d2dFactory = nullptr; }
}

static void RenderFrame() {
    if (!g_renderTarget) return;
    
    auto& ctx = AppContext::getInstance();
    
    // Get capture region info
    int captureSize = ctx.config.profile().detection_resolution;
    int centerX = g_screenWidth / 2;
    int centerY = g_screenHeight / 2;
    int captureX = centerX - captureSize / 2;
    int captureY = centerY - captureSize / 2;
    
    g_captureX = captureX;
    g_captureY = captureY;
    g_captureSize = captureSize;
    
    // Get targets
    std::vector<Target> targets = ctx.getAllTargets();
    Target bestTarget = ctx.getBestTarget();
    bool hasBest = ctx.hasValidTarget();
    
    // Begin drawing
    g_renderTarget->BeginDraw();
    g_renderTarget->Clear(D2D1::ColorF(0, 0, 0, 0));  // Transparent background
    
    // Draw capture region frame
    {
        D2D1_RECT_F frameRect = D2D1::RectF(
            static_cast<float>(captureX),
            static_cast<float>(captureY),
            static_cast<float>(captureX + captureSize),
            static_cast<float>(captureY + captureSize)
        );
        g_renderTarget->DrawRectangle(frameRect, g_brushWhite, 1.0f);
        
        // Draw crosshair at center
        float cx = static_cast<float>(centerX);
        float cy = static_cast<float>(centerY);
        g_renderTarget->DrawLine(D2D1::Point2F(cx - 10, cy), D2D1::Point2F(cx + 10, cy), g_brushWhite, 1.0f);
        g_renderTarget->DrawLine(D2D1::Point2F(cx, cy - 10), D2D1::Point2F(cx, cy + 10), g_brushWhite, 1.0f);
    }
    
    // Draw all detection boxes
    float confThreshold = ctx.config.profile().confidence_threshold;
    
    for (const auto& target : targets) {
        if (target.width <= 0 || target.height <= 0) continue;
        
        // Convert from capture region coordinates to screen coordinates
        float screenX = captureX + target.x;
        float screenY = captureY + target.y;
        float screenW = target.width;
        float screenH = target.height;
        
        // Determine if this is the best target
        bool isBest = hasBest && 
            (std::abs(target.x - bestTarget.x) < 1.0f) &&
            (std::abs(target.y - bestTarget.y) < 1.0f);
        
        // Choose color based on status
        ID2D1SolidColorBrush* boxBrush;
        float thickness;
        if (isBest) {
            boxBrush = g_brushGreen;
            thickness = 3.0f;
        } else if (target.confidence >= confThreshold) {
            boxBrush = g_brushYellow;
            thickness = 2.0f;
        } else {
            boxBrush = g_brushRed;
            thickness = 1.0f;
        }
        
        // Draw bounding box
        D2D1_RECT_F boxRect = D2D1::RectF(screenX, screenY, screenX + screenW, screenY + screenH);
        g_renderTarget->DrawRectangle(boxRect, boxBrush, thickness);
        
        // Draw confidence label
        char labelBuf[64];
        snprintf(labelBuf, sizeof(labelBuf), "%.0f%%", target.confidence * 100.0f);
        
        // Convert to wide string
        wchar_t wlabel[64];
        MultiByteToWideChar(CP_UTF8, 0, labelBuf, -1, wlabel, 64);
        
        // Draw background for text
        D2D1_RECT_F textBgRect = D2D1::RectF(screenX, screenY - 18, screenX + 50, screenY);
        g_renderTarget->FillRectangle(textBgRect, g_brushBackground);
        
        // Draw text
        D2D1_RECT_F textRect = D2D1::RectF(screenX + 2, screenY - 16, screenX + 48, screenY);
        g_renderTarget->DrawTextW(wlabel, static_cast<UINT32>(wcslen(wlabel)), g_textFormat, textRect, boxBrush);
        
        // If best target, add marker
        if (isBest) {
            wchar_t targetLabel[] = L"TARGET";
            D2D1_RECT_F targetBgRect = D2D1::RectF(screenX, screenY - 36, screenX + 60, screenY - 18);
            g_renderTarget->FillRectangle(targetBgRect, g_brushBackground);
            D2D1_RECT_F targetTextRect = D2D1::RectF(screenX + 2, screenY - 34, screenX + 58, screenY - 18);
            g_renderTarget->DrawTextW(targetLabel, static_cast<UINT32>(wcslen(targetLabel)), g_textFormat, targetTextRect, g_brushGreen);
        }
    }
    
    // Draw status info at top-left
    {
        wchar_t statusBuf[256];
        swprintf_s(statusBuf, L"Targets: %zu | Capture: %dx%d | Conf: %.0f%%",
            targets.size(),
            captureSize, captureSize,
            confThreshold * 100.0f);
        
        D2D1_RECT_F statusBgRect = D2D1::RectF(10, 10, 350, 30);
        g_renderTarget->FillRectangle(statusBgRect, g_brushBackground);
        
        D2D1_RECT_F statusTextRect = D2D1::RectF(15, 12, 345, 28);
        g_renderTarget->DrawTextW(statusBuf, static_cast<UINT32>(wcslen(statusBuf)), g_textFormat, statusTextRect, g_brushWhite);
    }
    
    g_renderTarget->EndDraw();
}

static LRESULT CALLBACK DebugWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_PAINT:
            ValidateRect(hwnd, nullptr);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

} // namespace DebugOverlay
