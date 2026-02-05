#include "../core/windows_headers.h"

#include "preview_window.h"
#include "AppContext.h"
#include "../core/constants.h"
#include "../cuda/detection/postProcess.h"
#include "../cuda/simple_cuda_mat.h"
#include "../cuda/unified_graph_pipeline.h"

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
#include <cmath>

#include <d2d1.h>
#include <dwrite.h>
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")

namespace PreviewWindow {

// State
static std::atomic<bool> g_running{false};
static std::atomic<bool> g_visible{false};
static std::thread g_thread;

// Window
static HWND g_hwnd = nullptr;
static const int INITIAL_WIDTH = 800;
static const int INITIAL_HEIGHT = 600;
static std::atomic<bool> g_windowCreated{false};

// D2D resources
static ID2D1Factory* g_d2dFactory = nullptr;
static ID2D1HwndRenderTarget* g_renderTarget = nullptr;
static IDWriteFactory* g_dwriteFactory = nullptr;
static IDWriteTextFormat* g_textFormat = nullptr;
static IDWriteTextFormat* g_textFormatSmall = nullptr;
static ID2D1SolidColorBrush* g_brushGreen = nullptr;
static ID2D1SolidColorBrush* g_brushYellow = nullptr;
static ID2D1SolidColorBrush* g_brushRed = nullptr;
static ID2D1SolidColorBrush* g_brushWhite = nullptr;
static ID2D1SolidColorBrush* g_brushBackground = nullptr;
static ID2D1SolidColorBrush* g_brushGray = nullptr;
static ID2D1SolidColorBrush* g_brushBlue = nullptr;
static ID2D1Bitmap* g_frameBitmap = nullptr;
static int g_bitmapW = 0;
static int g_bitmapH = 0;

// UI state
static int g_activeTab = 0;       // 0=Aim Settings, 1=Color Filter
static int g_activeSlider = -1;   // Which slider is being dragged
static int g_activeCheckbox = -1; // Which checkbox is being clicked
static POINT g_mousePos = {0, 0};
static bool g_mouseDown = false;

// Forward declarations
static LRESULT CALLBACK PreviewWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
static bool CreatePreviewWindow();
static bool InitD2D();
static void CleanupD2D();
static void ResizeRenderTarget();
static void RenderFrame();
static void RenderSettingsPanel(float panelX, float panelY, float panelW, float panelH);
static void HandleMouseClick(int x, int y);
static void HandleMouseMove(int x, int y);
static void HandleMouseRelease();
static void PreviewThreadFunc();

bool IsVisible() {
    return g_visible.load();
}

bool IsRunning() {
    return g_running.load();
}

void SetVisible(bool visible) {
    g_visible.store(visible);
    
    // Wait for window creation if needed (max 1 second)
    int retries = 0;
    while (!g_hwnd && g_running.load() && retries < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        retries++;
    }
    
    if (g_hwnd) {
        PostMessage(g_hwnd, WM_APP + 1, visible ? 1 : 0, 0);
    }
}

void Start() {
    if (g_running.load()) return;

    g_running.store(true);
    g_visible.store(false);

    g_thread = std::thread(PreviewThreadFunc);
}

void Stop() {
    if (!g_running.load()) return;

    g_running.store(false);
    g_visible.store(false);

    if (g_hwnd) {
        PostMessage(g_hwnd, WM_CLOSE, 0, 0);
    }

    if (g_thread.joinable()) {
        g_thread.join();
    }
}

static void PreviewThreadFunc() {
    if (!CreatePreviewWindow()) {
        g_running.store(false);
        return;
    }

    if (!InitD2D()) {
        DestroyWindow(g_hwnd);
        g_running.store(false);
        return;
    }

    MSG msg = {};
    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    while (g_running.load()) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                g_running.store(false);
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (!g_running.load()) break;

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrameTime).count();

        if (elapsed < 33) {  // ~30 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(33 - elapsed));
        }
        lastFrameTime = std::chrono::high_resolution_clock::now();

        // Render if window is visible (check actual window state, not just flag)
        if (g_visible.load() && g_hwnd && IsWindowVisible(g_hwnd)) {
            RenderFrame();
        } else if (g_visible.load() && g_hwnd && !IsWindowVisible(g_hwnd)) {
            // Window should be visible but isn't - force show
            ShowWindow(g_hwnd, SW_SHOW);
            SetWindowPos(g_hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
        }
    }

    CleanupD2D();

    if (g_hwnd) {
        DestroyWindow(g_hwnd);
        g_hwnd = nullptr;
    }

    UnregisterClassW(L"PreviewWindowClass", GetModuleHandle(nullptr));
}

static bool CreatePreviewWindow() {
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = PreviewWndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"PreviewWindowClass";

    if (!RegisterClassExW(&wc)) {
        DWORD err = GetLastError();
        if (err != ERROR_CLASS_ALREADY_EXISTS) {
            return false;
        }
    }

    // Independent window (always on top, completely separate from main overlay)
    g_hwnd = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_APPWINDOW | WS_EX_WINDOWEDGE,  // Always on top, appears in taskbar
        L"PreviewWindowClass",
        L"Preview - NeedAimBot",
        WS_OVERLAPPEDWINDOW,  // Initially hidden
        100, 100,  // Fixed position instead of CW_USEDEFAULT
        INITIAL_WIDTH, INITIAL_HEIGHT,
        nullptr, nullptr,
        GetModuleHandle(nullptr),
        nullptr
    );

    if (!g_hwnd) {
        return false;
    }

    g_windowCreated.store(true);
    
    // Initially hidden
    ShowWindow(g_hwnd, SW_HIDE);

    return true;
}

static bool InitD2D() {
    HRESULT hr;

    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &g_d2dFactory);
    if (FAILED(hr)) return false;

    hr = DWriteCreateFactory(
        DWRITE_FACTORY_TYPE_SHARED,
        __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(&g_dwriteFactory)
    );
    if (FAILED(hr)) return false;

    hr = g_dwriteFactory->CreateTextFormat(
        L"Segoe UI",
        nullptr,
        DWRITE_FONT_WEIGHT_SEMI_BOLD,
        DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL,
        14.0f,
        L"en-us",
        &g_textFormat
    );
    if (FAILED(hr)) return false;

    g_textFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    g_textFormat->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);
    
    // Small text format for settings
    hr = g_dwriteFactory->CreateTextFormat(
        L"Segoe UI",
        nullptr,
        DWRITE_FONT_WEIGHT_NORMAL,
        DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL,
        11.0f,
        L"en-us",
        &g_textFormatSmall
    );
    if (FAILED(hr)) return false;
    
    g_textFormatSmall->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    g_textFormatSmall->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);

    RECT rc;
    GetClientRect(g_hwnd, &rc);
    D2D1_SIZE_U size = D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top);

    D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED)
    );
    D2D1_HWND_RENDER_TARGET_PROPERTIES hwndProps = D2D1::HwndRenderTargetProperties(g_hwnd, size);

    hr = g_d2dFactory->CreateHwndRenderTarget(rtProps, hwndProps, &g_renderTarget);
    if (FAILED(hr)) return false;

    g_renderTarget->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_CLEARTYPE);

    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 1.0f, 0.0f, 1.0f), &g_brushGreen);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 0.0f, 1.0f), &g_brushYellow);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 0.0f, 0.0f, 1.0f), &g_brushRed);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 1.0f, 1.0f), &g_brushWhite);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.8f), &g_brushBackground);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.5f, 0.5f, 0.5f, 1.0f), &g_brushGray);
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.3f, 0.5f, 0.8f, 1.0f), &g_brushBlue);

    return true;
}

static void CleanupD2D() {
    if (g_frameBitmap) { g_frameBitmap->Release(); g_frameBitmap = nullptr; }
    if (g_brushBlue) { g_brushBlue->Release(); g_brushBlue = nullptr; }
    if (g_brushGray) { g_brushGray->Release(); g_brushGray = nullptr; }
    if (g_brushBackground) { g_brushBackground->Release(); g_brushBackground = nullptr; }
    if (g_brushWhite) { g_brushWhite->Release(); g_brushWhite = nullptr; }
    if (g_brushRed) { g_brushRed->Release(); g_brushRed = nullptr; }
    if (g_brushYellow) { g_brushYellow->Release(); g_brushYellow = nullptr; }
    if (g_brushGreen) { g_brushGreen->Release(); g_brushGreen = nullptr; }
    if (g_textFormatSmall) { g_textFormatSmall->Release(); g_textFormatSmall = nullptr; }
    if (g_textFormat) { g_textFormat->Release(); g_textFormat = nullptr; }
    if (g_renderTarget) { g_renderTarget->Release(); g_renderTarget = nullptr; }
    if (g_dwriteFactory) { g_dwriteFactory->Release(); g_dwriteFactory = nullptr; }
    if (g_d2dFactory) { g_d2dFactory->Release(); g_d2dFactory = nullptr; }
    g_bitmapW = 0;
    g_bitmapH = 0;
}

static void ResizeRenderTarget() {
    if (!g_renderTarget) return;
    RECT rc;
    GetClientRect(g_hwnd, &rc);
    D2D1_SIZE_U size = D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top);
    g_renderTarget->Resize(size);
}

// Upload SimpleMat (RGBA) to D2D bitmap
static bool UpdateBitmap(const SimpleMat& frame) {
    if (frame.empty() || !g_renderTarget) return false;

    int w = frame.cols();
    int h = frame.rows();
    if (w <= 0 || h <= 0 || w > 10000 || h > 10000) return false;

    // RGBA -> BGRA conversion buffer
    // D2D expects BGRA, SimpleMat provides RGBA
    size_t rowBytes = w * 4;
    std::vector<uint8_t> bgraData(rowBytes * h);

    const uint8_t* src = frame.data();
    size_t srcStep = frame.step();

    for (int y = 0; y < h; ++y) {
        const uint8_t* srcRow = src + srcStep * y;
        uint8_t* dstRow = bgraData.data() + rowBytes * y;
        for (int x = 0; x < w; ++x) {
            dstRow[x * 4 + 0] = srcRow[x * 4 + 2]; // B <- R
            dstRow[x * 4 + 1] = srcRow[x * 4 + 1]; // G
            dstRow[x * 4 + 2] = srcRow[x * 4 + 0]; // R <- B
            dstRow[x * 4 + 3] = srcRow[x * 4 + 3]; // A
        }
    }

    // Recreate bitmap if size changed
    if (!g_frameBitmap || w != g_bitmapW || h != g_bitmapH) {
        if (g_frameBitmap) { g_frameBitmap->Release(); g_frameBitmap = nullptr; }

        D2D1_BITMAP_PROPERTIES bmpProps = D2D1::BitmapProperties(
            D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED)
        );

        HRESULT hr = g_renderTarget->CreateBitmap(
            D2D1::SizeU(w, h),
            bgraData.data(),
            static_cast<UINT32>(rowBytes),
            bmpProps,
            &g_frameBitmap
        );
        if (FAILED(hr)) return false;

        g_bitmapW = w;
        g_bitmapH = h;
    } else {
        D2D1_RECT_U rect = D2D1::RectU(0, 0, w, h);
        g_frameBitmap->CopyFromMemory(&rect, bgraData.data(), static_cast<UINT32>(rowBytes));
    }

    return true;
}

static void RenderFrame() {
    if (!g_renderTarget) return;

    auto& ctx = AppContext::getInstance();

    // Get preview frame from pipeline
    static SimpleMat previewFrame;
    static auto lastUpdate = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate);

    auto& pipelineManager = gpa::PipelineManager::getInstance();
    auto* pipeline = pipelineManager.getPipeline();

    if (pipeline && pipeline->isPreviewAvailable() && elapsed.count() >= 33) {
        if (pipeline->getPreviewSnapshot(previewFrame) &&
            !previewFrame.empty() &&
            previewFrame.cols() > 0 && previewFrame.rows() > 0 &&
            previewFrame.cols() <= 10000 && previewFrame.rows() <= 10000) {
            lastUpdate = now;
        }
    }

    bool hasFrame = UpdateBitmap(previewFrame);

    // Get window client size
    RECT rc;
    GetClientRect(g_hwnd, &rc);
    float clientW = static_cast<float>(rc.right - rc.left);
    float clientH = static_cast<float>(rc.bottom - rc.top);

    // Layout: Image LEFT, Settings RIGHT
    const float settingsPanelW = 280.0f;
    const float imageAreaW = clientW - settingsPanelW;
    const float statusBarH = 24.0f;
    const float availH = clientH - statusBarH;

    g_renderTarget->BeginDraw();
    g_renderTarget->Clear(D2D1::ColorF(0.1f, 0.1f, 0.1f, 1.0f));  // Dark background

    float scale = 1.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;

    if (hasFrame && g_frameBitmap && g_bitmapW > 0 && g_bitmapH > 0) {
        // Fit image into LEFT area, maintaining aspect ratio
        float scaleX = imageAreaW / static_cast<float>(g_bitmapW);
        float scaleY = availH / static_cast<float>(g_bitmapH);
        scale = (scaleX < scaleY) ? scaleX : scaleY;
        if (scale <= 0) scale = 1.0f;

        float drawW = g_bitmapW * scale;
        float drawH = g_bitmapH * scale;
        offsetX = (clientW - drawW) / 2.0f;
        offsetY = (availH - drawH) / 2.0f;

        D2D1_RECT_F destRect = D2D1::RectF(offsetX, offsetY, offsetX + drawW, offsetY + drawH);
        g_renderTarget->DrawBitmap(g_frameBitmap, destRect);

        // Draw crosshair at center of image
        float cx = offsetX + drawW / 2.0f;
        float cy = offsetY + drawH / 2.0f;
        g_renderTarget->DrawLine(D2D1::Point2F(cx - 10, cy), D2D1::Point2F(cx + 10, cy), g_brushWhite, 1.0f);
        g_renderTarget->DrawLine(D2D1::Point2F(cx, cy - 10), D2D1::Point2F(cx, cy + 10), g_brushWhite, 1.0f);

        // Draw detection boxes
        std::vector<Target> targets = ctx.getAllTargets();
        Target bestTarget = ctx.getBestTarget();
        bool hasBest = ctx.hasValidTarget();
        float confThreshold = ctx.config.profile().confidence_threshold;
        int iconClassFilter = ctx.config.global().preview_icon_class;

        for (const auto& target : targets) {
            if (target.width <= 0 || target.height <= 0) continue;

            // Class filter: skip if filter is set and class doesn't match
            if (iconClassFilter >= 0 && target.classId != iconClassFilter) continue;

            float sx = offsetX + target.x * scale;
            float sy = offsetY + target.y * scale;
            float sw = target.width * scale;
            float sh = target.height * scale;

            bool isBest = hasBest &&
                (std::abs(target.x - bestTarget.x) < 1.0f) &&
                (std::abs(target.y - bestTarget.y) < 1.0f);

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

            D2D1_RECT_F boxRect = D2D1::RectF(sx, sy, sx + sw, sy + sh);
            g_renderTarget->DrawRectangle(boxRect, boxBrush, thickness);

            // Confidence label
            char labelBuf[64];
            snprintf(labelBuf, sizeof(labelBuf), "%.0f%%", target.confidence * 100.0f);
            wchar_t wlabel[64];
            MultiByteToWideChar(CP_UTF8, 0, labelBuf, -1, wlabel, 64);

            D2D1_RECT_F textBgRect = D2D1::RectF(sx, sy - 16, sx + 44, sy);
            g_renderTarget->FillRectangle(textBgRect, g_brushBackground);
            D2D1_RECT_F textRect = D2D1::RectF(sx + 2, sy - 15, sx + 42, sy);
            g_renderTarget->DrawTextW(wlabel, static_cast<UINT32>(wcslen(wlabel)), g_textFormat, textRect, boxBrush);

            if (isBest) {
                wchar_t targetLabel[] = L"TARGET";
                D2D1_RECT_F tBgRect = D2D1::RectF(sx, sy - 32, sx + 56, sy - 16);
                g_renderTarget->FillRectangle(tBgRect, g_brushBackground);
                D2D1_RECT_F tTextRect = D2D1::RectF(sx + 2, sy - 31, sx + 54, sy - 16);
                g_renderTarget->DrawTextW(targetLabel, static_cast<UINT32>(wcslen(targetLabel)), g_textFormat, tTextRect, g_brushGreen);
            }
        }
    } else {
        // No frame yet
        wchar_t waitText[] = L"Waiting for capture...";
        D2D1_RECT_F textRect = D2D1::RectF(10, availH / 2 - 10, clientW - 10, availH / 2 + 10);
        g_renderTarget->DrawTextW(waitText, static_cast<UINT32>(wcslen(waitText)), g_textFormat, textRect, g_brushWhite);
    }

    // Status bar at bottom
    {
        std::vector<Target> targets = ctx.getAllTargets();
        int captureSize = ctx.config.profile().detection_resolution;
        float confThreshold = ctx.config.profile().confidence_threshold;

        D2D1_RECT_F barRect = D2D1::RectF(0, clientH - statusBarH, clientW, clientH);
        g_renderTarget->FillRectangle(barRect, g_brushBackground);

        wchar_t statusBuf[256];
        swprintf_s(statusBuf, L"Targets: %zu | Capture: %dx%d | Conf: %.0f%%",
            targets.size(), captureSize, captureSize, confThreshold * 100.0f);

        D2D1_RECT_F statusTextRect = D2D1::RectF(8, clientH - statusBarH + 4, imageAreaW - 8, clientH);
        g_renderTarget->DrawTextW(statusBuf, static_cast<UINT32>(wcslen(statusBuf)), g_textFormat, statusTextRect, g_brushWhite);
    }
    
    // Settings panel on the RIGHT
    RenderSettingsPanel(imageAreaW, 0, settingsPanelW, clientH);

    g_renderTarget->EndDraw();
}

// Helper function to draw a slider
static bool DrawSlider(float x, float y, float w, const wchar_t* label, float* value, float minVal, float maxVal, int sliderId) {
    const float sliderH = 20.0f;
    const float labelH = 16.0f;
    bool changed = false;
    
    // Label
    D2D1_RECT_F labelRect = D2D1::RectF(x, y, x + w, y + labelH);
    g_renderTarget->DrawTextW(label, static_cast<UINT32>(wcslen(label)), g_textFormatSmall, labelRect, g_brushWhite);
    
    // Slider track
    D2D1_RECT_F trackRect = D2D1::RectF(x, y + labelH + 2, x + w, y + labelH + 2 + sliderH);
    g_renderTarget->FillRectangle(trackRect, g_brushGray);
    
    // Slider handle
    float normalizedValue = (*value - minVal) / (maxVal - minVal);
    if (normalizedValue < 0) normalizedValue = 0;
    if (normalizedValue > 1) normalizedValue = 1;
    float handleX = x + normalizedValue * w;
    D2D1_RECT_F handleRect = D2D1::RectF(handleX - 4, y + labelH, handleX + 4, y + labelH + 4 + sliderH);
    g_renderTarget->FillRectangle(handleRect, g_brushBlue);
    
    // Value text
    wchar_t valueBuf[32];
    swprintf_s(valueBuf, L"%.2f", *value);
    D2D1_RECT_F valueRect = D2D1::RectF(x, y + labelH + sliderH + 4, x + w, y + labelH + sliderH + 20);
    g_renderTarget->DrawTextW(valueBuf, static_cast<UINT32>(wcslen(valueBuf)), g_textFormatSmall, valueRect, g_brushWhite);
    
    // Handle mouse interaction
    if (g_activeSlider == sliderId && g_mouseDown) {
        float mouseX = static_cast<float>(g_mousePos.x);
        float newNormalized = (mouseX - x) / w;
        if (newNormalized < 0) newNormalized = 0;
        if (newNormalized > 1) newNormalized = 1;
        float newValue = minVal + newNormalized * (maxVal - minVal);
        if (newValue != *value) {
            *value = newValue;
            changed = true;
        }
    }
    
    return changed;
}

// Helper function to draw an integer slider
static bool DrawIntSlider(float x, float y, float w, const wchar_t* label, int* value, int minVal, int maxVal, int sliderId) {
    const float sliderH = 20.0f;
    const float labelH = 16.0f;
    bool changed = false;
    
    // Label
    D2D1_RECT_F labelRect = D2D1::RectF(x, y, x + w, y + labelH);
    g_renderTarget->DrawTextW(label, static_cast<UINT32>(wcslen(label)), g_textFormatSmall, labelRect, g_brushWhite);
    
    // Slider track
    D2D1_RECT_F trackRect = D2D1::RectF(x, y + labelH + 2, x + w, y + labelH + 2 + sliderH);
    g_renderTarget->FillRectangle(trackRect, g_brushGray);
    
    // Slider handle
    float normalizedValue = static_cast<float>(*value - minVal) / static_cast<float>(maxVal - minVal);
    if (normalizedValue < 0) normalizedValue = 0;
    if (normalizedValue > 1) normalizedValue = 1;
    float handleX = x + normalizedValue * w;
    D2D1_RECT_F handleRect = D2D1::RectF(handleX - 4, y + labelH, handleX + 4, y + labelH + 4 + sliderH);
    g_renderTarget->FillRectangle(handleRect, g_brushBlue);
    
    // Value text
    wchar_t valueBuf[32];
    swprintf_s(valueBuf, L"%d", *value);
    D2D1_RECT_F valueRect = D2D1::RectF(x, y + labelH + sliderH + 4, x + w, y + labelH + sliderH + 20);
    g_renderTarget->DrawTextW(valueBuf, static_cast<UINT32>(wcslen(valueBuf)), g_textFormatSmall, valueRect, g_brushWhite);
    
    // Handle mouse interaction
    if (g_activeSlider == sliderId && g_mouseDown) {
        float mouseX = static_cast<float>(g_mousePos.x);
        float newNormalized = (mouseX - x) / w;
        if (newNormalized < 0) newNormalized = 0;
        if (newNormalized > 1) newNormalized = 1;
        int newValue = minVal + static_cast<int>(newNormalized * static_cast<float>(maxVal - minVal));
        if (newValue != *value) {
            *value = newValue;
            changed = true;
        }
    }
    
    return changed;
}

// Helper function to draw a checkbox
static bool DrawCheckbox(float x, float y, const wchar_t* label, bool* value, int checkboxId) {
    const float boxSize = 16.0f;
    const float labelMargin = 6.0f;
    bool changed = false;
    
    // Checkbox box
    D2D1_RECT_F boxRect = D2D1::RectF(x, y, x + boxSize, y + boxSize);
    g_renderTarget->FillRectangle(boxRect, g_brushGray);
    
    // Check mark (X) if enabled
    if (*value) {
        g_renderTarget->DrawLine(D2D1::Point2F(x + 3, y + 3), D2D1::Point2F(x + boxSize - 3, y + boxSize - 3), g_brushGreen, 2.0f);
        g_renderTarget->DrawLine(D2D1::Point2F(x + boxSize - 3, y + 3), D2D1::Point2F(x + 3, y + boxSize - 3), g_brushGreen, 2.0f);
    }
    
    // Label
    D2D1_RECT_F labelRect = D2D1::RectF(x + boxSize + labelMargin, y, x + 250, y + boxSize);
    g_renderTarget->DrawTextW(label, static_cast<UINT32>(wcslen(label)), g_textFormatSmall, labelRect, g_brushWhite);
    
    // Handle mouse click
    if (g_activeCheckbox == checkboxId && g_mouseDown) {
        *value = !(*value);
        changed = true;
        g_activeCheckbox = -1; // Reset to prevent multiple toggles
    }
    
    return changed;
}

// Helper function to draw a tab button
static bool DrawTabButton(float x, float y, float w, float h, const wchar_t* label, bool isActive) {
    // Button background
    ID2D1SolidColorBrush* bgBrush = isActive ? g_brushBlue : g_brushGray;
    D2D1_RECT_F btnRect = D2D1::RectF(x, y, x + w, y + h);
    g_renderTarget->FillRectangle(btnRect, bgBrush);
    
    // Border
    g_renderTarget->DrawRectangle(btnRect, g_brushWhite, 1.0f);
    
    // Label
    g_renderTarget->DrawTextW(label, static_cast<UINT32>(wcslen(label)), g_textFormatSmall, btnRect, g_brushWhite);
    
    return false; // Handled in mouse click
}

// Render Aim Settings Tab
static void RenderAimTab(float panelX, float panelY, float panelW, float panelH) {
    auto& ctx = AppContext::getInstance();
    float cursorY = panelY + 50.0f; // Below tabs
    const float padding = 12.0f;
    const float itemW = panelW - padding * 2;
    
    // Crosshair Offsets
    if (DrawSlider(panelX + padding, cursorY, itemW, L"Crosshair X", &ctx.config.profile().crosshair_offset_x, -100.0f, 100.0f, 0)) {
        ctx.config.saveConfig();
    }
    cursorY += 55;
    
    if (DrawSlider(panelX + padding, cursorY, itemW, L"Crosshair Y", &ctx.config.profile().crosshair_offset_y, -100.0f, 100.0f, 1)) {
        ctx.config.saveConfig();
    }
    cursorY += 60;
    
    // Body/Head Offsets
    if (DrawSlider(panelX + padding, cursorY, itemW, L"Body Y Offset", &ctx.config.profile().body_y_offset, -1.0f, 1.0f, 2)) {
        ctx.config.saveConfig();
    }
    cursorY += 55;
    
    if (DrawSlider(panelX + padding, cursorY, itemW, L"Head Y Offset", &ctx.config.profile().head_y_offset, -1.0f, 1.0f, 3)) {
        ctx.config.saveConfig();
    }
    cursorY += 60;
    
    // Aim+Shoot Offset Enable
    if (DrawCheckbox(panelX + padding, cursorY, L"Enable Aim+Shoot Offset", &ctx.config.profile().enable_aim_shoot_offset, 0)) {
        ctx.config.saveConfig();
    }
    cursorY += 30;
    
    // Aim+Shoot Offsets (only if enabled)
    if (ctx.config.profile().enable_aim_shoot_offset) {
        if (DrawSlider(panelX + padding + 20, cursorY, itemW - 20, L"Aim+Shoot X", &ctx.config.profile().aim_shoot_offset_x, -50.0f, 50.0f, 4)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        
        if (DrawSlider(panelX + padding + 20, cursorY, itemW - 20, L"Aim+Shoot Y", &ctx.config.profile().aim_shoot_offset_y, -50.0f, 50.0f, 5)) {
            ctx.config.saveConfig();
        }
    }
}

// Render Color Filter Tab
static void RenderColorFilterTab(float panelX, float panelY, float panelW, float panelH) {
    auto& ctx = AppContext::getInstance();
    float cursorY = panelY + 50.0f; // Below tabs
    const float padding = 12.0f;
    const float itemW = panelW - padding * 2;
    
    // Enable checkbox
    if (DrawCheckbox(panelX + padding, cursorY, L"Enable Color Filter", &ctx.config.profile().color_filter_enabled, 1)) {
        ctx.config.saveConfig();
    }
    cursorY += 30;
    
    if (!ctx.config.profile().color_filter_enabled) {
        // Show disabled message
        wchar_t msg[] = L"(Filter disabled)";
        D2D1_RECT_F msgRect = D2D1::RectF(panelX + padding, cursorY, panelX + panelW - padding, cursorY + 20);
        g_renderTarget->DrawTextW(msg, static_cast<UINT32>(wcslen(msg)), g_textFormatSmall, msgRect, g_brushGray);
        return;
    }
    
    // Mode selection (RGB=0, HSV=1)
    wchar_t modeLabel[32];
    swprintf_s(modeLabel, L"Mode: %s", ctx.config.profile().color_filter_mode == 0 ? L"RGB" : L"HSV");
    D2D1_RECT_F modeRect = D2D1::RectF(panelX + padding, cursorY, panelX + panelW - padding, cursorY + 18);
    g_renderTarget->DrawTextW(modeLabel, static_cast<UINT32>(wcslen(modeLabel)), g_textFormatSmall, modeRect, g_brushWhite);
    cursorY += 25;
    
    // Toggle mode button (simple click area)
    wchar_t toggleBtn[] = L"[Switch Mode]";
    D2D1_RECT_F toggleRect = D2D1::RectF(panelX + padding, cursorY, panelX + panelW - padding, cursorY + 18);
    g_renderTarget->DrawTextW(toggleBtn, static_cast<UINT32>(wcslen(toggleBtn)), g_textFormatSmall, toggleRect, g_brushBlue);
    // TODO: Add click detection in HandleMouseClick
    cursorY += 30;
    
    if (ctx.config.profile().color_filter_mode == 0) {
        // RGB Mode
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"R Min", &ctx.config.profile().color_filter_r_min, 0, 255, 10)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"R Max", &ctx.config.profile().color_filter_r_max, 0, 255, 11)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"G Min", &ctx.config.profile().color_filter_g_min, 0, 255, 12)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"G Max", &ctx.config.profile().color_filter_g_max, 0, 255, 13)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"B Min", &ctx.config.profile().color_filter_b_min, 0, 255, 14)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"B Max", &ctx.config.profile().color_filter_b_max, 0, 255, 15)) {
            ctx.config.saveConfig();
        }
    } else {
        // HSV Mode
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"H Min", &ctx.config.profile().color_filter_h_min, 0, 179, 20)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"H Max", &ctx.config.profile().color_filter_h_max, 0, 179, 21)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"S Min", &ctx.config.profile().color_filter_s_min, 0, 255, 22)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"S Max", &ctx.config.profile().color_filter_s_max, 0, 255, 23)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"V Min", &ctx.config.profile().color_filter_v_min, 0, 255, 24)) {
            ctx.config.saveConfig();
        }
        cursorY += 55;
        if (DrawIntSlider(panelX + padding, cursorY, itemW, L"V Max", &ctx.config.profile().color_filter_v_max, 0, 255, 25)) {
            ctx.config.saveConfig();
        }
    }
}

static void RenderSettingsPanel(float panelX, float panelY, float panelW, float panelH) {
    // Panel background (dark gray)
    D2D1_RECT_F panelRect = D2D1::RectF(panelX, panelY, panelX + panelW, panelY + panelH);
    
    // Create temporary brush for panel background
    ID2D1SolidColorBrush* tempBrush = nullptr;
    g_renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.15f, 0.15f, 0.15f, 1.0f), &tempBrush);
    if (tempBrush) {
        g_renderTarget->FillRectangle(panelRect, tempBrush);
        tempBrush->Release();
    }
    
    // Draw vertical divider
    g_renderTarget->DrawLine(D2D1::Point2F(panelX, 0), D2D1::Point2F(panelX, panelH), g_brushGray, 1.0f);
    
    const float padding = 12.0f;
    const float tabH = 32.0f;
    const float tabW = panelW / 2.0f;
    
    // Tab buttons
    DrawTabButton(panelX, panelY + 5, tabW, tabH, L"Aim Settings", g_activeTab == 0);
    DrawTabButton(panelX + tabW, panelY + 5, tabW, tabH, L"Color Filter", g_activeTab == 1);
    
    // Render active tab content
    if (g_activeTab == 0) {
        RenderAimTab(panelX, panelY, panelW, panelH);
    } else if (g_activeTab == 1) {
        RenderColorFilterTab(panelX, panelY, panelW, panelH);
    }
}

static void HandleMouseClick(int x, int y) {
    RECT rc;
    GetClientRect(g_hwnd, &rc);
    float clientW = static_cast<float>(rc.right - rc.left);
    const float settingsPanelW = 280.0f;
    const float imageAreaW = clientW - settingsPanelW;
    
    g_activeSlider = -1;
    g_activeCheckbox = -1;
    
    // Check if click is in settings panel area
    if (x >= imageAreaW) {
        float panelX = imageAreaW;
        const float tabH = 32.0f;
        const float tabW = settingsPanelW / 2.0f;
        
        // Check tab clicks (Y: 5 to 37)
        if (y >= 5 && y <= 5 + tabH) {
            if (x >= panelX && x < panelX + tabW) {
                g_activeTab = 0; // Aim Settings
                g_mouseDown = true;
                return;
            } else if (x >= panelX + tabW && x < panelX + settingsPanelW) {
                g_activeTab = 1; // Color Filter
                g_mouseDown = true;
                return;
            }
        }
        
        // Tab content clicks (below tabs, Y > 50)
        if (g_activeTab == 0) {
            // Aim Settings Tab
            // Slider positions (approximate based on cursorY increments)
            float y1 = 50;   // Crosshair X
            float y2 = 105;  // Crosshair Y
            float y3 = 165;  // Body Y
            float y4 = 220;  // Head Y
            float y5 = 280;  // Aim+Shoot Enable checkbox
            float y6 = 310;  // Aim+Shoot X slider
            float y7 = 365;  // Aim+Shoot Y slider
            
            if (y >= y1 && y < y1 + 40) g_activeSlider = 0;
            else if (y >= y2 && y < y2 + 40) g_activeSlider = 1;
            else if (y >= y3 && y < y3 + 40) g_activeSlider = 2;
            else if (y >= y4 && y < y4 + 40) g_activeSlider = 3;
            else if (y >= y5 && y < y5 + 20) g_activeCheckbox = 0; // Enable Aim+Shoot
            else if (y >= y6 && y < y6 + 40) g_activeSlider = 4;
            else if (y >= y7 && y < y7 + 40) g_activeSlider = 5;
        } else if (g_activeTab == 1) {
            // Color Filter Tab
            float y1 = 50;  // Enable checkbox
            
            if (y >= y1 && y < y1 + 20) {
                g_activeCheckbox = 1; // Color Filter Enable
            } else {
                // Mode switch button
                float y2 = 105; // Mode switch area
                if (y >= y2 && y < y2 + 18) {
                    // Toggle color filter mode
                    auto& ctx = AppContext::getInstance();
                    ctx.config.profile().color_filter_mode = (ctx.config.profile().color_filter_mode == 0) ? 1 : 0;
                    ctx.config.saveConfig();
                    g_mouseDown = true;
                    return;
                }
                
                // Sliders (RGB or HSV based on mode)
                float sliderStart = 135;
                for (int i = 0; i < 6; i++) {
                    float yPos = sliderStart + i * 55;
                    if (y >= yPos && y < yPos + 40) {
                        g_activeSlider = 10 + i; // Slider IDs 10-15 for RGB, 20-25 for HSV
                        break;
                    }
                }
            }
        }
    }
    
    g_mouseDown = true;
}

static void HandleMouseMove(int x, int y) {
    g_mousePos.x = x;
    g_mousePos.y = y;
}

static void HandleMouseRelease() {
    g_activeSlider = -1;
    g_mouseDown = false;
}

static LRESULT CALLBACK PreviewWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_LBUTTONDOWN:
            {
                int x = LOWORD(lParam);
                int y = HIWORD(lParam);
                HandleMouseClick(x, y);
            }
            return 0;
        case WM_MOUSEMOVE:
            {
                int x = LOWORD(lParam);
                int y = HIWORD(lParam);
                HandleMouseMove(x, y);
            }
            return 0;
        case WM_LBUTTONUP:
            HandleMouseRelease();
            return 0;
        case WM_SIZE:
            ResizeRenderTarget();
            return 0;
        case WM_CLOSE:
            // User closed the window -> hide and update config
            g_visible.store(false);
            ShowWindow(hwnd, SW_HIDE);
            {
                auto& ctx = AppContext::getInstance();
                ctx.config.global().show_preview_window = false;
                ctx.config.saveConfig();
            }
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_PAINT:
            ValidateRect(hwnd, nullptr);
            return 0;
        case (WM_APP + 1):
            // Custom message for show/hide
            if (wParam) {
                ShowWindow(hwnd, SW_SHOW);
                SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
                SetForegroundWindow(hwnd);
            } else {
                ShowWindow(hwnd, SW_HIDE);
            }
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

} // namespace PreviewWindow
