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
static const int INITIAL_WIDTH = 480;
static const int INITIAL_HEIGHT = 520;

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
static ID2D1Bitmap* g_frameBitmap = nullptr;
static int g_bitmapW = 0;
static int g_bitmapH = 0;

// Forward declarations
static LRESULT CALLBACK PreviewWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
static bool CreatePreviewWindow();
static bool InitD2D();
static void CleanupD2D();
static void ResizeRenderTarget();
static void RenderFrame();
static void PreviewThreadFunc();

bool IsVisible() {
    return g_visible.load();
}

bool IsRunning() {
    return g_running.load();
}

void SetVisible(bool visible) {
    g_visible.store(visible);
    if (g_hwnd) {
        PostMessage(g_hwnd, WM_APP + 1, visible ? 1 : 0, 0);
    }
}

void Start() {
    if (g_running.load()) return;

    g_running.store(true);
    g_visible.store(false);

    g_thread = std::thread(PreviewThreadFunc);

    std::cout << "[PreviewWindow] Started" << std::endl;
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

    std::cout << "[PreviewWindow] Stopped" << std::endl;
}

static void PreviewThreadFunc() {
    if (!CreatePreviewWindow()) {
        std::cerr << "[PreviewWindow] Failed to create window" << std::endl;
        g_running.store(false);
        return;
    }

    if (!InitD2D()) {
        std::cerr << "[PreviewWindow] Failed to init D2D" << std::endl;
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

        if (g_visible.load()) {
            RenderFrame();
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

    // Normal resizable window
    g_hwnd = CreateWindowExW(
        WS_EX_TOOLWINDOW,
        L"PreviewWindowClass",
        L"Preview",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        INITIAL_WIDTH, INITIAL_HEIGHT,
        nullptr, nullptr,
        GetModuleHandle(nullptr),
        nullptr
    );

    if (!g_hwnd) {
        return false;
    }

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

    return true;
}

static void CleanupD2D() {
    if (g_frameBitmap) { g_frameBitmap->Release(); g_frameBitmap = nullptr; }
    if (g_brushBackground) { g_brushBackground->Release(); g_brushBackground = nullptr; }
    if (g_brushWhite) { g_brushWhite->Release(); g_brushWhite = nullptr; }
    if (g_brushRed) { g_brushRed->Release(); g_brushRed = nullptr; }
    if (g_brushYellow) { g_brushYellow->Release(); g_brushYellow = nullptr; }
    if (g_brushGreen) { g_brushGreen->Release(); g_brushGreen = nullptr; }
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

    // Reserve space for status bar
    float statusBarH = 24.0f;
    float availH = clientH - statusBarH;
    if (availH < 10) availH = 10;

    g_renderTarget->BeginDraw();
    g_renderTarget->Clear(D2D1::ColorF(0.1f, 0.1f, 0.1f, 1.0f));  // Dark background

    float scale = 1.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;

    if (hasFrame && g_frameBitmap && g_bitmapW > 0 && g_bitmapH > 0) {
        // Fit image into available area, maintaining aspect ratio
        float scaleX = clientW / static_cast<float>(g_bitmapW);
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

        D2D1_RECT_F statusTextRect = D2D1::RectF(8, clientH - statusBarH + 4, clientW - 8, clientH);
        g_renderTarget->DrawTextW(statusBuf, static_cast<UINT32>(wcslen(statusBuf)), g_textFormat, statusTextRect, g_brushWhite);
    }

    g_renderTarget->EndDraw();
}

static LRESULT CALLBACK PreviewWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
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
                ShowWindow(hwnd, SW_SHOWNA);
            } else {
                ShowWindow(hwnd, SW_HIDE);
            }
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

} // namespace PreviewWindow
