#include "../core/windows_headers.h"

#include <tchar.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <d3d11.h>
#include <dxgi.h>
#include <filesystem>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_dx11.h>
#include <imgui/imgui_impl_win32.h>
#include <imgui/imgui_internal.h>

#include "AppContext.h"
#include "overlay.h"
#include "mouse/mouse.h"
#include "overlay/draw_settings.h"
#include "overlay/preview_window.h"
#include "overlay/draw_offset.h"
#include "overlay/ui_helpers.h"
#include "config.h"
#include "keycodes.h"
#include "needaimbot.h"
// #include "capture/capture.h" - removed, using GPU capture now
#include "keyboard_listener.h"
#include "other_tools.h"
#include "../core/constants.h"
#include "../cuda/unified_graph_pipeline.h"
#include "../cuda/cuda_resource_manager.h"
#include <cuda_runtime.h>
#include <cstring>  // For std::memcpy

std::atomic<bool> g_config_optical_flow_changed{false};

// Auto-save state instance
AutoSaveState g_autoSaveState;

// Preview window state - now uses separate PreviewWindow class with its own HWND

ID3D11Device* g_pd3dDevice = NULL;
ID3D11DeviceContext* g_pd3dDeviceContext = NULL;
IDXGISwapChain* g_pSwapChain = NULL;
ID3D11RenderTargetView* g_mainRenderTargetView = NULL;
HWND g_hwnd = NULL;

// UI-Pipeline separation: GPU read infrastructure
struct UIGPUReader {
    cudaStream_t uiStream = nullptr;
    std::vector<Target> h_targets;  // Host buffer for targets
    int h_targetCount = 0;
    Target h_bestTarget = {};
    int h_bestTargetIndex = -1;
    cudaEvent_t copyCompleteEvent = nullptr;
    bool copyInProgress = false;
    uint64_t lastFrameRead = 0;
    
    void initialize() {
        if (!uiStream) {
            cudaStreamCreate(&uiStream);
        }
        if (!copyCompleteEvent) {
            cudaEventCreate(&copyCompleteEvent);
        }
        if (h_targets.empty()) {
            h_targets.resize(Constants::MAX_DETECTIONS);
        }
    }
    
    void cleanup() {
        if (uiStream) {
            cudaStreamDestroy(uiStream);
            uiStream = nullptr;
        }
        if (copyCompleteEvent) {
            cudaEventDestroy(copyCompleteEvent);
            copyCompleteEvent = nullptr;
        }
    }
    
    // Non-blocking read from GPU buffers
    bool tryReadFromGPU(gpa::UnifiedGraphPipeline* pipeline) {
        if (!pipeline || !uiStream) return false;
        
        // Check if new data is available
        if (!pipeline->hasNewFrameData()) {
            return false;  // No new data
        }
        
        // Check if previous copy is complete
        if (copyInProgress) {
            if (cudaEventQuery(copyCompleteEvent) != cudaSuccess) {
                return false;  // Previous copy still in progress
            }
            copyInProgress = false;
        }
        
        // Get GPU buffers
        auto gpuBuffers = pipeline->getUIGPUBuffers();
        
        // Start async copy from GPU to host
        cudaMemcpyAsync(&h_targetCount, gpuBuffers.finalTargetsCount, 
                       sizeof(int), cudaMemcpyDeviceToHost, uiStream);
        cudaMemcpyAsync(&h_bestTargetIndex, gpuBuffers.bestTargetIndex,
                       sizeof(int), cudaMemcpyDeviceToHost, uiStream);
        cudaMemcpyAsync(&h_bestTarget, gpuBuffers.bestTarget,
                       sizeof(Target), cudaMemcpyDeviceToHost, uiStream);
        
        // Copy targets (only up to MAX_DETECTIONS)
        int copyCount = (Constants::MAX_DETECTIONS < (int)h_targets.size()) ? Constants::MAX_DETECTIONS : (int)h_targets.size();
        cudaMemcpyAsync(h_targets.data(), gpuBuffers.finalTargets,
                       copyCount * sizeof(Target), cudaMemcpyDeviceToHost, uiStream);
        
        // Record event for completion check
        cudaEventRecord(copyCompleteEvent, uiStream);
        copyInProgress = true;
        
        // Mark frame as read
        pipeline->markUIFrameRead();
        lastFrameRead++;
        
        return true;
    }
    
    // Get the latest read data (if ready)
    bool getLatestData(std::vector<Target>& targets, int& targetCount, Target& bestTarget) {
        if (copyInProgress) {
            if (cudaEventQuery(copyCompleteEvent) != cudaSuccess) {
                return false;  // Copy not complete
            }
            copyInProgress = false;
        }
        
        if (h_targetCount > 0 && h_targetCount <= Constants::MAX_DETECTIONS) {
            // OPTIMIZATION: Resize instead of clear+push_back to avoid reallocations
            targets.resize(h_targetCount);
            // Direct memory copy instead of individual push_backs
            std::memcpy(targets.data(), h_targets.data(), h_targetCount * sizeof(Target));
            targetCount = h_targetCount;
            bestTarget = h_bestTarget;
            return true;
        }
        
        return false;
    }
};

static UIGPUReader g_uiGPUReader;

extern std::atomic<bool> should_exit;

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();

ID3D11BlendState* g_pBlendState = nullptr;
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int overlayWidth = 0;
int overlayHeight = 0;

std::vector<std::string> availableModels;
std::vector<std::string> key_names;
std::vector<const char*> key_names_cstrs;

ID3D11ShaderResourceView* body_texture = nullptr;

bool InitializeBlendState()
{
    D3D11_BLEND_DESC blendDesc;
    ZeroMemory(&blendDesc, sizeof(blendDesc));

    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    HRESULT hr = g_pd3dDevice->CreateBlendState(&blendDesc, &g_pBlendState);
    if (FAILED(hr))
    {
        return false;
    }

    float blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
    g_pd3dDeviceContext->OMSetBlendState(g_pBlendState, blendFactor, 0xffffffff);

    return true;
}

bool CreateDeviceD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 1; // Single buffer for better performance
    sd.BufferDesc.Width = overlayWidth;
    sd.BufferDesc.Height = overlayHeight;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 0; // Unlimited refresh rate
    sd.BufferDesc.RefreshRate.Denominator = 0;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1; // No multisampling for performance
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD; // Fastest swap effect
    sd.Flags = 0; // Remove unnecessary flags for performance

    UINT createDeviceFlags = 0;

    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] =
    {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_0,
    };

    HRESULT res = D3D11CreateDeviceAndSwapChain(NULL,
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        createDeviceFlags,
        featureLevelArray,
        2,
        D3D11_SDK_VERSION,
        &sd,
        &g_pSwapChain,
        &g_pd3dDevice,
        &featureLevel,
        &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    if (!InitializeBlendState())
        return false;

    CreateRenderTarget();
    return true;
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer = NULL;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = NULL; }
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
    if (g_pBlendState) { g_pBlendState->Release(); g_pBlendState = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            RECT rect;
            GetWindowRect(hWnd, &rect);
            UINT width = rect.right - rect.left;
            UINT height = rect.bottom - rect.top;

            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_DESTROY:
        should_exit = true;
        AppContext::getInstance().should_exit = true;
        AppContext::getInstance().frame_cv.notify_all();  // Wake up main thread
        ::PostQuitMessage(0);
        return 0;
    default:
        return ::DefWindowProc(hWnd, msg, wParam, lParam);
    }
}

// RoseDark theme
void ApplyTheme_RoseDark()
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Window
    style.WindowPadding = ImVec2(8.0f, 8.0f);
    style.WindowRounding = 6.0f;
    style.WindowBorderSize = 1.0f;
    style.WindowMinSize = ImVec2(32.0f, 32.0f);
    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);

    // Frame
    style.FramePadding = ImVec2(5.0f, 3.0f);
    style.FrameRounding = 4.0f;
    style.FrameBorderSize = 0.0f;

    // Items
    style.ItemSpacing = ImVec2(6.0f, 4.0f);
    style.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
    style.IndentSpacing = 20.0f;

    // Scrollbar
    style.ScrollbarSize = 12.0f;
    style.ScrollbarRounding = 9.0f;

    // Grab
    style.GrabMinSize = 8.0f;
    style.GrabRounding = 3.0f;

    // Tab
    style.TabRounding = 4.0f;
    style.TabBorderSize = 0.0f;

    // Child
    style.ChildRounding = 4.0f;
    style.ChildBorderSize = 1.0f;

    // Popup
    style.PopupRounding = 4.0f;
    style.PopupBorderSize = 1.0f;

    // Alpha
    style.Alpha = 1.0f;
    style.DisabledAlpha = 0.60f;

    // Colors - Rose Dark theme
    colors[ImGuiCol_Text] = ImVec4(0.95f, 0.92f, 0.93f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.45f, 0.47f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.08f, 0.09f, 0.98f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.12f, 0.10f, 0.11f, 0.90f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.06f, 0.07f, 0.98f);
    colors[ImGuiCol_Border] = ImVec4(0.35f, 0.20f, 0.25f, 0.50f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    colors[ImGuiCol_FrameBg] = ImVec4(0.18f, 0.12f, 0.14f, 0.90f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.28f, 0.18f, 0.22f, 0.95f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.38f, 0.22f, 0.28f, 1.00f);

    colors[ImGuiCol_TitleBg] = ImVec4(0.12f, 0.08f, 0.10f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.20f, 0.12f, 0.15f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.12f, 0.08f, 0.10f, 0.75f);

    colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.10f, 0.12f, 1.00f);

    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.08f, 0.09f, 0.60f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.45f, 0.25f, 0.32f, 0.80f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.55f, 0.30f, 0.38f, 0.90f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.65f, 0.35f, 0.45f, 1.00f);

    colors[ImGuiCol_CheckMark] = ImVec4(0.90f, 0.50f, 0.60f, 1.00f);

    colors[ImGuiCol_SliderGrab] = ImVec4(0.75f, 0.40f, 0.50f, 0.90f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.90f, 0.50f, 0.60f, 1.00f);

    colors[ImGuiCol_Button] = ImVec4(0.55f, 0.25f, 0.35f, 0.80f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.70f, 0.35f, 0.45f, 0.95f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.45f, 0.20f, 0.28f, 1.00f);

    colors[ImGuiCol_Header] = ImVec4(0.50f, 0.25f, 0.32f, 0.70f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.65f, 0.32f, 0.42f, 0.85f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.75f, 0.40f, 0.50f, 1.00f);

    colors[ImGuiCol_Separator] = ImVec4(0.35f, 0.20f, 0.25f, 0.60f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.70f, 0.40f, 0.50f, 0.78f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.85f, 0.50f, 0.60f, 1.00f);

    colors[ImGuiCol_ResizeGrip] = ImVec4(0.70f, 0.40f, 0.50f, 0.30f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.80f, 0.50f, 0.60f, 0.67f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.90f, 0.55f, 0.65f, 0.95f);

    colors[ImGuiCol_Tab] = ImVec4(0.18f, 0.12f, 0.14f, 0.90f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.65f, 0.35f, 0.45f, 0.80f);
    colors[ImGuiCol_TabActive] = ImVec4(0.55f, 0.28f, 0.38f, 1.00f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.12f, 0.08f, 0.10f, 0.97f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.40f, 0.22f, 0.30f, 1.00f);

    colors[ImGuiCol_PlotLines] = ImVec4(0.75f, 0.45f, 0.55f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.55f, 0.65f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.85f, 0.50f, 0.60f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.70f, 1.00f);

    colors[ImGuiCol_TableHeaderBg] = ImVec4(0.18f, 0.12f, 0.14f, 1.00f);
    colors[ImGuiCol_TableBorderStrong] = ImVec4(0.35f, 0.20f, 0.25f, 1.00f);
    colors[ImGuiCol_TableBorderLight] = ImVec4(0.25f, 0.15f, 0.18f, 1.00f);
    colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.04f);

    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.70f, 0.40f, 0.50f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 0.60f, 0.70f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ImVec4(0.85f, 0.50f, 0.60f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.10f, 0.15f, 0.35f);
}

// Cyber Minimal theme - Modern minimalist with Cyber Blue (#00D4FF)
void ApplyTheme_CyberMinimal()
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Modern minimalist spacing
    style.WindowPadding = ImVec2(12.0f, 12.0f);
    style.WindowRounding = 8.0f;
    style.WindowBorderSize = 1.0f;
    style.WindowMinSize = ImVec2(32.0f, 32.0f);
    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);

    // Clean frame styling
    style.FramePadding = ImVec2(8.0f, 4.0f);
    style.FrameRounding = 6.0f;
    style.FrameBorderSize = 0.0f;

    // Comfortable item spacing
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.IndentSpacing = 22.0f;

    // Sleek scrollbar
    style.ScrollbarSize = 10.0f;
    style.ScrollbarRounding = 12.0f;

    // Subtle grab
    style.GrabMinSize = 10.0f;
    style.GrabRounding = 4.0f;

    // Modern tabs
    style.TabRounding = 6.0f;
    style.TabBorderSize = 0.0f;

    // Clean child windows
    style.ChildRounding = 6.0f;
    style.ChildBorderSize = 1.0f;

    // Smooth popups
    style.PopupRounding = 6.0f;
    style.PopupBorderSize = 1.0f;

    style.Alpha = 1.0f;
    style.DisabledAlpha = 0.50f;

    // Cyber Blue color palette (#00D4FF based)
    const ImVec4 cyberBlue = ImVec4(0.00f, 0.83f, 1.00f, 1.00f);      // #00D4FF
    const ImVec4 cyberBlueDark = ImVec4(0.00f, 0.55f, 0.75f, 1.00f);  // Darker variant
    const ImVec4 cyberBlueGlow = ImVec4(0.00f, 0.83f, 1.00f, 0.15f);  // Glow effect

    // Dark background palette
    const ImVec4 bgDark = ImVec4(0.06f, 0.07f, 0.09f, 0.98f);
    const ImVec4 bgMid = ImVec4(0.09f, 0.10f, 0.13f, 0.95f);
    const ImVec4 bgLight = ImVec4(0.12f, 0.14f, 0.17f, 0.90f);

    // Text colors
    colors[ImGuiCol_Text] = ImVec4(0.93f, 0.94f, 0.96f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.45f, 0.48f, 0.52f, 1.00f);

    // Backgrounds
    colors[ImGuiCol_WindowBg] = bgDark;
    colors[ImGuiCol_ChildBg] = bgMid;
    colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.09f, 0.11f, 0.98f);
    colors[ImGuiCol_Border] = ImVec4(0.00f, 0.83f, 1.00f, 0.20f);  // Subtle cyber blue border
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame (input boxes, sliders background)
    colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.14f, 0.17f, 0.90f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.16f, 0.18f, 0.22f, 0.95f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.00f, 0.55f, 0.75f, 0.40f);

    // Title bar
    colors[ImGuiCol_TitleBg] = ImVec4(0.06f, 0.07f, 0.09f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.00f, 0.35f, 0.50f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.06f, 0.07f, 0.09f, 0.75f);

    colors[ImGuiCol_MenuBarBg] = ImVec4(0.08f, 0.09f, 0.11f, 1.00f);

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.06f, 0.07f, 0.09f, 0.60f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.00f, 0.55f, 0.75f, 0.60f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.00f, 0.70f, 0.90f, 0.80f);
    colors[ImGuiCol_ScrollbarGrabActive] = cyberBlue;

    // Checkmark
    colors[ImGuiCol_CheckMark] = cyberBlue;

    // Slider
    colors[ImGuiCol_SliderGrab] = ImVec4(0.00f, 0.70f, 0.90f, 0.90f);
    colors[ImGuiCol_SliderGrabActive] = cyberBlue;

    // Buttons
    colors[ImGuiCol_Button] = ImVec4(0.00f, 0.45f, 0.60f, 0.70f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.00f, 0.60f, 0.80f, 0.85f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.00f, 0.75f, 1.00f, 1.00f);

    // Headers (collapsing headers, tree nodes)
    colors[ImGuiCol_Header] = ImVec4(0.00f, 0.45f, 0.60f, 0.50f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.00f, 0.60f, 0.80f, 0.70f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.00f, 0.70f, 0.90f, 0.90f);

    // Separators
    colors[ImGuiCol_Separator] = ImVec4(0.00f, 0.83f, 1.00f, 0.25f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.00f, 0.83f, 1.00f, 0.60f);
    colors[ImGuiCol_SeparatorActive] = cyberBlue;

    // Resize grip
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.55f, 0.75f, 0.30f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.00f, 0.70f, 0.90f, 0.60f);
    colors[ImGuiCol_ResizeGripActive] = cyberBlue;

    // Tabs
    colors[ImGuiCol_Tab] = ImVec4(0.10f, 0.12f, 0.15f, 0.90f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.00f, 0.60f, 0.80f, 0.80f);
    colors[ImGuiCol_TabActive] = ImVec4(0.00f, 0.50f, 0.70f, 1.00f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.08f, 0.09f, 0.11f, 0.97f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.00f, 0.40f, 0.55f, 1.00f);

    // Plots
    colors[ImGuiCol_PlotLines] = cyberBlue;
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.00f, 0.70f, 0.90f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = cyberBlue;

    // Tables
    colors[ImGuiCol_TableHeaderBg] = ImVec4(0.10f, 0.12f, 0.15f, 1.00f);
    colors[ImGuiCol_TableBorderStrong] = ImVec4(0.00f, 0.55f, 0.75f, 0.50f);
    colors[ImGuiCol_TableBorderLight] = ImVec4(0.00f, 0.45f, 0.60f, 0.30f);
    colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.00f, 0.83f, 1.00f, 0.03f);

    // Selection & interaction
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.00f, 0.55f, 0.75f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = cyberBlue;
    colors[ImGuiCol_NavHighlight] = cyberBlue;
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00f, 0.10f, 0.15f, 0.50f);
}

void SetupImGui()
{
    auto& ctx = AppContext::getInstance();
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = ctx.config.global().overlay_ui_scale;

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Load a font with Korean glyphs to avoid '??' for Hangul
    {
        ImGuiIO& io = ImGui::GetIO();
        const ImWchar* ranges = io.Fonts->GetGlyphRangesKorean();
        // Try Malgun Gothic (Windows default Korean font)
        ImFont* font = io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\malgun.ttf", 18.0f, nullptr, ranges);
        if (!font) {
            // Fallback to default font if loading fails
            io.Fonts->AddFontDefault();
        }
    }

    // Apply Cyber Minimal theme (Modern minimalist with Cyber Blue)
    ApplyTheme_CyberMinimal();
}

bool CreateOverlayWindow()
{
    auto& ctx = AppContext::getInstance();
    
    overlayWidth = static_cast<int>((std::max)(Constants::MIN_OVERLAY_WIDTH, static_cast<int>(Constants::BASE_OVERLAY_WIDTH * ctx.config.global().overlay_ui_scale)));
    overlayHeight = static_cast<int>((std::max)(Constants::MIN_OVERLAY_HEIGHT, static_cast<int>(Constants::BASE_OVERLAY_HEIGHT * ctx.config.global().overlay_ui_scale)));

    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      _T("Edge"), NULL };
    ::RegisterClassEx(&wc);

    // Remove WS_EX_TOPMOST to avoid anti-cheat detection
    // Use WS_EX_LAYERED only for transparency support
    DWORD exStyle = WS_EX_LAYERED;
    if (ctx.config.global().always_on_top) {
        exStyle |= WS_EX_TOPMOST;
    }

    g_hwnd = ::CreateWindowEx(
        exStyle,
        wc.lpszClassName, _T("Chrome"),
        WS_POPUP | WS_SIZEBOX | WS_MAXIMIZEBOX, 0, 0, overlayWidth, overlayHeight,
        NULL, NULL, wc.hInstance, NULL);

    if (g_hwnd == NULL)
        return false;
    
    if (ctx.config.global().overlay_opacity <= 20)
    {
        ctx.config.global().overlay_opacity = 20;
        ctx.config.saveConfig();
    }

    if (ctx.config.global().overlay_opacity >= 256)
    {
        ctx.config.global().overlay_opacity = 255;
        ctx.config.saveConfig();
    }

    BYTE opacity = ctx.config.global().overlay_opacity;

    SetLayeredWindowAttributes(g_hwnd, 0, opacity, LWA_ALPHA);

    if (!CreateDeviceD3D(g_hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return false;
    }

    return true;
}

void OverlayThread()
{
    auto& ctx = AppContext::getInstance();
    
    if (!CreateOverlayWindow())
    {
        std::cout << "[Overlay] Can't create overlay window!" << std::endl;
        return;
    }

    SetupImGui();

    // Initialize GPU reader for UI
    g_uiGPUReader.initialize();
    
    // Start independent Preview Window (separate HWND)
    PreviewWindow::Start();

    // Load body texture after D3D device is created
    load_body_texture();

    bool show_overlay = false;

    // Only track values that need actual comparison for state changes
    int prev_detection_resolution = ctx.config.profile().detection_resolution;
    bool prev_capture_borders = ctx.config.profile().capture_borders;
    bool prev_capture_cursor = ctx.config.profile().capture_cursor;
    int prev_opacity = ctx.config.global().overlay_opacity;
    bool prev_show_window = ctx.config.global().show_window;
    bool prev_always_on_top = ctx.config.global().always_on_top;

    for (const auto& pair : KeyCodes::key_code_map)
    {
        key_names.push_back(pair.first);
    }
    std::sort(key_names.begin(), key_names.end());

    key_names_cstrs.reserve(key_names.size());
    for (const auto& name : key_names)
    {
        key_names_cstrs.push_back(name.c_str());
    }

    // Cache available models - only refresh every 5 seconds to avoid filesystem overhead
    static std::vector<std::string> availableModels;
    static auto lastModelRefresh = std::chrono::high_resolution_clock::now();
    auto now_for_models = std::chrono::high_resolution_clock::now();
    if (availableModels.empty() || 
        std::chrono::duration_cast<std::chrono::seconds>(now_for_models - lastModelRefresh).count() >= 5)
    {
        availableModels = getAvailableModels();
        lastModelRefresh = now_for_models;
    }

    MSG msg;
    ZeroMemory(&msg, sizeof(msg));

    // Overlay rendering frame timing - Fixed 30 FPS for UI
    auto lastOverlayFrameTime = std::chrono::high_resolution_clock::now();
    
    // Config saves immediately on change - no batching needed

    while (!should_exit && !AppContext::getInstance().should_exit)
    {
        // Track frame start time for proper FPS limiting
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        
        // UI-Pipeline separation: Try to read GPU data at UI's own pace (30 FPS)
        // This is completely independent from pipeline's processing speed
        if (ctx.config.global().show_window && ctx.preview_enabled) {
            // Get pipeline instance
            auto& pipelineManager = gpa::PipelineManager::getInstance();
            auto* pipeline = pipelineManager.getPipeline();
            
            if (pipeline) {
                // Non-blocking GPU read attempt
                g_uiGPUReader.tryReadFromGPU(pipeline);
                
                // Update UI targets if data is ready
                // OPTIMIZATION: Use static vector to avoid repeated allocations
                static std::vector<Target> uiTargets;
                static int targetCount = 0;
                static Target bestTarget = {};
                
                if (g_uiGPUReader.getLatestData(uiTargets, targetCount, bestTarget)) {
                    // Update AppContext with the latest targets for UI display
                    ctx.updateTargets(uiTargets);
                } 
            }
        }
        
        // Handle Windows messages
        while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
            {
                should_exit = true;
                AppContext::getInstance().should_exit = true;
                AppContext::getInstance().frame_cv.notify_all();  // Wake up main thread
                return;
            }
        }

        if (isAnyKeyPressed(ctx.config.global().button_open_overlay) & 0x1)
        {
            show_overlay = !show_overlay;

            if (show_overlay)
            {
                ShowWindow(g_hwnd, SW_SHOW);
                SetForegroundWindow(g_hwnd);
            }
            else
            {
                ShowWindow(g_hwnd, SW_HIDE);
            }
            // Note: PreviewWindow is completely independent (separate HWND)

            std::this_thread::sleep_for(std::chrono::milliseconds(Constants::OVERLAY_INIT_RETRY_SLEEP_MS));
        }

        // When overlay is hidden, reduce CPU usage
        if (!show_overlay)
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - frame_start_time);
            int target_frame_time_ms = 33; // Fixed 30 FPS for overlay
            int remaining_time_ms = target_frame_time_ms - static_cast<int>(elapsed.count());
            
            if (remaining_time_ms > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(remaining_time_ms));
            }
        }

        if (show_overlay)
        {
            // Optimized ImGui frame setup
            ImGui_ImplDX11_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
            
            // Disable unnecessary ImGui features for performance
            ImGuiIO& io = ImGui::GetIO();
            io.MouseDrawCursor = false; // Disable software cursor rendering

            // === MAIN WINDOW ===
            RECT rect;
            GetClientRect(g_hwnd, &rect);
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2((float)(rect.right - rect.left), (float)(rect.bottom - rect.top)));

            ImGui::Begin("Options", &show_overlay, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
            {
                std::lock_guard<std::mutex> lock(ctx.configMutex);
                
                // Rapidfire pause removed - MouseThread no longer exists

                // === NEW HIERARCHICAL UI LAYOUT ===
                
                // Get target count for status display
                int targetCount = static_cast<int>(ctx.num_targets_);
                bool isPaused = ctx.detection_paused.load();
                
                // Static state for collapsible sections
                static bool s_fineTuningOpen = false;
                static bool s_detectionOpen = false;
                static bool s_hotkeysOpen = false;
                static bool s_systemOpen = false;
                
                // Status Header (Paused state only)
                UIHelpers::StatusHeader(targetCount, isPaused);
                
                // === QUICK CONTROLS ===
                UIHelpers::SectionHeader("QUICK CONTROLS");
                
                float labelWidth = 70.0f;
                float inputWidth = (ImGui::GetContentRegionAvail().x - labelWidth - 10.0f) * 0.5f;
                
                // PID Kp (Proportional - Response Speed)
                ImGui::Text("Kp");
                ImGui::SameLine(labelWidth);
                ImGui::SetNextItemWidth(inputWidth);
                if (ImGui::DragFloat("##kp_x", &ctx.config.profile().pid_kp_x, 0.005f, 0.0f, 2.0f, "X: %.3f")) {
                    MARK_CONFIG_DIRTY();
                }
                ImGui::SameLine();
                ImGui::SetNextItemWidth(inputWidth);
                if (ImGui::DragFloat("##kp_y", &ctx.config.profile().pid_kp_y, 0.005f, 0.0f, 2.0f, "Y: %.3f")) {
                    MARK_CONFIG_DIRTY();
                }
                
                // PID Ki (Integral - Tracking)
                ImGui::Text("Ki");
                ImGui::SameLine(labelWidth);
                ImGui::SetNextItemWidth(inputWidth);
                if (ImGui::DragFloat("##ki_x", &ctx.config.profile().pid_ki_x, 0.001f, 0.0f, 0.3f, "X: %.4f")) {
                    MARK_CONFIG_DIRTY();
                }
                ImGui::SameLine();
                ImGui::SetNextItemWidth(inputWidth);
                if (ImGui::DragFloat("##ki_y", &ctx.config.profile().pid_ki_y, 0.001f, 0.0f, 0.3f, "Y: %.4f")) {
                    MARK_CONFIG_DIRTY();
                }
                
                // PID Kd (Derivative - Damping)
                ImGui::Text("Kd");
                ImGui::SameLine(labelWidth);
                ImGui::SetNextItemWidth(inputWidth);
                if (ImGui::DragFloat("##kd_x", &ctx.config.profile().pid_kd_x, 0.005f, 0.0f, 1.0f, "X: %.3f")) {
                    MARK_CONFIG_DIRTY();
                }
                ImGui::SameLine();
                ImGui::SetNextItemWidth(inputWidth);
                if (ImGui::DragFloat("##kd_y", &ctx.config.profile().pid_kd_y, 0.005f, 0.0f, 1.0f, "Y: %.3f")) {
                    MARK_CONFIG_DIRTY();
                }
                
                // NoRecoil with profile selector
                ImGui::Text("NoRecoil");
                ImGui::SameLine(labelWidth);
                
                auto& profile = ctx.config.profile();
                float comboWidth = inputWidth * 0.8f;
                float strengthWidth = inputWidth * 1.2f - 5.0f;
                
                ImGui::SetNextItemWidth(comboWidth);
                if (ImGui::BeginCombo("##norecoil_profile", 
                    profile.input_profiles.empty() ? "None" : profile.input_profiles[profile.active_input_profile_index].profile_name.c_str())) {
                    for (int i = 0; i < (int)profile.input_profiles.size(); i++) {
                        bool isSelected = (profile.active_input_profile_index == i);
                        if (ImGui::Selectable(profile.input_profiles[i].profile_name.c_str(), isSelected)) {
                            profile.active_input_profile_index = i;
                            MARK_CONFIG_DIRTY();
                        }
                        if (isSelected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                
                ImGui::SameLine();
                auto* inputProfile = ctx.config.getCurrentInputProfile();
                if (inputProfile) {
                    ImGui::SetNextItemWidth(strengthWidth);
                    if (ImGui::DragFloat("##base_strength", &inputProfile->base_strength, 0.1f, 0.0f, 20.0f, "%.1f")) {
                        MARK_CONFIG_DIRTY();
                    }
                }
                
                UIHelpers::Spacer(8.0f);
                
                // Preview Window toggle button (uses separate HWND window)
                bool previewVisible = PreviewWindow::IsVisible();
                ImVec4 btnColor = previewVisible ? UIHelpers::GetAccentColor(1.0f) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
                ImGui::PushStyleColor(ImGuiCol_Button, btnColor);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(btnColor.x + 0.1f, btnColor.y + 0.1f, btnColor.z + 0.1f, 1.0f));
                if (ImGui::Button(previewVisible ? "Preview ON" : "Preview OFF", ImVec2(-1, 28))) {
                    PreviewWindow::SetVisible(!previewVisible);
                }
                ImGui::PopStyleColor(2);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Open Preview Window (independent window)");
                }
                
                UIHelpers::Spacer(16.0f);
                
                // === COLLAPSIBLE SECTIONS ===
                
                // Aim Settings Section
                if (UIHelpers::CollapsibleSection("Aim Settings", &s_fineTuningOpen)) {
                    ImGui::Indent(10.0f);
                    draw_mouse();
                    UIHelpers::Spacer(10.0f);
                    draw_stabilizer();
                    UIHelpers::Spacer(10.0f);
                    renderOffsetTab();
                    UIHelpers::Spacer(10.0f);
                    
                    // Trigger Bot settings
                    ImGui::SeparatorText("Trigger Bot");
                    ImGui::Text("Action Area");
                    ImGui::SameLine(100.0f);
                    ImGui::SetNextItemWidth(-1);
                    if (ImGui::SliderFloat("##action_area", &ctx.config.profile().bScope_multiplier, 0.1f, 2.0f, "%.2f")) {
                        MARK_CONFIG_DIRTY();
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Trigger Bot activation area size.\nSmaller = larger area, Larger = smaller area");
                    }
                    
                    ImGui::Unindent(10.0f);
                    UIHelpers::Spacer(8.0f);
                }
                
                // AI Section
                if (UIHelpers::CollapsibleSection("AI", &s_detectionOpen)) {
                    ImGui::Indent(10.0f);
                    ImGui::SeparatorText("AI Model");
                    draw_ai();
                    UIHelpers::Spacer(10.0f);
                    ImGui::SeparatorText("Screen Capture");
                    draw_capture_settings();
                    ImGui::Unindent(10.0f);
                    UIHelpers::Spacer(8.0f);
                }
                
                // Hotkeys Section
                if (UIHelpers::CollapsibleSection("Hotkeys", &s_hotkeysOpen)) {
                    ImGui::Indent(10.0f);
                    draw_buttons();
                    ImGui::Unindent(10.0f);
                    UIHelpers::Spacer(8.0f);
                }
                
                // System Section
                if (UIHelpers::CollapsibleSection("System", &s_systemOpen)) {
                    ImGui::Indent(10.0f);
                    draw_profile();
                    UIHelpers::Spacer(10.0f);
                    draw_overlay();
                    UIHelpers::Spacer(10.0f);
                    ImGui::SeparatorText("Debug");
                    draw_debug();
#ifdef ENABLE_DEPTH_ESTIMATION
                    UIHelpers::Spacer(10.0f);
                    ImGui::SeparatorText("Depth Estimation");
                    draw_depth_settings();
#endif
                    ImGui::Unindent(10.0f);
                    UIHelpers::Spacer(8.0f);
                }

                // Efficient config change detection - only check values that trigger actions
                {
                    if (prev_detection_resolution != ctx.config.profile().detection_resolution)
                    {
                        prev_detection_resolution = ctx.config.profile().detection_resolution;
                        detection_resolution_changed = true;
                        ctx.model_changed = true;
                    }

                    if (prev_capture_cursor != ctx.config.profile().capture_cursor)
                    {
                        capture_cursor_changed = true;
                        prev_capture_cursor = ctx.config.profile().capture_cursor;
                    }

                    if (prev_capture_borders != ctx.config.profile().capture_borders)
                    {
                        capture_borders_changed = true;
                        prev_capture_borders = ctx.config.profile().capture_borders;
                    }

                    if (prev_opacity != ctx.config.global().overlay_opacity)
                    {
                        prev_opacity = ctx.config.global().overlay_opacity;
                        SetLayeredWindowAttributes(g_hwnd, 0, static_cast<BYTE>(ctx.config.global().overlay_opacity), LWA_ALPHA);
                    }

                    if (prev_show_window != ctx.config.global().show_window || prev_always_on_top != ctx.config.global().always_on_top)
                    {
                        prev_always_on_top = ctx.config.global().always_on_top;
                        prev_show_window = ctx.config.global().show_window;
                        show_window_changed = true;
                        ctx.preview_enabled = ctx.config.global().show_window;
                        
                        // Update window TOPMOST state dynamically
                        HWND insertAfter = ctx.config.global().always_on_top ? HWND_TOPMOST : HWND_NOTOPMOST;
                        SetWindowPos(g_hwnd, insertAfter, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
                    }
                }

                // Auto-save: check if dirty and delay has passed
                if (g_autoSaveState.shouldSave())
                {
                    ctx.config.saveConfig();
                    g_autoSaveState.reset();
                }

                ImGui::EndTabBar();
            }

            
            ImGui::Separator();
            ImGui::TextColored(ImVec4(255, 255, 255, 100), "Do not test shooting and aiming with the overlay and debug window is open.");

            ImGui::End();

            // Note: Preview Window is now a separate HWND managed by PreviewWindow class

            ImGui::Render();

            // Optimized rendering
            const float clear_color_with_alpha[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
            g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

            // Use immediate present for better performance (no VSync)
            HRESULT result = g_pSwapChain->Present(0, DXGI_PRESENT_DO_NOT_WAIT);
            auto present_now = std::chrono::high_resolution_clock::now();
            auto present_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(present_now - lastOverlayFrameTime);
            
            if (result == DXGI_STATUS_OCCLUDED || result == DXGI_ERROR_ACCESS_LOST)
            {
                // If occluded, back off
                std::this_thread::sleep_for(std::chrono::milliseconds(Constants::OVERLAY_OCCLUDED_SLEEP_MS));
            }
            else
            {
                // Use configurable overlay FPS for better control
                auto targetFrameTime = std::chrono::milliseconds(static_cast<long long>(1000.0f / Constants::OVERLAY_TARGET_FPS));
                
                if (present_elapsed < targetFrameTime)
                {
                    std::this_thread::sleep_for(targetFrameTime - present_elapsed);
                }
            }
            lastOverlayFrameTime = std::chrono::high_resolution_clock::now();
        }
        else
        {
            // Window not visible: slow down loop
            std::this_thread::sleep_for(std::chrono::milliseconds(Constants::OVERLAY_HIDDEN_SLEEP_MS));
        }
    }

    // Cleanup GPU reader
    g_uiGPUReader.cleanup();
    
    // Stop Preview Window
    PreviewWindow::Stop();

    // Release any overlay textures we created
    release_body_texture();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(g_hwnd);
    ::UnregisterClass(_T("Edge"), GetModuleHandle(NULL));
}

