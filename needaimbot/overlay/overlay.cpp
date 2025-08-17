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
#include "mouse/rapidfire.h"
#include "overlay/draw_settings.h"
#include "overlay/draw_offset.h"
#include "overlay/draw_tracker.h"
#include "overlay/ui_helpers.h"
#include "config.h"
#include "keycodes.h"
#include "needaimbot.h"
// #include "capture/capture.h" - removed, using GPU capture now
#include "keyboard_listener.h"
#include "other_tools.h"


std::atomic<bool> g_config_optical_flow_changed{false};

ID3D11Device* g_pd3dDevice = NULL;
ID3D11DeviceContext* g_pd3dDeviceContext = NULL;
IDXGISwapChain* g_pSwapChain = NULL;
ID3D11RenderTargetView* g_mainRenderTargetView = NULL;
HWND g_hwnd = NULL;

extern std::mutex configMutex;
extern std::atomic<bool> should_exit;

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();

ID3D11BlendState* g_pBlendState = nullptr;
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

const int BASE_OVERLAY_WIDTH = 800;
const int BASE_OVERLAY_HEIGHT = 600;
const int MIN_OVERLAY_WIDTH = 600;  // Minimum width
const int MIN_OVERLAY_HEIGHT = 400;  // Minimum height
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
        ::PostQuitMessage(0);
        return 0;
    default:
        return ::DefWindowProc(hWnd, msg, wParam, lParam);
    }
}

void SetupImGui()
{
    auto& ctx = AppContext::getInstance();
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = ctx.config.overlay_ui_scale;

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    ImGui::StyleColorsDark();

    
    ImGuiStyle& style = ImGui::GetStyle();
    
    style.WindowPadding = ImVec2(6.0f, 6.0f);
    style.FramePadding = ImVec2(4.0f, 3.0f);
    style.ItemSpacing = ImVec2(5.0f, 4.0f);
    style.ItemInnerSpacing = ImVec2(3.0f, 3.0f);
    style.IndentSpacing = 15.0f;
    style.ScrollbarSize = 12.0f;
    style.GrabMinSize = 8.0f;

    style.WindowBorderSize = 0.0f;
    style.ChildBorderSize = 0.0f;
    style.PopupBorderSize = 0.0f;
    style.FrameBorderSize = 0.0f;
    style.TabBorderSize = 0.0f;

    style.WindowRounding = 8.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 6.0f;
    style.ScrollbarRounding = 8.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 4.0f;

    style.Alpha = 0.98f;
    style.DisabledAlpha = 0.60f;

    
    ImVec4* colors = style.Colors;
    
    colors[ImGuiCol_Text] = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.08f, 0.98f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.08f, 0.08f, 0.10f, 0.90f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.05f, 0.05f, 0.07f, 0.98f);
    colors[ImGuiCol_Border] = ImVec4(0.20f, 0.20f, 0.25f, 0.50f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    
    colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.12f, 0.15f, 0.90f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.18f, 0.22f, 0.95f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.24f, 0.24f, 0.28f, 1.00f);
    
    colors[ImGuiCol_TitleBg] = ImVec4(0.04f, 0.04f, 0.06f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.06f, 0.06f, 0.08f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.04f, 0.04f, 0.06f, 0.75f);
    
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
    
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.08f, 0.08f, 0.10f, 0.60f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.25f, 0.25f, 0.30f, 0.80f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.30f, 0.30f, 0.35f, 0.90f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    
    colors[ImGuiCol_CheckMark] = ImVec4(0.20f, 0.70f, 0.90f, 1.00f);
    
    colors[ImGuiCol_SliderGrab] = ImVec4(0.20f, 0.60f, 0.90f, 0.90f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.25f, 0.75f, 1.00f, 1.00f);
    
    colors[ImGuiCol_Button] = ImVec4(0.15f, 0.45f, 0.75f, 0.80f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.20f, 0.55f, 0.85f, 0.95f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.10f, 0.35f, 0.65f, 1.00f);
    
    colors[ImGuiCol_Header] = ImVec4(0.12f, 0.40f, 0.70f, 0.70f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.15f, 0.50f, 0.80f, 0.85f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.60f, 0.90f, 1.00f);
    
    colors[ImGuiCol_Separator] = ImVec4(0.20f, 0.20f, 0.25f, 0.60f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.30f, 0.60f, 0.85f, 0.78f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.35f, 0.70f, 0.95f, 1.00f);
    
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.20f, 0.60f, 0.90f, 0.30f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.25f, 0.70f, 0.95f, 0.67f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.30f, 0.80f, 1.00f, 0.95f);
    
    colors[ImGuiCol_Tab] = ImVec4(0.12f, 0.12f, 0.15f, 0.90f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.20f, 0.50f, 0.80f, 0.80f);
    colors[ImGuiCol_TabActive] = ImVec4(0.15f, 0.45f, 0.75f, 1.00f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.08f, 0.08f, 0.10f, 0.97f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.12f, 0.35f, 0.60f, 1.00f);
    
    colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    
    colors[ImGuiCol_TableHeaderBg] = ImVec4(0.12f, 0.12f, 0.15f, 1.00f);
    colors[ImGuiCol_TableBorderStrong] = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
    colors[ImGuiCol_TableBorderLight] = ImVec4(0.15f, 0.15f, 0.18f, 1.00f);
    colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.20f, 0.60f, 0.90f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ImVec4(0.20f, 0.60f, 0.90f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f); 

    load_body_texture();
}

bool CreateOverlayWindow()
{
    auto& ctx = AppContext::getInstance();
    auto& config = ctx.config;  // Reference to avoid global config confusion
    
    overlayWidth = static_cast<int>((std::max)(MIN_OVERLAY_WIDTH, static_cast<int>(BASE_OVERLAY_WIDTH * ctx.config.overlay_ui_scale)));
    overlayHeight = static_cast<int>((std::max)(MIN_OVERLAY_HEIGHT, static_cast<int>(BASE_OVERLAY_HEIGHT * ctx.config.overlay_ui_scale)));

    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      _T("Edge"), NULL };
    ::RegisterClassEx(&wc);

    g_hwnd = ::CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_LAYERED,
        wc.lpszClassName, _T("Chrome"),
        WS_POPUP | WS_SIZEBOX | WS_MAXIMIZEBOX, 0, 0, overlayWidth, overlayHeight,
        NULL, NULL, wc.hInstance, NULL);

    if (g_hwnd == NULL)
        return false;
    
    if (ctx.config.overlay_opacity <= 20)
    {
        ctx.config.overlay_opacity = 20;
        ctx.config.saveConfig("config.ini");
    }

    if (ctx.config.overlay_opacity >= 256)
    {
        ctx.config.overlay_opacity = 255;
        ctx.config.saveConfig("config.ini");
    }

    BYTE opacity = ctx.config.overlay_opacity;

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

    bool show_overlay = false;

    
    int prev_detection_resolution = ctx.config.detection_resolution;
    int prev_monitor_idx = ctx.config.monitor_idx;
    bool prev_circle_mask = ctx.config.circle_mask;
    bool prev_capture_borders = ctx.config.capture_borders;
    bool prev_capture_cursor = ctx.config.capture_cursor;

    
    float prev_body_y_offset = ctx.config.body_y_offset;
    float prev_head_y_offset = ctx.config.head_y_offset;
    bool prev_ignore_third_person = ctx.config.ignore_third_person;
    bool prev_shooting_range_targets = ctx.config.shooting_range_targets;
    bool prev_auto_aim = ctx.config.auto_aim;

    
    bool prev_easynorecoil = ctx.config.easynorecoil;
    float prev_easynorecoilstrength = ctx.config.easynorecoilstrength;

    

    
    float prev_bScope_multiplier = ctx.config.bScope_multiplier;

    
    float prev_confidence_threshold = ctx.config.confidence_threshold;
    float prev_nms_threshold = ctx.config.nms_threshold;
    int prev_max_detections = ctx.config.max_detections;

    
    int prev_opacity = ctx.config.overlay_opacity;

    
    bool prev_show_window = ctx.config.show_window;
    bool prev_show_fps = ctx.config.show_fps;
    int prev_window_size = ctx.config.window_size;
    int prev_screenshot_delay = ctx.config.screenshot_delay;
    bool prev_always_on_top = ctx.config.always_on_top;

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

    int input_method_index = 0;
    if (ctx.config.input_method == "WIN32")
        input_method_index = 0;
    else if (ctx.config.input_method == "GHUB")
        input_method_index = 1;
    else if (ctx.config.input_method == "ARDUINO")
        input_method_index = 2;
    else
        input_method_index = 0;
    
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

    static auto lastTime = std::chrono::high_resolution_clock::now();

    MSG msg;
    ZeroMemory(&msg, sizeof(msg));

    // Overlay rendering frame timing - Use ctx.config.target_fps for user control
    auto lastOverlayFrameTime = std::chrono::high_resolution_clock::now();
    
    // Config save batching to reduce I/O
    bool config_needs_save = false;
    auto last_config_save_time = std::chrono::high_resolution_clock::now();
    const std::chrono::milliseconds config_save_interval(500); // Save config every 500ms max

    while (!should_exit && !AppContext::getInstance().should_exit)
    {
        // Track frame start time for proper FPS limiting
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        
        // Handle Windows messages
        while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
            {
                should_exit = true;
                return;
            }
        }

        if (isAnyKeyPressed(ctx.config.button_open_overlay) & 0x1)
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

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        // When overlay is hidden, reduce CPU usage based on target_fps setting
        if (!show_overlay)
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - frame_start_time);
            int target_frame_time_ms = static_cast<int>(1000.0f / ctx.config.target_fps);
            int remaining_time_ms = target_frame_time_ms - static_cast<int>(elapsed.count());
            
            if (remaining_time_ms > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(remaining_time_ms));
            }
        }

        if (show_overlay)
        {
            // Control frame rate to 30 FPS when visible
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastOverlayFrameTime);
            if (elapsed.count() < 33) { // ~30 FPS
                std::this_thread::sleep_for(std::chrono::milliseconds(33 - elapsed.count()));
            }
            lastOverlayFrameTime = std::chrono::high_resolution_clock::now();
            
            // Optimized ImGui frame setup
            ImGui_ImplDX11_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
            
            // Disable unnecessary ImGui features for performance
            ImGuiIO& io = ImGui::GetIO();
            io.MouseDrawCursor = false; // Disable software cursor rendering

            RECT rect;
            GetClientRect(g_hwnd, &rect);
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2((float)(rect.right - rect.left), (float)(rect.bottom - rect.top)));

            ImGui::Begin("Options", &show_overlay, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
            {
                std::lock_guard<std::mutex> lock(configMutex);
                
                // Pause rapidfire when UI is shown
                if (ctx.mouseThread) {
                    auto rapidfire = ctx.mouseThread->getRapidFire();
                    if (rapidfire) {
                        rapidfire->setUIActive(true);
                    }
                }

                if (ImGui::BeginTabBar("Options tab bar", ImGuiTabBarFlags_FittingPolicyResizeDown))
                {
                    // Main Controls - Essential aimbot/triggerbot settings
                    if (ImGui::BeginTabItem("Main"))
                    {
                        // Main aimbot and triggerbot controls from draw_target
                        UIHelpers::BeginSettingsSection("Main Controls", "Essential aimbot and triggerbot settings");
                        draw_target();  // This now contains the enable checkboxes
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                    // Offset Settings
                    if (ImGui::BeginTabItem("Offset"))
                    {
                        renderOffsetTab();
                        ImGui::EndTabItem();
                    }

                    // Mouse Movement
                    if (ImGui::BeginTabItem("Mouse"))
                    {
                        UIHelpers::BeginSettingsSection("Mouse Movement", "Configure mouse sensitivity and movement behavior");
                        draw_mouse();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                    // Tracker Settings
                    if (ImGui::BeginTabItem("Tracker"))
                    {
                        UIHelpers::BeginSettingsSection("Target Tracking", "Configure tracking and prediction systems");
                        draw_tracker();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                    // Recoil Control
                    if (ImGui::BeginTabItem("Recoil"))
                    {
                        UIHelpers::BeginSettingsSection("Recoil Control", "Configure automatic recoil compensation");
                        draw_rcs_settings();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                    // Key Bindings
                    if (ImGui::BeginTabItem("Keybinds"))
                    {
                        UIHelpers::BeginSettingsSection("Key Bindings", "Configure all hotkeys and control buttons");
                        draw_buttons();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                    // AI Model
                    if (ImGui::BeginTabItem("AI Model"))
                    {
                        UIHelpers::BeginSettingsSection("AI Configuration", "Configure AI detection model and parameters");
                        draw_ai();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                    // Screen Capture
                    if (ImGui::BeginTabItem("Capture"))
                    {
                        UIHelpers::BeginSettingsSection("Screen Capture", "Configure capture area and performance");
                        draw_capture_settings();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                    // Visual Settings
                    if (ImGui::BeginTabItem("Visual"))
                    {
                        // Color Filter (RGB/HSV)
                        UIHelpers::BeginSettingsSection("Color Filter", "Configure color-based filtering");
                        draw_color_filter_settings();
                        UIHelpers::EndSettingsSection();
                        
                        // Overlay Settings
                        UIHelpers::BeginSettingsSection("Overlay", "Configure overlay appearance");
                        draw_overlay();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                    // Profile Management
                    if (ImGui::BeginTabItem("Profiles"))
                    {
                        UIHelpers::BeginSettingsSection("Profile Management", "Save and load configurations");
                        draw_profile();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }


                    // Monitoring
                    if (ImGui::BeginTabItem("Monitor"))
                    {
                        // Performance Stats
                        UIHelpers::BeginSettingsSection("Performance", "Real-time performance metrics");
                        draw_stats();
                        UIHelpers::EndSettingsSection();
                        
                        // Debug Info
                        UIHelpers::BeginSettingsSection("Debug", "System information and troubleshooting");
                        draw_debug();
                        UIHelpers::EndSettingsSection();
                        ImGui::EndTabItem();
                    }

                }

                // Reduce frequency of config change detection to every 5 frames for performance
                static int config_check_frame_counter = 0;
                config_check_frame_counter++;
                bool should_check_config = (config_check_frame_counter % 5 == 0);
                
                if (should_check_config) {
                    
                    if (prev_detection_resolution != ctx.config.detection_resolution)
                    {
                        prev_detection_resolution = ctx.config.detection_resolution;
                        detection_resolution_changed.store(true);
                        detector_model_changed.store(true); 

                        
                        if (AppContext::getInstance().mouseThread) {
                            AppContext::getInstance().mouseThread->updateConfig(
                                ctx.config.detection_resolution,
                                ctx.config.bScope_multiplier,
                                ctx.config.norecoil_ms
                            );
                        }
                        config_needs_save = true;
                    }

                    
                    if (prev_capture_cursor != ctx.config.capture_cursor)
                    {
                        capture_cursor_changed.store(true);
                        prev_capture_cursor = ctx.config.capture_cursor;
                        config_needs_save = true;
                    }

                    
                    if (prev_capture_borders != ctx.config.capture_borders)
                    {
                        capture_borders_changed.store(true);
                        prev_capture_borders = ctx.config.capture_borders;
                        config_needs_save = true;
                    }

                    
                    if (prev_monitor_idx != ctx.config.monitor_idx)
                    {
                        prev_monitor_idx = ctx.config.monitor_idx;
                        config_needs_save = true;
                    }

                    
                    if (
                        prev_body_y_offset != ctx.config.body_y_offset ||
                        prev_head_y_offset != ctx.config.head_y_offset ||
                        prev_ignore_third_person != ctx.config.ignore_third_person ||
                        prev_shooting_range_targets != ctx.config.shooting_range_targets ||
                        prev_auto_aim != ctx.config.auto_aim ||
                        prev_easynorecoil != ctx.config.easynorecoil ||
                        prev_easynorecoilstrength != ctx.config.easynorecoilstrength)
                    {
                        
                        prev_body_y_offset = ctx.config.body_y_offset;
                        prev_head_y_offset = ctx.config.head_y_offset;
                        prev_ignore_third_person = ctx.config.ignore_third_person;
                        prev_shooting_range_targets = ctx.config.shooting_range_targets;
                        prev_auto_aim = ctx.config.auto_aim;
                        prev_easynorecoil = ctx.config.easynorecoil;
                        prev_easynorecoilstrength = ctx.config.easynorecoilstrength;
                        config_needs_save = true;
                    }

                    
                    if (prev_bScope_multiplier != ctx.config.bScope_multiplier)
                    {
                        prev_bScope_multiplier = ctx.config.bScope_multiplier;

                        if (AppContext::getInstance().mouseThread) {
                            AppContext::getInstance().mouseThread->updateConfig(
                            ctx.config.detection_resolution,
                            ctx.config.bScope_multiplier,
                            ctx.config.norecoil_ms
                            );
                        }

                        config_needs_save = true;
                    }

                    
                    if (prev_opacity != ctx.config.overlay_opacity)
                    {
                        BYTE opacity = ctx.config.overlay_opacity;
                        SetLayeredWindowAttributes(g_hwnd, 0, opacity, LWA_ALPHA);
                        config_needs_save = true;
                    }

                    
                    if (prev_confidence_threshold != ctx.config.confidence_threshold ||
                        prev_nms_threshold != ctx.config.nms_threshold ||
                        prev_max_detections != ctx.config.max_detections)
                    {
                        prev_nms_threshold = ctx.config.nms_threshold;
                        prev_confidence_threshold = ctx.config.confidence_threshold;
                        prev_max_detections = ctx.config.max_detections;
                        config_needs_save = true;
                    }

                    
                    if (prev_show_window != ctx.config.show_window ||
                        prev_always_on_top != ctx.config.always_on_top)
                    {
                        prev_always_on_top = ctx.config.always_on_top;
                        show_window_changed.store(true);
                        prev_show_window = ctx.config.show_window;
                        config_needs_save = true;
                    }
                    
                    
                    if (prev_show_fps != ctx.config.show_fps ||
                        prev_window_size != ctx.config.window_size ||
                        prev_screenshot_delay != ctx.config.screenshot_delay)
                    {
                        prev_show_fps = ctx.config.show_fps;
                        prev_window_size = ctx.config.window_size;
                        prev_screenshot_delay = ctx.config.screenshot_delay;
                        config_needs_save = true;
                    }
                }

                ImGui::EndTabBar();
            }

            
            ImGui::Separator();
            ImGui::TextColored(ImVec4(255, 255, 255, 100), "Do not test shooting and aiming with the overlay and debug window is open.");

            ImGui::End();
            
            // Resume rapidfire when UI is closed
            if (ctx.mouseThread) {
                auto rapidfire = ctx.mouseThread->getRapidFire();
                if (rapidfire) {
                    rapidfire->setUIActive(false);
                }
            }
            
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
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            else
            {
                // Limit overlay FPS to 30 for better performance while maintaining responsiveness
                float overlay_target_fps = 30.0f;
                auto targetFrameTime = std::chrono::milliseconds(static_cast<long long>(1000.0f / overlay_target_fps));
                
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
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Batched config saving to reduce I/O overhead
        auto now = std::chrono::high_resolution_clock::now();
        if (config_needs_save && 
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_config_save_time) >= config_save_interval)
        {
            ctx.config.saveConfig();
            config_needs_save = false;
            last_config_save_time = now;
        }
    }

    release_body_texture();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(g_hwnd);
    ::UnregisterClass(_T("Edge"), GetModuleHandle(NULL));
}

