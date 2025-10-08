#include "../core/windows_headers.h"
#include <Psapi.h>

#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include <cstring>
#include <vector>
#include <string>
#include <filesystem>

#include "AppContext.h"
#include "../core/constants.h"
#include "config/config.h"
#include "needaimbot.h"
// #include "../capture/capture.h" - removed, using GPU capture now
#include "include/other_tools.h"
#include "draw_settings.h"
#include "ui_helpers.h"
#include "../cuda/unified_graph_pipeline.h"

// Monitor count is now simplified - just count all monitors
int monitors = GetSystemMetrics(SM_CMONITORS);

static void draw_capture_area_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Capture Area & Resolution");
    
    // Use temporary float for the slider
    float detection_res_float = static_cast<float>(ctx.config.detection_resolution);
    
    // Store the old value to detect changes
    int old_resolution = ctx.config.detection_resolution;
    
    UIHelpers::CompactSlider("Detection Resolution", &detection_res_float, 50.0f, 1280.0f, "%.0f");
    UIHelpers::InfoTooltip("Size (in pixels) of the square area around the cursor to capture for detection.\nSmaller values improve performance but may miss targets further from the crosshair.");
    
    // Check if value changed
    int new_resolution = static_cast<int>(detection_res_float);
    if (new_resolution != old_resolution) {
        ctx.config.detection_resolution = new_resolution;
        SAVE_PROFILE();
        
        // Force update flags
        extern std::atomic<bool> detection_resolution_changed;
        // detector_model_changed is now managed by DetectionState
        detection_resolution_changed.store(true);
        ctx.model_changed = true;
    }
    
    if (ctx.config.detection_resolution >= Constants::DETECTION_RESOLUTION_HIGH_PERF)
    {
        UIHelpers::BeautifulText("WARNING: Large detection resolution can impact performance.", UIHelpers::GetWarningColor());
    }
    
    UIHelpers::CompactSpacer();
    
    if (UIHelpers::BeautifulToggle("Circle Mask", &ctx.config.circle_mask, "Applies a circular mask to the captured area, ignoring corners."))
    {
        SAVE_PROFILE();
    }
    
    UIHelpers::EndCard();

    // Capture Debug Info
    {
        auto* pipeline = needaimbot::PipelineManager::getInstance().getPipeline();
        if (pipeline) {
            needaimbot::UnifiedGraphPipeline::CaptureStats stats{};
            pipeline->getCaptureStats(stats);
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.6f,0.9f,0.6f,1.0f), "Capture Debug");
            ImGui::Text("Backend: %s", stats.backend ? stats.backend : "?");
            ImGui::Text("Frame: %dx%d (%s)", stats.lastWidth, stats.lastHeight, stats.gpuDirect ? "GPU-direct" : "CPU");
            ImGui::Text("ROI: left=%d top=%d size=%d", stats.roiLeft, stats.roiTop, stats.roiSize);
            ImGui::Text("Flags: hasFrame=%s previewEnabled=%s previewHasHost=%s",
                        stats.hasFrame?"true":"false",
                        stats.previewEnabled?"true":"false",
                        stats.previewHasHost?"true":"false");
        }
    }
}

static void draw_capture_behavior_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Capture Behavior");

    // Backend selection
    const char* backends[] = { "DDA", "OBS_HOOK" };
    int current = (_stricmp(ctx.config.capture_method.c_str(), "OBS_HOOK") == 0) ? 1 : 0;
    int prev_backend = current;
    UIHelpers::CompactCombo("Capture Backend", &current, backends, IM_ARRAYSIZE(backends));
    if (current != prev_backend) {
        ctx.config.capture_method = (current == 1) ? "OBS_HOOK" : "DDA";
        SAVE_PROFILE();
    }
    if (current == 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Desktop Duplication Capture Active");
        UIHelpers::InfoTooltip("The Windows Desktop Duplication API (DDA) powers high-quality capture with low latency.\nThis path delivers consistent results across modern GPUs without requiring vendor-specific drivers.");
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.6f, 1.0f), "OBS Hook Capture Selected");
        UIHelpers::InfoTooltip("Uses an OBS-style game capture hook. Requires specifying the game window title.\nGPU-direct is not available in this path.");
        static char obs_title_buf[256] = {};
        static bool obs_title_init = false;
        if (!obs_title_init) {
            strncpy(obs_title_buf, ctx.config.obs_window_title.c_str(), sizeof(obs_title_buf) - 1);
            obs_title_init = true;
        }
        (void)ImGui::InputText("Game Window Title", obs_title_buf, IM_ARRAYSIZE(obs_title_buf));
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            ctx.config.obs_window_title = obs_title_buf;
            SAVE_PROFILE();
        }

        // OBS hook source folder (for auto-copy)
        static char obs_src_buf[512] = {};
        static bool obs_src_init = false;
        if (!obs_src_init) {
            strncpy(obs_src_buf, ctx.config.obs_hook_source_dir.c_str(), sizeof(obs_src_buf) - 1);
            obs_src_init = true;
        }
        (void)ImGui::InputText("OBS Hook Source Folder", obs_src_buf, IM_ARRAYSIZE(obs_src_buf));
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            ctx.config.obs_hook_source_dir = obs_src_buf;
            SAVE_PROFILE();
        }

        // Check OBS hook binaries availability
        {
            namespace fs = std::filesystem;
            bool ok = fs::exists("obs_stuff\\inject-helper64.exe") &&
                      fs::exists("obs_stuff\\graphics-hook64.dll") &&
                      fs::exists("obs_stuff\\get-graphics-offsets64.exe");
            if (!ok) {
                UIHelpers::BeautifulText("OBS hook binaries not found in obs_stuff/. Please add inject-helper64.exe, graphics-hook64.dll, get-graphics-offsets64.exe", UIHelpers::GetWarningColor());
            } else {
                ImGui::TextDisabled("OBS hook binaries detected.");
            }
        }

        // Enumerate open windows and provide a dropdown to pick one
        UIHelpers::CompactSpacer();
        ImGui::Text("Pick From Open Windows:");
        UIHelpers::InfoTooltip("Enumerates top-level visible windows and lets you choose the title.");

        struct EnumCtx { std::vector<std::string>* out; };
        auto enumProc = [](HWND hWnd, LPARAM lParam) -> BOOL {
            if (!IsWindowVisible(hWnd)) return TRUE;
            wchar_t titleW[512]{};
            int n = GetWindowTextW(hWnd, titleW, 511);
            if (n <= 0) return TRUE;
            // Skip tool windows
            LONG ex = GetWindowLongW(hWnd, GWL_EXSTYLE);
            if (ex & WS_EX_TOOLWINDOW) return TRUE;
            EnumCtx* ec = reinterpret_cast<EnumCtx*>(lParam);
            std::wstring w(titleW, n);
            // Convert to UTF-8 without capturing outer state
            std::string u8;
            if (!w.empty()) {
                int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
                if (len > 0) {
                    u8.resize(len);
                    (void)WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), u8.data(), len, nullptr, nullptr);
                }
            }
            if (!u8.empty()) ec->out->push_back(u8);
            return TRUE;
        };

        static std::vector<std::string> s_windowTitles;
        static std::vector<const char*> s_windowItems;
        if (UIHelpers::BeautifulButton("Refresh Window List")) {
            s_windowTitles.clear();
            s_windowItems.clear();
            EnumCtx ec{ &s_windowTitles };
            EnumWindows(enumProc, reinterpret_cast<LPARAM>(&ec));
            s_windowItems.reserve(s_windowTitles.size());
            for (auto& t : s_windowTitles) s_windowItems.push_back(t.c_str());
        }

        // Auto-copy required binaries from source folder
        ImGui::SameLine();
        if (UIHelpers::BeautifulButton("Auto-Copy Binaries")) {
            namespace fs = std::filesystem;
            try {
                fs::create_directories("obs_stuff");
                auto copy_from_base = [&](const std::string& base) {
                    // Try common known subpaths first
                    std::vector<std::pair<std::string, std::string>> direct = {
                        {base + "\\bin\\64bit\\inject-helper64.exe", "obs_stuff\\inject-helper64.exe"},
                        {base + "\\obs-plugins\\64bit\\graphics-hook64.dll", "obs_stuff\\graphics-hook64.dll"},
                        {base + "\\bin\\64bit\\get-graphics-offsets64.exe", "obs_stuff\\get-graphics-offsets64.exe"}
                    };
                    for (auto& f : direct) {
                        if (fs::exists(f.first)) {
                            fs::copy_file(f.first, f.second, fs::copy_options::overwrite_existing);
                        }
                    }
                    auto find_and_copy = [&](const char* name, const char* dst) {
                        if (fs::exists(dst)) return true;
                        std::error_code ec;
                        for (fs::recursive_directory_iterator it(base, fs::directory_options::skip_permission_denied, ec), end; it != end; it.increment(ec)) {
                            if (ec) continue;
                            if (!it->is_regular_file(ec)) continue;
                            if (it->path().filename().string() == name) {
                                fs::copy_file(it->path(), dst, fs::copy_options::overwrite_existing, ec);
                                return fs::exists(dst);
                            }
                        }
                        return false;
                    };
                    (void)find_and_copy("inject-helper64.exe", "obs_stuff\\inject-helper64.exe");
                    (void)find_and_copy("graphics-hook64.dll", "obs_stuff\\graphics-hook64.dll");
                    (void)find_and_copy("get-graphics-offsets64.exe", "obs_stuff\\get-graphics-offsets64.exe");
                };
                std::string base = obs_src_buf[0] ? std::string(obs_src_buf) : std::string("C:\\Program Files\\obs-studio");
                copy_from_base(base);
            } catch (...) {
                // best-effort; UI warning below will still show if missing
            }
        }

        if (!s_windowItems.empty()) {
            int sel = -1;
            for (size_t i = 0; i < s_windowTitles.size(); ++i) {
                if (s_windowTitles[i] == ctx.config.obs_window_title) { sel = (int)i; break; }
            }
            UIHelpers::CompactCombo("Open Windows", &sel, s_windowItems.data(), (int)s_windowItems.size());
            if (sel >= 0 && sel < (int)s_windowTitles.size()) {
                if (ctx.config.obs_window_title != s_windowTitles[sel]) {
                    ctx.config.obs_window_title = s_windowTitles[sel];
                    strncpy(obs_title_buf, ctx.config.obs_window_title.c_str(), sizeof(obs_title_buf) - 1);
                    SAVE_PROFILE();
                }
            }
        } else {
            ImGui::TextDisabled("(List is empty. Click Refresh to scan windows.)");
        }
    }
    
    UIHelpers::CompactSpacer();
    
    
    ImGui::Columns(2, "capture_options", false);
    
    if (UIHelpers::BeautifulToggle("Capture Borders", &ctx.config.capture_borders, "Includes window borders in the screen capture (if applicable)."))
    {
        SAVE_PROFILE();
    }
    
    ImGui::NextColumn();
    
    if (UIHelpers::BeautifulToggle("Capture Cursor", &ctx.config.capture_cursor, "Includes the mouse cursor in the screen capture."))
    {
        SAVE_PROFILE();
    }
    
    ImGui::Columns(1);
    
    UIHelpers::EndCard();
}

static void draw_capture_source_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Capture Source (CUDA Only)");
    
    std::vector<std::string> monitorNames;
    if (monitors == -1)
    {
        monitorNames.push_back("Monitor 1");
    }
    else
    {
        for (int i = -1; i < monitors; ++i)
        {
            monitorNames.push_back("Monitor " + std::to_string(i + 1));
        }
    }

    std::vector<const char*> monitorItems;
    for (const auto& name : monitorNames)
    {
        monitorItems.push_back(name.c_str());
    }

    UIHelpers::CompactCombo("Capture Monitor", &ctx.config.monitor_idx, monitorItems.data(), static_cast<int>(monitorItems.size()));
    UIHelpers::InfoTooltip("Select which monitor to capture from when using CUDA-based screen capture.");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        SAVE_PROFILE();
    }
    
    UIHelpers::EndCard();
}


void draw_capture_settings()
{
    auto& ctx = AppContext::getInstance();

    draw_capture_area_settings();
    UIHelpers::Spacer();

    draw_capture_behavior_settings();
    UIHelpers::Spacer();

    draw_capture_source_settings();
}

