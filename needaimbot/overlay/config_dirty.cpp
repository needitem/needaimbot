#include "config_dirty.h"

#include "../config/config.h"
#include "../AppContext.h"
#include <imgui/imgui.h>

namespace {
    bool g_configDirty = false;
    double g_configDirtyTime = 0.0;
    constexpr double kSaveDelaySec = 0.35;  // 350ms delay before saving
}

void OverlayConfig_MarkDirty()
{
    g_configDirty = true;
    g_configDirtyTime = ImGui::GetTime();
}

void OverlayConfig_TrySave(const char* filename)
{
    if (!g_configDirty)
        return;

    const double now = ImGui::GetTime();
    if ((now - g_configDirtyTime) < kSaveDelaySec)
        return;

    // Don't save while user is actively editing
    if (ImGui::IsAnyItemActive())
        return;

    auto& ctx = AppContext::getInstance();
    ctx.config.saveConfig(filename ? filename : "");
    g_configDirty = false;
}
