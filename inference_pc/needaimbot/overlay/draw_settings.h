#ifndef DRAW_SETTINGS_H
#define DRAW_SETTINGS_H

#include <chrono>

// Auto-save system: tracks dirty state and saves after delay
struct AutoSaveState {
    bool isDirty = false;
    std::chrono::steady_clock::time_point lastChangeTime;
    static constexpr float SAVE_DELAY_SECONDS = 0.5f;  // Save 0.5s after last change
    
    void markDirty() {
        isDirty = true;
        lastChangeTime = std::chrono::steady_clock::now();
    }
    
    bool shouldSave() const {
        if (!isDirty) return false;
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - lastChangeTime).count();
        return elapsed >= SAVE_DELAY_SECONDS;
    }
    
    void reset() {
        isDirty = false;
    }
};

// Global auto-save state
extern AutoSaveState g_autoSaveState;

// Macro for marking config as dirty (triggers auto-save after delay)
#define MARK_CONFIG_DIRTY() do { \
    g_autoSaveState.markDirty(); \
} while(0)

// Macro for immediately saving config
#define SAVE_PROFILE() do { \
    AppContext::getInstance().config.saveConfig(); \
} while(0)

// Macro for immediately saving input profile
#define SAVE_INPUT_PROFILE() do { \
    AppContext::getInstance().config.saveConfig(); \
} while(0)

void draw_capture_settings();
void draw_target();
void draw_mouse();
void draw_ai();
void draw_buttons();
void draw_overlay();
void draw_debug();
void draw_profile();
void draw_color_filter();
void draw_stabilizer();
void load_body_texture();
void release_body_texture();

#endif 
