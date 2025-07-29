#ifndef DRAW_SETTINGS_H
#define DRAW_SETTINGS_H

// Macro for saving settings to active profile
#define SAVE_PROFILE() do { \
    AppContext::getInstance().config.saveActiveProfile(); \
} while(0)

// Macro for saving weapon profiles (now saves with the active profile)
#define SAVE_WEAPON_PROFILE() do { \
    AppContext::getInstance().config.saveActiveProfile(); \
} while(0)

void draw_capture_settings();
void draw_target();
void draw_mouse();
void draw_ai();
void draw_buttons();
void draw_overlay();
void draw_debug();
void draw_profile();
void draw_stats();
void draw_color_filter_settings();
void draw_rcs_settings();
void load_body_texture();
void release_body_texture();

#endif 
