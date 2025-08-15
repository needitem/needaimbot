#pragma once

#include <stdint.h>

// OBS Hook Info structures
// These must match exactly with OBS's internal structures

struct hook_info {
    uint32_t offsets_version;
    uint32_t feeder_version;
    uint32_t version;
    uint32_t window;
    uint32_t type;      // 0 = CAPTURE_TYPE_MEMORY, 1 = CAPTURE_TYPE_TEXTURE
    uint32_t format;
    uint32_t flip;
    uint32_t map_id;
    uint32_t map_size;
    bool capture_overlay;
    bool force_shmem;
    bool use_scale;
    bool active;
};

struct shtex_data {
    uint32_t tex_handle[2];  // Shared texture handles
};

// Capture types
enum capture_type {
    CAPTURE_TYPE_MEMORY = 0,
    CAPTURE_TYPE_TEXTURE = 1
};

// Graphics offsets structure (from OBS)
struct graphics_offsets {
    uint32_t d3d11;
    uint32_t d3d12;
    uint32_t dxgi;
};

// Event names used by OBS
#define HOOK_RESTART_EVENT L"CaptureHook_Restart"
#define HOOK_STOP_EVENT L"CaptureHook_Stop"
#define HOOK_READY_EVENT L"CaptureHook_Ready"
#define HOOK_EXIT_EVENT L"CaptureHook_Exit"
#define HOOK_INIT_EVENT L"CaptureHook_Init"

#define HOOK_INFO_MAP L"CaptureHook_HookInfo"
#define HOOK_DATA_MAP L"CaptureHook_Data"
#define KEEPALIVE_MUTEX L"CaptureHook_KeepAlive"
#define TEXTURE_MUTEX L"CaptureHook_TextureMutex"