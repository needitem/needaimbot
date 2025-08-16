#pragma once
#include <cstdint>

struct d3d8_offsets {
    uint32_t present{};
};

struct d3d9_offsets {
    uint32_t present{};
    uint32_t present_ex{};
    uint32_t present_swap{};
    uint32_t d3d9_clsoff{};
    uint32_t is_d3d9ex_clsoff{};
};

struct d3d12_offsets {
    uint32_t execute_command_lists{};
};

struct dxgi_offsets {
    uint32_t present{};
    uint32_t resize{};
    uint32_t present1{};
};

struct dxgi_offsets2 {
    uint32_t release{};
};

struct ddraw_offsets {
    uint32_t surface_create{};
    uint32_t surface_restore{};
    uint32_t surface_release{};
    uint32_t surface_unlock{};
    uint32_t surface_blt{};
    uint32_t surface_flip{};
    uint32_t surface_set_palette{};
    uint32_t palette_set_entries{};
};

struct graphics_offsets {
    d3d8_offsets d3d8;
    d3d9_offsets d3d9;
    dxgi_offsets dxgi;
    ddraw_offsets ddraw;
    dxgi_offsets2 dxgi2;
    d3d12_offsets d3d12;
};

struct shmem_data {
    volatile int last_tex;
    uint32_t tex1_offset;
    uint32_t tex2_offset;
};


struct shtex_data {
    uint32_t tex_handle;
};


enum capture_type {
    CAPTURE_TYPE_MEMORY,
    CAPTURE_TYPE_TEXTURE,
};

struct hook_info {
    /* hook version */
    uint32_t hook_ver_major;
    uint32_t hook_ver_minor;

    /* capture info */
    enum capture_type type;
    uint32_t window;
    uint32_t format;
    uint32_t cx;
    uint32_t cy;
    uint32_t UNUSED_base_cx;
    uint32_t UNUSED_base_cy;
    uint32_t pitch;
    uint32_t map_id;
    uint32_t map_size;
    bool flip;

    /* additional options */
    uint64_t frame_interval;
    bool UNUSED_use_scale;
    bool force_shmem;
    bool capture_overlay;
    bool allow_srgb_alias;

    /* hook addresses */
    graphics_offsets offsets;

    uint32_t reserved[126];
};
