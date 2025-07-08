#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <detours.h>
#include <iostream>
#include <memory>
#include "hook_info.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "detours.lib")

static HMODULE g_hModule = nullptr;
static hook_info* g_hook_info = nullptr;
static shtex_data* g_shtex_data = nullptr;
static HANDLE g_texture_mutex = nullptr;
static HANDLE g_hook_ready = nullptr;
static HANDLE g_hook_stop = nullptr;
static HANDLE g_hook_info_map = nullptr;
static HANDLE g_hook_data_map = nullptr;
static ID3D11Device* g_device = nullptr;
static ID3D11DeviceContext* g_context = nullptr;
static ID3D11Texture2D* g_shared_texture = nullptr;
static DXGI_FORMAT g_format = DXGI_FORMAT_B8G8R8A8_UNORM;
static uint32_t g_cx = 0, g_cy = 0;
static uint32_t g_window = 0;
static uint32_t g_map_id = 0;
static bool g_initialized = false;

// Original function pointers
typedef HRESULT(WINAPI* PFN_PRESENT)(IDXGISwapChain* pSwapChain, UINT SyncInterval, UINT Flags);
typedef HRESULT(WINAPI* PFN_RESIZE_BUFFERS)(IDXGISwapChain* pSwapChain, UINT BufferCount, UINT Width, UINT Height, DXGI_FORMAT NewFormat, UINT SwapChainFlags);

static PFN_PRESENT g_original_present = nullptr;
static PFN_RESIZE_BUFFERS g_original_resize_buffers = nullptr;

std::wstring GetEventName(const wchar_t* base_name, DWORD process_id) {
    return std::wstring(base_name) + std::to_wstring(process_id);
}

std::wstring GetTextureName(uint32_t window, uint32_t map_id) {
    return L"CaptureHook_Texture_" + std::to_wstring(window) + L"_" + std::to_wstring(map_id);
}

bool InitializeSharedMemory() {
    DWORD process_id = GetCurrentProcessId();
    
    // Create minimal events for performance
    g_hook_stop = CreateEventW(nullptr, FALSE, FALSE, GetEventName(L"CaptureHook_Stop", process_id).c_str());
    g_hook_ready = CreateEventW(nullptr, FALSE, FALSE, GetEventName(L"CaptureHook_HookReady", process_id).c_str());
    
    // Create single mutex for texture synchronization
    g_texture_mutex = CreateMutexW(nullptr, FALSE, GetEventName(L"CaptureHook_TextureMutex", process_id).c_str());
    
    // Create hook info shared memory
    g_hook_info_map = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(hook_info), GetEventName(L"CaptureHook_HookInfo", process_id).c_str());
    if (!g_hook_info_map) return false;
    
    g_hook_info = static_cast<hook_info*>(MapViewOfFile(g_hook_info_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(hook_info)));
    if (!g_hook_info) return false;
    
    memset(g_hook_info, 0, sizeof(hook_info));
    g_hook_info->hook_ver_major = 1;
    g_hook_info->hook_ver_minor = 0;
    g_hook_info->type = CAPTURE_TYPE_TEXTURE;
    g_hook_info->format = g_format;
    g_hook_info->flip = false;
    g_hook_info->force_shmem = false;
    g_hook_info->capture_overlay = false;
    g_hook_info->allow_srgb_alias = true;
    
    return true;
}

bool CreateSharedTexture(uint32_t width, uint32_t height) {
    if (!g_device || !g_context) return false;
    
    // Create shared texture
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = g_format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    HRESULT hr = g_device->CreateTexture2D(&desc, nullptr, &g_shared_texture);
    if (FAILED(hr)) return false;
    
    // Get shared handle
    IDXGIResource* dxgi_resource = nullptr;
    hr = g_shared_texture->QueryInterface(__uuidof(IDXGIResource), (void**)&dxgi_resource);
    if (FAILED(hr)) return false;
    
    HANDLE shared_handle = nullptr;
    hr = dxgi_resource->GetSharedHandle(&shared_handle);
    dxgi_resource->Release();
    if (FAILED(hr)) return false;
    
    // Update hook info
    g_hook_info->cx = width;
    g_hook_info->cy = height;
    g_hook_info->window = g_window;
    g_hook_info->map_id = g_map_id;
    g_hook_info->map_size = sizeof(shtex_data);
    
    // Create texture data shared memory
    std::wstring texture_name = GetTextureName(g_window, g_map_id);
    g_hook_data_map = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(shtex_data), texture_name.c_str());
    if (!g_hook_data_map) return false;
    
    g_shtex_data = static_cast<shtex_data*>(MapViewOfFile(g_hook_data_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(shtex_data)));
    if (!g_shtex_data) return false;
    
    g_shtex_data->tex_handle = (uint32_t)(uintptr_t)shared_handle;
    
    return true;
}

bool CaptureFrame(IDXGISwapChain* swap_chain) {
    if (!g_initialized) return false;
    
    // Get back buffer
    ID3D11Texture2D* back_buffer = nullptr;
    HRESULT hr = swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer);
    if (FAILED(hr)) return false;
    
    // Copy to shared texture with immediate flush for low latency
    g_context->CopyResource(g_shared_texture, back_buffer);
    g_context->Flush(); // Force immediate GPU processing
    back_buffer->Release();
    
    return true;
}

HRESULT WINAPI HookPresent(IDXGISwapChain* pSwapChain, UINT SyncInterval, UINT Flags) {
    if (g_initialized) {
        CaptureFrame(pSwapChain);
    }
    return g_original_present(pSwapChain, SyncInterval, Flags);
}

HRESULT WINAPI HookResizeBuffers(IDXGISwapChain* pSwapChain, UINT BufferCount, UINT Width, UINT Height, DXGI_FORMAT NewFormat, UINT SwapChainFlags) {
    if (g_initialized && Width > 0 && Height > 0) {
        // Release old texture
        if (g_shared_texture) {
            g_shared_texture->Release();
            g_shared_texture = nullptr;
        }
        
        // Create new texture with new dimensions
        CreateSharedTexture(Width, Height);
    }
    return g_original_resize_buffers(pSwapChain, BufferCount, Width, Height, NewFormat, SwapChainFlags);
}

bool InitializeHook() {
    // Get D3D11 device and context from swap chain
    IDXGISwapChain* swap_chain = nullptr;
    // This is a simplified version - in practice you'd need to find the swap chain
    // through various methods like COM hooking or by intercepting D3D11 creation
    
    if (!InitializeSharedMemory()) return false;
    
    // Skip initialization wait for immediate startup
    
    // Set up window and map ID
    g_window = (uint32_t)GetCurrentProcessId();
    g_map_id = GetTickCount();
    
    // Try to get device from current context
    // This is simplified - normally you'd hook D3D11 creation functions
    
    // Signal ready
    SetEvent(g_hook_ready);
    
    g_initialized = true;
    return true;
}

void Cleanup() {
    if (g_shared_texture) g_shared_texture->Release();
    if (g_context) g_context->Release();
    if (g_device) g_device->Release();
    if (g_shtex_data) UnmapViewOfFile(g_shtex_data);
    if (g_hook_info) UnmapViewOfFile(g_hook_info);
    if (g_hook_data_map) CloseHandle(g_hook_data_map);
    if (g_hook_info_map) CloseHandle(g_hook_info_map);
    if (g_texture_mutex) CloseHandle(g_texture_mutex);
    if (g_hook_stop) CloseHandle(g_hook_stop);
    if (g_hook_ready) CloseHandle(g_hook_ready);
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hModule);
        g_hModule = hModule;
        
        // Start initialization in a separate thread
        CreateThread(nullptr, 0, [](LPVOID) -> DWORD {
            if (InitializeHook()) {
                // Hook DXGI Present function
                // This requires proper COM vtable hooking
                // Simplified version here
            }
            return 0;
        }, nullptr, 0, nullptr);
        break;
        
    case DLL_PROCESS_DETACH:
        Cleanup();
        break;
    }
    return TRUE;
}