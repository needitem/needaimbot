#pragma once

#include <windows.h>
#include <d3d11.h>
#include <string>
#include <cuda_runtime.h>
#include "hook_info.h"

struct Image {
    int width;
    int height;
    int pitch;
    BYTE* data;
};

class GameCapture {
public:
    GameCapture(int fov_width, int fov_height, int screen_width, int screen_height, const std::string& game_name);
    ~GameCapture();
    
    bool initialize();
    bool StartCapture();
    void StopCapture();
    bool WaitForNextFrame();
    cudaGraphicsResource_t GetCudaResource() const;
    
    // Legacy interface for compatibility
    Image get_frame();
    
private:
    // OBS hook paths (will be auto-detected or use fallback)
    std::wstring inject_path;
    std::wstring hook_path;
    std::string get_graphics_offsets64;
    
    // Screen and capture dimensions
    int screen_width, screen_height;
    int width, height;
    std::string game_name;
    
    // Window and process info
    HWND hwnd;
    DWORD process_id, thread_id;
    
    // Hook synchronization handles
    HANDLE hook_restart, hook_stop, hook_ready, hook_exit, hook_init;
    HANDLE keepalive_mutex, hook_info_map, hook_data_map;
    HANDLE texture_mutexes[2];
    
    // Shared memory structures
    hook_info* shared_hook_info;
    shtex_data* shared_shtex_data;
    
    // D3D11 resources
    ID3D11Device* pDevice;
    ID3D11DeviceContext* pContext;
    ID3D11Resource* pSharedResource;
    ID3D11Texture2D* pStagingTexture;
    ID3D11Texture2D* m_cudaTexture;  // For CUDA interop
    
    // CUDA interop
    cudaGraphicsResource_t m_cudaResource;
    cudaStream_t m_captureStream;
    
    // Capture region
    D3D11_BOX sourceRegion;
    
    // Frame buffer (legacy)
    BYTE* FrameBuffer;
    
    // Helper methods
    HANDLE inject_hook(DWORD target_id);
    HANDLE OpenMapPlusId(const std::wstring& base_name, DWORD id);
    HANDLE OpenDataMap(uint32_t window, uint32_t map_id);
    HANDLE OpenEventPlusId(const std::wstring& base_name, DWORD id);
    HANDLE OpenMutexPlusId(const std::wstring& base_name, DWORD id);
    HANDLE CreateKeepaliveMutex(int pid);
    D3D11_BOX get_region();
    void initialize_offsets();
    std::string run_get_graphics_offsets();
    bool initializeCUDAInterop();
    void cleanup();
    bool findOBSFiles();
    
    // Enhanced error handling and detection
    bool detectAntiCheat();
    void diagnoseInjectionError(DWORD error);
    bool checkOBSVersion();
    bool isProcessElevated(HANDLE hProcess);
    
    // Performance optimization
    uint64_t last_frame_time;
    uint32_t frame_skip_count;
    static constexpr uint32_t MAX_FRAME_SKIP = 3;
};