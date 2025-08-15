#pragma once

#include <windows.h>
#include <d3d11.h>
#include <string>
#include <memory>
#include <atomic>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

// OBS hook structures
struct hook_info {
    uint32_t offsets_version;
    uint32_t feeder_version;
    uint32_t version;
    uint32_t window;
    uint32_t type;
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
    uint32_t tex_handle[2];
};

struct Image {
    int width;
    int height;
    int pitch;   
    BYTE* data;  
};

using Microsoft::WRL::ComPtr;

class OBSGameCapture {
private:
    // OBS paths
    std::wstring inject_path = L"obs_stuff\\inject-helper64.exe";
    std::wstring hook_path = L"obs_stuff\\graphics-hook64.dll";
    const char* get_graphics_offsets64 = R"(obs_stuff\get-graphics-offsets64.exe)";

    // Window and process info
    HWND hwnd;
    DWORD process_id, thread_id;
    std::string game_name;
    int width, height;
    int screen_width, screen_height;
    
    // OBS hook handles
    HANDLE hook_restart, hook_stop, hook_ready, hook_exit, hook_init;
    HANDLE keepalive_mutex, hook_info_map, hook_data_map;
    HANDLE texture_mutexes[2];
    
    // Shared memory
    hook_info* shared_hook_info;
    shtex_data* shared_shtex_data;
    
    // D3D11 resources
    ComPtr<ID3D11Device> pDevice;
    ComPtr<ID3D11DeviceContext> pContext;
    ComPtr<ID3D11Resource> pSharedResource;
    ComPtr<ID3D11Texture2D> pStagingTexture;
    ComPtr<ID3D11Texture2D> pGPUTexture;  // For CUDA interop
    
    // CUDA interop
    cudaGraphicsResource_t m_cudaResource;
    cudaStream_t m_captureStream;
    
    // Frame data
    Image frame;
    std::unique_ptr<BYTE[]> FrameBuffer;
    D3D11_BOX sourceRegion;
    
    // State
    std::atomic<bool> m_isCapturing;
    std::atomic<bool> m_frameAvailable;
    std::atomic<int> m_currentTextureIndex;
    
    // Helper functions
    HANDLE inject_hook(DWORD target_id);
    HANDLE OpenMapPlusId(const std::wstring& base_name, DWORD id);
    HANDLE OpenDataMap(uint32_t window, uint32_t map_id);
    HANDLE OpenEventPlusId(const std::wstring& base_name, DWORD id);
    HANDLE OpenMutexPlusId(const std::wstring& base_name, DWORD id);
    HANDLE CreateKeepaliveMutex(int pid);
    D3D11_BOX get_region();
    void initialize_offsets();
    std::string run_get_graphics_offsets();
    
    bool InitializeD3D11();
    bool InitializeCUDAInterop();
    bool SetupSharedTexture(HANDLE sharedHandle);
    
public:
    OBSGameCapture(int fov_width, int fov_height, const std::string& game_name);
    ~OBSGameCapture();
    
    bool Initialize();
    bool StartCapture();
    void StopCapture();
    bool WaitForNextFrame();
    
    cudaGraphicsResource_t GetCudaResource() const { return m_cudaResource; }
    bool IsCapturing() const { return m_isCapturing; }
    
    // Legacy interface
    Image get_frame();
    bool SaveBMP(const char* filename, const Image& img);
};