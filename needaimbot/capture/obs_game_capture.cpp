// OBS Game Capture Hook Implementation
// Uses OBS's internal hooking mechanism for ultra-low latency capture

#include "obs_game_capture.h"
#include "../AppContext.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

OBSGameCapture::OBSGameCapture(int fov_width, int fov_height, const std::string& game_name)
    : width(fov_width), height(fov_height), game_name(game_name),
      hwnd(nullptr), process_id(0), thread_id(0),
      shared_hook_info(nullptr), shared_shtex_data(nullptr),
      m_cudaResource(nullptr), m_captureStream(nullptr),
      m_isCapturing(false), m_frameAvailable(false),
      m_currentTextureIndex(0) {
    
    // Get screen dimensions
    screen_width = GetSystemMetrics(SM_CXSCREEN);
    screen_height = GetSystemMetrics(SM_CYSCREEN);
    
    // Allocate frame buffer
    int bufferSize = width * height * 4;  // BGRA format
    FrameBuffer = std::make_unique<BYTE[]>(bufferSize);
    
    frame.width = width;
    frame.height = height;
    frame.pitch = width * 4;
    frame.data = FrameBuffer.get();
}

OBSGameCapture::~OBSGameCapture() {
    StopCapture();
    
    // Cleanup CUDA resources
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    
    if (m_captureStream) {
        cudaStreamDestroy(m_captureStream);
        m_captureStream = nullptr;
    }
    
    // Cleanup shared memory
    if (shared_hook_info) {
        UnmapViewOfFile(shared_hook_info);
        shared_hook_info = nullptr;
    }
    
    if (shared_shtex_data) {
        UnmapViewOfFile(shared_shtex_data);
        shared_shtex_data = nullptr;
    }
    
    // Close handles
    if (hook_info_map) CloseHandle(hook_info_map);
    if (hook_data_map) CloseHandle(hook_data_map);
    if (keepalive_mutex) CloseHandle(keepalive_mutex);
    if (hook_restart) CloseHandle(hook_restart);
    if (hook_stop) CloseHandle(hook_stop);
    if (hook_ready) CloseHandle(hook_ready);
    if (hook_exit) CloseHandle(hook_exit);
    if (hook_init) CloseHandle(hook_init);
    
    for (int i = 0; i < 2; i++) {
        if (texture_mutexes[i]) CloseHandle(texture_mutexes[i]);
    }
}

bool OBSGameCapture::Initialize() {
    std::cout << "[OBSGameCapture] Initializing for game: " << game_name << std::endl;
    
    // Find game window
    hwnd = FindWindowA(nullptr, game_name.c_str());
    if (!hwnd) {
        std::cerr << "[OBSGameCapture] Failed to find game window: " << game_name << std::endl;
        return false;
    }
    
    // Get process ID
    thread_id = GetWindowThreadProcessId(hwnd, &process_id);
    if (!process_id) {
        std::cerr << "[OBSGameCapture] Failed to get process ID" << std::endl;
        return false;
    }
    
    std::cout << "[OBSGameCapture] Found game window - PID: " << process_id << std::endl;
    
    // Initialize graphics offsets
    initialize_offsets();
    
    // Initialize D3D11
    if (!InitializeD3D11()) {
        std::cerr << "[OBSGameCapture] Failed to initialize D3D11" << std::endl;
        return false;
    }
    
    // Open/Create shared memory and events
    hook_restart = OpenEventPlusId(HOOK_RESTART_EVENT, process_id);
    hook_stop = OpenEventPlusId(HOOK_STOP_EVENT, process_id);
    hook_ready = OpenEventPlusId(HOOK_READY_EVENT, process_id);
    hook_exit = OpenEventPlusId(HOOK_EXIT_EVENT, process_id);
    hook_init = OpenEventPlusId(HOOK_INIT_EVENT, process_id);
    
    if (!hook_restart || !hook_stop || !hook_ready || !hook_exit || !hook_init) {
        std::cerr << "[OBSGameCapture] Failed to open hook events" << std::endl;
        return false;
    }
    
    // Create keepalive mutex
    keepalive_mutex = CreateKeepaliveMutex(process_id);
    if (!keepalive_mutex) {
        std::cerr << "[OBSGameCapture] Failed to create keepalive mutex" << std::endl;
        return false;
    }
    
    // Open hook info map
    hook_info_map = OpenMapPlusId(HOOK_INFO_MAP, process_id);
    if (!hook_info_map) {
        std::cerr << "[OBSGameCapture] Failed to open hook info map" << std::endl;
        return false;
    }
    
    // Map shared hook info
    shared_hook_info = (hook_info*)MapViewOfFile(hook_info_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(hook_info));
    if (!shared_hook_info) {
        std::cerr << "[OBSGameCapture] Failed to map hook info" << std::endl;
        return false;
    }
    
    std::cout << "[OBSGameCapture] Initialization complete" << std::endl;
    return true;
}

bool OBSGameCapture::StartCapture() {
    std::cout << "[OBSGameCapture] Starting capture..." << std::endl;
    
    // Inject hook into game process
    HANDLE inject_process = inject_hook(process_id);
    if (!inject_process) {
        std::cerr << "[OBSGameCapture] Failed to inject hook" << std::endl;
        return false;
    }
    
    // Wait for hook to be ready
    std::cout << "[OBSGameCapture] Waiting for hook to initialize..." << std::endl;
    DWORD result = WaitForSingleObject(hook_ready, 5000);
    if (result != WAIT_OBJECT_0) {
        std::cerr << "[OBSGameCapture] Hook initialization timeout" << std::endl;
        CloseHandle(inject_process);
        return false;
    }
    
    // Setup shared hook info
    shared_hook_info->offsets_version = 0;
    shared_hook_info->feeder_version = 0;
    shared_hook_info->version = 0;
    shared_hook_info->window = (uint32_t)(uintptr_t)hwnd;
    shared_hook_info->type = CAPTURE_TYPE_TEXTURE;  // Use texture mode for GPU
    shared_hook_info->format = 0;
    shared_hook_info->flip = 0;
    shared_hook_info->map_id = 0;
    shared_hook_info->map_size = 0;
    shared_hook_info->capture_overlay = true;
    shared_hook_info->force_shmem = false;
    shared_hook_info->use_scale = false;
    shared_hook_info->active = true;
    
    // Signal hook to start
    SetEvent(hook_init);
    
    // Wait for hook to setup shared textures
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Open data map
    hook_data_map = OpenDataMap((uint32_t)(uintptr_t)hwnd, shared_hook_info->map_id);
    if (!hook_data_map) {
        std::cerr << "[OBSGameCapture] Failed to open data map" << std::endl;
        CloseHandle(inject_process);
        return false;
    }
    
    // Map shared texture data
    shared_shtex_data = (shtex_data*)MapViewOfFile(hook_data_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(shtex_data));
    if (!shared_shtex_data) {
        std::cerr << "[OBSGameCapture] Failed to map texture data" << std::endl;
        CloseHandle(inject_process);
        return false;
    }
    
    // Open texture mutexes
    for (int i = 0; i < 2; i++) {
        wchar_t mutex_name[256];
        swprintf_s(mutex_name, L"%s%u_%u", TEXTURE_MUTEX, process_id, i);
        texture_mutexes[i] = OpenMutexW(SYNCHRONIZE, FALSE, mutex_name);
        if (!texture_mutexes[i]) {
            std::cerr << "[OBSGameCapture] Failed to open texture mutex " << i << std::endl;
            CloseHandle(inject_process);
            return false;
        }
    }
    
    // Setup shared textures
    bool textureSetup = false;
    for (int i = 0; i < 10; i++) {  // Try multiple times
        if (shared_shtex_data->tex_handle[0] != 0) {
            if (SetupSharedTexture((HANDLE)(uintptr_t)shared_shtex_data->tex_handle[0])) {
                textureSetup = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (!textureSetup) {
        std::cerr << "[OBSGameCapture] Failed to setup shared texture" << std::endl;
        CloseHandle(inject_process);
        return false;
    }
    
    // Initialize CUDA interop
    if (!InitializeCUDAInterop()) {
        std::cerr << "[OBSGameCapture] Failed to initialize CUDA interop" << std::endl;
        CloseHandle(inject_process);
        return false;
    }
    
    CloseHandle(inject_process);
    m_isCapturing = true;
    
    std::cout << "[OBSGameCapture] Capture started successfully" << std::endl;
    return true;
}

void OBSGameCapture::StopCapture() {
    if (m_isCapturing) {
        std::cout << "[OBSGameCapture] Stopping capture..." << std::endl;
        
        // Signal hook to stop
        if (hook_stop) {
            SetEvent(hook_stop);
        }
        
        // Mark as inactive
        if (shared_hook_info) {
            shared_hook_info->active = false;
        }
        
        m_isCapturing = false;
    }
}

bool OBSGameCapture::WaitForNextFrame() {
    if (!m_isCapturing || !shared_hook_info || !shared_shtex_data) {
        return false;
    }
    
    // Check which texture is current
    int currentIdx = m_currentTextureIndex.load();
    int nextIdx = (currentIdx + 1) % 2;
    
    // Try to acquire mutex for next texture
    DWORD result = WaitForSingleObject(texture_mutexes[nextIdx], 0);
    if (result == WAIT_OBJECT_0) {
        // New frame available
        HANDLE sharedHandle = (HANDLE)(uintptr_t)shared_shtex_data->tex_handle[nextIdx];
        if (sharedHandle != 0) {
            // Open shared texture
            ComPtr<ID3D11Texture2D> sharedTexture;
            HRESULT hr = pDevice->OpenSharedResource(sharedHandle, IID_PPV_ARGS(&sharedTexture));
            
            if (SUCCEEDED(hr)) {
                // Copy to our texture
                pContext->CopySubresourceRegion(
                    pGPUTexture.Get(), 0,
                    0, 0, 0,
                    sharedTexture.Get(), 0,
                    &sourceRegion
                );
                
                m_currentTextureIndex.store(nextIdx);
                m_frameAvailable.store(true);
                
                ReleaseMutex(texture_mutexes[nextIdx]);
                
                static int frameCount = 0;
                frameCount++;
                if (frameCount <= 10 || frameCount % 100 == 0) {
                    std::cout << "[OBSGameCapture] Frame #" << frameCount << " captured" << std::endl;
                }
                
                return true;
            }
        }
        
        ReleaseMutex(texture_mutexes[nextIdx]);
    }
    
    return false;
}

bool OBSGameCapture::InitializeD3D11() {
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    
    D3D_FEATURE_LEVEL featureLevel;
    UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    
#ifdef _DEBUG
    createFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        createFlags,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &pDevice,
        &featureLevel,
        &pContext
    );
    
    if (FAILED(hr)) {
        std::cerr << "[OBSGameCapture] Failed to create D3D11 device: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // Create GPU texture for CUDA interop
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    hr = pDevice->CreateTexture2D(&desc, nullptr, &pGPUTexture);
    if (FAILED(hr)) {
        std::cerr << "[OBSGameCapture] Failed to create GPU texture: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // Create staging texture for CPU access (if needed)
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    
    hr = pDevice->CreateTexture2D(&desc, nullptr, &pStagingTexture);
    if (FAILED(hr)) {
        std::cerr << "[OBSGameCapture] Failed to create staging texture: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // Setup source region for capture
    sourceRegion = get_region();
    
    return true;
}

bool OBSGameCapture::InitializeCUDAInterop() {
    // Create CUDA stream
    cudaError_t err = cudaStreamCreate(&m_captureStream);
    if (err != cudaSuccess) {
        std::cerr << "[OBSGameCapture] Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Register D3D11 texture with CUDA
    err = cudaGraphicsD3D11RegisterResource(
        &m_cudaResource,
        pGPUTexture.Get(),
        cudaGraphicsRegisterFlagsNone
    );
    
    if (err != cudaSuccess) {
        std::cerr << "[OBSGameCapture] Failed to register D3D11 resource with CUDA: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool OBSGameCapture::SetupSharedTexture(HANDLE sharedHandle) {
    if (!sharedHandle || !pDevice) {
        return false;
    }
    
    // Open shared texture from game
    HRESULT hr = pDevice->OpenSharedResource(sharedHandle, IID_PPV_ARGS(&pSharedResource));
    if (FAILED(hr)) {
        std::cerr << "[OBSGameCapture] Failed to open shared resource: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    std::cout << "[OBSGameCapture] Shared texture opened successfully" << std::endl;
    return true;
}

HANDLE OBSGameCapture::inject_hook(DWORD target_id) {
    // Build command line for inject helper
    std::wostringstream commandLine;
    commandLine << inject_path << L" ";
    commandLine << hook_path << L" ";
    commandLine << target_id << L" ";
    commandLine << process_id;  // Parent process ID
    
    // Start inject helper process
    STARTUPINFOW si = {};
    PROCESS_INFORMATION pi = {};
    si.cb = sizeof(si);
    
    std::wstring cmdLine = commandLine.str();
    
    std::wcout << L"[OBSGameCapture] Injecting with command: " << cmdLine << std::endl;
    
    BOOL result = CreateProcessW(
        nullptr,
        (LPWSTR)cmdLine.c_str(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &si,
        &pi
    );
    
    if (!result) {
        std::cerr << "[OBSGameCapture] Failed to start inject helper: " << GetLastError() << std::endl;
        return nullptr;
    }
    
    CloseHandle(pi.hThread);
    return pi.hProcess;
}

HANDLE OBSGameCapture::OpenMapPlusId(const std::wstring& base_name, DWORD id) {
    wchar_t name[256];
    swprintf_s(name, L"%s%u", base_name.c_str(), id);
    
    HANDLE handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (!handle) {
        // Try to create it
        handle = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            0,
            sizeof(hook_info),
            name
        );
    }
    
    return handle;
}

HANDLE OBSGameCapture::OpenDataMap(uint32_t window, uint32_t map_id) {
    wchar_t name[256];
    swprintf_s(name, L"%s%u_%u", HOOK_DATA_MAP, window, map_id);
    
    return OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name);
}

HANDLE OBSGameCapture::OpenEventPlusId(const std::wstring& base_name, DWORD id) {
    wchar_t name[256];
    swprintf_s(name, L"%s%u", base_name.c_str(), id);
    
    HANDLE handle = OpenEventW(EVENT_ALL_ACCESS, FALSE, name);
    if (!handle) {
        handle = CreateEventW(nullptr, FALSE, FALSE, name);
    }
    
    return handle;
}

HANDLE OBSGameCapture::OpenMutexPlusId(const std::wstring& base_name, DWORD id) {
    wchar_t name[256];
    swprintf_s(name, L"%s%u", base_name.c_str(), id);
    
    HANDLE handle = OpenMutexW(SYNCHRONIZE, FALSE, name);
    if (!handle) {
        handle = CreateMutexW(nullptr, FALSE, name);
    }
    
    return handle;
}

HANDLE OBSGameCapture::CreateKeepaliveMutex(int pid) {
    wchar_t name[256];
    swprintf_s(name, L"%s%u", KEEPALIVE_MUTEX, pid);
    
    return CreateMutexW(nullptr, FALSE, name);
}

D3D11_BOX OBSGameCapture::get_region() {
    D3D11_BOX region = {};
    
    // Calculate center region
    int centerX = screen_width / 2;
    int centerY = screen_height / 2;
    
    region.left = centerX - (width / 2);
    region.top = centerY - (height / 2);
    region.right = region.left + width;
    region.bottom = region.top + height;
    region.front = 0;
    region.back = 1;
    
    // Apply offset if needed
    auto& ctx = AppContext::getInstance();
    region.left += (UINT)ctx.config.crosshair_offset_x;
    region.right += (UINT)ctx.config.crosshair_offset_x;
    region.top += (UINT)ctx.config.crosshair_offset_y;
    region.bottom += (UINT)ctx.config.crosshair_offset_y;
    
    return region;
}

void OBSGameCapture::initialize_offsets() {
    // Run get-graphics-offsets to get hook offsets
    std::string output = run_get_graphics_offsets();
    if (!output.empty()) {
        std::cout << "[OBSGameCapture] Graphics offsets: " << output << std::endl;
    }
}

std::string OBSGameCapture::run_get_graphics_offsets() {
    // This would run the get-graphics-offsets64.exe to get proper offsets
    // For now, return empty (OBS will use defaults)
    return "";
}

Image OBSGameCapture::get_frame() {
    // Legacy interface for compatibility
    if (WaitForNextFrame()) {
        // Copy from GPU texture to CPU if needed
        pContext->CopyResource(pStagingTexture.Get(), pGPUTexture.Get());
        
        D3D11_MAPPED_SUBRESOURCE mapped;
        HRESULT hr = pContext->Map(pStagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (SUCCEEDED(hr)) {
            BYTE* src = (BYTE*)mapped.pData;
            BYTE* dst = frame.data;
            
            for (int y = 0; y < height; y++) {
                memcpy(dst + y * frame.pitch, 
                       src + y * mapped.RowPitch, 
                       frame.pitch);
            }
            
            pContext->Unmap(pStagingTexture.Get(), 0);
        }
    }
    
    return frame;
}

bool OBSGameCapture::SaveBMP(const char* filename, const Image& img) {
    // BMP save implementation (if needed)
    return false;
}