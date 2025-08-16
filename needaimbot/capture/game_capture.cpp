#include <sstream>
#include <windows.h>
#include <d3d11.h>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <vector>
#include <algorithm>
#include <tlhelp32.h>
#include <chrono>
#include "game_capture.h"
#include "../AppContext.h"

GameCapture::GameCapture(int fw, int fh, int sw, int sh, const std::string& game) :
    screen_width(sw), screen_height(sh), width(fw), height(fh), game_name(game), hwnd(nullptr), process_id(0), thread_id(0),
    hook_restart(nullptr), hook_stop(nullptr), hook_ready(nullptr), hook_exit(nullptr), hook_init(nullptr),
    keepalive_mutex(nullptr), hook_info_map(nullptr), hook_data_map(nullptr),
    shared_hook_info(nullptr), shared_shtex_data(nullptr),
    pDevice(nullptr), pContext(nullptr), pSharedResource(nullptr), pStagingTexture(nullptr), 
    texture_mutexes{ nullptr, nullptr },
    m_cudaResource(nullptr), m_captureStream(nullptr), m_cudaTexture(nullptr),
    last_frame_time(0), frame_skip_count(0)
{
    FrameBuffer = nullptr;
    
    // Try to find OBS installation paths
    if (!findOBSFiles()) {
        std::cerr << "[GameCapture] WARNING: Could not find OBS files automatically" << std::endl;
        std::cerr << "[GameCapture] Please ensure OBS Studio is installed or place files in obs_stuff folder" << std::endl;
    }
    
    // Check OBS version compatibility
    if (!checkOBSVersion()) {
        std::cerr << "[GameCapture] WARNING: OBS version compatibility check failed" << std::endl;
    }
}

GameCapture::~GameCapture() {
    cleanup();
}

void GameCapture::cleanup() {
    // Cleanup CUDA resources
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    if (m_captureStream) {
        cudaStreamDestroy(m_captureStream);
        m_captureStream = nullptr;
    }
    if (m_cudaTexture) {
        m_cudaTexture->Release();
        m_cudaTexture = nullptr;
    }
    
    // Cleanup D3D and Hook resources
    if (hook_stop) {
        SetEvent(hook_stop);
        CloseHandle(hook_stop);
    }
    if (pStagingTexture) pStagingTexture->Release();
    if (pSharedResource) pSharedResource->Release();
    if (pContext) pContext->Release();
    if (pDevice) pDevice->Release();
    if (shared_hook_info) UnmapViewOfFile(shared_hook_info);
    if (shared_shtex_data) UnmapViewOfFile(shared_shtex_data);
    if (texture_mutexes[0]) CloseHandle(texture_mutexes[0]);
    if (texture_mutexes[1]) CloseHandle(texture_mutexes[1]);
    if (hook_data_map) CloseHandle(hook_data_map);
    if (hook_info_map) CloseHandle(hook_info_map);
    if (hook_restart) CloseHandle(hook_restart);
    if (hook_ready) CloseHandle(hook_ready);
    if (hook_exit) CloseHandle(hook_exit);
    if (hook_init) CloseHandle(hook_init);
    if (keepalive_mutex) CloseHandle(keepalive_mutex);
    if (FrameBuffer) {
        free(FrameBuffer);
        FrameBuffer = nullptr;
    }

    hook_stop = nullptr;
    hook_ready = nullptr;
    hook_exit = nullptr;
    hook_init = nullptr;
    shared_hook_info = nullptr;
    shared_shtex_data = nullptr;
    texture_mutexes[0] = nullptr;
    texture_mutexes[1] = nullptr;
    hook_data_map = nullptr;
    hook_info_map = nullptr;
    pStagingTexture = nullptr;
    pSharedResource = nullptr;
    pContext = nullptr;
    pDevice = nullptr;
}

bool GameCapture::initialize() {
    std::cout << "[GameCapture] Initializing for game: " << game_name << std::endl;
    
    // First collect all available windows
    std::vector<std::pair<HWND, std::string>> windows;
    
    EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
        auto* windows = reinterpret_cast<std::vector<std::pair<HWND, std::string>>*>(lParam);
        char title[256];
        GetWindowTextA(hwnd, title, sizeof(title));
        
        if (strlen(title) > 0 && IsWindowVisible(hwnd)) {
            // Skip some system windows
            std::string windowTitle(title);
            if (windowTitle != "Program Manager" && 
                windowTitle != "Default IME" &&
                windowTitle != "MSCTFIME UI") {
                windows->push_back({hwnd, windowTitle});
            }
        }
        return TRUE; // Continue enumeration
    }, reinterpret_cast<LPARAM>(&windows));
    
    // Try to auto-find the game window first
    hwnd = nullptr;
    for (const auto& [wnd, title] : windows) {
        std::string titleLower = title;
        std::transform(titleLower.begin(), titleLower.end(), titleLower.begin(), ::tolower);
        
        // Check for PUBG-related titles
        if (titleLower.find("pubg") != std::string::npos || 
            titleLower.find("battlegrounds") != std::string::npos ||
            titleLower.find("배틀그라운드") != std::string::npos) {
            hwnd = wnd;
            game_name = title; // Update game_name to actual window title
            std::cout << "[GameCapture] Auto-detected game window: " << title << std::endl;
            break;
        }
    }
    
    // If not auto-detected, show selection UI
    if (!hwnd && !windows.empty()) {
        std::cout << "\n[GameCapture] Available windows to capture:" << std::endl;
        std::cout << "========================================" << std::endl;
        
        for (size_t i = 0; i < windows.size(); i++) {
            std::cout << "[" << i << "] " << windows[i].second << std::endl;
        }
        
        std::cout << "========================================" << std::endl;
        std::cout << "Enter the number of the window to capture (or -1 to cancel): ";
        
        int selection;
        std::cin >> selection;
        
        if (selection >= 0 && selection < static_cast<int>(windows.size())) {
            hwnd = windows[selection].first;
            game_name = windows[selection].second;
            std::cout << "[GameCapture] Selected window: " << game_name << std::endl;
        } else if (selection == -1) {
            std::cout << "[GameCapture] Window selection cancelled" << std::endl;
            return false;
        } else {
            std::cout << "[GameCapture] Invalid selection" << std::endl;
            return false;
        }
    }
    
    if (!hwnd) {
        std::cout << "[GameCapture] ERROR: No window selected or found" << std::endl;
        return false;
    }

    thread_id = GetWindowThreadProcessId(hwnd, &process_id);
    std::cout << "[GameCapture] Found game window - PID: " << process_id << ", TID: " << thread_id << std::endl;

    keepalive_mutex = CreateKeepaliveMutex(process_id);
    if (!keepalive_mutex) {
        std::cout << "[GameCapture] ERROR: CreateKeepaliveMutex failed!" << std::endl;
        return false;
    }

    // First, try to stop any existing hook
    hook_stop = OpenEventPlusId(L"CaptureHook_Stop", process_id);
    if (hook_stop) {
        std::cout << "[GameCapture] Stopping existing hook..." << std::endl;
        SetEvent(hook_stop);
        CloseHandle(hook_stop);
        hook_stop = nullptr;
        Sleep(1000); // Wait for hook to stop
    }
    
    hook_restart = OpenEventPlusId(L"CaptureHook_Restart", process_id);

    if (hook_restart) {
        std::cout << "[GameCapture] Hook still exists after stop attempt, forcing restart..." << std::endl;
        
        // Try to force close the existing hook
        hook_exit = OpenEventPlusId(L"CaptureHook_Exit", process_id);
        if (hook_exit) {
            SetEvent(hook_exit);
            CloseHandle(hook_exit);
            hook_exit = nullptr;
            Sleep(1000);
        }
        
        // Close and re-open handles
        CloseHandle(hook_restart);
        hook_restart = nullptr;
        
        // Try injecting fresh
        std::cout << "[GameCapture] Attempting fresh injection..." << std::endl;
    }
    
    // Always try to inject (whether hook existed or not)
    {
        std::cout << "[GameCapture] Injecting hook..." << std::endl;
        
        // Check if we have admin rights (might be needed for some games)
        BOOL isElevated = FALSE;
        HANDLE token = NULL;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
            TOKEN_ELEVATION elevation;
            DWORD size = sizeof(TOKEN_ELEVATION);
            if (GetTokenInformation(token, TokenElevation, &elevation, size, &size)) {
                isElevated = elevation.TokenIsElevated;
            }
            CloseHandle(token);
        }
        
        if (!isElevated) {
            std::cout << "[GameCapture] WARNING: Not running as administrator. Some games may require admin rights for hook injection." << std::endl;
        }
        
        HANDLE injector = inject_hook(process_id);  // Pass process_id, inject_hook will use thread_id internally
        if (injector) {
            DWORD wait_result = WaitForSingleObject(injector, 5000); // 5 second timeout
            if (wait_result == WAIT_TIMEOUT) {
                std::cout << "[GameCapture] WARNING: inject-helper is taking too long" << std::endl;
            }
            
            DWORD exit_code = 0;
            GetExitCodeProcess(injector, &exit_code);
            CloseHandle(injector);
            
            if (exit_code != 0) {
                std::cout << "[GameCapture] ERROR: Hook injection failed with code: " << exit_code << std::endl;
                std::cout << "[GameCapture] Try running the application as administrator." << std::endl;
                return false;
            }
            std::cout << "[GameCapture] Hook injection completed" << std::endl;
        } else {
            std::cout << "[GameCapture] ERROR: Failed to start inject-helper" << std::endl;
            return false;
        }
    }

    // IMPORTANT: Open these BEFORE trying to use them
    hook_info_map = OpenMapPlusId(L"CaptureHook_HookInfo", process_id);
    if (!hook_info_map) {
        std::cout << "[GameCapture] ERROR: OpenMapPlusId (hook_info_map) failed!" << std::endl;
        return false;
    }

    shared_hook_info = static_cast<hook_info*>(MapViewOfFile(hook_info_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(hook_info)));
    if (!shared_hook_info) {
        std::cout << "[GameCapture] ERROR: MapViewOfFile (shared_hook_info) failed!" << std::endl;
        return false;
    }

    initialize_offsets();

    // Check hook version compatibility
    if (shared_hook_info->hook_ver_major > 0) {
        std::cout << "[GameCapture] OBS Hook version: " 
                  << shared_hook_info->hook_ver_major << "." 
                  << shared_hook_info->hook_ver_minor << std::endl;
        
        // Warn if version is too old or too new
        if (shared_hook_info->hook_ver_major < 1) {
            std::cout << "[GameCapture] WARNING: Hook version is too old, may have compatibility issues" << std::endl;
        } else if (shared_hook_info->hook_ver_major > 2) {
            std::cout << "[GameCapture] WARNING: Hook version is newer than expected, may have compatibility issues" << std::endl;
        }
    }

    shared_hook_info->capture_overlay = false;
    shared_hook_info->UNUSED_use_scale = false;
    shared_hook_info->allow_srgb_alias = true;
    shared_hook_info->force_shmem = false;
    shared_hook_info->frame_interval = 0;

    texture_mutexes[0] = OpenMutexPlusId(L"CaptureHook_TextureMutex1", process_id);
    texture_mutexes[1] = OpenMutexPlusId(L"CaptureHook_TextureMutex2", process_id);
    if (!texture_mutexes[0] || !texture_mutexes[1]) {
        std::cout << "[GameCapture] ERROR: OpenMutexPlusId failed!" << std::endl;
        return false;
    }

    // Re-open the event handles after injection
    hook_stop = OpenEventPlusId(L"CaptureHook_Stop", process_id);
    hook_ready = OpenEventPlusId(L"CaptureHook_HookReady", process_id);
    hook_exit = OpenEventPlusId(L"CaptureHook_Exit", process_id);
    hook_init = OpenEventPlusId(L"CaptureHook_Initialize", process_id);
    if (!hook_stop || !hook_ready || !hook_exit || !hook_init) {
        std::cout << "[GameCapture] ERROR: OpenEventPlusId (hook setup) failed!" << std::endl;
        return false;
    }

    if (!SetEvent(hook_init)) {
        std::cout << "[GameCapture] ERROR: SetEvent (hook_init) failed!" << std::endl;
        return false;
    }

    std::cout << "[GameCapture] Waiting for hook ready signal..." << std::endl;
    DWORD wait_result = WaitForSingleObject(hook_ready, 10000);  // 10 second timeout
    
    if (wait_result == WAIT_TIMEOUT) {
        std::cout << "[GameCapture] ERROR: Timeout waiting for hook ready signal" << std::endl;
        std::cout << "[GameCapture] This might happen if:" << std::endl;
        std::cout << "  - The game is using anti-cheat that blocks hooks" << std::endl;
        std::cout << "  - The game is running in fullscreen exclusive mode" << std::endl;
        std::cout << "  - The game is using DirectX 12 or Vulkan (OBS hook only supports DX9/10/11)" << std::endl;
        return false;
    } else if (wait_result == WAIT_FAILED) {
        std::cout << "[GameCapture] ERROR: WaitForSingleObject failed: " << GetLastError() << std::endl;
        return false;
    }
    
    std::cout << "[GameCapture] Hook is ready" << std::endl;
    
    // Give the hook some time to create the shared memory after signaling ready
    std::cout << "[GameCapture] Waiting for shared memory initialization..." << std::endl;
    Sleep(2000);  // Wait 2 seconds for hook to fully initialize shared memory

    int try_count = 0;
    HRESULT hr;

    while (true) {
        shared_shtex_data = nullptr;
        try_count++;

        // Keep retrying until we get the hook_data_map
        int retry_count = 0;
        while (!shared_shtex_data && retry_count < 30) {  // Increase retry count
            if (hook_data_map) {
                CloseHandle(hook_data_map);
                hook_data_map = nullptr;
            }

            // Try different map_ids in case the first one isn't ready yet
            uint32_t map_id_to_try = shared_hook_info->map_id + (retry_count / 10);
            
            // Debug: Print the values we're using
            if (retry_count == 0 || retry_count % 5 == 0) {
                std::cout << "[GameCapture] Attempting to open data map - window: 0x" << std::hex << shared_hook_info->window 
                          << ", map_id: " << map_id_to_try 
                          << ", map_size: " << std::dec << shared_hook_info->map_size << std::endl;
            }

            hook_data_map = OpenDataMap(shared_hook_info->window, map_id_to_try);

            if (hook_data_map) {
                shared_shtex_data = static_cast<shtex_data*>(MapViewOfFile(hook_data_map, FILE_MAP_ALL_ACCESS, 0, 0, shared_hook_info->map_size));
                if (!shared_shtex_data) {
                    DWORD error = GetLastError();
                    std::cout << "[GameCapture] MapViewOfFile failed with error: " << error << std::endl;
                } else {
                    // Successfully opened! Update the map_id if it changed
                    if (map_id_to_try != shared_hook_info->map_id) {
                        std::cout << "[GameCapture] Successfully opened with map_id: " << map_id_to_try 
                                  << " (original was " << shared_hook_info->map_id << ")" << std::endl;
                        shared_hook_info->map_id = map_id_to_try;
                    }
                }
            } else {
                DWORD error = GetLastError();
                if (retry_count == 0 || retry_count % 10 == 0) {
                    std::cout << "[GameCapture] OpenDataMap failed with error: " << error 
                              << " (2=FILE_NOT_FOUND, 5=ACCESS_DENIED)" << std::endl;
                }
            }
            
            if (!shared_shtex_data) {
                retry_count++;
                Sleep(500);  // Shorter sleep for faster retries
            }
        }
        
        if (!shared_shtex_data) {
            std::cout << "[GameCapture] ERROR: Failed to get shared texture data after " << retry_count << " attempts" << std::endl;
            return false;
        }

        // Debug: Check shared texture data
        std::cout << "[GameCapture] Shared texture handle: 0x" << std::hex << shared_shtex_data->tex_handle << std::dec << std::endl;
        
        std::cout << "[GameCapture] Creating D3D11 device..." << std::endl;
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &pDevice, nullptr, &pContext);
        if (FAILED(hr)) {
            std::cout << "[GameCapture] ERROR: D3D11CreateDevice failed!" << std::endl;
            return false;
        }

        hr = pDevice->OpenSharedResource((HANDLE)(uintptr_t)shared_shtex_data->tex_handle, __uuidof(ID3D11Resource), (void**)&pSharedResource);
        if (FAILED(hr)) {
            if (pSharedResource) {
                pSharedResource->Release();
                pSharedResource = nullptr;
            }
            if (pContext) {
                pContext->Release();
                pContext = nullptr;
            }
            if (pDevice) {
                pDevice->Release();
                pDevice = nullptr;
            }

            if (try_count >= 5) {
                std::cout << "[GameCapture] ERROR: Failed to open shared D3D resource after 5 attempts" << std::endl;
                return false;
            }

            std::cout << "[GameCapture] Retrying to open shared resource (attempt " << try_count << "/5)..." << std::endl;
            Sleep(50);
            continue;
        }

        break;
    }

    // Create staging texture for CPU access
    D3D11_TEXTURE2D_DESC desc;
    ((ID3D11Texture2D*)pSharedResource)->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;

    hr = pDevice->CreateTexture2D(&desc, nullptr, &pStagingTexture);
    if (FAILED(hr)) {
        std::cout << "[GameCapture] ERROR: CreateTexture2D (staging) failed!" << std::endl;
        return false;
    }

    // Create texture for CUDA interop
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    hr = pDevice->CreateTexture2D(&desc, nullptr, &m_cudaTexture);
    if (FAILED(hr)) {
        std::cout << "[GameCapture] ERROR: CreateTexture2D (CUDA) failed!" << std::endl;
        return false;
    }

    // Initialize CUDA interop
    if (!initializeCUDAInterop()) {
        std::cout << "[GameCapture] ERROR: CUDA interop initialization failed!" << std::endl;
        return false;
    }

    sourceRegion = get_region();

    std::cout << "[GameCapture] Initialized successfully for game: " << game_name << std::endl;
    std::cout << "[GameCapture] Capture region: " << width << "x" << height 
              << " from screen " << screen_width << "x" << screen_height << std::endl;
    return true;
}

bool GameCapture::initializeCUDAInterop() {
    // Create CUDA stream for capture operations
    cudaError_t err = cudaStreamCreate(&m_captureStream);
    if (err != cudaSuccess) {
        std::cerr << "[GameCapture] Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Register D3D11 texture with CUDA
    err = cudaGraphicsD3D11RegisterResource(
        &m_cudaResource,
        m_cudaTexture,
        cudaGraphicsRegisterFlagsNone
    );
    
    if (err != cudaSuccess) {
        std::cerr << "[GameCapture] Failed to register D3D11 resource with CUDA: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "[GameCapture] CUDA interop initialized successfully" << std::endl;
    return true;
}

bool GameCapture::WaitForNextFrame() {
    if (WaitForSingleObject(hook_restart, 0) == WAIT_OBJECT_0) {
        std::cout << "[GameCapture] Hook restart signal received, re-initializing..." << std::endl;
        cleanup();
        if (!initialize()) {
            std::cout << "[GameCapture] ERROR: Re-initialization failed" << std::endl;
            return false;
        }
    }

    if (!pContext || !pSharedResource || !pStagingTexture || !m_cudaTexture) {
        std::cout << "[GameCapture] ERROR: Resources not initialized!" << std::endl;
        return false;
    }

    // Remove all frame limiting - let the game's frame rate dictate our capture rate
    // This allows us to capture at the game's actual FPS (300+ if available)
    
    // Wait for the texture to be available (double buffering synchronization)
    // OBS uses two texture mutexes for double buffering
    static int current_texture = 0;
    
    // Use short timeout to prevent indefinite blocking but not limit FPS
    constexpr DWORD FRAME_TIMEOUT_MS = 16; // 16ms timeout (allows up to ~60 attempts per second)
    DWORD wait_result = WaitForSingleObject(texture_mutexes[current_texture], FRAME_TIMEOUT_MS);
    
    if (wait_result == WAIT_OBJECT_0) {
        // We got the mutex, the texture is ready
        // Copy the region we need from shared resource to CUDA texture
        pContext->CopySubresourceRegion(m_cudaTexture, 0, 0, 0, 0, pSharedResource, 0, &sourceRegion);
        
        // Release the mutex immediately to allow OBS to continue
        ReleaseMutex(texture_mutexes[current_texture]);
        
        // Switch to the other texture for next frame
        current_texture = 1 - current_texture;
        
        // Update frame timing for statistics only (no limiting)
        last_frame_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        return true;
    } else if (wait_result == WAIT_TIMEOUT) {
        // Timeout occurred - try the other buffer immediately
        // This can happen when one buffer is stuck, switching might get us a fresh frame
        current_texture = 1 - current_texture;
        
        // Try the other buffer with zero timeout (non-blocking)
        wait_result = WaitForSingleObject(texture_mutexes[current_texture], 0);
        if (wait_result == WAIT_OBJECT_0) {
            // Got the alternate buffer
            pContext->CopySubresourceRegion(m_cudaTexture, 0, 0, 0, 0, pSharedResource, 0, &sourceRegion);
            ReleaseMutex(texture_mutexes[current_texture]);
            current_texture = 1 - current_texture;
            last_frame_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            return true;
        }
        
        // Both buffers unavailable - this is normal when game isn't rendering new frames
        // Don't log spam, just return false
    }
    
    return false;
}

cudaGraphicsResource_t GameCapture::GetCudaResource() const {
    return m_cudaResource;
}

bool GameCapture::StartCapture() {
    if (!initialize()) {
        std::cout << "[GameCapture] ERROR: Failed to initialize capture" << std::endl;
        return false;
    }
    std::cout << "[GameCapture] Capture started successfully" << std::endl;
    return true;
}

void GameCapture::StopCapture() {
    cleanup();
    std::cout << "[GameCapture] Capture stopped" << std::endl;
}

Image GameCapture::get_frame() {
    Image img = {};

    if (!WaitForNextFrame()) {
        return img;
    }

    // For compatibility - not used when using CUDA
    img.width = width;
    img.height = height;
    img.pitch = width * 4;
    img.data = nullptr;

    return img;
}

HANDLE GameCapture::inject_hook(DWORD target_id) {
    // OBS inject-helper expects exactly 3 arguments (plus program name = 4 total):
    // inject-helper.exe [dll_path] [anti_cheat] [target_id]
    // For anti-cheat games: anti_cheat=1 and target_id=thread_id
    // For normal games: anti_cheat=0 and target_id=process_id
    
    // Enhanced anti-cheat detection
    bool is_anticheat_game = detectAntiCheat();
    
    // Use appropriate injection mode based on anti-cheat detection
    std::wstring command_line;
    if (is_anticheat_game) {
        command_line = L"\"" + inject_path + L"\" \"" + hook_path + L"\" 1 " + 
                      std::to_wstring(thread_id);  // Use thread_id for anti-cheat mode
        std::cout << "[GameCapture] Anti-cheat detected, using thread injection mode" << std::endl;
    } else {
        command_line = L"\"" + inject_path + L"\" \"" + hook_path + L"\" 0 " + 
                      std::to_wstring(process_id);  // Use process_id for normal mode
        std::cout << "[GameCapture] Normal game detected, using process injection mode" << std::endl;
    }
    
    std::wcout << L"[GameCapture] Executing: " << command_line << std::endl;
    
    STARTUPINFOW si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);
    
    // Create the inject-helper process with proper arguments
    BOOL success = CreateProcessW(
        inject_path.c_str(),     // Application path
        &command_line[0],         // Command line arguments 
        nullptr,                  // Process security attributes
        nullptr,                  // Thread security attributes
        FALSE,                    // Don't inherit handles
        CREATE_NO_WINDOW,         // Creation flags
        nullptr,                  // Environment
        nullptr,                  // Current directory
        &si,                      // Startup info
        &pi                       // Process information
    );
    
    if (!success) {
        DWORD error = GetLastError();
        std::cout << "[GameCapture] ERROR: CreateProcessW failed! Error code: 0x" 
                  << std::hex << error << std::dec << std::endl;
        
        // Enhanced error diagnostics
        diagnoseInjectionError(error);
        
        return nullptr;
    }
    
    CloseHandle(pi.hThread);
    return pi.hProcess;
}

HANDLE GameCapture::OpenMapPlusId(const std::wstring& base_name, DWORD id) {
    std::wstring full_name = base_name + std::to_wstring(id);
    return OpenFileMappingW((FILE_MAP_READ | FILE_MAP_WRITE), FALSE, full_name.c_str());
}

HANDLE GameCapture::OpenDataMap(uint32_t window, uint32_t map_id) {
    std::wstringstream ss;
    ss << L"CaptureHook_Texture_" << window << L"_" << map_id;
    return OpenFileMappingW((FILE_MAP_READ | FILE_MAP_WRITE), FALSE, ss.str().c_str());
}

HANDLE GameCapture::OpenEventPlusId(const std::wstring& base_name, DWORD id) {
    std::wstring full_name = base_name + std::to_wstring(id);
    return OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, full_name.c_str());
}

HANDLE GameCapture::OpenMutexPlusId(const std::wstring& base_name, DWORD id) {
    std::wstring full_name = base_name + std::to_wstring(id);
    return OpenMutexW(SYNCHRONIZE, FALSE, full_name.c_str());
}

HANDLE GameCapture::CreateKeepaliveMutex(int pid) {
    std::wstringstream ss;
    ss << L"CaptureHook_KeepAlive" << pid;
    return CreateMutexW(nullptr, FALSE, ss.str().c_str());
}

D3D11_BOX GameCapture::get_region() {
    // Calculate capture area (center + offset)
    auto& ctx = AppContext::getInstance();
    
    // Apply offset from config
    bool useAimShootOffset = ctx.aiming && ctx.shooting;
    int offsetX = useAimShootOffset ? 
                  static_cast<int>(ctx.config.aim_shoot_offset_x) : 
                  static_cast<int>(ctx.config.crosshair_offset_x);
    int offsetY = useAimShootOffset ? 
                  static_cast<int>(ctx.config.aim_shoot_offset_y) : 
                  static_cast<int>(ctx.config.crosshair_offset_y);
    
    D3D11_BOX box;
    box.left = (screen_width - width) / 2 + offsetX;
    box.top = (screen_height - height) / 2 + offsetY;
    box.front = 0;
    box.right = box.left + width;
    box.bottom = box.top + height;
    box.back = 1;
    
    // Clamp to screen bounds
    if (box.left < 0) {
        box.right -= box.left;
        box.left = 0;
    }
    if (box.top < 0) {
        box.bottom -= box.top;
        box.top = 0;
    }
    if (box.right > screen_width) {
        box.left -= (box.right - screen_width);
        box.right = screen_width;
    }
    if (box.bottom > screen_height) {
        box.top -= (box.bottom - screen_height);
        box.bottom = screen_height;
    }
    
    return box;
}

std::string GameCapture::run_get_graphics_offsets() {
    std::string result;
    // Wrap the path in quotes to handle spaces
    std::string command = "\"" + get_graphics_offsets64 + "\"";
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
    if (!pipe) {
        std::cout << "[GameCapture] ERROR: Failed to run get-graphics-offsets64.exe" << std::endl;
        return result;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe.get()))
        result += buffer;
    return result;
}

bool GameCapture::findOBSFiles() {
    // Check local obs_stuff folder first
    std::wstring localInject = L"obs_stuff\\inject-helper64.exe";
    std::wstring localHook = L"obs_stuff\\graphics-hook64.dll";
    std::string localOffsets = "obs_stuff\\get-graphics-offsets64.exe";
    
    // Check if local files exist
    if (GetFileAttributesW(localInject.c_str()) != INVALID_FILE_ATTRIBUTES &&
        GetFileAttributesW(localHook.c_str()) != INVALID_FILE_ATTRIBUTES &&
        GetFileAttributesA(localOffsets.c_str()) != INVALID_FILE_ATTRIBUTES) {
        inject_path = localInject;
        hook_path = localHook;
        get_graphics_offsets64 = localOffsets;
        std::cout << "[GameCapture] Using local OBS files from obs_stuff folder" << std::endl;
        return true;
    }
    
    // Try to find OBS installation
    std::vector<std::wstring> obsBasePaths = {
        L"C:\\Program Files\\obs-studio",
        L"C:\\Program Files (x86)\\obs-studio",
        L"D:\\Program Files\\obs-studio",
        L"D:\\Program Files (x86)\\obs-studio"
    };
    
    // Check environment variable for custom OBS path
    wchar_t* obsPath = nullptr;
    size_t obsPathSize = 0;
    if (_wdupenv_s(&obsPath, &obsPathSize, L"OBS_STUDIO_PATH") == 0 && obsPath != nullptr) {
        obsBasePaths.insert(obsBasePaths.begin(), std::wstring(obsPath));
        free(obsPath);
    }
    
    // Also check registry for OBS installation path
    HKEY hKey;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\OBS Studio", 0, KEY_READ, &hKey) == ERROR_SUCCESS ||
        RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\WOW6432Node\\OBS Studio", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        wchar_t installPath[MAX_PATH];
        DWORD size = sizeof(installPath);
        DWORD type;
        if (RegQueryValueExW(hKey, L"", NULL, &type, (LPBYTE)installPath, &size) == ERROR_SUCCESS) {
            obsBasePaths.insert(obsBasePaths.begin(), std::wstring(installPath));
        }
        RegCloseKey(hKey);
    }
    
    // Debug: Print all paths being checked
    std::cout << "[GameCapture] Checking " << obsBasePaths.size() << " potential OBS locations..." << std::endl;
    
    // Search for OBS files in potential locations
    for (const auto& basePath : obsBasePaths) {
        std::wcout << L"[GameCapture] Checking: " << basePath << std::endl;
        
        // Check if base path exists
        if (GetFileAttributesW(basePath.c_str()) == INVALID_FILE_ATTRIBUTES) {
            std::cout << "  -> Directory does not exist" << std::endl;
            continue;
        }
        std::cout << "  -> Directory exists!" << std::endl;
        
        // Construct full paths - FIXED: Files are in data/obs-plugins/win-capture/
        std::wstring testInject = basePath + L"\\data\\obs-plugins\\win-capture\\inject-helper64.exe";
        std::wstring testHook = basePath + L"\\data\\obs-plugins\\win-capture\\graphics-hook64.dll";
        std::wstring testOffsets = basePath + L"\\data\\obs-plugins\\win-capture\\get-graphics-offsets64.exe";
        
        // Check each file individually for better debugging
        bool injectExists = (GetFileAttributesW(testInject.c_str()) != INVALID_FILE_ATTRIBUTES);
        bool hookExists = (GetFileAttributesW(testHook.c_str()) != INVALID_FILE_ATTRIBUTES);
        bool offsetsExists = (GetFileAttributesW(testOffsets.c_str()) != INVALID_FILE_ATTRIBUTES);
        
        std::wcout << L"  -> inject-helper64.exe: " << (injectExists ? L"FOUND" : L"NOT FOUND") << L" at " << testInject << std::endl;
        std::wcout << L"  -> graphics-hook64.dll: " << (hookExists ? L"FOUND" : L"NOT FOUND") << L" at " << testHook << std::endl;
        std::wcout << L"  -> get-graphics-offsets64.exe: " << (offsetsExists ? L"FOUND" : L"NOT FOUND") << L" at " << testOffsets << std::endl;
        
        // Check if all files exist
        if (injectExists && hookExists && offsetsExists) {
            inject_path = testInject;
            hook_path = testHook;
            
            // Convert wide string to narrow string for offsets path
            char offsetsPath[MAX_PATH];
            WideCharToMultiByte(CP_UTF8, 0, testOffsets.c_str(), -1, offsetsPath, MAX_PATH, NULL, NULL);
            get_graphics_offsets64 = std::string(offsetsPath);
            
            std::wcout << L"[GameCapture] SUCCESS! Found OBS installation at: " << basePath << std::endl;
            std::wcout << L"[GameCapture] Using files directly from OBS installation:" << std::endl;
            std::wcout << L"  -> inject-helper: " << testInject << std::endl;
            std::wcout << L"  -> graphics-hook: " << testHook << std::endl;
            std::wcout << L"  -> offsets tool: " << testOffsets << std::endl;
            
            return true;
        }
    }
    
    // If not found, use default paths (will likely fail but provides clear error)
    inject_path = L"obs_stuff\\inject-helper64.exe";
    hook_path = L"obs_stuff\\graphics-hook64.dll";
    get_graphics_offsets64 = "obs_stuff\\get-graphics-offsets64.exe";
    
    std::cerr << "[GameCapture] Could not find OBS installation." << std::endl;
    std::cerr << "[GameCapture] Please install OBS Studio or manually copy the following files to 'obs_stuff' folder:" << std::endl;
    std::cerr << "  - inject-helper64.exe (from OBS/data/obs-plugins/win-capture/)" << std::endl;
    std::cerr << "  - graphics-hook64.dll (from OBS/data/obs-plugins/win-capture/)" << std::endl;
    std::cerr << "  - get-graphics-offsets64.exe (from OBS/data/obs-plugins/win-capture/)" << std::endl;
    
    return false;
}

// Enhanced Anti-cheat detection
bool GameCapture::detectAntiCheat() {
    if (!hwnd || !process_id) return false;
    
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, process_id);
    if (!hProcess) return true; // If we can't open, assume anti-cheat
    
    bool hasAntiCheat = false;
    
    // Check for common anti-cheat processes
    const std::vector<std::wstring> antiCheatProcesses = {
        L"BEService.exe",      // BattlEye
        L"EasyAntiCheat.exe",  // EAC
        L"vgc.exe",           // Riot Vanguard
        L"xigncode3.exe",     // XIGNCODE3
        L"NGS.exe"            // nProtect GameGuard
    };
    
    // Check if any anti-cheat process is running
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32W pe32;
        pe32.dwSize = sizeof(pe32);
        
        if (Process32FirstW(hSnapshot, &pe32)) {
            do {
                for (const auto& acProcess : antiCheatProcesses) {
                    if (_wcsicmp(pe32.szExeFile, acProcess.c_str()) == 0) {
                        hasAntiCheat = true;
                        std::wcout << L"[GameCapture] Anti-cheat detected: " << acProcess << std::endl;
                        break;
                    }
                }
            } while (!hasAntiCheat && Process32NextW(hSnapshot, &pe32));
        }
        CloseHandle(hSnapshot);
    }
    
    // Check for elevated process (often indicates anti-cheat)
    if (!hasAntiCheat) {
        hasAntiCheat = isProcessElevated(hProcess);
        if (hasAntiCheat) {
            std::cout << "[GameCapture] Process is elevated, likely has anti-cheat" << std::endl;
        }
    }
    
    // Check for known anti-cheat game names
    std::string gameLower = game_name;
    std::transform(gameLower.begin(), gameLower.end(), gameLower.begin(), ::tolower);
    
    if (gameLower.find("pubg") != std::string::npos ||
        gameLower.find("valorant") != std::string::npos ||
        gameLower.find("fortnite") != std::string::npos ||
        gameLower.find("apex") != std::string::npos ||
        gameLower.find("rainbow six") != std::string::npos) {
        hasAntiCheat = true;
        std::cout << "[GameCapture] Known anti-cheat protected game: " << game_name << std::endl;
    }
    
    CloseHandle(hProcess);
    return hasAntiCheat;
}

// Enhanced error diagnostics
void GameCapture::diagnoseInjectionError(DWORD error) {
    switch (error) {
        case ERROR_FILE_NOT_FOUND:
            std::cout << "[GameCapture] DIAGNOSIS: inject-helper64.exe not found" << std::endl;
            std::wcout << L"  Expected path: " << inject_path << std::endl;
            std::cout << "  Solution: Install OBS Studio or copy files to obs_stuff folder" << std::endl;
            break;
            
        case ERROR_ACCESS_DENIED:
            std::cout << "[GameCapture] DIAGNOSIS: Access denied - insufficient privileges" << std::endl;
            std::cout << "  Solutions:" << std::endl;
            std::cout << "    1. Run this application as Administrator" << std::endl;
            std::cout << "    2. Check if target process has anti-cheat protection" << std::endl;
            std::cout << "    3. Ensure Windows Defender is not blocking the injection" << std::endl;
            break;
            
        case ERROR_ELEVATION_REQUIRED:
            std::cout << "[GameCapture] DIAGNOSIS: Administrator privileges required" << std::endl;
            std::cout << "  Solution: Right-click and 'Run as Administrator'" << std::endl;
            break;
            
        case ERROR_VIRUS_INFECTED:
        case ERROR_VIRUS_DELETED:
            std::cout << "[GameCapture] DIAGNOSIS: Antivirus is blocking the injection" << std::endl;
            std::cout << "  Solutions:" << std::endl;
            std::cout << "    1. Add an exception for this application in your antivirus" << std::endl;
            std::cout << "    2. Add OBS hook files to antivirus whitelist" << std::endl;
            break;
            
        case ERROR_BAD_EXE_FORMAT:
            std::cout << "[GameCapture] DIAGNOSIS: Architecture mismatch" << std::endl;
            std::cout << "  Ensure you're using 64-bit hooks for 64-bit games" << std::endl;
            break;
            
        default:
            std::cout << "[GameCapture] DIAGNOSIS: Unknown error" << std::endl;
            std::cout << "  Error details:" << std::endl;
            std::cout << "    Code: 0x" << std::hex << error << std::dec << std::endl;
            
            // Get system error message
            LPWSTR messageBuffer = nullptr;
            size_t size = FormatMessageW(
                FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (LPWSTR)&messageBuffer, 0, NULL);
            
            if (size) {
                std::wcout << L"    Message: " << messageBuffer << std::endl;
                LocalFree(messageBuffer);
            }
            break;
    }
    
    // Additional diagnostics
    std::cout << "\n[GameCapture] System diagnostics:" << std::endl;
    
    // Check if running as admin
    BOOL isAdmin = FALSE;
    HANDLE hToken = NULL;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
        TOKEN_ELEVATION elevation;
        DWORD size = sizeof(TOKEN_ELEVATION);
        if (GetTokenInformation(hToken, TokenElevation, &elevation, sizeof(elevation), &size)) {
            isAdmin = elevation.TokenIsElevated;
        }
        CloseHandle(hToken);
    }
    std::cout << "  Running as Administrator: " << (isAdmin ? "Yes" : "No") << std::endl;
    
    // Check target process info
    if (process_id) {
        HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, process_id);
        if (hProcess) {
            std::cout << "  Target process accessible: Yes" << std::endl;
            CloseHandle(hProcess);
        } else {
            std::cout << "  Target process accessible: No (may have anti-cheat)" << std::endl;
        }
    }
}

// Check OBS version compatibility
bool GameCapture::checkOBSVersion() {
    // Try to run get-graphics-offsets with version check
    std::string versionCmd = "\"" + get_graphics_offsets64 + "\" --version 2>&1";
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(versionCmd.c_str(), "r"), _pclose);
    
    if (!pipe) {
        // If version check fails, assume compatible
        return true;
    }
    
    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        result += buffer;
    }
    
    // Check if version is compatible (OBS 27+ recommended)
    if (!result.empty()) {
        std::cout << "[GameCapture] OBS hook version info: " << result << std::endl;
        
        // Parse version if available
        // Format expected: "OBS Graphics Hook v[major].[minor]"
        size_t vPos = result.find(" v");
        if (vPos != std::string::npos) {
            std::string version = result.substr(vPos + 2);
            int major = 0;
            sscanf(version.c_str(), "%d", &major);
            
            if (major < 27) {
                std::cout << "[GameCapture] WARNING: OBS version " << major 
                          << " detected. Version 27+ recommended for best compatibility" << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

// Check if process is elevated
bool GameCapture::isProcessElevated(HANDLE hProcess) {
    HANDLE hToken = NULL;
    BOOL isElevated = FALSE;
    
    if (OpenProcessToken(hProcess, TOKEN_QUERY, &hToken)) {
        TOKEN_ELEVATION elevation;
        DWORD size = sizeof(TOKEN_ELEVATION);
        if (GetTokenInformation(hToken, TokenElevation, &elevation, sizeof(elevation), &size)) {
            isElevated = elevation.TokenIsElevated;
        }
        CloseHandle(hToken);
    }
    
    return isElevated;
}

void GameCapture::initialize_offsets() {
    const std::string output = run_get_graphics_offsets();
    if (output.empty()) {
        std::cout << "[GameCapture] WARNING: No offsets retrieved, using defaults" << std::endl;
        return;
    }
    
    std::istringstream iss(output);
    std::string line, current_section;

    while (std::getline(iss, line)) {
        if (line.empty()) continue;

        if (line.front() == '[') {
            current_section = line.substr(1, line.find(']') - 1);
            continue;
        }

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value_str = line.substr(eq_pos + 1);

        uint32_t value = 0;
        try {
            value = std::stoul(value_str, nullptr, 16);
        }
        catch (...) {
            continue;
        }
        
        auto& offsets = shared_hook_info->offsets;
        if (current_section == "d3d8") {
            if (key == "present") offsets.d3d8.present = value;
        }
        else if (current_section == "d3d9") {
            if (key == "present") offsets.d3d9.present = value;
            else if (key == "present_ex") offsets.d3d9.present_ex = value;
            else if (key == "present_swap") offsets.d3d9.present_swap = value;
            else if (key == "d3d9_clsoff") offsets.d3d9.d3d9_clsoff = value;
            else if (key == "is_d3d9ex_clsoff") offsets.d3d9.is_d3d9ex_clsoff = value;
        }
        else if (current_section == "dxgi") {
            if (key == "present") offsets.dxgi.present = value;
            else if (key == "resize") offsets.dxgi.resize = value;
            else if (key == "present1") offsets.dxgi.present1 = value;
            else if (key == "release") offsets.dxgi2.release = value;
        }
    }
    
    std::cout << "[GameCapture] Offsets initialized" << std::endl;
}