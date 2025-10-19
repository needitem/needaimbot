#include <sstream>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <iostream>
#include <stdio.h>
#include <cwctype>
#include <algorithm>
#include "game_capture.h"

GameCapture::GameCapture(int fw, int fh, int sw, int sh, const std::string& game) :
	screen_width(sw), screen_height(sh), width(fw), height(fh), game_name(game), hwnd(nullptr), process_id(0), thread_id(0),
    hook_restart(nullptr), hook_stop(nullptr), hook_ready(nullptr), hook_exit(nullptr), hook_init(nullptr),
    keepalive_mutex(nullptr), hook_info_map(nullptr), hook_data_map(nullptr),
    shared_hook_info(nullptr), shared_data(nullptr),
	pDevice(nullptr), pContext(nullptr), pSharedResource(nullptr), pStagingTexture(nullptr), texture_mutexes{ nullptr, nullptr }
{
	if (!initialize()) {
		std::cout << "ERROR: GameCapture initialization failed for game: " << game_name << std::endl;
		throw std::runtime_error("GameCapture initialization failed");
	}
}

GameCapture::~GameCapture() {
    if (hook_stop) {
        SetEvent(hook_stop);
        CloseHandle(hook_stop);
    }
    // First, clean up CUDA interop to drop references before releasing D3D texture
    if (m_cudaGraphicsResource) {
        cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
        m_cudaGraphicsResource = nullptr;
    }
    m_cudaMappedArray = nullptr;
    m_cudaInteropEnabled = false;

    if (pStagingTexture) pStagingTexture->Release();
    if (pCudaSharedTexture) pCudaSharedTexture->Release();
    if (pSharedResource) pSharedResource->Release();
    if (pContext) pContext->Release();
    if (pDevice) pDevice->Release();
    if (shared_hook_info) UnmapViewOfFile(shared_hook_info);
    if (shared_data) UnmapViewOfFile(shared_data);
    if (texture_mutexes[0]) CloseHandle(texture_mutexes[0]);
    if (texture_mutexes[1]) CloseHandle(texture_mutexes[1]);
    if (hook_data_map) CloseHandle(hook_data_map);
    if (hook_info_map) CloseHandle(hook_info_map);
    if (hook_restart) CloseHandle(hook_restart);

    if (hook_ready) CloseHandle(hook_ready);
    if (hook_exit) CloseHandle(hook_exit);
    if (hook_init) CloseHandle(hook_init);
    if (keepalive_mutex) CloseHandle(keepalive_mutex);

    hook_stop = nullptr;
    hook_ready = nullptr;
    hook_exit = nullptr;
    hook_init = nullptr;
    shared_hook_info = nullptr;
    shared_data = nullptr;
    texture_mutexes[0] = nullptr;
    texture_mutexes[1] = nullptr;
    hook_data_map = nullptr;
    hook_info_map = nullptr;
    pStagingTexture = nullptr;
    pCudaSharedTexture = nullptr;
    pSharedResource = nullptr;
    pContext = nullptr;
    pDevice = nullptr;

    // Already unregistered above
}

bool GameCapture::initialize() {
    // Convert UTF-8 game_name to wide for Unicode FindWindow
    std::wstring wname;
    if (!game_name.empty()) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, game_name.c_str(), -1, nullptr, 0);
        if (wlen > 0) {
            wname.resize(static_cast<size_t>(wlen > 0 ? wlen - 1 : 0));
            if (!wname.empty()) {
                MultiByteToWideChar(CP_UTF8, 0, game_name.c_str(), -1, &wname[0], wlen);
            }
        }
    }

    // First try exact match via FindWindowW
    hwnd = FindWindowW(nullptr, wname.empty() ? nullptr : wname.c_str());

    // Fallback: enumerate visible top-level windows and try exact/ci/substring match
    if (!hwnd && !wname.empty()) {
        struct EnumCtx {
            const std::wstring* target;
            HWND found{nullptr};
        } ec{ &wname, nullptr };

        auto enumProc = [](HWND hWnd, LPARAM lParam) -> BOOL {
            EnumCtx* ctx = reinterpret_cast<EnumCtx*>(lParam);
            if (!IsWindowVisible(hWnd)) return TRUE;
            int len = GetWindowTextLengthW(hWnd);
            if (len <= 0) return TRUE;
            std::wstring title;
            title.resize(static_cast<size_t>(len + 1));
            int copied = GetWindowTextW(hWnd, &title[0], len + 1);
            if (copied <= 0) return TRUE;
            title.resize(static_cast<size_t>(copied));
            if (!title.empty()) {
                bool match = false;
                // Exact match
                match = (title == *ctx->target);
                // Case-insensitive match
                if (!match) {
                    int ci = CompareStringOrdinal(title.c_str(), static_cast<int>(title.size()), ctx->target->c_str(), static_cast<int>(ctx->target->size()), TRUE);
                    match = (ci == CSTR_EQUAL);
                }
                // Substring (case-insensitive)
                if (!match) {
                    auto toLower = [](const std::wstring& s){
                        std::wstring t=s; for (auto& ch : t) ch = static_cast<wchar_t>(towlower(ch)); return t;
                    };
                    std::wstring lt = toLower(title);
                    std::wstring tt = toLower(*ctx->target);
                    match = (lt.find(tt) != std::wstring::npos) || (tt.find(lt) != std::wstring::npos);
                }
                if (match) {
                    ctx->found = hWnd;
                    return FALSE; // stop enumeration
                }
            }
            return TRUE;
        };

        EnumWindows(enumProc, reinterpret_cast<LPARAM>(&ec));
        hwnd = ec.found;
    }

    if (!hwnd) {
        std::cout << "ERROR: Invalid game window name!" << std::endl;
        return false;
    }

    thread_id = GetWindowThreadProcessId(hwnd, &process_id);

    keepalive_mutex = CreateKeepaliveMutex(process_id);
    if (!keepalive_mutex) {
        std::cout << "ERROR: CreateKeepaliveMutex failed!" << std::endl;
        return false;
    }

    hook_restart = OpenEventPlusId(L"CaptureHook_Restart", process_id);

    if (hook_restart) {
        if (!SetEvent(hook_restart)) 
            std::cout << "ERROR: SetEvent (hook_restart) failed!" << std::endl;
    }
    else {
        HANDLE injector = inject_hook(thread_id);
        if (injector) {
            DWORD w = WaitForSingleObject(injector, 15000); // 15s timeout
            DWORD exit_code = (DWORD)-1;
            if (w == WAIT_OBJECT_0) {
                GetExitCodeProcess(injector, &exit_code);
            } else {
                std::cout << "ERROR: Injector timed out" << std::endl;
            }
            CloseHandle(injector);
            if (w != WAIT_OBJECT_0 || exit_code != 0) return false;
        }
    }

    texture_mutexes[0] = OpenMutexPlusId(L"CaptureHook_TextureMutex1", process_id);
    texture_mutexes[1] = OpenMutexPlusId(L"CaptureHook_TextureMutex2", process_id);
    if (!texture_mutexes[0] || !texture_mutexes[1]) {
        std::cout << "ERROR: OpenMutexPlusId failed!" << std::endl;
        return false;
    }

    hook_info_map = OpenMapPlusId(L"CaptureHook_HookInfo", process_id);
    if (!hook_info_map) {
        std::cout << "ERROR: OpenMapPlusId (hook_info_map) failed!" << std::endl;
        return false;
    }

    shared_hook_info = static_cast<hook_info*>(MapViewOfFile(hook_info_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(hook_info)));
    if (!shared_hook_info) {
        std::cout << "ERROR: MapViewOfFile (shared_hook_info) failed!" << std::endl;
        return false;
    }

	initialize_offsets();

    shared_hook_info->capture_overlay = false;
    shared_hook_info->UNUSED_use_scale = false;
    shared_hook_info->allow_srgb_alias = true;
    shared_hook_info->force_shmem = false;
    shared_hook_info->frame_interval = 0;

    hook_stop = OpenEventPlusId(L"CaptureHook_Stop", process_id);
    hook_ready = OpenEventPlusId(L"CaptureHook_HookReady", process_id);
    hook_exit = OpenEventPlusId(L"CaptureHook_Exit", process_id);
    hook_init = OpenEventPlusId(L"CaptureHook_Initialize", process_id);
    if (!hook_stop || !hook_ready || !hook_exit || !hook_init) {
        std::cout << "ERROR: OpenEventPlusId (hook setup) failed!" << std::endl;
        return false;
    }

    if (!SetEvent(hook_init)) {
        std::cout << "ERROR: SetEvent (hook_init) failed!" << std::endl;
        return false;
    }

    // Use adaptive timeout: long for first init, short for re-init
    DWORD hookTimeout = m_hasInitializedOnce ? m_hookReadyTimeoutReinitMs : m_hookReadyTimeoutInitialMs;
    if (WaitForSingleObject(hook_ready, hookTimeout) != WAIT_OBJECT_0) { // adaptive timeout
        std::cout << "ERROR: Hook not ready (timeout)" << std::endl;
        return false;
    }

    hook_info_map = OpenMapPlusId(L"CaptureHook_HookInfo", process_id);
    if (!hook_info_map) {
        std::cout << "ERROR: OpenMapPlusId (hook_info_map) failed!" << std::endl;
        return false;
    }

    shared_hook_info = static_cast<hook_info*>(MapViewOfFile(hook_info_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(hook_info)));
    if (!shared_hook_info) {
        std::cout << "ERROR: MapViewOfFile (shared_hook_info) failed!" << std::endl;
        return false;
    }

    // Mark that we have successfully initialized at least once
    m_hasInitializedOnce = true;

    // std::cout << shared_hook_info->window << " " << shared_hook_info->map_id << std::endl;

    int try_count = 0;
    HRESULT hr;

    while (true) {
        // Unmap previous shared_data view to avoid leaks on retry
        if (shared_data) {
            UnmapViewOfFile(shared_data);
            shared_data = nullptr;
        }
        try_count++;

        // Keep retrying until we get the hook_data_map
        int inner_tries = 0;
        while (!shared_data) {
            if (hook_data_map) {
                CloseHandle(hook_data_map);
                hook_data_map = nullptr;
            }

            hook_data_map = OpenDataMap(shared_hook_info->window, shared_hook_info->map_id);

            if (hook_data_map)
                shared_data = MapViewOfFile(hook_data_map, FILE_MAP_ALL_ACCESS, 0, 0, shared_hook_info->map_size);
            if (shared_data) break;
            Sleep(100);
            inner_tries++;
            if (inner_tries > 100) { // ~10s max
                break;
            }
        }

        if (!shared_data) {
            std::cout << "Failed to map shared data" << std::endl;
            return false;
        }

        // Release previous D3D resources before recreating to prevent handle churn
        if (pContext) {
            pContext->Release();
            pContext = nullptr;
        }
        if (pDevice) {
            pDevice->Release();
            pDevice = nullptr;
        }

        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &pDevice, nullptr, &pContext);
        if (FAILED(hr)) {
            std::cout << "ERROR: D3D11CreateDevice failed!" << std::endl;
            return false;
        }

        // Handle different capture types
        if (shared_hook_info->type == CAPTURE_TYPE_TEXTURE) {
            shtex_data* shtex = static_cast<shtex_data*>(shared_data);
            hr = pDevice->OpenSharedResource((HANDLE)(uintptr_t)shtex->tex_handle, __uuidof(ID3D11Resource), (void**)&pSharedResource);
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
                    std::cout << "ERROR: Failed to open shared D3D resource (texture mode)" << std::endl;
                    return false;
                }

                // Keep retrying to get the new valid tex_handle,
                // since minimizing the game window sometimes causes the tex_handle to become invalid
                Sleep(50);
                continue;
            }
        } else if (shared_hook_info->type == CAPTURE_TYPE_MEMORY) {
            shmem_data* shmem = static_cast<shmem_data*>(shared_data);
            // For memory mode, we'll handle texture access in get_frame() using offsets
            // For now, we need to create a texture based on hook_info dimensions
            D3D11_TEXTURE2D_DESC memDesc = {};
            memDesc.Width = shared_hook_info->cx;
            memDesc.Height = shared_hook_info->cy;
            memDesc.MipLevels = 1;
            memDesc.ArraySize = 1;
            memDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            memDesc.SampleDesc.Count = 1;
            memDesc.Usage = D3D11_USAGE_DYNAMIC;
            memDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            memDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

            ID3D11Texture2D* tempTex = nullptr;
            hr = pDevice->CreateTexture2D(&memDesc, nullptr, &tempTex);
            if (FAILED(hr)) {
                std::cout << "ERROR: Failed to create texture for memory mode" << std::endl;
                if (pContext) pContext->Release();
                if (pDevice) pDevice->Release();
                pContext = nullptr;
                pDevice = nullptr;
                if (try_count >= 5) return false;
                Sleep(50);
                continue;
            }
            pSharedResource = tempTex;
        }

        break;
    }

    D3D11_TEXTURE2D_DESC desc;
    ((ID3D11Texture2D*)pSharedResource)->GetDesc(&desc);
    // Update capture surface size to source texture size to compute region correctly
    UINT srcW = desc.Width;
    UINT srcH = desc.Height;
    if (width > (int)srcW) width = (int)srcW;
    if (height > (int)srcH) height = (int)srcH;
    screen_width = (int)srcW;
    screen_height = (int)srcH;

    // Create CPU staging texture with ROI size
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
        std::cout << "ERROR: CreateTexture2D (pDevice) failed!" << std::endl;
        return false;
    }

    // Compute center crop region within the source texture
    sourceRegion = get_region();
    // Clamp region just in case
    if ((int)sourceRegion.right > (int)srcW) sourceRegion.right = srcW;
    if ((int)sourceRegion.bottom > (int)srcH) sourceRegion.bottom = srcH;

    // Prepare CUDA interop shared texture for zero-copy GPU path
    if (!ensureCudaSharedTexture(static_cast<unsigned int>(width), static_cast<unsigned int>(height), DXGI_FORMAT_B8G8R8A8_UNORM)) {
        // Not fatal: we keep CPU fallback
        std::cout << "[GameCapture] CUDA interop not available; using CPU fallback" << std::endl;
    }

	std::cout << "GameCapture initialized successfully for game: " << game_name << std::endl;
    return true;
}

Image GameCapture::get_frame() {
    Image img = {};

    if (WaitForSingleObject(hook_restart, 0) == WAIT_OBJECT_0) {
        // Apply exponential backoff if we've had recent failures
        auto now = std::chrono::steady_clock::now();
        if (m_failureCount > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastFailureTime).count();
            int backoffTime = BASE_BACKOFF_MS * (1 << (m_failureCount - 1));  // 50ms, 100ms, 200ms, 400ms, 800ms
            backoffTime = std::min(backoffTime, 2000);  // Cap at 2 seconds

            if (elapsed < backoffTime) {
                // Still in backoff period - don't retry yet
                return img;
            }
        }

        if (!initialize()) {
            std::cout << "ERROR: Re-initialization failed (attempt " << (m_failureCount + 1) << "/" << MAX_RETRIES << ")" << std::endl;
            m_failureCount++;
            m_lastFailureTime = now;

            if (m_failureCount >= MAX_RETRIES) {
                throw std::runtime_error("Persistent capture failure after " + std::to_string(MAX_RETRIES) + " retries");
            }
            return img; // empty on failure
        }

        // Success - reset failure counter and backoff timing baseline
        std::cout << "Re-initialization successful after " << m_failureCount << " failures" << std::endl;
        m_failureCount = 0;
        // Push last failure time sufficiently into the past so next failure doesn't inherit short backoff
        m_lastFailureTime = std::chrono::steady_clock::time_point{};
    }

    if (!pContext || !pSharedResource || !pStagingTexture) {
        std::cout << "ERROR: D3D resources not initialized!" << std::endl;
        return img;
    }

    // Handle different capture types
    if (shared_hook_info->type == CAPTURE_TYPE_TEXTURE) {
        // Texture mode: direct GPU texture copy (original implementation)
        pContext->CopySubresourceRegion(pStagingTexture, 0, 0, 0, 0, pSharedResource, 0, &sourceRegion);
    } else if (shared_hook_info->type == CAPTURE_TYPE_MEMORY) {
        // Memory mode: CPU fallback not supported in get_frame()
        // Use GetLatestFrameGPU() for GPU-direct path with CUDA interop
        std::cout << "ERROR: Memory mode requires GPU-direct path (use GetLatestFrameGPU)" << std::endl;
        return img;
    }

    D3D11_MAPPED_SUBRESOURCE mapped = {};
    HRESULT hr = pContext->Map(pStagingTexture, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cout << "ERROR: Map (pContext) failed!" << std::endl;
        return img;
    }

    int row_size = width * 4;
    int data_size = height * row_size;

    // Try to allocate pinned memory for better CUDA performance, but keep malloc fallback
    if (!m_frameBufferPinned || m_frameBufferPinned->size() != (size_t)data_size) {
        try {
            m_frameBufferPinned = std::make_unique<CudaPinnedMemory<unsigned char>>(
                data_size, cudaHostAllocWriteCombined | cudaHostAllocPortable);
            // Free old malloc buffer if we successfully allocated pinned memory
            if (FrameBuffer) {
                free(FrameBuffer);
                FrameBuffer = nullptr;
            }
        } catch (const std::exception&) {
            // If pinned memory fails, fall back to malloc
            m_frameBufferPinned.reset();
            static int prev_data_size = 0;
            if (!FrameBuffer || prev_data_size != data_size) {
                if (FrameBuffer) free(FrameBuffer);
                FrameBuffer = (BYTE*)malloc(data_size);
                prev_data_size = data_size;
            }
        }
    }

    // Use pinned memory if available, otherwise use malloc buffer
    BYTE* targetBuffer = m_frameBufferPinned ? m_frameBufferPinned->get() : FrameBuffer;

    for (int y = 0; y < height; ++y) {
        memcpy(targetBuffer + y * row_size, (BYTE*)mapped.pData + y * mapped.RowPitch, row_size);
    }

    pContext->Unmap(pStagingTexture, 0);

    img.width = width;
    img.height = height;
    img.pitch = row_size;
    img.data = targetBuffer;

    return img;
}

bool GameCapture::GetLatestFrameGPU(cudaArray_t* cudaArray, unsigned int* outWidth, unsigned int* outHeight) {
    if (!pContext || !pSharedResource) {
        return false;
    }

    // Ensure CUDA shared texture is ready and sized to current ROI
    if (!ensureCudaSharedTexture(static_cast<unsigned int>(width), static_cast<unsigned int>(height), DXGI_FORMAT_B8G8R8A8_UNORM)) {
        return false;
    }

    // Ensure CUDA interop sync: unmap before D3D writes, then remap for CUDA reads
    if (m_cudaGraphicsResource && m_cudaMappedArray) {
        cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0);
        m_cudaMappedArray = nullptr;
    }

    // Handle different capture types
    if (shared_hook_info->type == CAPTURE_TYPE_TEXTURE) {
        // Texture mode: direct GPU texture copy (original implementation)
        pContext->CopySubresourceRegion(pCudaSharedTexture, 0, 0, 0, 0, pSharedResource, 0, &sourceRegion);
    } else if (shared_hook_info->type == CAPTURE_TYPE_MEMORY) {
        // Memory mode: copy directly from shared memory to CUDA-mapped array, avoiding D3D staging
        shmem_data* shmem = static_cast<shmem_data*>(shared_data);
        int tex_index = shmem->last_tex;
        if (tex_index < 0 || tex_index >= 2) {
            return false;
        }

        HANDLE mutex = texture_mutexes[tex_index];
        // Reduced timeout with fast retry for better responsiveness
        DWORD wait_result = WaitForSingleObject(mutex, 5);
        if (wait_result != WAIT_OBJECT_0) {
            // Quick retry once before giving up
            wait_result = WaitForSingleObject(mutex, 2);
            if (wait_result != WAIT_OBJECT_0) {
                return false;
            }
        }

        // Get pointer to shared memory texture data
        uint32_t offset = (tex_index == 0) ? shmem->tex1_offset : shmem->tex2_offset;
        BYTE* shmem_ptr = reinterpret_cast<BYTE*>(shared_data) + offset;

        // Map CUDA resource if needed
        if (m_cudaGraphicsResource && !m_cudaMappedArray) {
            if (cudaGraphicsMapResources(1, &m_cudaGraphicsResource, 0) != cudaSuccess) {
                ReleaseMutex(mutex);
                return false;
            }
            if (cudaGraphicsSubResourceGetMappedArray(&m_cudaMappedArray, m_cudaGraphicsResource, 0, 0) != cudaSuccess) {
                cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0);
                m_cudaMappedArray = nullptr;
                ReleaseMutex(mutex);
                return false;
            }
        }

        if (!m_cudaMappedArray) {
            ReleaseMutex(mutex);
            return false;
        }

        // Apply ROI offsets from sourceRegion to honor crop region
        int x0 = static_cast<int>(sourceRegion.left);
        int y0 = static_cast<int>(sourceRegion.top);
        int copy_width = (std::min)(width, (int)shared_hook_info->cx - x0);
        int copy_height = (std::min)(height, (int)shared_hook_info->cy - y0);
        size_t src_pitch = static_cast<size_t>(shared_hook_info->pitch);
        size_t row_bytes = static_cast<size_t>(copy_width) * 4;

        // Offset into shared memory to start from ROI origin
        BYTE* roi_shmem_ptr = shmem_ptr + (y0 * src_pitch) + (x0 * 4);

        cudaError_t cpy = cudaMemcpy2DToArray(
            m_cudaMappedArray,
            0, 0,
            roi_shmem_ptr,
            src_pitch,
            row_bytes,
            static_cast<size_t>(copy_height),
            cudaMemcpyHostToDevice);

        ReleaseMutex(mutex);

        if (cpy != cudaSuccess) {
            // On failure, clear mapped array to trigger remap/retry next call
            cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0);
            m_cudaMappedArray = nullptr;
            return false;
        }
    }

    // Remap so CUDA sees the updated contents (if not already mapped in memory mode)
    if (m_cudaGraphicsResource && !m_cudaMappedArray) {
        if (cudaGraphicsMapResources(1, &m_cudaGraphicsResource, 0) != cudaSuccess) {
            return false;
        }
        if (cudaGraphicsSubResourceGetMappedArray(&m_cudaMappedArray, m_cudaGraphicsResource, 0, 0) != cudaSuccess) {
            cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0);
            m_cudaMappedArray = nullptr;
            return false;
        }
    }

    if (!m_cudaInteropEnabled || !m_cudaMappedArray) {
        return false;
    }

    if (cudaArray) *cudaArray = m_cudaMappedArray;
    if (outWidth) *outWidth = static_cast<unsigned int>(width);
    if (outHeight) *outHeight = static_cast<unsigned int>(height);
    return true;
}

HANDLE GameCapture::inject_hook(DWORD target_id) {
    // Build absolute paths based on executable directory to avoid CWD issues
    wchar_t modulePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, modulePath, MAX_PATH);
    std::wstring exeDir(modulePath);
    size_t pos = exeDir.find_last_of(L"/\\");
    if (pos != std::wstring::npos) exeDir = exeDir.substr(0, pos);
    std::wstring inject_full = exeDir + L"\\" + inject_path;  // inject-helper64.exe
    std::wstring hook_full = exeDir + L"\\" + hook_path;      // graphics-hook64.dll

    std::wstring command_line = L"\"" + inject_full + L"\" \"" + hook_full + L"\" 1 " + std::to_wstring(target_id);
    STARTUPINFOW si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);
    BOOL success = CreateProcessW(inject_full.c_str(), &command_line[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    if (!success) {
        std::cout << "ERROR: CreateProcessW failed!" << std::endl;
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
    D3D11_BOX box;
    box.left = (screen_width - width) / 2;
    box.top = (screen_height - height) / 2;
    box.front = 0;
    box.right = box.left + width;
    box.bottom = box.top + height;
    box.back = 1;
    return box;
}

std::string GameCapture::run_get_graphics_offsets() {
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(get_graphics_offsets64, "r"), _pclose);
    if (!pipe) throw std::runtime_error("Failed to run exe");

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe.get()))
        result += buffer;
    return result;
}

bool GameCapture::SaveBMP(const char* filename, const Image& img) {
    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    int padding = (4 - (img.width * 3) % 4) % 4;
    int rowSize = img.width * 3 + padding;
    int imageSize = rowSize * img.height;

    BITMAPFILEHEADER bfh = {};
    bfh.bfType = 0x4D42;
    bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    bfh.bfSize = bfh.bfOffBits + imageSize;

    BITMAPINFOHEADER bih = {};
    bih.biSize = sizeof(BITMAPINFOHEADER);
    bih.biWidth = img.width;
    bih.biHeight = -img.height;  // Negative for top-down DIB
    bih.biPlanes = 1;
    bih.biBitCount = 24;
    bih.biCompression = BI_RGB;
    bih.biSizeImage = imageSize;

    fwrite(&bfh, sizeof(bfh), 1, f);
    fwrite(&bih, sizeof(bih), 1, f);

    for (int y = 0; y < img.height; ++y) {
        BYTE* row = img.data + y * img.pitch;
        for (int x = 0; x < img.width; ++x) {
            // BGRA -> BGR (drop alpha)
            BYTE pixel[3] = { row[x * 4 + 0], row[x * 4 + 1], row[x * 4 + 2] };
            fwrite(pixel, 1, 3, f);
        }
        BYTE pad[3] = { 0, 0, 0 };
        fwrite(pad, 1, padding, f);
    }

    fclose(f);
    return true;
}

void GameCapture::initialize_offsets() {
    const std::string output = run_get_graphics_offsets();
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
            else if (key == "release") offsets.dxgi2.release = value;  // dxgi_offsets2
        }
    }
}

bool GameCapture::SetCaptureRegion(int x, int y, int w, int h) {
    // Validate and clamp to screen bounds
    if (w <= 0 || h <= 0) return false;
    if (x < 0) x = 0; if (y < 0) y = 0;
    if (x + w > screen_width) w = screen_width - x;
    if (y + h > screen_height) h = screen_height - y;

    // Store new capture size for row pitch and copies
    bool sizeChanged = (width != w) || (height != h);
    width = w;
    height = h;

    // Update region box
    sourceRegion.left = static_cast<UINT>(x);
    sourceRegion.top = static_cast<UINT>(y);
    sourceRegion.front = 0;
    sourceRegion.right = static_cast<UINT>(x + w);
    sourceRegion.bottom = static_cast<UINT>(y + h);
    sourceRegion.back = 1;

    // Recreate staging texture if size changed
    if (sizeChanged && pDevice) {
        if (pStagingTexture) {
            pStagingTexture->Release();
            pStagingTexture = nullptr;
        }
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.MiscFlags = 0;
        HRESULT hr = pDevice->CreateTexture2D(&desc, nullptr, &pStagingTexture);
        if (FAILED(hr)) {
            std::cout << "ERROR: Recreate pStagingTexture failed!" << std::endl;
            return false;
        }

        // Recreate CUDA shared texture for GPU path
        if (!ensureCudaSharedTexture(static_cast<unsigned int>(width), static_cast<unsigned int>(height), DXGI_FORMAT_B8G8R8A8_UNORM)) {
            // If failed, keep CPU fallback; don't fail region update.
            m_cudaInteropEnabled = false;
        }
    }
    return true;
}

void GameCapture::GetCaptureRegion(int* x, int* y, int* w, int* h) const {
    if (x) *x = static_cast<int>(sourceRegion.left);
    if (y) *y = static_cast<int>(sourceRegion.top);
    if (w) *w = static_cast<int>(sourceRegion.right - sourceRegion.left);
    if (h) *h = static_cast<int>(sourceRegion.bottom - sourceRegion.top);
}

bool GameCapture::ensureCudaSharedTexture(unsigned int w, unsigned int h, DXGI_FORMAT format) {
    // If texture exists and matches, nothing to do
    if (pCudaSharedTexture) {
        D3D11_TEXTURE2D_DESC cur{};
        pCudaSharedTexture->GetDesc(&cur);
        if (cur.Width == w && cur.Height == h && cur.Format == format) {
            return m_cudaInteropEnabled && m_cudaMappedArray != nullptr;
        }
    }

    // Cleanup previous CUDA interop objects
    if (m_cudaGraphicsResource) {
        cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
        m_cudaGraphicsResource = nullptr;
    }
    m_cudaMappedArray = nullptr;
    m_cudaInteropEnabled = false;
    if (pCudaSharedTexture) {
        pCudaSharedTexture->Release();
        pCudaSharedTexture = nullptr;
    }

    if (!pDevice) return false;

    // Create a DEFAULT-usage texture suitable for CUDA interop
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    HRESULT hr = pDevice->CreateTexture2D(&desc, nullptr, &pCudaSharedTexture);
    if (FAILED(hr) || !pCudaSharedTexture) {
        std::cout << "[GameCapture] Failed to create CUDA shared texture" << std::endl;
        return false;
    }

    // Register and map with CUDA
    cudaError_t cerr = cudaGraphicsD3D11RegisterResource(&m_cudaGraphicsResource, pCudaSharedTexture, cudaGraphicsRegisterFlagsNone);
    if (cerr != cudaSuccess) {
        std::cout << "[GameCapture] cudaGraphicsD3D11RegisterResource failed: " << cudaGetErrorString(cerr) << std::endl;
        return false;
    }

    cerr = cudaGraphicsMapResources(1, &m_cudaGraphicsResource, 0);
    if (cerr != cudaSuccess) {
        std::cout << "[GameCapture] cudaGraphicsMapResources failed: " << cudaGetErrorString(cerr) << std::endl;
        cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
        m_cudaGraphicsResource = nullptr;
        return false;
    }

    cerr = cudaGraphicsSubResourceGetMappedArray(&m_cudaMappedArray, m_cudaGraphicsResource, 0, 0);
    if (cerr != cudaSuccess) {
        std::cout << "[GameCapture] cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorString(cerr) << std::endl;
        cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0);
        cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
        m_cudaGraphicsResource = nullptr;
        return false;
    }

    m_cudaInteropEnabled = true;
    return true;
}
