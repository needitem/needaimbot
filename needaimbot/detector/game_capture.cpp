#include <sstream>
#include <windows.h>
#include <d3d11.h>
#include <iostream>
#include <stdio.h>
#include "game_capture.h"

GameCapture::GameCapture(int fw, int fh, int sw, int sh, const std::string& game) :
	screen_width(sw), screen_height(sh), width(fw), height(fh), game_name(game), hwnd(nullptr), process_id(0), thread_id(0),
    hook_restart(nullptr), hook_stop(nullptr), hook_ready(nullptr), hook_exit(nullptr), hook_init(nullptr),
    keepalive_mutex(nullptr), hook_info_map(nullptr), hook_data_map(nullptr),
    shared_hook_info(nullptr), shared_shtex_data(nullptr),
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
    hwnd = FindWindowA(nullptr, game_name.c_str());
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
            WaitForSingleObject(injector, INFINITE);
            DWORD exit_code;
            GetExitCodeProcess(injector, &exit_code);
            CloseHandle(injector);
            if (exit_code != 0) return false;
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

    WaitForSingleObject(hook_ready, INFINITE);

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

    // std::cout << shared_hook_info->window << " " << shared_hook_info->map_id << std::endl;

    int try_count = 0;
    HRESULT hr;

    while (true) {
        shared_shtex_data = nullptr;
        try_count++;

        // Keep retrying until we get the hook_data_map
        while (!shared_shtex_data) {
            if (hook_data_map) {
                CloseHandle(hook_data_map);
                hook_data_map = nullptr;
            }

            hook_data_map = OpenDataMap(shared_hook_info->window, shared_hook_info->map_id);

            if (hook_data_map)
                shared_shtex_data = static_cast<shtex_data*>(MapViewOfFile(hook_data_map, FILE_MAP_ALL_ACCESS, 0, 0, shared_hook_info->map_size));
            Sleep(1000);
        }

        if (!shared_shtex_data) {
            std::cout << "Failed to map shared texture data" << std::endl;
            return false;
        }

        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &pDevice, nullptr, &pContext);
        if (FAILED(hr)) {
            std::cout << "ERROR: D3D11CreateDevice failed!" << std::endl;
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
                std::cout << "ERROR: Failed to open shared D3D resource" << std::endl;
                return false;
            }

            // Keep retrying to get the new valid tex_handle, 
            // since minimizing the game window sometimes causes the tex_handle to become invalid
            Sleep(50);
            continue;
        }

        break;
    }

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
        std::cout << "ERROR: CreateTexture2D (pDevice) failed!" << std::endl;
        return false;
    }

    sourceRegion = get_region();

	std::cout << "GameCapture initialized successfully for game: " << game_name << std::endl;
    return true;
}

Image GameCapture::get_frame() {
    Image img = {};

    if (WaitForSingleObject(hook_restart, 0) == WAIT_OBJECT_0) {
        if (!initialize()) {
            std::cout << "ERROR: Re-initialization failed" << std::endl;
            return img;
        }
    }

    if (!pContext || !pSharedResource || !pStagingTexture) {
        std::cout << "ERROR: D3D resources not initialized!" << std::endl;
        return img;
    }

    pContext->CopySubresourceRegion(pStagingTexture, 0, 0, 0, 0, pSharedResource, 0, &sourceRegion);

    D3D11_MAPPED_SUBRESOURCE mapped = {};
    HRESULT hr = pContext->Map(pStagingTexture, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cout << "ERROR: Map (pContext) failed!" << std::endl;
        return img;
    }

    int row_size = width * 4;  
    int data_size = height * row_size;
    static int prev_data_size = 0;

    if (!FrameBuffer || prev_data_size != data_size) {
        if (FrameBuffer)
            free(FrameBuffer);

        FrameBuffer = (BYTE*)malloc(data_size);
        prev_data_size = data_size;
    }

    for (int y = 0; y < height; ++y) {
        memcpy(FrameBuffer + y * row_size, (BYTE*)mapped.pData + y * mapped.RowPitch, row_size);
    }

    pContext->Unmap(pStagingTexture, 0);

    img.width = width;
    img.height = height;
    img.pitch = row_size;
    img.data = FrameBuffer;

    return img;
}

HANDLE GameCapture::inject_hook(DWORD target_id) {
    std::wstring command_line = L"\"" + inject_path + L"\" \"" + hook_path + L"\" 1 " + std::to_wstring(target_id);
    STARTUPINFOW si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);
    BOOL success = CreateProcessW(inject_path.c_str(), &command_line[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
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