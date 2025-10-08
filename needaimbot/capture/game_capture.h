#pragma once

#include <windows.h>
#include <d3d11.h>
#include "hook_info.h"
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <memory>

// CUDA utilities for pinned memory
#include <cuda_runtime.h>
#include "../utils/cuda_utils.h"

#ifdef _MSC_VER
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif

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
    Image get_frame();
    bool SaveBMP(const char* filename, const Image& img);
    // Added helpers for pipeline compatibility
    bool SetCaptureRegion(int x, int y, int w, int h);
    void GetCaptureRegion(int* x, int* y, int* w, int* h) const;
    int GetScreenWidth() const { return screen_width; }
    int GetScreenHeight() const { return screen_height; }
private:
	// Edit these paths as needed
	std::wstring inject_path = L"obs_stuff\\inject-helper64.exe";
	std::wstring hook_path = L"obs_stuff\\graphics-hook64.dll";
	const char* get_graphics_offsets64 = R"("obs_stuff\\get-graphics-offsets64.exe")";

	int screen_width, screen_height;
	HWND hwnd;
	DWORD process_id, thread_id;
	HANDLE hook_restart, hook_stop, hook_ready, hook_exit, hook_init;
	HANDLE keepalive_mutex, hook_info_map, hook_data_map;
	HANDLE texture_mutexes[2];
	hook_info* shared_hook_info;
	shtex_data* shared_shtex_data;
	ID3D11Device* pDevice;
	ID3D11DeviceContext* pContext;
	ID3D11Resource* pSharedResource;
	ID3D11Texture2D* pStagingTexture;
    D3D11_BOX sourceRegion;
	Image frame;
	BYTE* FrameBuffer;  // Legacy fallback
	int width, height;
	std::string game_name;

	// Pinned memory for faster CPU transfers
	std::unique_ptr<CudaPinnedMemory<unsigned char>> m_frameBufferPinned;
	HANDLE inject_hook(DWORD target_id);
	HANDLE OpenMapPlusId(const std::wstring& base_name, DWORD id);
	HANDLE OpenDataMap(uint32_t window, uint32_t map_id);
	HANDLE OpenEventPlusId(const std::wstring& base_name, DWORD id);
	HANDLE OpenMutexPlusId(const std::wstring& base_name, DWORD id);
	HANDLE CreateKeepaliveMutex(int pid);
	D3D11_BOX get_region();
	void initialize_offsets();
	std::string run_get_graphics_offsets();
};
