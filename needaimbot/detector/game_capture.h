#pragma once
#include <windows.h>
#include <d3d11.h>
#include <opencv2/opencv.hpp>
#include "hook_info.h"

class GameCapture {
public:
	GameCapture(int fov_width, int fov_height, int screen_width, int screen_height, const std::string& game_name, bool use_1ms_capture = false);
	~GameCapture();
	bool initialize();
	cv::Mat get_frame();
private:
	// Edit these paths as needed
	std::wstring inject_path = L"C:\\Program Files\\obs-studio\\data\\obs-plugins\\win-capture\\inject-helper64.exe";
	std::wstring hook_path = L"C:\\Program Files\\obs-studio\\data\\obs-plugins\\win-capture\\graphics-hook64.dll";
	const char* get_graphics_offsets64 = R"("C:\Program Files\obs-studio\data\obs-plugins\win-capture\get-graphics-offsets64.exe")";

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
	cv::Mat frame;
	int width, height;
	std::string game_name;
	bool use_1ms_capture;
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
