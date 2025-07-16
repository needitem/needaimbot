#include <windows.h>
#include <d3d11.h>
#include "hook_info.h"
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#ifdef _MSC_VER
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib") 
#endif

struct Image {
	int width;
	int height;
	int pitch;   
	BYTE* data;
	
	// For frame pooling
	Image() : width(0), height(0), pitch(0), data(nullptr) {}
	Image(int w, int h, int p) : width(w), height(h), pitch(p), data(nullptr) {}
};

class GameCapture {
public:
	GameCapture(int fov_width, int fov_height, int screen_width, int screen_height, const std::string& game_name);
	~GameCapture();
	bool initialize();
	Image get_frame();
	bool SaveBMP(const char* filename, const Image& img);
	
	// Performance optimization methods
	void enable_async_capture(bool enable) { async_capture_enabled = enable; }
	void set_frame_pool_size(size_t size) { max_frame_pool_size = size; }
	
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
	BYTE* FrameBuffer;
	int width, height;
	std::string game_name;
	
	// Performance optimization members
	std::vector<Image> frame_pool;
	std::mutex frame_pool_mutex;
	size_t max_frame_pool_size = 5;
	
	// Double buffering for async capture
	ID3D11Texture2D* pStagingTexture2;
	std::atomic<int> current_staging_buffer{0};
	BYTE* FrameBuffer2;
	
	// Async capture thread
	std::thread capture_thread;
	std::atomic<bool> capture_running{false};
	std::atomic<bool> async_capture_enabled{false};
	std::mutex capture_mutex;
	std::condition_variable capture_cv;
	Image ready_frame;
	std::atomic<bool> new_frame_ready{false};
	
	// Memory management
	std::vector<BYTE*> allocated_buffers;
	std::mutex buffer_mutex;
	HANDLE inject_hook(DWORD target_id);
	HANDLE OpenMapPlusId(const std::wstring& base_name, DWORD id);
	HANDLE OpenDataMap(uint32_t window, uint32_t map_id);
	HANDLE OpenEventPlusId(const std::wstring& base_name, DWORD id);
	HANDLE OpenMutexPlusId(const std::wstring& base_name, DWORD id);
	HANDLE CreateKeepaliveMutex(int pid);
	D3D11_BOX get_region();
	void initialize_offsets();
	std::string run_get_graphics_offsets();
	
	// Performance optimization methods
	Image get_pooled_frame();
	void return_to_pool(const Image& img);
	void capture_thread_func();
	Image get_frame_internal();
	void cleanup_buffers();
	BYTE* allocate_buffer(size_t size);
};
