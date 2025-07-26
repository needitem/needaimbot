#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

// OpenCV removed - using custom CUDA processing

#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cstdio> 

#include "visuals.h"
#include "config.h"
#include "needaimbot.h" 
#include "../capture/capture.h"   


extern std::mutex configMutex; 









