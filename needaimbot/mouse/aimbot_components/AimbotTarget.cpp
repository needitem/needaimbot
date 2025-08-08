#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <limits>
// OpenCV removed - using standard C++ types

#include "needaimbot.h"
#include "AimbotTarget.h"
#include "config.h"

// Constructor is now provided by Target structure
// AimbotTarget uses the Target(int x, int y, int width, int height, float conf, int cls) constructor

static AimbotTarget s_targetInstance(0, 0, 0, 0, -1.0f, 0);
