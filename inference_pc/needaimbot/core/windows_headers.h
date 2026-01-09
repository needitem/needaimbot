// Standardized Windows headers include
// This file ensures correct order of Windows headers to avoid conflicts

#ifndef WINDOWS_HEADERS_H
#define WINDOWS_HEADERS_H

// Define these BEFORE including any Windows headers
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Prevent old winsock.h from being included
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif

// Prevent windows.h from defining conflicting macros
#ifndef WIN32
#define WIN32
#endif

// Ensure std::min/std::max remain functions instead of macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

// CRITICAL: Include winsock2 first, then ws2tcpip, then windows
#include <winsock2.h>
#include <ws2tcpip.h>
#include <Windows.h>

// Commonly used Windows headers
#include <shellapi.h>
#include <timeapi.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winmm.lib")

#endif // WINDOWS_HEADERS_H
