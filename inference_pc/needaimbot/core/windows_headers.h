// Standardized platform headers include
// This file ensures correct order of headers to avoid conflicts

#ifndef WINDOWS_HEADERS_H
#define WINDOWS_HEADERS_H

#ifdef _WIN32
// =============================================================================
// Windows Platform
// =============================================================================

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

#else
// =============================================================================
// Linux/Unix Platform
// =============================================================================

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/time.h>
#include <pthread.h>

// Windows compatibility types for Linux
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket close
#define WSAGetLastError() errno
#define WSAEWOULDBLOCK EWOULDBLOCK
#define WSAEINPROGRESS EINPROGRESS

// Sleep compatibility
#define Sleep(ms) usleep((ms) * 1000)

// Windows types
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned int DWORD;
typedef int BOOL;
typedef void* HANDLE;
typedef void* HWND;
typedef void* HINSTANCE;
typedef long LONG;
typedef unsigned long ULONG;
typedef long long LONGLONG;
typedef unsigned long long ULONGLONG;

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

// LARGE_INTEGER for timing
typedef union _LARGE_INTEGER {
    struct {
        DWORD LowPart;
        LONG HighPart;
    };
    struct {
        DWORD LowPart;
        LONG HighPart;
    } u;
    LONGLONG QuadPart;
} LARGE_INTEGER;

// QueryPerformanceCounter/Frequency for Linux
inline int QueryPerformanceCounter(LARGE_INTEGER* lpPerformanceCount) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    lpPerformanceCount->QuadPart = ts.tv_sec * 1000000000LL + ts.tv_nsec;
    return 1;
}

inline int QueryPerformanceFrequency(LARGE_INTEGER* lpFrequency) {
    lpFrequency->QuadPart = 1000000000LL; // nanoseconds
    return 1;
}

// GetTickCount64 equivalent
inline unsigned long long GetTickCount64() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;
}

#endif // _WIN32

#endif // WINDOWS_HEADERS_H
