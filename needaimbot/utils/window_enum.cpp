#include "window_enum.h"
#include <dwmapi.h>
#include <set>

#pragma comment(lib, "dwmapi.lib")

// Common game window keywords
static const std::set<std::string> gameKeywords = {
    "Apex Legends", "Valorant", "Overwatch", "Call of Duty", "Fortnite",
    "Counter-Strike", "CS2", "PUBG", "Rainbow Six", "Battlefield",
    "Destiny", "Warzone", "Halo", "League of Legends", "Rocket League",
    "Unity", "Unreal", "Game", "DirectX", "OpenGL"
};

// System windows to exclude
static const std::set<std::string> excludeKeywords = {
    "Program Manager", "Windows Input Experience", "Settings", "Microsoft Store",
    "Mail", "Calendar", "Calculator", "Cortana", "Search", "Start",
    "NVIDIA", "AMD", "Intel", "Task Manager", "File Explorer"
};

BOOL CALLBACK WindowEnumerator::EnumWindowsProc(HWND hwnd, LPARAM lParam) {
    auto* windows = reinterpret_cast<std::vector<WindowInfo>*>(lParam);
    
    if (!IsWindowVisible(hwnd)) {
        return TRUE;
    }
    
    // Check if it's an alt-tab window
    if (!IsAltTabWindow(hwnd)) {
        return TRUE;
    }
    
    // Get window title
    char title[256] = {0};
    GetWindowTextA(hwnd, title, sizeof(title));
    
    // Skip empty titles
    if (strlen(title) == 0) {
        return TRUE;
    }
    
    // Get process ID
    DWORD processId = 0;
    GetWindowThreadProcessId(hwnd, &processId);
    
    WindowInfo info;
    info.hwnd = hwnd;
    info.title = std::string(title);
    info.processId = processId;
    info.isVisible = true;
    
    // Filter out some system windows
    bool shouldExclude = false;
    for (const auto& keyword : excludeKeywords) {
        if (info.title.find(keyword) != std::string::npos) {
            shouldExclude = true;
            break;
        }
    }
    
    if (!shouldExclude) {
        windows->push_back(info);
    }
    
    return TRUE;
}

bool WindowEnumerator::IsAltTabWindow(HWND hwnd) {
    // Check if window should appear in Alt-Tab
    LONG exStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
    
    // Skip tool windows and no-activate windows
    if (exStyle & WS_EX_TOOLWINDOW) return false;
    if (exStyle & WS_EX_NOACTIVATE) return false;
    
    // Check if window has an owner
    HWND owner = GetWindow(hwnd, GW_OWNER);
    if (owner && IsWindowVisible(owner)) {
        return false;
    }
    
    // Check if window is cloaked (Windows 10+ virtual desktops)
    BOOL isCloaked = FALSE;
    DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &isCloaked, sizeof(isCloaked));
    if (isCloaked) return false;
    
    return true;
}

std::vector<WindowInfo> WindowEnumerator::GetVisibleWindows() {
    std::vector<WindowInfo> windows;
    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&windows));
    
    // Sort by title for consistent ordering
    std::sort(windows.begin(), windows.end(), 
        [](const WindowInfo& a, const WindowInfo& b) {
            return a.title < b.title;
        });
    
    return windows;
}

std::vector<std::string> WindowEnumerator::GetWindowTitles() {
    auto windows = GetVisibleWindows();
    std::vector<std::string> titles;
    titles.reserve(windows.size());
    
    for (const auto& window : windows) {
        titles.push_back(window.title);
    }
    
    return titles;
}

bool WindowEnumerator::IsGameWindow(const std::string& title) {
    // Check if title contains any game keywords
    for (const auto& keyword : gameKeywords) {
        if (title.find(keyword) != std::string::npos) {
            return true;
        }
    }
    
    // Check for common game executable patterns
    if (title.find(".exe") != std::string::npos) {
        return false;  // Actual exe names in title are usually not games
    }
    
    // Check if it might be a fullscreen application
    HWND hwnd = FindWindowA(nullptr, title.c_str());
    if (hwnd) {
        RECT rect;
        GetWindowRect(hwnd, &rect);
        
        // Check if window is fullscreen or near fullscreen
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);
        
        int windowWidth = rect.right - rect.left;
        int windowHeight = rect.bottom - rect.top;
        
        // If window takes up most of the screen, it might be a game
        if (windowWidth >= screenWidth * 0.8 && windowHeight >= screenHeight * 0.8) {
            return true;
        }
    }
    
    return false;
}