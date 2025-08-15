#pragma once

#include <windows.h>
#include <vector>
#include <string>
#include <algorithm>

struct WindowInfo {
    HWND hwnd;
    std::string title;
    DWORD processId;
    bool isVisible;
};

class WindowEnumerator {
public:
    static std::vector<WindowInfo> GetVisibleWindows();
    static std::vector<std::string> GetWindowTitles();
    static bool IsGameWindow(const std::string& title);
    
private:
    static BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam);
    static bool IsAltTabWindow(HWND hwnd);
};