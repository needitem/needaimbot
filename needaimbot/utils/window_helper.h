#pragma once

#include <windows.h>
#include <vector>
#include <string>
#include <psapi.h>
#include <tlhelp32.h>

#pragma comment(lib, "psapi.lib")

struct GameWindow {
    HWND hwnd;
    DWORD processId;
    std::string windowTitle;
    std::string processName;
};

class WindowHelper {
public:
    // Enumerate all visible windows that might be games
    static std::vector<GameWindow> EnumerateGameWindows();
    
    // Find window by partial title match
    static HWND FindWindowByPartialTitle(const std::string& partialTitle);
    
    // Get list of common game window titles
    static std::vector<std::string> GetCommonGameTitles();
    
private:
    static BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam);
    static std::string GetWindowTitle(HWND hwnd);
    static std::string GetProcessName(DWORD processId);
    static bool IsGameWindow(HWND hwnd, const std::string& title, const std::string& processName);
};