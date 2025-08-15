#include "window_helper.h"
#include <algorithm>
#include <iostream>

std::vector<GameWindow> WindowHelper::EnumerateGameWindows() {
    std::vector<GameWindow>* windows = new std::vector<GameWindow>();
    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(windows));
    
    std::vector<GameWindow> result = *windows;
    delete windows;
    return result;
}

BOOL CALLBACK WindowHelper::EnumWindowsProc(HWND hwnd, LPARAM lParam) {
    auto* windows = reinterpret_cast<std::vector<GameWindow>*>(lParam);
    
    // Skip invisible windows
    if (!IsWindowVisible(hwnd)) {
        return TRUE;
    }
    
    std::string title = GetWindowTitle(hwnd);
    if (title.empty()) {
        return TRUE;
    }
    
    DWORD processId;
    GetWindowThreadProcessId(hwnd, &processId);
    std::string processName = GetProcessName(processId);
    
    // Check if this looks like a game window
    if (IsGameWindow(hwnd, title, processName)) {
        GameWindow gameWindow;
        gameWindow.hwnd = hwnd;
        gameWindow.processId = processId;
        gameWindow.windowTitle = title;
        gameWindow.processName = processName;
        windows->push_back(gameWindow);
        
        std::cout << "[WindowHelper] Found game: " << title 
                  << " (Process: " << processName << ", PID: " << processId << ")" << std::endl;
    }
    
    return TRUE;
}

std::string WindowHelper::GetWindowTitle(HWND hwnd) {
    char title[256];
    int length = GetWindowTextA(hwnd, title, sizeof(title));
    if (length > 0) {
        return std::string(title);
    }
    return "";
}

std::string WindowHelper::GetProcessName(DWORD processId) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processId);
    if (hProcess) {
        char processName[MAX_PATH];
        if (GetModuleBaseNameA(hProcess, NULL, processName, sizeof(processName))) {
            CloseHandle(hProcess);
            return std::string(processName);
        }
        CloseHandle(hProcess);
    }
    return "";
}

bool WindowHelper::IsGameWindow(HWND hwnd, const std::string& title, const std::string& processName) {
    // Skip common non-game windows
    if (title.find("Microsoft") != std::string::npos ||
        title.find("Windows") != std::string::npos ||
        title.find("Chrome") != std::string::npos ||
        title.find("Firefox") != std::string::npos ||
        title.find("Edge") != std::string::npos ||
        title.find("Visual Studio") != std::string::npos ||
        title.find("Discord") != std::string::npos ||
        title.find("Steam") != std::string::npos ||
        title == "Program Manager" ||
        title == "Task Manager") {
        return false;
    }
    
    // Common game executables
    std::vector<std::string> gameExes = {
        "r5apex.exe",           // Apex Legends
        "cs2.exe",              // Counter-Strike 2
        "VALORANT.exe",         // Valorant
        "Overwatch.exe",        // Overwatch 2
        "FortniteClient.exe",   // Fortnite
        "RainbowSix.exe",       // Rainbow Six Siege
        "cod.exe",              // Call of Duty
        "bf2042.exe",           // Battlefield 2042
        "TslGame.exe",          // PUBG: BATTLEGROUNDS (Steam)
        "PUBG.exe",             // PUBG: BATTLEGROUNDS (Alternative)
        "ExecPubg.exe",         // PUBG: BATTLEGROUNDS (Kakao)
        "RogueCompany.exe",     // Rogue Company
        "Paladins.exe",         // Paladins
        "javaw.exe",            // Minecraft Java
        "GTA5.exe",             // GTA V
        "RDR2.exe"              // Red Dead Redemption 2
    };
    
    // Check if process name matches known games
    std::string lowerProcessName = processName;
    std::transform(lowerProcessName.begin(), lowerProcessName.end(), lowerProcessName.begin(), ::tolower);
    
    for (const auto& gameExe : gameExes) {
        std::string lowerGameExe = gameExe;
        std::transform(lowerGameExe.begin(), lowerGameExe.end(), lowerGameExe.begin(), ::tolower);
        if (lowerProcessName == lowerGameExe) {
            return true;
        }
    }
    
    // Check common game window title patterns
    std::vector<std::string> gameTitles = GetCommonGameTitles();
    for (const auto& gameTitle : gameTitles) {
        if (title.find(gameTitle) != std::string::npos) {
            return true;
        }
    }
    
    // Check if window is fullscreen or borderless (common for games)
    RECT rect;
    GetWindowRect(hwnd, &rect);
    int width = rect.right - rect.left;
    int height = rect.bottom - rect.top;
    
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    // If window covers most of the screen, might be a game
    if (width >= screenWidth * 0.9 && height >= screenHeight * 0.9) {
        // Additional check: games usually have specific window styles
        LONG style = GetWindowLong(hwnd, GWL_STYLE);
        if (!(style & WS_CAPTION) || !(style & WS_THICKFRAME)) {
            return true;  // Borderless or fullscreen window
        }
    }
    
    return false;
}

HWND WindowHelper::FindWindowByPartialTitle(const std::string& partialTitle) {
    auto windows = EnumerateGameWindows();
    
    // Convert partial title to lowercase for case-insensitive search
    std::string lowerPartial = partialTitle;
    std::transform(lowerPartial.begin(), lowerPartial.end(), lowerPartial.begin(), ::tolower);
    
    for (const auto& window : windows) {
        std::string lowerTitle = window.windowTitle;
        std::transform(lowerTitle.begin(), lowerTitle.end(), lowerTitle.begin(), ::tolower);
        
        if (lowerTitle.find(lowerPartial) != std::string::npos) {
            std::cout << "[WindowHelper] Found matching window: " << window.windowTitle << std::endl;
            return window.hwnd;
        }
    }
    
    return nullptr;
}

std::vector<std::string> WindowHelper::GetCommonGameTitles() {
    return {
        "Apex Legends",
        "Counter-Strike",
        "VALORANT",
        "Overwatch",
        "Fortnite",
        "Rainbow Six",
        "Call of Duty",
        "Battlefield",
        "PUBG: BATTLEGROUNDS",  // Full PUBG title
        "PUBG",                 // Short version
        "배틀그라운드",           // Korean title
        "Rogue Company",
        "Paladins",
        "Minecraft",
        "Grand Theft Auto",
        "Red Dead Redemption",
        "League of Legends",
        "Dota 2",
        "Team Fortress",
        "Rocket League",
        "Among Us",
        "Fall Guys",
        "Destiny",
        "Warframe",
        "World of Warcraft",
        "Final Fantasy",
        "Genshin Impact"
    };
}