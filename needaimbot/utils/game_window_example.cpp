// Example usage of WindowHelper to find game windows
#include "window_helper.h"
#include <iostream>

void ExampleUsage() {
    // Method 1: List all detected game windows
    std::cout << "\n=== Detected Game Windows ===" << std::endl;
    auto gameWindows = WindowHelper::EnumerateGameWindows();
    
    for (size_t i = 0; i < gameWindows.size(); i++) {
        std::cout << i + 1 << ". " << gameWindows[i].windowTitle 
                  << " (Process: " << gameWindows[i].processName 
                  << ", PID: " << gameWindows[i].processId << ")" << std::endl;
    }
    
    if (gameWindows.empty()) {
        std::cout << "No game windows detected" << std::endl;
    } else {
        // Let user select which game to capture
        std::cout << "\nSelect game window (1-" << gameWindows.size() << "): ";
        int choice;
        std::cin >> choice;
        
        if (choice > 0 && choice <= gameWindows.size()) {
            GameWindow selected = gameWindows[choice - 1];
            std::cout << "Selected: " << selected.windowTitle << std::endl;
            // Use selected.hwnd for capture
        }
    }
    
    // Method 2: Find specific game by partial name
    std::cout << "\n=== Finding Specific Game ===" << std::endl;
    HWND apexWindow = WindowHelper::FindWindowByPartialTitle("Apex");
    if (apexWindow) {
        std::cout << "Found Apex Legends window!" << std::endl;
    }
    
    // Method 3: Try common game names
    std::cout << "\n=== Searching Common Games ===" << std::endl;
    std::vector<std::string> gamesToSearch = {
        "Apex Legends",
        "Counter-Strike",
        "VALORANT",
        "Overwatch"
    };
    
    for (const auto& gameName : gamesToSearch) {
        HWND hwnd = WindowHelper::FindWindowByPartialTitle(gameName);
        if (hwnd) {
            std::cout << "Found: " << gameName << std::endl;
        }
    }
}