#include "AutoClicker.h"
#include <iostream>

AutoClicker::AutoClicker() 
    : isRunning(false), isClicking(false), clickDelay(50) {
}

AutoClicker::~AutoClicker() {
    Stop();
}

void AutoClicker::ClickLoop() {
    while (isRunning) {
        if (isClicking) {
            INPUT input = {0};
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            SendInput(1, &input, sizeof(INPUT));
            
            Sleep(15);  // Reduce CPU usage - 15ms is still very responsive
            
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
            
            Sleep(clickDelay - 5);
        } else {
            Sleep(10);
        }
    }
}

void AutoClicker::InputLoop() {
    bool leftButtonWasPressed = false;
    
    while (isRunning) {
        bool leftButtonPressed = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
        
        if (leftButtonPressed && !leftButtonWasPressed) {
            isClicking = true;
        } else if (!leftButtonPressed && leftButtonWasPressed) {
            isClicking = false;
        }
        
        leftButtonWasPressed = leftButtonPressed;
        
        // F1 키로 속도 증가
        if (GetAsyncKeyState(VK_F1) & 0x0001) {
            int newDelay = std::max(10, clickDelay.load() - 10);
            SetClickDelay(newDelay);
            std::cout << "Click speed increased: " << (1000.0 / newDelay) << " CPS\n";
        }
        
        // F2 키로 속도 감소
        if (GetAsyncKeyState(VK_F2) & 0x0001) {
            int newDelay = std::min(500, clickDelay.load() + 10);
            SetClickDelay(newDelay);
            std::cout << "Click speed decreased: " << (1000.0 / newDelay) << " CPS\n";
        }
        
        // ESC 키로 종료
        if (GetAsyncKeyState(VK_ESCAPE) & 0x0001) {
            std::cout << "Stopping auto clicker...\n";
            Stop();
            break;
        }
        
        Sleep(15);  // Reduce CPU usage - 15ms is still very responsive
    }
}

void AutoClicker::Start() {
    if (isRunning) return;
    
    isRunning = true;
    
    std::cout << "Auto Clicker Started!\n";
    std::cout << "Usage:\n";
    std::cout << "- Hold left mouse button for auto clicking\n";
    std::cout << "- F1: Increase click speed\n";
    std::cout << "- F2: Decrease click speed\n";
    std::cout << "- ESC: Stop program\n";
    std::cout << "Current speed: " << (1000.0 / clickDelay) << " CPS\n";
    
    clickThread = std::thread(&AutoClicker::ClickLoop, this);
    inputThread = std::thread(&AutoClicker::InputLoop, this);
}

void AutoClicker::Stop() {
    isRunning = false;
    isClicking = false;
    
    if (clickThread.joinable()) {
        clickThread.join();
    }
    if (inputThread.joinable()) {
        inputThread.join();
    }
}

void AutoClicker::SetClickDelay(int delayMs) {
    clickDelay = delayMs;
}