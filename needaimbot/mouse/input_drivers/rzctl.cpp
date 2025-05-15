#include "rzctl.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <windows.h>

void RZControl::setupFunctions() {
    init = reinterpret_cast<InitFunc>(GetProcAddress(dllHandle, "init"));
    mouse_move = reinterpret_cast<MouseMoveFunc>(GetProcAddress(dllHandle, "mouse_move"));
    mouse_click = reinterpret_cast<MouseClickFunc>(GetProcAddress(dllHandle, "mouse_click"));
    keyboard_input = reinterpret_cast<KeyboardInputFunc>(GetProcAddress(dllHandle, "keyboard_input"));
    
    if (!init || !mouse_move || !mouse_click || !keyboard_input) {
        throw std::runtime_error("Failed to get one or more function pointers from rzctl.dll");
    }
}

RZControl::RZControl(const std::wstring& dll_path) : dllHandle(nullptr) {
    dllHandle = LoadLibraryW(dll_path.c_str());
    if (!dllHandle) {
        DWORD error = GetLastError();
        throw std::runtime_error("Failed to load rzctl.dll from " + 
                               std::string(dll_path.begin(), dll_path.end()) +
                               ". Error code: " + std::to_string(error));
    }
    try {
        setupFunctions();
    } catch (...) {
        FreeLibrary(dllHandle); 
        dllHandle = nullptr;
        throw;
    }
}

RZControl::~RZControl() {
    if (dllHandle) {
        FreeLibrary(dllHandle);
    }
}

bool RZControl::initialize() {
    if (!init) {
        throw std::logic_error("RZControl::init function pointer is null.");
    }
    BOOL result = init();
    return result != 0;
}

void RZControl::moveMouse(int x, int y, bool from_start_point) {
    if (!mouse_move) {
        throw std::logic_error("RZControl::mouse_move function pointer is null.");
    }
    mouse_move(x, y, from_start_point);
}

void RZControl::processMoveInstructions(const std::vector<std::pair<int, int>>& instructions, 
                           bool from_start_point /* = false */,
                           DWORD delay_ms /* = 0 */) {
    for (const auto& [x, y] : instructions) {
        moveMouse(x, y, from_start_point);
        if (delay_ms > 0) {
            Sleep(delay_ms);
        }
    }
}

void RZControl::sendKeyboardInput(short scan_code, KeyboardInputType up_down) {
    if (!keyboard_input) {
        throw std::logic_error("RZControl::keyboard_input function pointer is null.");
    }
    keyboard_input(scan_code, static_cast<int>(up_down));
}

void RZControl::mouseClick(MouseClick click_type) {
    if (!mouse_click) {
        throw std::logic_error("RZControl::mouse_click function pointer is null.");
    }
    mouse_click(static_cast<int>(click_type));
}