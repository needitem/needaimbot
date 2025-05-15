#ifndef RZCTL_H
#define RZCTL_H

#include <windows.h>
#include <string>
#include <vector>
#include <utility> // For std::pair

// Define enums in the header so they are accessible
enum class MouseClick {
    LEFT_DOWN = 1,
    LEFT_UP = 2,
    RIGHT_DOWN = 4,
    RIGHT_UP = 8,
    SCROLL_CLICK_DOWN = 16,
    SCROLL_CLICK_UP = 32,
    BACK_DOWN = 64,
    BACK_UP = 128,
    FORWARD_DOWN = 256,
    FORWARD_UP = 512,
    SCROLL_DOWN = 4287104000, // Note: Check if these large uint values are correct
    SCROLL_UP = 7865344
};

enum class KeyboardInputType {
    KEYBOARD_DOWN = 0,
    KEYBOARD_UP = 1
};

class RZControl {
private:
    HINSTANCE dllHandle;

    // Define function pointer types
    using InitFunc = BOOL (*)();
    using MouseMoveFunc = void (*)(int, int, BOOL);
    using MouseClickFunc = void (*)(int);
    using KeyboardInputFunc = void (*)(SHORT, int);

    // Store function pointers
    InitFunc init;
    MouseMoveFunc mouse_move;
    MouseClickFunc mouse_click;
    KeyboardInputFunc keyboard_input;

    // Private helper to load function pointers
    void setupFunctions();

public:
    // Constructor loads the DLL and function pointers
    explicit RZControl(const std::wstring& dll_path);

    // Destructor frees the DLL
    ~RZControl();

    // Disable copy constructor and assignment operator
    RZControl(const RZControl&) = delete;
    RZControl& operator=(const RZControl&) = delete;

    // Initialize the RZCONTROL device
    bool initialize();

    // Move the mouse
    void moveMouse(int x, int y, bool from_start_point);

    // Process multiple movement instructions with optional delay
    void processMoveInstructions(const std::vector<std::pair<int, int>>& instructions,
                               bool from_start_point = false,
                               DWORD delay_ms = 0);

    // Send keyboard input
    void sendKeyboardInput(short scan_code, KeyboardInputType up_down);

    // Send mouse click events
    void mouseClick(MouseClick click_type);
};

#endif // RZCTL_H
