#ifndef RZCTL_H
#define RZCTL_H

#include <windows.h>
#include <string>
#include <vector>
#include <utility> 


enum class MouseClick : uint32_t {
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
    SCROLL_DOWN = 4287104000U, 
    SCROLL_UP = 7865344U
};

enum class KeyboardInputType {
    KEYBOARD_DOWN = 0,
    KEYBOARD_UP = 1
};

class RZControl {
private:
    HINSTANCE dllHandle;

    
    using InitFunc = BOOL (*)();
    using MouseMoveFunc = void (*)(int, int, BOOL);
    using MouseClickFunc = void (*)(int);
    using KeyboardInputFunc = void (*)(SHORT, int);

    
    InitFunc init;
    MouseMoveFunc mouse_move;
    MouseClickFunc mouse_click;
    KeyboardInputFunc keyboard_input;

    
    void setupFunctions();

public:
    
    explicit RZControl(const std::wstring& dll_path);

    
    ~RZControl();

    
    RZControl(const RZControl&) = delete;
    RZControl& operator=(const RZControl&) = delete;

    
    bool initialize();

    
    void moveMouse(int x, int y, bool from_start_point);

    
    void processMoveInstructions(const std::vector<std::pair<int, int>>& instructions,
                               bool from_start_point = false,
                               DWORD delay_ms = 0);

    
    void sendKeyboardInput(short scan_code, KeyboardInputType up_down);

    
    void mouseClick(MouseClick click_type);
};

#endif 

