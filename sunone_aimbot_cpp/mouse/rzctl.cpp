#include <windows.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

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
    SCROLL_DOWN = 4287104000,
    SCROLL_UP = 7865344
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

    void setupFunctions() {
        init = reinterpret_cast<InitFunc>(GetProcAddress(dllHandle, "init"));
        mouse_move = reinterpret_cast<MouseMoveFunc>(GetProcAddress(dllHandle, "mouse_move"));
        mouse_click = reinterpret_cast<MouseClickFunc>(GetProcAddress(dllHandle, "mouse_click"));
        keyboard_input = reinterpret_cast<KeyboardInputFunc>(GetProcAddress(dllHandle, "keyboard_input"));
        
        if (!init || !mouse_move || !mouse_click || !keyboard_input) {
            throw std::runtime_error("Failed to get function pointers from DLL");
        }
    }

public:
    explicit RZControl(const std::wstring& dll_path) {
        dllHandle = LoadLibraryW(dll_path.c_str());
        if (!dllHandle) {
            throw std::runtime_error("Failed to load DLL from " + 
                                   std::string(dll_path.begin(), dll_path.end()));
        }
        setupFunctions();
    }

    ~RZControl() {
        if (dllHandle) {
            FreeLibrary(dllHandle);
        }
    }

    RZControl(const RZControl&) = delete;
    RZControl& operator=(const RZControl&) = delete;

    bool initialize() {
        BOOL result = init();
        if (!result) {
            throw std::runtime_error("Failed to initialize RZCONTROL device.");
        }
        return result != 0;
    }

    void moveMouse(int x, int y, bool from_start_point) {
        if (!from_start_point && (x == 0 || y == 0)) {
            throw std::invalid_argument("When not from start point, x and y cannot be 0.");
        }
        mouse_move(x, y, from_start_point);
    }

    // New method to process multiple movement instructions
    void processMoveInstructions(const std::vector<std::pair<int, int>>& instructions, 
                               bool from_start_point = false, 
                               DWORD delay_ms = 0) {
        for (const auto& [x, y] : instructions) {
            moveMouse(x, y, from_start_point);
            if (delay_ms > 0) {
                Sleep(delay_ms); // Add delay between movements if specified
            }
        }
    }

    void sendKeyboardInput(short scan_code, KeyboardInputType up_down) {
        keyboard_input(scan_code, static_cast<int>(up_down));
    }

    void mouseClick(MouseClick click_type) {
        mouse_click(static_cast<int>(click_type));
    }
};