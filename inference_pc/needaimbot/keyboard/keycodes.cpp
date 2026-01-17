#ifdef _WIN32
#include "../core/windows_headers.h"
#else
// Linux: Define virtual key codes matching Windows values for config compatibility
#define VK_LBUTTON        0x01
#define VK_RBUTTON        0x02
#define VK_CANCEL         0x03
#define VK_MBUTTON        0x04
#define VK_XBUTTON1       0x05
#define VK_XBUTTON2       0x06
#define VK_BACK           0x08
#define VK_TAB            0x09
#define VK_CLEAR          0x0C
#define VK_RETURN         0x0D
#define VK_PAUSE          0x13
#define VK_CAPITAL        0x14
#define VK_ESCAPE         0x1B
#define VK_SPACE          0x20
#define VK_PRIOR          0x21
#define VK_NEXT           0x22
#define VK_END            0x23
#define VK_HOME           0x24
#define VK_LEFT           0x25
#define VK_UP             0x26
#define VK_RIGHT          0x27
#define VK_DOWN           0x28
#define VK_SELECT         0x29
#define VK_PRINT          0x2A
#define VK_EXECUTE        0x2B
#define VK_SNAPSHOT       0x2C
#define VK_INSERT         0x2D
#define VK_DELETE         0x2E
#define VK_HELP           0x2F
#define VK_LWIN           0x5B
#define VK_RWIN           0x5C
#define VK_APPS           0x5D
#define VK_SLEEP          0x5F
#define VK_NUMPAD0        0x60
#define VK_NUMPAD1        0x61
#define VK_NUMPAD2        0x62
#define VK_NUMPAD3        0x63
#define VK_NUMPAD4        0x64
#define VK_NUMPAD5        0x65
#define VK_NUMPAD6        0x66
#define VK_NUMPAD7        0x67
#define VK_NUMPAD8        0x68
#define VK_NUMPAD9        0x69
#define VK_MULTIPLY       0x6A
#define VK_ADD            0x6B
#define VK_SEPARATOR      0x6C
#define VK_SUBTRACT       0x6D
#define VK_DECIMAL        0x6E
#define VK_DIVIDE         0x6F
#define VK_F1             0x70
#define VK_F2             0x71
#define VK_F3             0x72
#define VK_F4             0x73
#define VK_F5             0x74
#define VK_F6             0x75
#define VK_F7             0x76
#define VK_F8             0x77
#define VK_F9             0x78
#define VK_F10            0x79
#define VK_F11            0x7A
#define VK_F12            0x7B
#define VK_NUMLOCK        0x90
#define VK_SCROLL         0x91
#define VK_LSHIFT         0xA0
#define VK_RSHIFT         0xA1
#define VK_LCONTROL       0xA2
#define VK_RCONTROL       0xA3
#define VK_LMENU          0xA4
#define VK_RMENU          0xA5
#define VK_BROWSER_BACK   0xA6
#define VK_BROWSER_REFRESH 0xA8
#define VK_BROWSER_STOP   0xA9
#define VK_BROWSER_SEARCH 0xAA
#define VK_BROWSER_FAVORITES 0xAB
#define VK_BROWSER_HOME   0xAC
#define VK_VOLUME_MUTE    0xAD
#define VK_VOLUME_DOWN    0xAE
#define VK_VOLUME_UP      0xAF
#define VK_MEDIA_NEXT_TRACK 0xB0
#define VK_MEDIA_PREV_TRACK 0xB1
#define VK_MEDIA_STOP     0xB2
#define VK_MEDIA_PLAY_PAUSE 0xB3
#define VK_LAUNCH_MAIL    0xB4
#define VK_LAUNCH_MEDIA_SELECT 0xB5
#define VK_LAUNCH_APP1    0xB6
#define VK_LAUNCH_APP2    0xB7
#endif

#include "keycodes.h"

#include <string>
#include <unordered_map>

std::unordered_map<std::string, int> KeyCodes::key_code_map =
{
    {"None", 0},
    {"LeftMouseButton", VK_LBUTTON},
    {"RightMouseButton", VK_RBUTTON},
    {"ControlBreak", VK_CANCEL},
    {"MiddleMouseButton", VK_MBUTTON},
    {"X1MouseButton", VK_XBUTTON1},
    {"X2MouseButton", VK_XBUTTON2},
    {"Backspace", VK_BACK},
    {"Tab", VK_TAB},
    {"Clear", VK_CLEAR},
    {"Enter", VK_RETURN},
    {"Pause", VK_PAUSE},
    {"CapsLock", VK_CAPITAL},
    {"Escape", VK_ESCAPE},
    {"Space", VK_SPACE},
    {"PageUp", VK_PRIOR},
    {"PageDown", VK_NEXT},
    {"End", VK_END},
    {"Home", VK_HOME},
    {"LeftArrow", VK_LEFT},
    {"UpArrow", VK_UP},
    {"RightArrow", VK_RIGHT},
    {"DownArrow", VK_DOWN},
    {"Select", VK_SELECT},
    {"Print", VK_PRINT},
    {"Execute", VK_EXECUTE},
    {"PrintScreen", VK_SNAPSHOT},
    {"Ins", VK_INSERT},
    {"Delete", VK_DELETE},
    {"Help", VK_HELP},
    {"Key0", '0'},
    {"Key1", '1'},
    {"Key2", '2'},
    {"Key3", '3'},
    {"Key4", '4'},
    {"Key5", '5'},
    {"Key6", '6'},
    {"Key7", '7'},
    {"Key8", '8'},
    {"Key9", '9'},
    {"A", 'A'},
    {"B", 'B'},
    {"C", 'C'},
    {"D", 'D'},
    {"E", 'E'},
    {"F", 'F'},
    {"G", 'G'},
    {"H", 'H'},
    {"I", 'I'},
    {"J", 'J'},
    {"K", 'K'},
    {"L", 'L'},
    {"M", 'M'},
    {"N", 'N'},
    {"O", 'O'},
    {"P", 'P'},
    {"Q", 'Q'},
    {"R", 'R'},
    {"S", 'S'},
    {"T", 'T'},
    {"U", 'U'},
    {"V", 'V'},
    {"W", 'W'},
    {"X", 'X'},
    {"Y", 'Y'},
    {"Z", 'Z'},
    {"LeftWindowsKey", VK_LWIN},
    {"RightWindowsKey", VK_RWIN},
    {"Application", VK_APPS},
    {"Sleep", VK_SLEEP},
    {"NumpadKey0", VK_NUMPAD0},
    {"NumpadKey1", VK_NUMPAD1},
    {"NumpadKey2", VK_NUMPAD2},
    {"NumpadKey3", VK_NUMPAD3},
    {"NumpadKey4", VK_NUMPAD4},
    {"NumpadKey5", VK_NUMPAD5},
    {"NumpadKey6", VK_NUMPAD6},
    {"NumpadKey7", VK_NUMPAD7},
    {"NumpadKey8", VK_NUMPAD8},
    {"NumpadKey9", VK_NUMPAD9},
    {"Multiply", VK_MULTIPLY},
    {"Add", VK_ADD},
    {"Separator", VK_SEPARATOR},
    {"Subtract", VK_SUBTRACT},
    {"Decimal", VK_DECIMAL},
    {"Divide", VK_DIVIDE},
    {"F1", VK_F1},
    {"F2", VK_F2},
    {"F3", VK_F3},
    {"F4", VK_F4},
    {"F5", VK_F5},
    {"F6", VK_F6},
    {"F7", VK_F7},
    {"F8", VK_F8},
    {"F9", VK_F9},
    {"F10", VK_F10},
    {"F11", VK_F11},
    {"F12", VK_F12},
    {"NumLock", VK_NUMLOCK},
    {"ScrollLock", VK_SCROLL},
    {"LeftShift", VK_LSHIFT},
    {"RightShift", VK_RSHIFT},
    {"LeftControl", VK_LCONTROL},
    {"RightControl", VK_RCONTROL},
    {"LeftAlt", VK_LMENU},
    {"RightAlt", VK_RMENU},
    {"BrowserBack", VK_BROWSER_BACK},
    {"BrowserRefresh", VK_BROWSER_REFRESH},
    {"BrowserStop", VK_BROWSER_STOP},
    {"BrowserSearch", VK_BROWSER_SEARCH},
    {"BrowserFavorites", VK_BROWSER_FAVORITES},
    {"BrowserHome", VK_BROWSER_HOME},
    {"VolumeMute", VK_VOLUME_MUTE},
    {"VolumeDown", VK_VOLUME_DOWN},
    {"VolumeUp", VK_VOLUME_UP},
    {"NextTrack", VK_MEDIA_NEXT_TRACK},
    {"PreviousTrack", VK_MEDIA_PREV_TRACK},
    {"StopMedia", VK_MEDIA_STOP},
    {"PlayMedia", VK_MEDIA_PLAY_PAUSE},
    {"StartMailKey", VK_LAUNCH_MAIL},
    {"SelectMedia", VK_LAUNCH_MEDIA_SELECT},
    {"StartApplication1", VK_LAUNCH_APP1},
    {"StartApplication2", VK_LAUNCH_APP2}
};

int KeyCodes::getKeyCode(const std::string& key_name) {
    auto it = key_code_map.find(key_name);
    if (it != key_code_map.end())
        return it->second;
    else
        return -1;
}
