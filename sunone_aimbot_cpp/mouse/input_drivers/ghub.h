#ifndef GHUB_H
#define GHUB_H

#include <filesystem>
#include <windows.h>

// Define function pointer types matching the DLL functions
typedef bool (*mouse_open_t)();
typedef bool (*moveR_t)(int, int);
typedef bool (*press_t)(int);
typedef bool (*release_t)();
typedef bool (*mouse_close_t)();

class GhubMouse
{
public:
    GhubMouse();
    ~GhubMouse();
    bool mouse_xy(int x, int y);
    bool mouse_down(int key = 1);
    bool mouse_up(int key = 1);
    bool mouse_close();

private:
    std::filesystem::path basedir;
    std::filesystem::path dlldir;
    HMODULE gm;
    bool gmok;

    // Cached function pointers
    mouse_open_t pfnMouseOpen = nullptr;
    moveR_t pfnMoveR = nullptr;
    press_t pfnPress = nullptr;
    release_t pfnRelease = nullptr;
    mouse_close_t pfnMouseClose = nullptr;

    static UINT _ghub_SendInput(UINT nInputs, LPINPUT pInputs);
    static INPUT _ghub_Input(MOUSEINPUT mi);
    static MOUSEINPUT _ghub_MouseInput(DWORD flags, LONG x, LONG y, DWORD data);
    static INPUT _ghub_Mouse(DWORD flags, LONG x = 0, LONG y = 0, DWORD data = 0);
};

#endif // GHUB_H