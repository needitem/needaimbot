#include "ghub.h"
#include <iostream>
#include <string>

UINT GhubMouse::_ghub_SendInput(UINT nInputs, LPINPUT pInputs)
{
    return SendInput(nInputs, pInputs, sizeof(INPUT));
}

INPUT GhubMouse::_ghub_Input(MOUSEINPUT mi)
{
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi = mi;
    return input;
}

MOUSEINPUT GhubMouse::_ghub_MouseInput(DWORD flags, LONG x, LONG y, DWORD data)
{
    MOUSEINPUT mi = { 0 };
    mi.dx = x;
    mi.dy = y;
    mi.mouseData = data;
    mi.dwFlags = flags;
    return mi;
}

INPUT GhubMouse::_ghub_Mouse(DWORD flags, LONG x, LONG y, DWORD data)
{
    return _ghub_Input(_ghub_MouseInput(flags, x, y, data));
}

GhubMouse::GhubMouse()
{
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    basedir = std::filesystem::path(buffer).parent_path();
    dlldir = basedir / "ghub_mouse.dll";
    gm = LoadLibraryA(dlldir.string().c_str());
    if (gm == NULL)
    {
        std::cerr << "[Ghub] Failed to load DLL" << std::endl;
        gmok = false;
    }
    else
    {
        // Cache function pointers
        pfnMouseOpen = reinterpret_cast<mouse_open_t>(GetProcAddress(gm, "mouse_open"));
        pfnMoveR = reinterpret_cast<moveR_t>(GetProcAddress(gm, "moveR"));
        pfnPress = reinterpret_cast<press_t>(GetProcAddress(gm, "press"));
        pfnRelease = reinterpret_cast<release_t>(GetProcAddress(gm, "release"));
        pfnMouseClose = reinterpret_cast<mouse_close_t>(GetProcAddress(gm, "mouse_close"));

        // Check if mouse_open function pointer is valid and call it
        if (pfnMouseOpen == NULL)
        {
            std::cerr << "[Ghub] Failed to get mouse_open function address!" << std::endl;
            gmok = false;
        }
        else
        {
            gmok = pfnMouseOpen();
            if (!gmok) {
                 std::cerr << "[Ghub] mouse_open() failed!" << std::endl;
            } else {
                 // Check if other critical functions were loaded
                 if (pfnMoveR == nullptr || pfnPress == nullptr || pfnRelease == nullptr) {
                     std::cerr << "[Ghub] Warning: Failed to load one or more core mouse functions (moveR, press, release)." << std::endl;
                     // Decide if this constitutes failure - perhaps gmok should be false?
                     // For now, we'll leave gmok as true if mouse_open succeeded, 
                     // but the functions using the null pointers will fallback to SendInput.
                 }
            }
        }
    }
}

GhubMouse::~GhubMouse()
{
    if (gm != NULL)
    {
        FreeLibrary(gm);
    }
}

bool GhubMouse::mouse_xy(int x, int y)
{
    // Use cached function pointer if available and DLL is okay
    if (gmok && pfnMoveR != nullptr)
    {
        return pfnMoveR(x, y);
    }
    // Fallback to SendInput
    INPUT input = _ghub_Mouse(MOUSEEVENTF_MOVE, x, y);
    return _ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_down(int key)
{
    // Use cached function pointer if available and DLL is okay
    if (gmok && pfnPress != nullptr)
    {
        return pfnPress(key);
    }
    // Fallback to SendInput
    DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_RIGHTDOWN;
    INPUT input = _ghub_Mouse(flag);
    return _ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_up(int key)
{
    // Use cached function pointer if available and DLL is okay
    if (gmok && pfnRelease != nullptr)
    {
        // Assuming release takes no arguments based on previous GetProcAddress call
        return pfnRelease(); 
    }
    // Fallback to SendInput
    DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTUP : MOUSEEVENTF_RIGHTUP;
    INPUT input = _ghub_Mouse(flag);
    return _ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_close()
{
    // Use cached function pointer if available and DLL is okay
    if (gmok && pfnMouseClose != nullptr)
    {
        return pfnMouseClose();
    }
    return false; // No SendInput fallback for close
}