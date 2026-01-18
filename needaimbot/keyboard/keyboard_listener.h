#ifndef KEYBOARD_LISTENER_H
#define KEYBOARD_LISTENER_H

#include <vector>
#include <string>

// Check if any key is pressed (supports hardware device state when available)
bool isAnyKeyPressed(const std::vector<std::string>& keys);

// Check if any key is pressed using Win32 API only (ignores hardware device state)
// Use this for system functions like Exit, Pause, Reload that should always work
bool isAnyKeyPressedWin32Only(const std::vector<std::string>& keys);

void keyboardListener();

#endif 

