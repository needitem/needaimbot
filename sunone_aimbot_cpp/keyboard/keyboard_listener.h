#ifndef KEYBOARD_LISTENER_H
#define KEYBOARD_LISTENER_H

#include <vector>
#include <string>

// Function to check if any key in the list is pressed
bool isAnyKeyPressed(const std::vector<std::string>& keys);

void keyboardListener();

#endif // KEYBOARD_LISTENER_H