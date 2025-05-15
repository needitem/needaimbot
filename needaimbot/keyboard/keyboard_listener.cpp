#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "../config/config.h"
#include "../mouse/input_drivers/SerialConnection.h"
#include "keyboard_listener.h"
#include "../mouse/mouse.h"
#include "keycodes.h"
#include "needaimbot.h"
#include "capture.h"

// Constants for offset and norecoil strength adjustments
const float MIN_OFFSET_Y = 0.0f;
const float MAX_OFFSET_Y = 1.0f;
const float MIN_NORECOIL_STRENGTH = 0.1f;
const float MAX_NORECOIL_STRENGTH = 500.0f;

extern std::atomic<bool> shouldExit;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;
extern std::atomic<bool> detectionPaused;
extern std::atomic<bool> auto_shoot_active;

extern MouseThread* globalMouseThread;

// Arrow key vectors
const std::vector<std::string> upArrowKeys = { "UpArrow" };
const std::vector<std::string> downArrowKeys = { "DownArrow" };
const std::vector<std::string> leftArrowKeys = { "LeftArrow" };
const std::vector<std::string> rightArrowKeys = { "RightArrow" };
const std::vector<std::string> shiftKeys = { "LeftShift", "RightShift" };

// Previous key states
bool prevUpArrow = false;
bool prevDownArrow = false;
bool prevLeftArrow = false;
bool prevRightArrow = false;

bool isAnyKeyPressed(const std::vector<std::string>& keys)
{
    for (const auto& key_name : keys)
    {
        int key_code = KeyCodes::getKeyCode(key_name);

        if (key_code != -1 && (GetAsyncKeyState(key_code) & 0x8000))
        {
            return true;
        }
    }
    return false;
}

void keyboardListener()
{
    while (!shouldExit)
    {
        // Aiming
        aiming = config.auto_aim ||
                 isAnyKeyPressed(config.button_targeting) ||
                 (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->aiming_active);

        // Shooting
        shooting = isAnyKeyPressed(config.button_shoot) ||
            (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->shooting_active);

        // Zooming
        zooming = isAnyKeyPressed(config.button_zoom) ||
            (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->zooming_active);

        // Exit
        if (isAnyKeyPressed(config.button_exit))
        {
            shouldExit = true;
            quick_exit(0);
        }

        // Pause detection
        static bool pausePressed = false;
        if (isAnyKeyPressed(config.button_pause))
        {
            if (!pausePressed)
            {
                detectionPaused = !detectionPaused;
                pausePressed = true;
            }
        }
        else
        {
            pausePressed = false;
        }

        // Reload config
        static bool reloadPressed = false;
        if (isAnyKeyPressed(config.button_reload_config))
        {
            if (!reloadPressed)
            {
                config.loadConfig();
                
                if (globalMouseThread)
                {
                    globalMouseThread->updateConfig(
                        config.detection_resolution,
                        config.kp_x,
                        config.ki_x,
                        config.kd_x,
                        config.kp_y,
                        config.ki_y,
                        config.kd_y,
                        config.bScope_multiplier,
                        config.norecoil_ms
                    );
                }
                reloadPressed = true;
            }
        }
        else
        {
            reloadPressed = false;
        }

        // --- Auto Shoot Activation (Hold Mode Only) --- 
        bool auto_shoot_key_pressed = isAnyKeyPressed(config.button_auto_shoot);
        auto_shoot_active.store(auto_shoot_key_pressed); // Set active state based on key press
        // Removed toggle logic and prev state tracking
        
        // --- End Auto Shoot Activation ---

        // Arrow key detection logic using isAnyKeyPressed
        bool upArrow = isAnyKeyPressed(upArrowKeys);
        bool downArrow = isAnyKeyPressed(downArrowKeys);
        bool leftArrow = isAnyKeyPressed(leftArrowKeys);
        bool rightArrow = isAnyKeyPressed(rightArrowKeys);
        bool shiftKey = isAnyKeyPressed(shiftKeys);

        // Adjust offsets based on arrow keys and shift combination
        if (upArrow && !prevUpArrow)
        {
            if (shiftKey)
            {
                // Shift + Up Arrow: Decrease head offset
                config.head_y_offset = std::max(MIN_OFFSET_Y, config.head_y_offset - config.offset_step);
            }
            else
            {
                // Up Arrow: Decrease body offset
                config.body_y_offset = std::max(MIN_OFFSET_Y, config.body_y_offset - config.offset_step);
            }
        }
        if (downArrow && !prevDownArrow)
        {
            if (shiftKey)
            {
                // Shift + Down Arrow: Increase head offset
                config.head_y_offset = std::min(MAX_OFFSET_Y, config.head_y_offset + config.offset_step);
            }
            else
            {
                // Down Arrow: Increase body offset
                config.body_y_offset = std::min(MAX_OFFSET_Y, config.body_y_offset + config.offset_step);
            }
        }


        // Adjust norecoil strength based on left and right arrow keys
        if (leftArrow && !prevLeftArrow)
        {
            config.easynorecoilstrength = std::max(MIN_NORECOIL_STRENGTH, config.easynorecoilstrength - config.norecoil_step);
        }

        if (rightArrow && !prevRightArrow)
        {
            config.easynorecoilstrength = std::min(MAX_NORECOIL_STRENGTH, config.easynorecoilstrength + config.norecoil_step);
        }
        
        // Update previous key states
        prevUpArrow = upArrow;
        prevDownArrow = downArrow;
        prevLeftArrow = leftArrow;
        prevRightArrow = rightArrow;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}