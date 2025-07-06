#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <condition_variable>

#include "needaimbot.h"
#include "SerialConnection.h"
#include "../../AppContext.h"

SerialConnection::SerialConnection(const std::string& port, unsigned int baud_rate)
    : is_open_(false),
    listening_(false)
{
    try
    {
        serial_.setPort(port);
        serial_.setBaudrate(baud_rate);
        serial_.open();

        if (serial_.isOpen())
        {
            is_open_ = true;
            std::cout << "[Arduino] Connected! PORT: " << port << std::endl;

            if (AppContext::getInstance().config.arduino_enable_keys)
            {
                startListening();
            }
        }
        else
        {
            std::cerr << "[Arduino] Unable to connect to the port: " << port << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "[Arduino] Error: " << e.what() << std::endl;
    }
}

SerialConnection::~SerialConnection()
{
    listening_ = false;
    if (serial_.isOpen())
    {
        try { serial_.close(); }
        catch (...) {}
    }
    if (listening_thread_.joinable())
    {
        listening_thread_.join();
    }
    is_open_ = false;
}

bool SerialConnection::isOpen() const
{
    return is_open_;
}

void SerialConnection::write(const std::string& data)
{
    if (!is_open_)
        return;

    static thread_local std::string buffer;
    buffer.reserve(64);
    
    try {
        buffer = data;
        size_t bytes_written = serial_.write(buffer);
        if (bytes_written != buffer.length()) {
            std::cerr << "[Arduino] Warning: Serial write might be incomplete. Expected " << buffer.length() << ", wrote " << bytes_written << std::endl;
        }
        buffer.clear();
    } catch (const std::exception& e) {
        std::cerr << "[Arduino] Error during serial write: " << e.what() << std::endl;
        is_open_ = false;
    }
}

std::string SerialConnection::read()
{
    if (!is_open_)
        return std::string();

    std::string result;
    try
    {
        result = serial_.readline(65536, "\n");
    }
    catch (...)
    {
        is_open_ = false;
    }
    return result;
}

void SerialConnection::click()
{
    sendCommand("c");
}

void SerialConnection::press()
{
    sendCommand("p");
}

void SerialConnection::release()
{
    sendCommand("r");
}

void SerialConnection::move(int x, int y)
{
    if (!is_open_)
        return;

    if (AppContext::getInstance().config.arduino_16_bit_mouse)
    {
        
        char buffer[32]; 
        int len = snprintf(buffer, sizeof(buffer), "m%d,%d\n", x, y);
        
        if (len > 0 && len < sizeof(buffer)) {
            
            
            write(std::string(buffer, len)); 
        }
    }
    else
    {
        char buffer[32]; 
        
        int current_x = x;
        int current_y = y;
        int sign_x = (x > 0) ? 1 : ((x < 0) ? -1 : 0);
        int sign_y = (y > 0) ? 1 : ((y < 0) ? -1 : 0);
        int abs_x = std::abs(x);
        int abs_y = std::abs(y);
        
        while (abs_x > 0 || abs_y > 0)
        {
            int move_part_x = 0;
            int move_part_y = 0;

            if (abs_x > 0) {
                move_part_x = std::min(abs_x, 127) * sign_x;
                abs_x -= std::min(abs_x, 127);
            }
            if (abs_y > 0) {
                move_part_y = std::min(abs_y, 127) * sign_y;
                abs_y -= std::min(abs_y, 127);
            }
            
            
            int len = snprintf(buffer, sizeof(buffer), "m%d,%d\n", move_part_x, move_part_y);
            
            if (len > 0 && len < sizeof(buffer)) {
                write(std::string(buffer, len)); 
            }
        }
    }
}

void SerialConnection::sendCommand(const std::string& command)
{
    write(command + "\n");
}

std::vector<int> SerialConnection::splitValue(int value)
{
    std::vector<int> values;
    int sign = (value < 0) ? -1 : 1;
    int absVal = (value < 0) ? -value : value;

    if (value == 0)
    {
        values.push_back(0);
        return values;
    }

    while (absVal > 127)
    {
        values.push_back(sign * 127);
        absVal -= 127;
    }

    if (absVal != 0)
    {
        values.push_back(sign * absVal);
    }

    return values;
}

void SerialConnection::timerThreadFunc()
{
    while (timer_running_)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!is_open_)
            continue;

        bool arduino_enable_keys_local;
        {
            arduino_enable_keys_local = AppContext::getInstance().config.arduino_enable_keys;
        }

        if (arduino_enable_keys_local)
        {
            if (!listening_)
            {
                startListening();
            }
        }
        else
        {
            if (listening_)
            {
                listening_ = false;
                if (listening_thread_.joinable())
                {
                    listening_thread_.join();
                }
            }
        }
    }
}

void SerialConnection::startListening()
{
    listening_ = true;
    if (listening_thread_.joinable())
        listening_thread_.join();

    listening_thread_ = std::thread(&SerialConnection::listeningThreadFunc, this);
}

void SerialConnection::listeningThreadFunc()
{
    std::string buffer;
    while (listening_ && is_open_)
    {
        try
        {
            size_t available = serial_.available();
            if (available > 0)
            {
                std::string data = serial_.read(available);
                buffer += data;
                size_t pos;
                while ((pos = buffer.find('\n')) != std::string::npos)
                {
                    std::string line = buffer.substr(0, pos);
                    buffer.erase(0, pos + 1);
                    if (!line.empty() && line.back() == '\r')
                        line.pop_back();
                    processIncomingLine(line);
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        catch (...)
        {
            is_open_ = false;
            break;
        }
    }
}

void SerialConnection::processIncomingLine(const std::string& line)
{
    try
    {
        if (line.rfind("BD:", 0) == 0)
        {
            uint16_t buttonId = static_cast<uint16_t>(std::stoi(line.substr(3)));
            switch (buttonId)
            {
            case 2:
                // aiming_active = true;
                // aiming.store(true); // Aiming state removed
                break;
            case 1:
                shooting_active = true;
                AppContext::getInstance().shooting.store(true);
                break;
            }
        }
        else if (line.rfind("BU:", 0) == 0)
        {
            uint16_t buttonId = static_cast<uint16_t>(std::stoi(line.substr(3)));
            switch (buttonId)
            {
            case 2:
                // aiming_active = false;
                // aiming.store(false); // Aiming state removed
                break;
            case 1:
                shooting_active = false;
                AppContext::getInstance().shooting.store(false);
                break;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Arduino] Error processing line '" << line << "': " << e.what() << std::endl;
    }
}
