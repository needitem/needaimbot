// MakcuConnection.h
#ifndef MAKCUCONNECTION_H
#define MAKCUCONNECTION_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

#include "serial/serial.h"

class MakcuConnection
{
public:
    MakcuConnection(const std::string& port, unsigned int baud_rate);
    ~MakcuConnection();

    bool isOpen() const;

    void write(const std::string& data);
    std::string read();

    void click(int button);
    void press(int button);
    void release(int button);
    void move(int x, int y);

    void start_boot();
    void reboot();
    void send_stop();

    bool aiming_active;
    bool shooting_active;
    bool zooming_active;

private:
    void sendCommand(const std::string& command);
    std::vector<int> splitValue(int value);

    void startListening();
    void listeningThreadFunc();
    void processIncomingLine(const std::string& line);

private:
    serial::Serial serial_;
    std::atomic<bool> is_open_;
    std::atomic<bool> listening_;
    std::thread       listening_thread_;
    std::mutex        write_mutex_;
};

#endif // MAKCUCONNECTION_H


// MakcuConnection.cpp
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <thread>
#include <mutex>

#include "MakcuConnection.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"

/* ---------- Makcu-specific constants ---------------------------- */
static const uint32_t BOOT_BAUD = 115200;      // baud rate after initial connection
static const uint32_t WORK_BAUD = 4000000;     // working baud rate – 4 Mbit/s

/* Baud change command (from a.py): 0xDEAD0500A500093D00 */
static const uint8_t  BAUD_CHANGE_CMD[9] =
{ 0xDE,0xAD,0x05,0x00,0xA5,0x00,0x09,0x3D,0x00 };

/* ===================================================================== */
/*                          CONSTRUCTOR                                  */
MakcuConnection::MakcuConnection(const std::string& port, unsigned int /*baud_rate*/)
    : is_open_(false), listening_(false),
      aiming_active(false), shooting_active(false), zooming_active(false)
{
    try {
        /* 1. open port at 115200 */
        serial_.setPort(port);
        serial_.setBaudrate(BOOT_BAUD);
        serial_.open();
        if (!serial_.isOpen())
            throw std::runtime_error("open failed");

        /* 2. send "secret" packet -> MCU reboots at 4 Mbit */
        serial_.write(BAUD_CHANGE_CMD, sizeof(BAUD_CHANGE_CMD));
        serial_.close();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));   // MCU reset delay

        /* 3. reopen at 4 Mbit/s */
        serial_.setBaudrate(WORK_BAUD);
        serial_.open();
        if (!serial_.isOpen())
            throw std::runtime_error("re-open @4M failed");

        is_open_ = true;
        std::cout << "[Makcu] Connected @4 Mbps on " << port << '\n';

        /* 4. disable echo and enable button stream */
        sendCommand("km.echo(0)");
        sendCommand("km.buttons(1)");

        startListening();
    }
    catch (const std::exception& e) {
        std::cerr << "[Makcu] Error: " << e.what() << '\n';
    }
}

/* ===================================================================== */
/*                               DESTRUCTOR                               */
MakcuConnection::~MakcuConnection()
{
    listening_ = false;
    if (serial_.isOpen()) {
        try { serial_.close(); }
        catch (...) {}
    }
    if (listening_thread_.joinable())
        listening_thread_.join();
    is_open_ = false;
}

/* ===================================================================== */
/*                         HELPER METHODS                                 */

bool MakcuConnection::isOpen() const { return is_open_; }

void MakcuConnection::write(const std::string& data)
{
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (!is_open_) return;
    try { serial_.write(data); }
    catch (...) { is_open_ = false; }
}

std::string MakcuConnection::read()
{
    if (!is_open_)
        return std::string();

    std::string result;
    try
    {
        result = serial_.readline(65536, "\n");
        std::cout << result << std::endl;
    }
    catch (...)
    {
        is_open_ = false;
    }
    return result;
}

void MakcuConnection::move(int x, int y)
{
    if (!is_open_)
        return;

    std::string cmd = "km.move(" + std::to_string(x) + "," + std::to_string(y) + ")\r\n";
    write(cmd);
}

void MakcuConnection::click(int button /*= 0*/)
{
    std::string cmd = "km.click(" + std::to_string(button) + ")\r\n";
    sendCommand(cmd);
}

void MakcuConnection::press(int button)
{
    sendCommand("km.left_down()");
}

void MakcuConnection::release(int button)
{
    sendCommand("km.left_up()");
}

void MakcuConnection::start_boot()
{
    write("\x03\x03");  // send interrupt
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    write("exec(open('boot.py').read(),globals())\r\n");
}

void MakcuConnection::reboot()
{
    write("\x03\x03");  // send interrupt
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    write("km.reboot()");
}

void MakcuConnection::send_stop()
{
    write("\x03\x03");  // send interrupt
}

void MakcuConnection::sendCommand(const std::string& cmd) { write(cmd + "\r\n"); }

std::vector<int> MakcuConnection::splitValue(int value)
{
    std::vector<int> values;
    return values;
}

/* ===================================================================== */
/*                      START LISTENER THREAD                             */
void MakcuConnection::startListening()
{
    listening_ = true;
    if (listening_thread_.joinable())
        listening_thread_.join();

    listening_thread_ = std::thread(&MakcuConnection::listeningThreadFunc, this);
}

/* ===================================================================== */
/*                    MAIN LOOP FOR READING FROM MCU                     */
void MakcuConnection::listeningThreadFunc()
{
    /* valid bytes: 0x00 (none), 0x01 (LMB), 0x02 (RMB), 0x03 (both) */
    const std::array<uint8_t, 4> legal{ 0x00, 0x01, 0x02, 0x03 };

    while (listening_ && is_open_) {
        try {
            if (!serial_.available()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            uint8_t b = 0;
            serial_.read(&b, 1);

            if (std::find(legal.begin(), legal.end(), b) == legal.end())
                continue;  // discard noise

            /* update flags */
            shooting_active = b & 0x01;  // left mouse button
            aiming_active = b & 0x02;    // right mouse button
            shooting.store(shooting_active);
            aiming.store(aiming_active);

            /* print state */
            std::cout << "LMB: " << (shooting_active ? "PRESS" : "release")
                      << " | RMB: " << (aiming_active ? "PRESS" : "release")
                      << std::endl;
        }
        catch (...) { is_open_ = false; break; }
    }
}

void MakcuConnection::processIncomingLine(const std::string& line)
{
    try
    {
        if (line.rfind("BD:", 0) == 0)
        {
            int btnId = std::stoi(line.substr(3));
            switch (btnId)
            {
            case 1:
                shooting_active = true;
                shooting.store(true);
                break;
            case 2:
                aiming_active = true;
                aiming.store(true);
                break;
            }
        }
        else if (line.rfind("BU:", 0) == 0)
        {
            int btnId = std::stoi(line.substr(3));
            switch (btnId)
            {
            case 1:
                shooting_active = false;
                shooting.store(false);
                break;
            case 2:
                aiming_active = false;
                aiming.store(false);
                break;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Makcu_b] Error processing line '" << line << "': " << e.what() << std::endl;
    }
}
