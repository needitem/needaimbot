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
#include "../../config/config.h"
#include "../../needaimbot.h"
#include "../../AppContext.h"

/* ---------- Makcu-specific constants ---------------------------- */
static const uint32_t BOOT_BAUD = 115200;      // baud rate after initial connection
static const uint32_t WORK_BAUD = 4000000;     // working baud rate – 4 Mbit/s

/* Baud change command (from a.py): 0xDEAD0500A500093D00 */
static const uint8_t BAUD_CHANGE_CMD[9] =
{ 0xDE,0xAD,0x05,0x00,0xA5,0x00,0x09,0x3D,0x00 };

MakcuConnection::MakcuConnection(const std::string& port, unsigned int /*baud_rate*/)
    : serial_handle_(INVALID_HANDLE_VALUE),
      is_open_(false), 
      listening_(false),
      aiming_active(false), 
      shooting_active(false), 
      zooming_active(false)
{
    std::string full_port = "\\\\.\\" + port;
    
    // 1단계: 115200 baud로 초기 연결
    serial_handle_ = CreateFileA(
        full_port.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[Makcu] Unable to open port: " << port 
                  << " (Error: " << GetLastError() << ")" << std::endl;
        return;
    }

    // DCB 설정 - 115200 baud
    ZeroMemory(&dcb_config_, sizeof(DCB));
    dcb_config_.DCBlength = sizeof(DCB);
    
    if (!GetCommState(serial_handle_, &dcb_config_)) {
        std::cerr << "[Makcu] Failed to get comm state" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

    dcb_config_.BaudRate = BOOT_BAUD;
    dcb_config_.ByteSize = 8;
    dcb_config_.Parity = NOPARITY;
    dcb_config_.StopBits = ONESTOPBIT;
    dcb_config_.fBinary = TRUE;
    dcb_config_.fParity = FALSE;
    dcb_config_.fOutxCtsFlow = FALSE;
    dcb_config_.fOutxDsrFlow = FALSE;
    dcb_config_.fDtrControl = DTR_CONTROL_DISABLE;
    dcb_config_.fDsrSensitivity = FALSE;
    dcb_config_.fTXContinueOnXoff = FALSE;
    dcb_config_.fOutX = FALSE;
    dcb_config_.fInX = FALSE;
    dcb_config_.fErrorChar = FALSE;
    dcb_config_.fNull = FALSE;
    dcb_config_.fRtsControl = RTS_CONTROL_DISABLE;
    dcb_config_.fAbortOnError = FALSE;

    if (!SetCommState(serial_handle_, &dcb_config_)) {
        std::cerr << "[Makcu] Failed to set initial comm state" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

    // 2단계: baud rate 변경 명령 전송
    DWORD bytes_written = 0;
    if (!WriteFile(serial_handle_, BAUD_CHANGE_CMD, sizeof(BAUD_CHANGE_CMD), &bytes_written, NULL)) {
        std::cerr << "[Makcu] Failed to send baud change command" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

    // 포트 닫고 MCU 리셋 대기
    CloseHandle(serial_handle_);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 3단계: 4Mbps로 재연결
    serial_handle_ = CreateFileA(
        full_port.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[Makcu] Unable to reopen port at high speed" << std::endl;
        return;
    }

    // 4Mbps DCB 설정
    ZeroMemory(&dcb_config_, sizeof(DCB));
    dcb_config_.DCBlength = sizeof(DCB);
    GetCommState(serial_handle_, &dcb_config_);

    dcb_config_.BaudRate = WORK_BAUD;  // 4 Mbps
    dcb_config_.ByteSize = 8;
    dcb_config_.Parity = NOPARITY;
    dcb_config_.StopBits = ONESTOPBIT;
    dcb_config_.fBinary = TRUE;
    dcb_config_.fParity = FALSE;
    dcb_config_.fOutxCtsFlow = FALSE;
    dcb_config_.fOutxDsrFlow = FALSE;
    dcb_config_.fDtrControl = DTR_CONTROL_DISABLE;
    dcb_config_.fDsrSensitivity = FALSE;
    dcb_config_.fTXContinueOnXoff = FALSE;
    dcb_config_.fOutX = FALSE;
    dcb_config_.fInX = FALSE;
    dcb_config_.fErrorChar = FALSE;
    dcb_config_.fNull = FALSE;
    dcb_config_.fRtsControl = RTS_CONTROL_DISABLE;
    dcb_config_.fAbortOnError = FALSE;

    if (!SetCommState(serial_handle_, &dcb_config_)) {
        std::cerr << "[Makcu] Failed to set high-speed comm state" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

    // 초저지연 타임아웃 설정
    timeouts_.ReadIntervalTimeout = 1;
    timeouts_.ReadTotalTimeoutMultiplier = 0;
    timeouts_.ReadTotalTimeoutConstant = 1;
    timeouts_.WriteTotalTimeoutMultiplier = 0;
    timeouts_.WriteTotalTimeoutConstant = 1;

    if (!SetCommTimeouts(serial_handle_, &timeouts_)) {
        std::cerr << "[Makcu] Failed to set timeouts" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

    // 버퍼 최적화
    SetupComm(serial_handle_, 64, 64);
    PurgeComm(serial_handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);

    is_open_ = true;
    std::cout << "[Makcu] Connected at 4Mbps! PORT: " << port 
              << " (Native Windows API - Ultra Low Latency)" << std::endl;

    startListening();
}

MakcuConnection::~MakcuConnection()
{
    listening_ = false;
    
    if (listening_thread_.joinable()) {
        listening_thread_.join();
    }
    
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
    
    is_open_ = false;
}

bool MakcuConnection::isOpen() const
{
    return is_open_;
}

void MakcuConnection::write(const std::string& data)
{
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE)
        return;

    std::lock_guard<std::mutex> lock(write_mutex_);

    DWORD bytes_written = 0;
    OVERLAPPED overlapped = {0};
    overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    
    if (overlapped.hEvent == NULL) return;

    BOOL result = WriteFile(
        serial_handle_,
        data.c_str(),
        static_cast<DWORD>(data.length()),
        &bytes_written,
        &overlapped
    );

    if (!result && GetLastError() == ERROR_IO_PENDING) {
        WaitForSingleObject(overlapped.hEvent, 1);
        GetOverlappedResult(serial_handle_, &overlapped, &bytes_written, FALSE);
    }

    CloseHandle(overlapped.hEvent);
    FlushFileBuffers(serial_handle_);
}

std::string MakcuConnection::read()
{
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE)
        return "";

    char buffer[256];
    DWORD bytes_read = 0;
    OVERLAPPED overlapped = {0};
    overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    
    if (overlapped.hEvent == NULL) return "";

    BOOL result = ReadFile(
        serial_handle_,
        buffer,
        sizeof(buffer) - 1,
        &bytes_read,
        &overlapped
    );

    if (!result && GetLastError() == ERROR_IO_PENDING) {
        WaitForSingleObject(overlapped.hEvent, 1);
        GetOverlappedResult(serial_handle_, &overlapped, &bytes_read, FALSE);
    }

    CloseHandle(overlapped.hEvent);

    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        return std::string(buffer);
    }
    
    return "";
}

void MakcuConnection::click(int button)
{
    if (button == 1) {
        sendCommand("LC\n");  // Left Click
    } else if (button == 2) {
        sendCommand("RC\n");  // Right Click
    }
}

void MakcuConnection::press(int button)
{
    if (button == 1) {
        sendCommand("LP\n");  // Left Press
    } else if (button == 2) {
        sendCommand("RP\n");  // Right Press
    }
}

void MakcuConnection::release(int button)
{
    if (button == 1) {
        sendCommand("LR\n");  // Left Release
    } else if (button == 2) {
        sendCommand("RR\n");  // Right Release
    }
}

void MakcuConnection::move(int x, int y)
{
    if (x == 0 && y == 0) return;
    
    // Makcu 최적화된 이동 명령
    char command[32];
    snprintf(command, sizeof(command), "M%d,%d\n", x, y);
    sendCommand(std::string(command));
}

void MakcuConnection::start_boot()
{
    sendCommand("BOOT\n");
}

void MakcuConnection::reboot()
{
    sendCommand("REBOOT\n");
}

void MakcuConnection::send_stop()
{
    sendCommand("STOP\n");
}

void MakcuConnection::sendCommand(const std::string& command)
{
    write(command);
}

std::vector<int> MakcuConnection::splitValue(int value)
{
    std::vector<int> result;
    
    if (value == 0) {
        result.push_back(0);
        return result;
    }

    bool negative = value < 0;
    if (negative) value = -value;

    while (value > 127) {
        result.push_back(127);
        value -= 127;
    }
    
    if (value > 0) {
        result.push_back(value);
    }

    if (negative) {
        for (auto& v : result) {
            v = -v;
        }
    }

    return result;
}

void MakcuConnection::startListening()
{
    listening_ = true;
    listening_thread_ = std::thread(&MakcuConnection::listeningThreadFunc, this);
}

void MakcuConnection::listeningThreadFunc()
{
    std::string buffer;
    
    while (listening_) {
        if (!is_open_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::string data = read();
        if (!data.empty()) {
            buffer += data;
            
            size_t pos = 0;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                std::string line = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);
                
                if (!line.empty()) {
                    processIncomingLine(line);
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50)); // 50μs 대기
        }
    }
}

void MakcuConnection::processIncomingLine(const std::string& line)
{
    if (line.find("AIM") == 0) {
        if (line.find("ON") != std::string::npos) {
            aiming_active = true;
        } else if (line.find("OFF") != std::string::npos) {
            aiming_active = false;
        }
    }
    else if (line.find("SHOOT") == 0) {
        if (line.find("ON") != std::string::npos) {
            shooting_active = true;
        } else if (line.find("OFF") != std::string::npos) {
            shooting_active = false;
        }
    }
    else if (line.find("ZOOM") == 0) {
        if (line.find("ON") != std::string::npos) {
            zooming_active = true;
        } else if (line.find("OFF") != std::string::npos) {
            zooming_active = false;
        }
    }
}