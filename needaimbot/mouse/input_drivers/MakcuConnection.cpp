#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <algorithm>
#include <vector>
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
      zooming_active(false),
      port_name_(port),
      write_event_(NULL),
      read_event_(NULL)
{
    // Initialize overlapped structures
    ZeroMemory(&write_overlapped_, sizeof(write_overlapped_));
    ZeroMemory(&read_overlapped_, sizeof(read_overlapped_));
    
    // Create events for async operations
    write_event_ = CreateEvent(NULL, TRUE, FALSE, NULL);
    read_event_ = CreateEvent(NULL, TRUE, FALSE, NULL);
    
    if (write_event_ && read_event_) {
        write_overlapped_.hEvent = write_event_;
        read_overlapped_.hEvent = read_event_;
    }
    
    try {
        if (!initializeMakcuConnection()) {
            throw std::runtime_error("Failed to initialize Makcu connection to " + port);
        }
        
        std::cout << "[Makcu] Connected at 4Mbps! PORT: " << port 
                  << " (Native Windows API - Async I/O)" << std::endl;
                  
    } catch (const std::exception& e) {
        cleanup();
        if (write_event_) CloseHandle(write_event_);
        if (read_event_) CloseHandle(read_event_);
        std::cerr << "[Makcu] Initialization error: " << e.what() << std::endl;
        // 객체는 생성되지만 is_open_은 false로 유지됨
    }
}

bool MakcuConnection::initializeMakcuConnection() {
    std::string full_port = "\\\\.\\" + port_name_;
    
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
        std::cerr << "[Makcu] Unable to open port: " << port_name_ 
                  << " (Error: " << GetLastError() << ")" << std::endl;
        return false;
    }

    // DCB 설정 - 115200 baud
    if (!configureDCB(BOOT_BAUD)) {
        closeHandle();
        return false;
    }

    // 2단계: baud rate 변경 명령 전송 (async I/O)
    if (!writeAsync(BAUD_CHANGE_CMD, sizeof(BAUD_CHANGE_CMD))) {
        std::cerr << "[Makcu] Failed to send baud change command" << std::endl;
        closeHandle();
        return false;
    }

    // 포트 닫고 MCU 리셋 대기 (event-based wait)
    CloseHandle(serial_handle_);
    serial_handle_ = INVALID_HANDLE_VALUE;
    
    // Use event-based wait instead of sleep
    if (write_event_) {
        WaitForSingleObject(write_event_, 100);
    }

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
        return false;
    }

    // 4Mbps DCB 설정
    if (!configureDCB(WORK_BAUD)) {
        closeHandle();
        return false;
    }

    // 타임아웃 설정
    if (!configureTimeouts()) {
        closeHandle();
        return false;
    }

    // 버퍼 최적화 - 더 큰 버퍼로 Makcu 통신 효율성 개선
    constexpr DWORD BUFFER_SIZE = 256;  // Optimized for Makcu's high-speed communication
    if (!SetupComm(serial_handle_, BUFFER_SIZE, BUFFER_SIZE)) {
        std::cerr << "[Makcu] Warning: Failed to setup comm buffers" << std::endl;
    }
    PurgeComm(serial_handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);

    // 리스닝 스레드 시작
    try {
        startListening();
        is_open_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Makcu] Failed to start listening thread: " << e.what() << std::endl;
        closeHandle();
        return false;
    }
}

bool MakcuConnection::configureDCB(uint32_t baud_rate) {
    ZeroMemory(&dcb_config_, sizeof(DCB));
    dcb_config_.DCBlength = sizeof(DCB);
    
    if (!GetCommState(serial_handle_, &dcb_config_)) {
        std::cerr << "[Makcu] Failed to get comm state" << std::endl;
        return false;
    }

    dcb_config_.BaudRate = baud_rate;
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
        std::cerr << "[Makcu] Failed to set comm state for baud rate: " << baud_rate << std::endl;
        return false;
    }
    
    return true;
}

bool MakcuConnection::configureTimeouts() {
    // Optimized timeouts for 4Mbps communication
    timeouts_.ReadIntervalTimeout = 1;
    timeouts_.ReadTotalTimeoutMultiplier = 0;
    timeouts_.ReadTotalTimeoutConstant = 5;  // Slightly higher for 4Mbps stability
    timeouts_.WriteTotalTimeoutMultiplier = 0;
    timeouts_.WriteTotalTimeoutConstant = 10;  // Adequate for high-speed writes

    if (!SetCommTimeouts(serial_handle_, &timeouts_)) {
        std::cerr << "[Makcu] Failed to set timeouts" << std::endl;
        return false;
    }
    
    return true;
}

void MakcuConnection::safeMakcuClose() {
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) {
        return;
    }
    
    std::cout << "[Makcu] Starting safe Makcu port closure (wjwwood/serial method)..." << std::endl;
    
    try {
        // 1. 릴리스 명령 전송 (async I/O)
        const char* left_release = "LR\n";
        const char* right_release = "RR\n";
        const char* neutral_pos = "M0,0\n";
        const char* stop_cmd = "STOP\n";
        
        writeAsync(left_release, static_cast<DWORD>(strlen(left_release)));
        writeAsync(right_release, static_cast<DWORD>(strlen(right_release)));
        writeAsync(neutral_pos, static_cast<DWORD>(strlen(neutral_pos)));
        
        // Wait for commands to complete
        if (write_event_) {
            WaitForSingleObject(write_event_, 10);
        }
        
        // 2. 출력 버퍼 플러시 (전송 완료 보장)
        if (!FlushFileBuffers(serial_handle_)) {
            std::cerr << "[Makcu] Warning: FlushFileBuffers failed" << std::endl;
        }
        
        // 3. 입력 버퍼 정리
        if (!PurgeComm(serial_handle_, PURGE_RXCLEAR)) {
            std::cerr << "[Makcu] Warning: PURGE_RXCLEAR failed" << std::endl;
        }
        
        // 4. 출력 버퍼 정리  
        if (!PurgeComm(serial_handle_, PURGE_TXCLEAR)) {
            std::cerr << "[Makcu] Warning: PURGE_TXCLEAR failed" << std::endl;
        }
        
        // 5. Makcu 정지 명령 (async)
        writeAsync(stop_cmd, static_cast<DWORD>(strlen(stop_cmd)));
        
        // 6. 하드웨어 안정화 대기 (event-based)
        if (write_event_) {
            WaitForSingleObject(write_event_, 50);
        }
        
        // 7. 포트 안전하게 닫기 (wjwwood/serial 방식)
        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            int ret = CloseHandle(serial_handle_);
            if (ret == 0) {
                DWORD error = GetLastError();
                std::cerr << "[Makcu] Error closing Makcu port: " << error << std::endl;
            } else {
                serial_handle_ = INVALID_HANDLE_VALUE;
                std::cout << "[Makcu] Makcu port closed successfully." << std::endl;
            }
        }
        
        is_open_ = false;
        listening_ = false;
        
    } catch (const std::exception& e) {
        std::cerr << "[Makcu] Error during safe Makcu close: " << e.what() << std::endl;
    }
}

void MakcuConnection::cleanup() {
    listening_ = false;
    
    // wjwwood/serial 방식의 안전한 포트 종료
    safeMakcuClose();
    
    if (listening_thread_.joinable()) {
        try {
            listening_thread_.join();
        } catch (const std::exception& e) {
            std::cerr << "[Makcu] Error joining listening thread: " << e.what() << std::endl;
        }
    }
}


void MakcuConnection::closeHandle() {
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
    is_open_ = false;
}

MakcuConnection::~MakcuConnection()
{
    cleanup();
    
    // Clean up event handles
    if (write_event_) {
        CloseHandle(write_event_);
        write_event_ = NULL;
    }
    if (read_event_) {
        CloseHandle(read_event_);
        read_event_ = NULL;
    }
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
    
    // Use the async write helper function
    if (!writeAsync(data.c_str(), static_cast<DWORD>(data.length()))) {
        std::cerr << "[Makcu] Write operation failed" << std::endl;
    }
}

std::string MakcuConnection::read()
{
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE)
        return "";

    char buffer[256];
    DWORD bytes_read = 0;
    
    // Use async read helper function
    if (readAsync(buffer, sizeof(buffer) - 1, &bytes_read) && bytes_read > 0) {
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
            std::this_thread::sleep_for(std::chrono::milliseconds(2));  // Reduced from 10ms for better responsiveness
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
            std::this_thread::sleep_for(std::chrono::microseconds(25)); // Reduced from 50μs for lower latency
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

// Async I/O helper functions
bool MakcuConnection::writeAsync(const void* data, DWORD size)
{
    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    // Reset event
    ResetEvent(write_overlapped_.hEvent);
    
    DWORD bytes_written = 0;
    BOOL result = WriteFile(
        serial_handle_,
        data,
        size,
        &bytes_written,
        &write_overlapped_
    );
    
    if (!result) {
        DWORD error = GetLastError();
        if (error == ERROR_IO_PENDING) {
            // Wait for async operation to complete
            return waitForAsyncOperation(&write_overlapped_, 50);
        }
        return false;
    }
    
    return true;
}

bool MakcuConnection::readAsync(void* buffer, DWORD size, DWORD* bytesRead)
{
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    // Reset event
    ResetEvent(read_overlapped_.hEvent);
    
    BOOL result = ReadFile(
        serial_handle_,
        buffer,
        size,
        bytesRead,
        &read_overlapped_
    );
    
    if (!result) {
        DWORD error = GetLastError();
        if (error == ERROR_IO_PENDING) {
            // Wait for async operation to complete
            if (waitForAsyncOperation(&read_overlapped_, 50)) {
                GetOverlappedResult(serial_handle_, &read_overlapped_, bytesRead, FALSE);
                return true;
            }
            return false;
        }
        return false;
    }
    
    return true;
}

bool MakcuConnection::waitForAsyncOperation(OVERLAPPED* overlapped, DWORD timeout_ms)
{
    DWORD result = WaitForSingleObject(overlapped->hEvent, timeout_ms);
    if (result == WAIT_OBJECT_0) {
        DWORD bytes_transferred;
        return GetOverlappedResult(serial_handle_, overlapped, &bytes_transferred, FALSE) != 0;
    }
    
    if (result == WAIT_TIMEOUT) {
        // Cancel pending I/O if timeout
        CancelIo(serial_handle_);
    }
    
    return false;
}
