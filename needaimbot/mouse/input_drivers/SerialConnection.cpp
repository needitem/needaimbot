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
    : serial_handle_(INVALID_HANDLE_VALUE),
      is_open_(false),
      port_name_(port),
      baud_rate_(baud_rate),
      timer_running_(false),
      listening_(false),
      aiming_active(false),
      shooting_active(false), 
      zooming_active(false)
{
    // 초기 연결 시도
    if (openPort() && configurePort()) {
        std::cout << "[Arduino] Connected! PORT: " << port_name_ 
                  << " (Native Windows API - Ultra Low Latency)" << std::endl;
        
        if (AppContext::getInstance().config.arduino_enable_keys) {
            startListening();
        }
        startTimer();
    } else {
        std::cerr << "[Arduino] Failed to establish initial connection to " << port_name_ << std::endl;
    }
}

SerialConnection::~SerialConnection()
{
    close();
}

bool SerialConnection::isOpen() const
{
    return is_open_;
}

bool SerialConnection::isHealthy() const
{
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    // 통신 오류 상태 확인
    DWORD errors;
    COMSTAT stat;
    if (!ClearCommError(serial_handle_, &errors, &stat)) {
        return false;
    }
    
    // 심각한 오류가 있는지 확인
    if (errors & (CE_FRAME | CE_OVERRUN | CE_RXPARITY | CE_BREAK)) {
        std::cerr << "[Arduino] Communication errors detected: " << errors << std::endl;
        return false;
    }
    
    return true;
}

bool SerialConnection::reconnect()
{
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    
    // 기존 연결 정리
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
    is_open_ = false;
    
    // 잠시 대기 (Arduino 재시작 시간 확보)
    Sleep(100);
    
    // 재연결 시도
    if (openPort() && configurePort()) {
        return true;
    } else {
        return false;
    }
}

void SerialConnection::close()
{
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    timer_running_ = false;
    listening_ = false;
    
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }
    
    if (listening_thread_.joinable()) {
        listening_thread_.join();
    }
    
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
    
    is_open_ = false;
}

bool SerialConnection::openPort()
{
    std::string full_port = "\\\\.\\" + port_name_;
    
    serial_handle_ = CreateFileA(
        full_port.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,                          // 배타적 액세스
        NULL,                       // 보안 속성 없음
        OPEN_EXISTING,             // 존재하는 포트 열기
        FILE_ATTRIBUTE_NORMAL,     // 동기 I/O로 변경
        NULL
    );

    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        DWORD error = GetLastError();
        std::cerr << "[Arduino] Unable to open port: " << port_name_ 
                  << " (Error: " << error << ")" << std::endl;
        return false;
    }

    return true;
}

bool SerialConnection::configurePort()
{
    // DCB 구조체 초기화 및 설정
    ZeroMemory(&dcb_config_, sizeof(DCB));
    dcb_config_.DCBlength = sizeof(DCB);
    
    if (!GetCommState(serial_handle_, &dcb_config_)) {
        std::cerr << "[Arduino] Failed to get comm state" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // 초고속 통신을 위한 최적 설정
    dcb_config_.BaudRate = baud_rate_;
    dcb_config_.ByteSize = 8;
    dcb_config_.Parity = NOPARITY;
    dcb_config_.StopBits = ONESTOPBIT;
    dcb_config_.fBinary = TRUE;
    dcb_config_.fParity = FALSE;
    dcb_config_.fOutxCtsFlow = FALSE;
    dcb_config_.fOutxDsrFlow = FALSE;
    dcb_config_.fDtrControl = DTR_CONTROL_ENABLE;   // Arduino 호환을 위해 활성화
    dcb_config_.fDsrSensitivity = FALSE;
    dcb_config_.fTXContinueOnXoff = FALSE;
    dcb_config_.fOutX = FALSE;
    dcb_config_.fInX = FALSE;
    dcb_config_.fErrorChar = FALSE;
    dcb_config_.fNull = FALSE;
    dcb_config_.fRtsControl = RTS_CONTROL_ENABLE;   // Arduino 호환을 위해 활성화
    dcb_config_.fAbortOnError = FALSE;

    if (!SetCommState(serial_handle_, &dcb_config_)) {
        std::cerr << "[Arduino] Failed to set comm state" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // 즉시 응답을 위한 타임아웃 설정
    timeouts_.ReadIntervalTimeout = 1;          // 바이트 간 간격 1ms
    timeouts_.ReadTotalTimeoutMultiplier = 0;   // 총 읽기 시간 승수
    timeouts_.ReadTotalTimeoutConstant = 10;    // 총 읽기 시간 상수 10ms
    timeouts_.WriteTotalTimeoutMultiplier = 0;  // 총 쓰기 시간 승수
    timeouts_.WriteTotalTimeoutConstant = 50;   // 총 쓰기 시간 상수 50ms

    if (!SetCommTimeouts(serial_handle_, &timeouts_)) {
        std::cerr << "[Arduino] Failed to set timeouts" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // 버퍼 크기 최적화 - 작은 버퍼로 지연 최소화
    if (!SetupComm(serial_handle_, 64, 64)) {
        std::cerr << "[Arduino] Failed to setup comm buffers" << std::endl;
    }

    // 기존 버퍼 플러시
    PurgeComm(serial_handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);

    is_open_ = true;
    return true;
}

bool SerialConnection::testConnection()
{
    if (!isOpen()) return false;
    
    // 간단한 핑 테스트 (실제 Arduino 코드에 따라 조정 필요)
    const std::string ping_cmd = "PING\n";
    DWORD bytes_written = 0;
    
    BOOL result = WriteFile(
        serial_handle_,
        ping_cmd.c_str(),
        static_cast<DWORD>(ping_cmd.length()),
        &bytes_written,
        NULL
    );
    
    return result && (bytes_written == ping_cmd.length());
}

void SerialConnection::write(const std::string& data)
{
    // 연결 상태 확인 및 복구 시도
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) {
        if (!reconnect()) {
            return;
        }
    }

    // 연결 건강도 확인
    if (!isHealthy()) {
        if (!reconnect()) {
            return;
        }
    }

    // 동기 방식으로 단순화 - 즉시 전송
    DWORD bytes_written = 0;
    BOOL result = WriteFile(
        serial_handle_,
        data.c_str(),
        static_cast<DWORD>(data.length()),
        &bytes_written,
        NULL  // 동기 I/O
    );

    if (result && bytes_written == data.length()) {
        // 즉시 플러시하여 확실한 전송
        FlushFileBuffers(serial_handle_);
        return; // 성공
    }

    // 실패한 경우 재시도
    const int MAX_RETRIES = 2;
    for (int retry = 0; retry < MAX_RETRIES; ++retry) {
        Sleep(5); // 짧은 대기
        
        bytes_written = 0;
        result = WriteFile(
            serial_handle_,
            data.c_str(),
            static_cast<DWORD>(data.length()),
            &bytes_written,
            NULL
        );

        if (result && bytes_written == data.length()) {
            FlushFileBuffers(serial_handle_);
            return; // 성공
        }

    }

    // 모든 재시도 실패 - 마지막 재연결 시도 (조용히 처리)
    if (!reconnect()) {
        is_open_ = false;
    }
}

std::string SerialConnection::read()
{
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE)
        return "";

    char buffer[256];
    DWORD bytes_read = 0;
    OVERLAPPED overlapped = {0};
    overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    
    if (overlapped.hEvent == NULL) {
        return "";
    }

    BOOL result = ReadFile(
        serial_handle_,
        buffer,
        sizeof(buffer) - 1,
        &bytes_read,
        &overlapped
    );

    if (!result && GetLastError() == ERROR_IO_PENDING) {
        DWORD wait_result = WaitForSingleObject(overlapped.hEvent, 1);
        if (wait_result == WAIT_OBJECT_0) {
            GetOverlappedResult(serial_handle_, &overlapped, &bytes_read, FALSE);
        }
    }

    CloseHandle(overlapped.hEvent);

    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        return std::string(buffer);
    }
    
    return "";
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
    if (x == 0 && y == 0) return;
    
    // Arduino 호환 명령 포맷 (소문자 m)
    char command[32];
    snprintf(command, sizeof(command), "m%d,%d\n", x, y);
    
    
    sendCommand(std::string(command));
}

void SerialConnection::sendCommand(const std::string& command)
{
    write(command);
}

std::vector<int> SerialConnection::splitValue(int value)
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

void SerialConnection::startTimer()
{
    timer_running_ = true;
    timer_thread_ = std::thread(&SerialConnection::timerThreadFunc, this);
}

void SerialConnection::startListening()
{
    listening_ = true;
    listening_thread_ = std::thread(&SerialConnection::listeningThreadFunc, this);
}

void SerialConnection::timerThreadFunc()
{
    // TIMER 명령 제거 - Arduino에서 필요하지 않음
    while (timer_running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 대기만
    }
}

void SerialConnection::listeningThreadFunc()
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
            std::this_thread::sleep_for(std::chrono::microseconds(100)); // 100μs 대기
        }
    }
}

void SerialConnection::processIncomingLine(const std::string& line)
{
    // 기존 로직 유지
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