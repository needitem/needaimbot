#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <set>
#include <future>
#include <cstring>

#include "needaimbot.h"
#include "SerialConnection.h"
#include "../../AppContext.h"
#include "../../core/common_utils.h"

// 전역 정리 핸들러 클래스 (전방 선언)
class GlobalSerialCleanupHandler {
private:
    static std::set<std::string> used_ports;
    
public:
    static void registerPort(const std::string& port) {
        used_ports.insert(port);
    }
    
    static void unregisterPort(const std::string& port) {
        used_ports.erase(port);
    }
    
    ~GlobalSerialCleanupHandler();  // 정의는 나중에
};

// static 멤버 정의
std::set<std::string> GlobalSerialCleanupHandler::used_ports;

SerialConnection::SerialConnection(const std::string& port, unsigned int baud_rate)
    : serial_handle_(INVALID_HANDLE_VALUE),
      is_open_(false),
      port_name_(port),
      baud_rate_(baud_rate),
      listening_(false),
      aiming_active(false),
      shooting_active(false), 
      zooming_active(false)
{
    try {
        // 사용된 포트 등록 (정리 대상으로 추가)
        GlobalSerialCleanupHandler::registerPort(port_name_);
        
        // 원자적 초기화 시도
        if (!initializeSerial()) {
            throw std::runtime_error("Failed to initialize serial connection to " + port_name_);
        }
        
        std::cout << "[Arduino] Connected! PORT: " << port_name_ 
                  << " (Native Windows API - Ultra Low Latency)" << std::endl;
        
    } catch (const std::exception& e) {
        // 초기화 실패 시 간단한 정리만 수행 (스레드는 아직 생성되지 않음)
        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(serial_handle_);
            serial_handle_ = INVALID_HANDLE_VALUE;
        }
        is_open_ = false;
        std::cerr << "[Arduino] Initialization error: " << e.what() << std::endl;
        // 객체는 생성되지만 is_open_은 false로 유지됨
    }
}

bool SerialConnection::initializeSerial() {
    // 1. 포트 열기
    if (!openPort()) {
        return false;
    }
    
    // 2. 포트 설정
    if (!configurePort()) {
        closeHandle();
        return false;
    }
    
    // 3. 스레드 시작
    try {
        if (AppContext::getInstance().config.arduino_enable_keys) {
            startListening();
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Arduino] Thread initialization failed: " << e.what() << std::endl;
        closeHandle();
        return false;
    }
}

void SerialConnection::safeSerialClose() {
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) {
        return;
    }
    
    std::cout << "[Arduino] Starting safe serial port closure..." << std::endl;
    
    try {
        // 1. 즉시 I/O 취소 (강제 종료 대비)
        CancelIo(serial_handle_);
        
        // 2. Arduino를 안전한 상태로 리셋 - 여러 명령 전송
        std::cout << "[Arduino] Resetting Arduino to safe state..." << std::endl;
        sendCommand("r");  // 마우스 릴리스
        Sleep(10);
        sendCommand("m0,0\n");  // 움직임 중지
        Sleep(10);
        sendCommand("r");  // 한 번 더 릴리스 확인
        Sleep(50);
        
        // 3. 출력 버퍼 플러시 (전송 완료 보장)
        if (!FlushFileBuffers(serial_handle_)) {
            std::cerr << "[Arduino] Warning: FlushFileBuffers failed" << std::endl;
        }
        
        // 4. 모든 버퍼 강제 정리 (ABORT 포함)
        DWORD purge_flags = PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR;
        if (!PurgeComm(serial_handle_, purge_flags)) {
            std::cerr << "[Arduino] Warning: PurgeComm failed" << std::endl;
        }
        
        // 5. DTR/RTS 라인 상태를 명시적으로 설정 (Arduino 안정화)
        // DTR과 RTS를 LOW로 설정하여 Arduino가 깨끗한 상태를 유지하도록 함
        EscapeCommFunction(serial_handle_, CLRDTR);
        EscapeCommFunction(serial_handle_, CLRRTS);
        Sleep(10);
        
        // 7. 짧은 대기
        Sleep(50);
        
        // 8. 포트 안전하게 닫기 (강화된 방식)
        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            int ret = CloseHandle(serial_handle_);
            if (ret == 0) {
                DWORD error = GetLastError();
                std::cerr << "[Arduino] Error closing serial port: " << error << std::endl;
            } else {
                serial_handle_ = INVALID_HANDLE_VALUE;
                std::cout << "[Arduino] Serial port closed successfully." << std::endl;
            }
        }
        
        is_open_ = false;
        
    } catch (const std::exception& e) {
        std::cerr << "[Arduino] Error during safe serial close: " << e.what() << std::endl;
        // 예외 발생 시에도 핸들 정리 시도
        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(serial_handle_);
            serial_handle_ = INVALID_HANDLE_VALUE;
            is_open_ = false;
        }
    }
}


void SerialConnection::cleanup() {
    // 플래그 설정
    listening_ = false;
    
    std::cout << "[Arduino] Starting cleanup..." << std::endl;
    
    // I/O 작업 취소 먼저 수행
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CancelIo(serial_handle_);
    }
    
    // 타임아웃 기반 스레드 종료 함수
    auto cleanup_with_timeout = [this](std::thread& t, const std::string& name) {
        if (t.joinable()) {
            auto future = std::async(std::launch::async, [&t]() {
                t.join();
            });
            
            if (future.wait_for(std::chrono::milliseconds(NeedAimbot::Constants::THREAD_JOIN_TIMEOUT_MS)) == std::future_status::timeout) {
                std::cout << "[Arduino] " << name << " thread timeout - detaching" << std::endl;
                // TerminateThread 사용 회피 - 리소스 누수 가능
                t.detach();
            } else {
                std::cout << "[Arduino] " << name << " thread terminated gracefully" << std::endl;
            }
        }
    };
    
    // 스레드를 먼저 정리 (포트 닫기 전에)
    cleanup_with_timeout(listening_thread_, "Listening");
    
    // 그 다음 포트 안전하게 종료
    safeSerialClose();
    
    std::cout << "[Arduino] Cleanup completed" << std::endl;
}


void SerialConnection::closeHandle() {
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
    is_open_ = false;
}

SerialConnection::~SerialConnection()
{
    std::cout << "[Arduino] Destructor called for port: " << port_name_ << std::endl;
    cleanup();
    
    // 포트 등록 해제
    GlobalSerialCleanupHandler::unregisterPort(port_name_);
    std::cout << "[Arduino] Port " << port_name_ << " unregistered from cleanup handler" << std::endl;
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
    cleanup();
}

bool SerialConnection::openPort()
{
    std::string full_port = "\\\\.\\" + port_name_;
    
    // 첫 번째 시도: 독점 액세스 (기존 방식)
    serial_handle_ = CreateFileA(
        full_port.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,                         // 독점 액세스
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        DWORD error = GetLastError();
        
        if (error == ERROR_ACCESS_DENIED || error == ERROR_SHARING_VIOLATION) {
            std::cout << "[Arduino] Port in use, trying shared access..." << std::endl;
            
            // 두 번째 시도: 공유 액세스
            serial_handle_ = CreateFileA(
                full_port.c_str(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,  // 공유 액세스 허용
                NULL,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                NULL
            );
            
            if (serial_handle_ == INVALID_HANDLE_VALUE) {
                std::cerr << "[Arduino] Failed to open port even with shared access: " 
                          << port_name_ << " (Error: " << GetLastError() << ")" << std::endl;
                return false;
            } else {
                std::cout << "[Arduino] Port opened with shared access" << std::endl;
            }
        } else {
            std::cerr << "[Arduino] Unable to open port: " << port_name_ 
                      << " (Error: " << error << ")" << std::endl;
            return false;
        }
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
    dcb_config_.fDtrControl = DTR_CONTROL_DISABLE;   // Arduino 리셋 방지
    dcb_config_.fDsrSensitivity = FALSE;
    dcb_config_.fTXContinueOnXoff = FALSE;
    dcb_config_.fOutX = FALSE;
    dcb_config_.fInX = FALSE;
    dcb_config_.fErrorChar = FALSE;
    dcb_config_.fNull = FALSE;
    dcb_config_.fRtsControl = RTS_CONTROL_DISABLE;   // Arduino 리셋 방지
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
    timeouts_.ReadTotalTimeoutConstant = NeedAimbot::Constants::SERIAL_READ_TIMEOUT_MS;
    timeouts_.WriteTotalTimeoutMultiplier = 0;  // 총 쓰기 시간 승수
    timeouts_.WriteTotalTimeoutConstant = NeedAimbot::Constants::SERIAL_WRITE_TIMEOUT_MS;

    if (!SetCommTimeouts(serial_handle_, &timeouts_)) {
        std::cerr << "[Arduino] Failed to set timeouts" << std::endl;
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // 버퍼 크기 최적화 - 더 큰 버퍼로 효율성 개선
    constexpr DWORD BUFFER_SIZE = 512;  // Increased from 64 for better throughput
    if (!SetupComm(serial_handle_, BUFFER_SIZE, BUFFER_SIZE)) {
        std::cerr << "[Arduino] Failed to setup comm buffers" << std::endl;
    }

    // 기존 버퍼 플러시
    PurgeComm(serial_handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);

    is_open_ = true;
    
    // Arduino 초기화 명령 전송 (is_open_ = true 이후)
    Sleep(100);  // Arduino 부팅 대기
    
    // 직접 write 호출 (sendCommand는 write를 호출하는데 write는 is_open을 체크함)
    const char* release_cmd = "r";
    const char* move_cmd = "m0,0\n";
    
    DWORD bytes_written;
    WriteFile(serial_handle_, release_cmd, static_cast<DWORD>(strlen(release_cmd)), &bytes_written, NULL);
    Sleep(10);
    WriteFile(serial_handle_, move_cmd, static_cast<DWORD>(strlen(move_cmd)), &bytes_written, NULL);
    Sleep(10);
    
    return true;
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

    // 동기 방식으로 단순화
    BOOL result = ReadFile(
        serial_handle_,
        buffer,
        sizeof(buffer) - 1,
        &bytes_read,
        NULL  // 동기 I/O
    );

    if (result && bytes_read > 0) {
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

void SerialConnection::startListening()
{
    listening_ = true;
    listening_thread_ = std::thread(&SerialConnection::listeningThreadFunc, this);
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

// GlobalSerialCleanupHandler 소멸자 구현
GlobalSerialCleanupHandler::~GlobalSerialCleanupHandler() {
        // 프로그램 종료 시 추가 정리 비활성화
        // SerialConnection의 소멸자가 이미 적절한 정리를 수행했으므로
        // 추가적인 포트 접근은 시스템 상태를 방해할 수 있음
        
        if (!used_ports.empty()) {
            std::cout << "[Cleanup] Serial ports have been cleaned up by their respective destructors." << std::endl;
            std::cout << "[Cleanup] Registered ports were: ";
            for (const std::string& p : used_ports) {
                std::cout << p << " ";
            }
            std::cout << std::endl;
        }
    }

// 전역 정리 객체 (프로그램 종료 시 자동 실행)
static GlobalSerialCleanupHandler global_serial_cleanup;