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
#include "../../core/constants.h"

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
      writer_running_(false),
      aiming_active(false),
      shooting_active(false),
      zooming_active(false),
      accumulated_move_x_(0),
      accumulated_move_y_(0),
      has_accumulated_move_(false),
<<<<<<< ours
=======
      consecutive_write_failures_(0),
      last_reconnect_time_(std::chrono::steady_clock::now()),
>>>>>>> theirs
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
        // 사용된 포트 등록 (정리 대상으로 추가)
        GlobalSerialCleanupHandler::registerPort(port_name_);
        
        // 원자적 초기화 시도
        if (!initializeSerial()) {
            throw std::runtime_error("Failed to initialize serial connection to " + port_name_);
        }

        startWriterThread();
    } catch (const std::exception& e) {
        // 초기화 실패 시 간단한 정리만 수행 (스레드는 아직 생성되지 않음)
        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(serial_handle_);
            serial_handle_ = INVALID_HANDLE_VALUE;
        }
        if (write_event_) CloseHandle(write_event_);
        if (read_event_) CloseHandle(read_event_);
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
    
    
    try {
        // 1. 즉시 I/O 취소 (강제 종료 대비)
        CancelIo(serial_handle_);
        
        // 2. Arduino를 안전한 상태로 리셋 - 바이너리 명령 전송
        uint8_t release_cmd[3] = {0x02, 0, 0}; // release command
        uint8_t move_cmd[3] = {0x04, 0, 0};    // move 0,0 command

        writeAsync(release_cmd, 3);
        writeAsync(move_cmd, 3);
        writeAsync(release_cmd, 3); // 한 번 더 릴리스
        
        // Wait for async operations to complete
        if (write_event_) {
            WaitForSingleObject(write_event_, 20);
        }
        
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
        
        // 7. Wait for hardware stabilization using event
        if (write_event_) {
            WaitForSingleObject(write_event_, 30);
        }
        
        // 8. 포트 안전하게 닫기 (강화된 방식)
        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            int ret = CloseHandle(serial_handle_);
            if (ret == 0) {
                DWORD error = GetLastError();
                std::cerr << "[Arduino] Error closing serial port: " << error << std::endl;
            } else {
                serial_handle_ = INVALID_HANDLE_VALUE;
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


void SerialConnection::startWriterThread() {
    if (writer_running_.load()) {
        return;
    }

    accumulated_move_x_ = 0;
    accumulated_move_y_ = 0;
    has_accumulated_move_ = false;

    writer_running_.store(true);

    try {
        writer_thread_ = std::thread(&SerialConnection::writerThreadFunc, this);
    } catch (...) {
        writer_running_.store(false);
        throw;
    }

#ifdef _WIN32
    SetThreadDescription(writer_thread_.native_handle(), L"ArduinoSerialWriter");
<<<<<<< ours
    SetThreadPriority(writer_thread_.native_handle(), THREAD_PRIORITY_HIGHEST);
=======
    SetThreadPriority(writer_thread_.native_handle(), Constants::MOUSE_THREAD_PRIORITY);
>>>>>>> theirs
#endif
}

void SerialConnection::stopWriterThread() {
    if (!writer_running_.load()) {
        return;
    }

    writer_running_.store(false);
    writer_cv_.notify_all();

    if (writer_thread_.joinable()) {
        auto future = std::async(std::launch::async, [this]() {
            writer_thread_.join();
        });

        if (future.wait_for(std::chrono::milliseconds(Constants::THREAD_JOIN_TIMEOUT_MS)) == std::future_status::timeout) {
            writer_thread_.detach();
        }
    }

    std::lock_guard<std::mutex> lock(writer_mutex_);
    writer_queue_.clear();
    accumulated_move_x_ = 0;
    accumulated_move_y_ = 0;
    has_accumulated_move_ = false;
}

void SerialConnection::enqueueBinaryCommand(uint8_t cmd, int param1, int param2, bool coalesce) {
    {
        std::lock_guard<std::mutex> lock(writer_mutex_);
        if (coalesce) {
            accumulated_move_x_ = std::clamp(accumulated_move_x_ + param1, -4096, 4096);
            accumulated_move_y_ = std::clamp(accumulated_move_y_ + param2, -4096, 4096);
            has_accumulated_move_ = true;
        } else {
            writer_queue_.push_back(PendingBinaryCommand{cmd, param1, param2, false});
        }
    }

    writer_cv_.notify_one();
}

<<<<<<< ours
void SerialConnection::sendBinaryImmediate(uint8_t cmd, int param1, int param2) {
=======
bool SerialConnection::sendBinaryImmediate(uint8_t cmd, int param1, int param2) {
>>>>>>> theirs
    int clampedX = std::clamp(param1, -127, 127);
    int clampedY = std::clamp(param2, -127, 127);

    uint8_t buffer[3] = {
        cmd,
        static_cast<uint8_t>(static_cast<int8_t>(clampedX)),
        static_cast<uint8_t>(static_cast<int8_t>(clampedY))
    };

<<<<<<< ours
    writeBinary(buffer, 3);
=======
    return writeBinary(buffer, 3);
>>>>>>> theirs
}

void SerialConnection::writerThreadFunc() {
#ifdef _WIN32
<<<<<<< ours
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
=======
    SetThreadPriority(GetCurrentThread(), Constants::MOUSE_THREAD_PRIORITY);
>>>>>>> theirs
#endif

    while (writer_running_.load()) {
        PendingBinaryCommand command{0x00, 0, 0, false};
        bool hasCommand = false;

        {
            std::unique_lock<std::mutex> lock(writer_mutex_);
            writer_cv_.wait(lock, [this]() {
                return !writer_running_.load() || has_accumulated_move_ || !writer_queue_.empty();
            });

            if (!writer_running_.load()) {
                break;
            }

            if (has_accumulated_move_) {
                command = PendingBinaryCommand{0x04, accumulated_move_x_, accumulated_move_y_, true};
                accumulated_move_x_ = 0;
                accumulated_move_y_ = 0;
                has_accumulated_move_ = false;
                hasCommand = true;
            } else if (!writer_queue_.empty()) {
                command = writer_queue_.front();
                writer_queue_.pop_front();
                hasCommand = true;
            }
        }

        if (!hasCommand) {
            continue;
        }

        if (command.coalesce) {
            int remainingX = command.param1;
            int remainingY = command.param2;

            while (remainingX != 0 || remainingY != 0) {
                int stepX = std::clamp(remainingX, -127, 127);
                int stepY = std::clamp(remainingY, -127, 127);
<<<<<<< ours
                remainingX -= stepX;
                remainingY -= stepY;

                sendBinaryImmediate(command.cmd, stepX, stepY);
            }
        } else {
            sendBinaryImmediate(command.cmd, command.param1, command.param2);
=======

                if (sendBinaryImmediate(command.cmd, stepX, stepY)) {
                    remainingX -= stepX;
                    remainingY -= stepY;
                    continue;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                {
                    std::lock_guard<std::mutex> lock(writer_mutex_);
                    accumulated_move_x_ = std::clamp(accumulated_move_x_ + remainingX, -4096, 4096);
                    accumulated_move_y_ = std::clamp(accumulated_move_y_ + remainingY, -4096, 4096);
                    has_accumulated_move_ = true;
                }
                writer_cv_.notify_one();
                break;
            }
        } else {
            if (!sendBinaryImmediate(command.cmd, command.param1, command.param2)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                {
                    std::lock_guard<std::mutex> lock(writer_mutex_);
                    writer_queue_.push_front(command);
                }
                writer_cv_.notify_one();
            }
>>>>>>> theirs
        }
    }
}


void SerialConnection::cleanup() {
    // 플래그 설정
    listening_ = false;

    stopWriterThread();


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
            
            if (future.wait_for(std::chrono::milliseconds(Constants::THREAD_JOIN_TIMEOUT_MS)) == std::future_status::timeout) {
                // TerminateThread 사용 회피 - 리소스 누수 가능
                t.detach();
            } else {
            }
        }
    };
    
    // 스레드를 먼저 정리 (포트 닫기 전에)
    cleanup_with_timeout(listening_thread_, "Listening");
    
    // 그 다음 포트 안전하게 종료
    safeSerialClose();
    
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
    
    // 포트 등록 해제
    GlobalSerialCleanupHandler::unregisterPort(port_name_);
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
    
    // Use event-based wait for Arduino restart instead of Sleep
    if (write_event_) {
        WaitForSingleObject(write_event_, 100);
    }
    
    // 재연결 시도
    if (openPort() && configurePort()) {
        {
            std::lock_guard<std::mutex> failure_lock(failure_mutex_);
            last_reconnect_time_ = std::chrono::steady_clock::now();
            consecutive_write_failures_ = 0;
        }
        return true;
    }

    return false;
}

void SerialConnection::close()
{
    std::lock_guard<std::mutex> lock(connection_mutex_);
    cleanup();
}

bool SerialConnection::openPort()
{
    std::string full_port = "\\\\.\\" + port_name_;
    
    // 포트 열기 재시도 로직
    const int MAX_OPEN_ATTEMPTS = 3;
    DWORD last_error = 0;
    
    for (int attempt = 0; attempt < MAX_OPEN_ATTEMPTS; ++attempt) {
        if (attempt > 0) {
            // Use event-based wait instead of Sleep
            if (write_event_) {
                WaitForSingleObject(write_event_, 100); // Short non-blocking wait
            }
        }
        
        // Open with OVERLAPPED flag for async I/O
        serial_handle_ = CreateFileA(
            full_port.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,                         // 독점 액세스
            NULL,
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED,      // Enable async I/O
            NULL
        );

        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            return true;
        }
        
        last_error = GetLastError();
        
        // ERROR_ACCESS_DENIED (5) 또는 ERROR_SHARING_VIOLATION 처리
        if (last_error == ERROR_ACCESS_DENIED || last_error == ERROR_SHARING_VIOLATION) {
            
            // 공유 액세스 시도 with overlapped I/O
            serial_handle_ = CreateFileA(
                full_port.c_str(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,  // 공유 액세스 허용
                NULL,
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,      // Enable async I/O
                NULL
            );
            
            if (serial_handle_ != INVALID_HANDLE_VALUE) {
                return true;
            }
            
            last_error = GetLastError();
            
            // 그래도 실패하면 다시 시도
            if (last_error == ERROR_ACCESS_DENIED) {
            }
        } else if (last_error == ERROR_FILE_NOT_FOUND) {
            std::cerr << "[Arduino] Port " << port_name_ << " does not exist!" << std::endl;
            return false; // 포트가 없으면 재시도 무의미
        }
    }
    
    // 모든 시도 실패
    std::cerr << "[Arduino] Failed to open port " << port_name_ 
              << " after " << MAX_OPEN_ATTEMPTS << " attempts." << std::endl;
    std::cerr << "[Arduino] Last error code: " << last_error << std::endl;
    
    // Windows 에러 코드 설명
    switch(last_error) {
        case ERROR_ACCESS_DENIED:
            std::cerr << "  ERROR_ACCESS_DENIED (5): Access is denied. Try:" << std::endl;
            std::cerr << "  1. Close Arduino IDE and Serial Monitor" << std::endl;
            std::cerr << "  2. Run as Administrator" << std::endl;
            std::cerr << "  3. Check if another process is using the port" << std::endl;
            break;
        case ERROR_SHARING_VIOLATION:
            std::cerr << "  ERROR_SHARING_VIOLATION: Port is locked by another process" << std::endl;
            break;
        case ERROR_FILE_NOT_FOUND:
            std::cerr << "  ERROR_FILE_NOT_FOUND: Port does not exist" << std::endl;
            break;
        default:
            std::cerr << "  Unknown error. Check system logs." << std::endl;
    }
    
    return false;
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
    timeouts_.ReadTotalTimeoutConstant = Constants::SERIAL_READ_TIMEOUT_MS;
    timeouts_.WriteTotalTimeoutMultiplier = 0;  // 총 쓰기 시간 승수
    timeouts_.WriteTotalTimeoutConstant = Constants::SERIAL_WRITE_TIMEOUT_MS;

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
    
    // Send Arduino initialization commands using binary protocol
    uint8_t release_cmd[3] = {0x02, 0, 0}; // release command
    uint8_t move_cmd[3] = {0x04, 0, 0};    // move 0,0 command

    // Use async write for initialization
    writeAsync(release_cmd, 3);
    writeAsync(move_cmd, 3);
    
    return true;
}


void SerialConnection::write(const std::string& data)
{
    if (data.empty()) {
        return;
    }

    if (!ensureConnectionReady()) {
        return;
    }

    performWrite(data.c_str(), static_cast<DWORD>(data.length()));
}

bool SerialConnection::writeBinary(const uint8_t* data, size_t size)
{
    if (size == 0 || data == nullptr) {
        return true;
    }

    if (!ensureConnectionReady()) {
        return false;
    }

    return performWrite(data, static_cast<DWORD>(size));
}

bool SerialConnection::ensureConnectionReady()
{
    HANDLE handle_snapshot = INVALID_HANDLE_VALUE;
    {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        handle_snapshot = serial_handle_;
    }

    if (!is_open_.load() || handle_snapshot == INVALID_HANDLE_VALUE) {
        if (!reconnect()) {
            return false;
        }
    }

    if (!isHealthy()) {
        if (!reconnect()) {
            return false;
        }
    }

    HANDLE current_handle = INVALID_HANDLE_VALUE;
    {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        current_handle = serial_handle_;
    }

    return is_open_.load() && current_handle != INVALID_HANDLE_VALUE;
}

bool SerialConnection::performWrite(const void* data, DWORD size)
{
    if (size == 0) {
        std::lock_guard<std::mutex> failure_lock(failure_mutex_);
        consecutive_write_failures_ = 0;
        return true;
    }

    constexpr int MAX_ATTEMPTS = 3;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        if (writeAsync(data, size)) {
            std::lock_guard<std::mutex> failure_lock(failure_mutex_);
            consecutive_write_failures_ = 0;
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    auto now = std::chrono::steady_clock::now();
    bool shouldReconnect = false;
    {
        std::lock_guard<std::mutex> failure_lock(failure_mutex_);
        ++consecutive_write_failures_;

        if (consecutive_write_failures_ >= 3 &&
            (now - last_reconnect_time_) > std::chrono::milliseconds(250)) {
            shouldReconnect = true;
            last_reconnect_time_ = now;
            consecutive_write_failures_ = 0;
        }
    }

    if (shouldReconnect) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        reconnect();
    }

    return false;
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
    sendBinaryCommand(0x01, 0, 0);
}

void SerialConnection::press()
{
    sendBinaryCommand(0x03, 0, 0);
}

void SerialConnection::release()
{
    sendBinaryCommand(0x02, 0, 0);
}

void SerialConnection::move(int x, int y)
{
    if (x == 0 && y == 0) return;

    sendBinaryCommand(0x04, x, y);
}

void SerialConnection::sendCommand(const std::string& command)
{
    write(command);
}

void SerialConnection::sendBinaryCommand(uint8_t cmd, int param1, int param2)
{
    if (writer_running_.load()) {
        enqueueBinaryCommand(cmd, param1, param2, cmd == 0x04);
    } else {
        sendBinaryImmediate(cmd, param1, param2);
    }
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
    DWORD wait_time = 10;  // Start with 10ms wait
    int empty_reads = 0;
    
    while (listening_) {
        if (!is_open_) {
            // Use event wait instead of sleep when port is closed
            if (write_event_) {
                WaitForSingleObject(write_event_, 50);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Reduced from 50ms for better responsiveness
            }
            continue;
        }

        std::string data = read();
        if (!data.empty()) {
            buffer += data;
            empty_reads = 0;  // Reset counter on successful read
            wait_time = 1;    // Minimal wait when data is flowing
            
            size_t pos = 0;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                std::string line = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);
                
                if (!line.empty()) {
                    processIncomingLine(line);
                }
            }
        } else {
            empty_reads++;
            // Adaptive wait: increase delay when no data, decrease when active
            if (empty_reads < 10) {
                wait_time = 1;  // 1ms for first 10 empty reads
            } else if (empty_reads < 100) {
                wait_time = 5;  // 5ms for next 90 empty reads
            } else {
                wait_time = 5; // Reduced from 20ms for better responsiveness
            }
            
            if (read_event_) {
                // Use event-based wait with timeout for better responsiveness
                WaitForSingleObject(read_event_, wait_time);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
            }
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

// Async I/O helper functions
bool SerialConnection::writeAsync(const void* data, DWORD size)
{
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) {
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
<<<<<<< ours
            // Wait for async operation to complete with a very short deadline to avoid stalling input
            return waitForAsyncOperation(&write_overlapped_, 5);
=======
            // Wait for async operation to complete with a balanced deadline
            return waitForAsyncOperation(&write_overlapped_, 30);
>>>>>>> theirs
        }
        return false;
    }

    return true;
}

bool SerialConnection::readAsync(void* buffer, DWORD size, DWORD* bytesRead)
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

bool SerialConnection::waitForAsyncOperation(OVERLAPPED* overlapped, DWORD timeout_ms)
{
    DWORD result = WaitForSingleObject(overlapped->hEvent, timeout_ms);
    if (result == WAIT_OBJECT_0) {
        DWORD bytes_transferred;
        return GetOverlappedResult(serial_handle_, overlapped, &bytes_transferred, FALSE) != 0;
    }
    
    if (result == WAIT_TIMEOUT) {
        // Cancel only the specific overlapped operation to avoid disrupting other I/O
        CancelIoEx(serial_handle_, overlapped);
    }

    return false;
}

// 전역 정리 객체 (프로그램 종료 시 자동 실행)
static GlobalSerialCleanupHandler global_serial_cleanup;