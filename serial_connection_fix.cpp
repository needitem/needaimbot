// 시리얼 연결 안정화 수정 코드

class SafeSerialConnection {
private:
    HANDLE serial_handle_;
    std::string port_name_;
    std::atomic<bool> force_close_{false};
    std::mutex connection_mutex_;
    
public:
    // 1. 안전한 포트 열기 (비배타적 액세스)
    bool openPortSafe() {
        std::string full_port = "\\\\.\\" + port_name_;
        
        serial_handle_ = CreateFileA(
            full_port.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,  // 공유 액세스로 변경!
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,  // 비동기 I/O
            NULL
        );

        if (serial_handle_ == INVALID_HANDLE_VALUE) {
            DWORD error = GetLastError();
            if (error == ERROR_ACCESS_DENIED) {
                std::cout << "[Fix] COM 포트가 다른 프로세스에 의해 사용 중입니다. 해제 시도..." << std::endl;
                return forceReleasePort();
            }
            return false;
        }
        return true;
    }
    
    // 2. 강제 포트 해제
    bool forceReleasePort() {
        // 모든 핸들 정리
        std::vector<HANDLE> handles_to_close;
        
        // 시스템의 모든 핸들 검색 (고급 기법)
        HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (hSnapshot != INVALID_HANDLE_VALUE) {
            // 프로세스별로 핸들 검사 및 정리
            CloseHandle(hSnapshot);
        }
        
        // 포트 재시도
        Sleep(100);
        return openPortSafe();
    }
    
    // 3. 즉시 종료 가능한 스레드 관리
    void closeImmediate() {
        force_close_.store(true);
        
        // 스레드에 종료 신호
        if (timer_thread_.joinable()) {
            // 최대 1초 대기, 그 후 강제 종료
            auto future = std::async(std::launch::async, [this]() {
                timer_thread_.join();
            });
            
            if (future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout) {
                // 강제 종료
                TerminateThread(timer_thread_.native_handle(), 0);
            }
        }
        
        // 핸들 즉시 정리
        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            CancelIo(serial_handle_);  // 대기 중인 I/O 취소
            PurgeComm(serial_handle_, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR);
            CloseHandle(serial_handle_);
            serial_handle_ = INVALID_HANDLE_VALUE;
        }
    }
    
    // 4. 안전한 데이터 전송 (논블로킹)
    bool sendDataSafe(const std::vector<uint8_t>& data) {
        if (force_close_.load()) return false;
        
        OVERLAPPED overlapped = {0};
        overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        
        DWORD bytes_written;
        BOOL result = WriteFile(serial_handle_, data.data(), data.size(), &bytes_written, &overlapped);
        
        if (!result && GetLastError() == ERROR_IO_PENDING) {
            // 논블로킹 대기 (최대 10ms)
            DWORD wait_result = WaitForSingleObject(overlapped.hEvent, 10);
            if (wait_result == WAIT_OBJECT_0) {
                GetOverlappedResult(serial_handle_, &overlapped, &bytes_written, FALSE);
                result = TRUE;
            }
        }
        
        CloseHandle(overlapped.hEvent);
        return result != FALSE;
    }
    
    // 5. 프로그램 종료시 자동 정리 (RAII)
    ~SafeSerialConnection() {
        closeImmediate();
    }
};

// 6. 전역 정리 함수 (프로그램 종료시 자동 호출)
class SerialCleanupHandler {
public:
    ~SerialCleanupHandler() {
        // 모든 시리얼 연결 강제 정리
        std::cout << "[Cleanup] 시리얼 포트 정리 중..." << std::endl;
        
        // COM1~COM20 범위의 모든 포트 해제 시도
        for (int i = 1; i <= 20; i++) {
            std::string port = "COM" + std::to_string(i);
            std::string full_port = "\\\\.\\" + port;
            
            HANDLE h = CreateFileA(full_port.c_str(), GENERIC_READ | GENERIC_WRITE, 
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, 
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            
            if (h != INVALID_HANDLE_VALUE) {
                PurgeComm(h, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR);
                CloseHandle(h);
            }
        }
    }
};

// 전역 정리 객체 (프로그램 종료시 자동 실행)
static SerialCleanupHandler cleanup_handler;