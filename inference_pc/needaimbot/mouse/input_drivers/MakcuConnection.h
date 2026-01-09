#ifndef MAKCUCONNECTION_H
#define MAKCUCONNECTION_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

// Windows Native Serial API - 최고 성능을 위해

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
    
    // 초기화 및 정리 메서드
    bool initializeMakcuConnection();
    bool configureDCB(uint32_t baud_rate);
    bool configureTimeouts();
    void cleanup();
    void closeHandle();
    void safeMakcuClose();  // wjwwood/serial 방식의 안전한 종료
    
    // Async I/O functions
    bool writeAsync(const void* data, DWORD size);
    bool readAsync(void* buffer, DWORD size, DWORD* bytesRead);
    bool waitForAsyncOperation(OVERLAPPED* overlapped, DWORD timeout_ms = 100);

private:
    HANDLE serial_handle_;
    DCB dcb_config_;
    COMMTIMEOUTS timeouts_;
    std::atomic<bool> is_open_;
    std::atomic<bool> listening_;
    std::thread       listening_thread_;
    std::mutex        write_mutex_;
    std::string       port_name_;  // 포트 이름 저장
    
    // Overlapped I/O structures
    OVERLAPPED write_overlapped_;
    OVERLAPPED read_overlapped_;
    HANDLE write_event_;
    HANDLE read_event_;
};

#endif // MAKCUCONNECTION_H
