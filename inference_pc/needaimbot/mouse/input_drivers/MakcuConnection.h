#ifndef MAKCUCONNECTION_H
#define MAKCUCONNECTION_H

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#else
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#endif

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

    bool initializeMakcuConnection();
    void cleanup();
    void closeHandle();

#ifdef _WIN32
    bool configureDCB(uint32_t baud_rate);
    bool configureTimeouts();
    void safeMakcuClose();
    bool writeAsync(const void* data, DWORD size);
    bool readAsync(void* buffer, DWORD size, DWORD* bytesRead);
    bool waitForAsyncOperation(OVERLAPPED* overlapped, DWORD timeout_ms = 100);

    HANDLE serial_handle_;
    DCB dcb_config_;
    COMMTIMEOUTS timeouts_;
    OVERLAPPED write_overlapped_;
    OVERLAPPED read_overlapped_;
    HANDLE write_event_;
    HANDLE read_event_;
#else
    bool configurePort(int baud_rate);
    ssize_t writeSerial(const void* data, size_t size);
    ssize_t readSerial(void* buffer, size_t size);

    int serial_fd_;
    struct termios tty_config_;
#endif

    std::atomic<bool> is_open_;
    std::atomic<bool> listening_;
    std::thread listening_thread_;
    std::mutex write_mutex_;
    std::string port_name_;
};

#endif // MAKCUCONNECTION_H
