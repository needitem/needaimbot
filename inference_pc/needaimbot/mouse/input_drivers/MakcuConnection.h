#ifndef MAKCUCONNECTION_H
#define MAKCUCONNECTION_H

#include <cstddef>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

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
    void move(int x, int y);

    std::atomic<bool> aiming_active;
    std::atomic<bool> shooting_active;

private:
    void write(const std::string& data);
    void sendCommand(const std::string& command);
    void sendCommand(const char* command, size_t size);
    bool sendCommandFast(const char* command, size_t size);

    void startListening();
    void listeningThreadFunc();

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

    int serial_fd_;
    struct termios tty_config_;
#endif

    std::atomic<bool> is_open_;
    std::atomic<bool> listening_;
    std::thread listening_thread_;
    std::mutex write_mutex_;
    std::string port_name_;
    unsigned int baud_rate_;
};

#endif // MAKCUCONNECTION_H
