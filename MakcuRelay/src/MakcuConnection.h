#ifndef MAKCUCONNECTION_H
#define MAKCUCONNECTION_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#endif

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdint>
#include <functional>

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

    bool left_mouse_active;
    bool right_mouse_active;
    bool side1_active;
    bool side2_active;

    // State broadcast callback
    using StateChangeCallback = std::function<void(bool left_mouse, bool right_mouse, bool side1, bool side2)>;
    void setStateChangeCallback(StateChangeCallback callback);

    // Button state polling
    void startButtonPolling();
    void stopButtonPolling();

private:
    StateChangeCallback state_callback_;
    std::atomic<bool> polling_enabled_{false};
    std::thread polling_thread_;
    void buttonPollingThreadFunc();
    void sendCommand(const std::string& command);
    std::vector<int> splitValue(int value);

    void startListening();
    void listeningThreadFunc();
    void processIncomingLine(const std::string& line);

    bool initializeMakcuConnection();
#ifdef _WIN32
    bool configureDCB(uint32_t baud_rate);
    bool configureTimeouts();
    bool writeAsync(const void* data, DWORD size);
    bool readAsync(void* buffer, DWORD size, DWORD* bytesRead);
    bool waitForAsyncOperation(OVERLAPPED* overlapped, DWORD timeout_ms = 100);
#else
    bool configureBaudRate(int fd, uint32_t baud_rate);
#endif
    void cleanup();
    void closeHandle();
    void safeMakcuClose();

private:
#ifdef _WIN32
    HANDLE serial_handle_;
    DCB dcb_config_;
    COMMTIMEOUTS timeouts_;
    OVERLAPPED write_overlapped_;
    OVERLAPPED read_overlapped_;
    HANDLE write_event_;
    HANDLE read_event_;
#else
    int serial_fd_;
#endif
    std::atomic<bool> is_open_;
    std::atomic<bool> listening_;
    std::thread       listening_thread_;
    std::mutex        write_mutex_;
    std::string       port_name_;
};

#endif // MAKCUCONNECTION_H

