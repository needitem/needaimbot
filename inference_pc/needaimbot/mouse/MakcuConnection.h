#ifndef MAKCUCONNECTION_H
#define MAKCUCONNECTION_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <atomic>
#include <condition_variable>
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
    // move() always uses the ASCII "km.move(x,y)\r\n" encoding. The MAKCU binary
    // move frame was tried and deliberately dropped: every binary setter echoes
    // an ACK frame ([0x50][0x0D]...) back on the shared serial line, and at aiming
    // rates (~144 Hz) that flood starves/desyncs the button-mask parser so aim-key
    // presses (Side2 thumb, right-click) get dropped. ASCII echoes are printable
    // and skipped cleanly, so ASCII is the only supported encoding. Do NOT add a
    // binary path back without also solving the RX-flood problem (e.g. km.echo(0),
    // which on the test firmware did not reliably fix it). ~20us serial savings
    // are not worth the button-reliability risk.
    void move(int x, int y);
    uint8_t buttonMask() const { return button_mask_.load(std::memory_order_acquire); }
    uint64_t buttonSequence() const { return button_sequence_.load(std::memory_order_acquire); }
    bool waitForButtonEvent(uint64_t last_sequence, int timeout_ms);

private:
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
    std::atomic<uint8_t> button_mask_{0};
    std::atomic<uint64_t> button_sequence_{0};
    std::thread listening_thread_;
    std::mutex write_mutex_;
    std::mutex button_mutex_;
    std::condition_variable button_cv_;
    std::string port_name_;
    unsigned int baud_rate_;
};

#endif // MAKCUCONNECTION_H
