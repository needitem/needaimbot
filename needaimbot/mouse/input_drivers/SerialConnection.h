#ifndef SERIALCONNECTION_H
#define SERIALCONNECTION_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// Windows Native Serial API - 최고 성능을 위해

class SerialConnection
{
public:
    SerialConnection(const std::string& port, unsigned int baud_rate);
    ~SerialConnection();

    bool isOpen() const;
    bool isHealthy() const;
    bool reconnect();
    void close();

    void write(const std::string& data);
    std::string read();

    void click();
    void press();
    void release();
    void move(int x, int y);

    bool aiming_active;
    bool shooting_active;
    bool zooming_active;

private:
    void sendCommand(const std::string& command);
    std::vector<int> splitValue(int value);

    void startTimer();
    void startListening();
    void processIncomingLine(const std::string& line);

    void timerThreadFunc();
    void listeningThreadFunc();

private:
    bool openPort();
    bool configurePort();
    bool initializeSerial();
    void cleanup();
    void closeHandle();
    void safeSerialClose();  // wjwwood/serial 방식의 안전한 종료

    HANDLE serial_handle_;
    DCB dcb_config_;
    COMMTIMEOUTS timeouts_;
    std::atomic<bool> is_open_;
    std::string port_name_;
    unsigned int baud_rate_;
    mutable std::mutex connection_mutex_;

    std::thread timer_thread_;
    std::atomic<bool> timer_running_;

    std::thread listening_thread_;
    std::atomic<bool> listening_;

};

#endif 
