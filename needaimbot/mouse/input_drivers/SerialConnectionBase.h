#ifndef SERIAL_CONNECTION_BASE_H
#define SERIAL_CONNECTION_BASE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include "../../core/common_utils.h"

class SerialConnectionBase {
public:
    SerialConnectionBase(const std::string& port, unsigned int baud_rate)
        : port_name_(port), baud_rate_(baud_rate), 
          serial_handle_(INVALID_HANDLE_VALUE), is_open_(false),
          listening_(false), aiming_active(false), 
          shooting_active(false), zooming_active(false) {}
    
    virtual ~SerialConnectionBase() {
        cleanup();
    }

    // Public interface
    bool isOpen() const { return is_open_; }
    
    // Public state flags
    std::atomic<bool> aiming_active;
    std::atomic<bool> shooting_active;
    std::atomic<bool> zooming_active;

protected:
    // Pure virtual methods that derived classes must implement
    virtual bool configureDCB() = 0;
    virtual bool configureTimeouts() = 0;
    virtual void processIncomingLine(const std::string& line) = 0;
    
    // Common implementation methods
    bool openPort() {
        std::string full_port = "\\\\.\\" + port_name_;
        
        serial_handle_ = CreateFileA(
            full_port.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,  // No sharing
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
            NULL
        );

        if (serial_handle_ == INVALID_HANDLE_VALUE) {
            DWORD error = GetLastError();
            std::cerr << "[Serial] Unable to open port: " << port_name_ 
                      << " (Error: " << error << ")" << std::endl;
            return false;
        }

        return true;
    }
    
    void closeHandle() {
        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(serial_handle_);
            serial_handle_ = INVALID_HANDLE_VALUE;
        }
        is_open_ = false;
    }
    
    void write(const std::string& data) {
        if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) return;

        std::lock_guard<std::mutex> lock(write_mutex_);

        DWORD bytes_written = 0;
        OVERLAPPED overlapped = {0};
        overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        
        if (overlapped.hEvent == NULL) return;

        BOOL result = WriteFile(
            serial_handle_,
            data.c_str(),
            static_cast<DWORD>(data.length()),
            &bytes_written,
            &overlapped
        );

        if (!result && GetLastError() == ERROR_IO_PENDING) {
            DWORD waitResult = WaitForSingleObject(overlapped.hEvent, 
                NeedAimbot::Constants::SERIAL_WRITE_TIMEOUT_MS);
            if (waitResult == WAIT_OBJECT_0) {
                GetOverlappedResult(serial_handle_, &overlapped, &bytes_written, FALSE);
            }
        }

        CloseHandle(overlapped.hEvent);
        FlushFileBuffers(serial_handle_);
    }
    
    std::string read() {
        if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) return "";

        char buffer[NeedAimbot::Constants::DEFAULT_SERIAL_BUFFER_SIZE];
        DWORD bytes_read = 0;
        OVERLAPPED overlapped = {0};
        overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        
        if (overlapped.hEvent == NULL) return "";

        BOOL result = ReadFile(
            serial_handle_,
            buffer,
            sizeof(buffer) - 1,
            &bytes_read,
            &overlapped
        );

        if (!result && GetLastError() == ERROR_IO_PENDING) {
            DWORD waitResult = WaitForSingleObject(overlapped.hEvent, 
                NeedAimbot::Constants::SERIAL_READ_TIMEOUT_MS);
            if (waitResult == WAIT_OBJECT_0) {
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
    
    // Common utility function
    std::vector<int> splitValue(int value) {
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
    
    void startListening() {
        listening_ = true;
        listening_thread_ = std::thread(&SerialConnectionBase::listeningThreadFunc, this);
    }
    
    void listeningThreadFunc() {
        std::string buffer;
        
        while (listening_) {
            if (!is_open_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));  // Reduced from 10ms for better responsiveness
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
                std::this_thread::sleep_for(std::chrono::microseconds(50));  // Reduced from 100us for lower latency
            }
        }
    }
    
    virtual void cleanup() {
        listening_ = false;
        
        if (listening_thread_.joinable()) {
            listening_thread_.join();
        }
        
        closeHandle();
    }

protected:
    // Member variables
    HANDLE serial_handle_;
    DCB dcb_config_;
    COMMTIMEOUTS timeouts_;
    std::atomic<bool> is_open_;
    std::atomic<bool> listening_;
    std::thread listening_thread_;
    std::mutex write_mutex_;
    std::string port_name_;
    unsigned int baud_rate_;
};

#endif // SERIAL_CONNECTION_BASE_H