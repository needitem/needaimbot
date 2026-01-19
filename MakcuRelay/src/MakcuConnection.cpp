#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <cerrno>
#endif

#include "MakcuConnection.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstring>
#include <stdexcept>

static const uint32_t BOOT_BAUD = 115200;
static const uint32_t WORK_BAUD = 4000000;

static const uint8_t BAUD_CHANGE_CMD[9] =
{ 0xDE,0xAD,0x05,0x00,0xA5,0x00,0x09,0x3D,0x00 };

MakcuConnection::MakcuConnection(const std::string& port, unsigned int /*baud_rate*/)
    :
#ifdef _WIN32
      serial_handle_(INVALID_HANDLE_VALUE),
      write_event_(NULL),
      read_event_(NULL),
#else
      serial_fd_(-1),
#endif
      is_open_(false),
      listening_(false),
      left_mouse_active(false),
      right_mouse_active(false),
      side1_active(false),
      side2_active(false),
      port_name_(port),
      state_callback_(nullptr)
{
#ifdef _WIN32
    ZeroMemory(&write_overlapped_, sizeof(write_overlapped_));
    ZeroMemory(&read_overlapped_, sizeof(read_overlapped_));

    write_event_ = CreateEvent(NULL, TRUE, FALSE, NULL);
    read_event_ = CreateEvent(NULL, TRUE, FALSE, NULL);

    if (write_event_ && read_event_) {
        write_overlapped_.hEvent = write_event_;
        read_overlapped_.hEvent = read_event_;
    }
#endif

    try {
        if (!initializeMakcuConnection()) {
            throw std::runtime_error("Failed to initialize Makcu connection to " + port);
        }

#ifdef _WIN32
        std::cout << "[Makcu] Connected at 4Mbps! PORT: " << port
                  << " (Native Windows API - Async I/O)" << std::endl;
#else
        std::cout << "[Makcu] Connected at 4Mbps! PORT: " << port
                  << " (Linux termios)" << std::endl;
#endif

    } catch (const std::exception& e) {
        cleanup();
#ifdef _WIN32
        if (write_event_) CloseHandle(write_event_);
        if (read_event_) CloseHandle(read_event_);
#endif
        std::cerr << "[Makcu] Initialization error: " << e.what() << std::endl;
    }
}

#ifdef _WIN32
bool MakcuConnection::initializeMakcuConnection() {
    std::string full_port = "\\\\.\\" + port_name_;

    serial_handle_ = CreateFileA(
        full_port.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[Makcu] Unable to open port: " << port_name_
                  << " (Error: " << GetLastError() << ")" << std::endl;
        return false;
    }

    if (!configureDCB(BOOT_BAUD)) {
        closeHandle();
        return false;
    }

    if (!writeAsync(BAUD_CHANGE_CMD, sizeof(BAUD_CHANGE_CMD))) {
        std::cerr << "[Makcu] Failed to send baud change command" << std::endl;
        closeHandle();
        return false;
    }

    CloseHandle(serial_handle_);
    serial_handle_ = INVALID_HANDLE_VALUE;

    if (write_event_) {
        WaitForSingleObject(write_event_, 100);
    }

    serial_handle_ = CreateFileA(
        full_port.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[Makcu] Unable to reopen port at high speed" << std::endl;
        return false;
    }

    if (!configureDCB(WORK_BAUD)) {
        closeHandle();
        return false;
    }

    if (!configureTimeouts()) {
        closeHandle();
        return false;
    }

    constexpr DWORD BUFFER_SIZE = 256;
    if (!SetupComm(serial_handle_, BUFFER_SIZE, BUFFER_SIZE)) {
        std::cerr << "[Makcu] Warning: Failed to setup comm buffers" << std::endl;
    }
    PurgeComm(serial_handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);

    try {
        startListening();
        is_open_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Makcu] Failed to start listening thread: " << e.what() << std::endl;
        closeHandle();
        return false;
    }
}
#else // Linux implementation
bool MakcuConnection::configureBaudRate(int fd, uint32_t baud_rate) {
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "[Makcu] tcgetattr failed: " << strerror(errno) << std::endl;
        return false;
    }

    speed_t speed;
    switch (baud_rate) {
        case 115200: speed = B115200; break;
        case 4000000: speed = B4000000; break;
        default:
            std::cerr << "[Makcu] Unsupported baud rate: " << baud_rate << std::endl;
            return false;
    }

    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;

    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;

    tty.c_cc[VTIME] = 1;
    tty.c_cc[VMIN] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "[Makcu] tcsetattr failed: " << strerror(errno) << std::endl;
        return false;
    }

    tcflush(fd, TCIOFLUSH);
    return true;
}

bool MakcuConnection::initializeMakcuConnection() {
    serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);

    if (serial_fd_ < 0) {
        std::cerr << "[Makcu] Unable to open port: " << port_name_
                  << " (" << strerror(errno) << ")" << std::endl;
        return false;
    }

    if (!configureBaudRate(serial_fd_, BOOT_BAUD)) {
        close(serial_fd_);
        serial_fd_ = -1;
        return false;
    }

    ssize_t written = ::write(serial_fd_, BAUD_CHANGE_CMD, sizeof(BAUD_CHANGE_CMD));
    if (written != sizeof(BAUD_CHANGE_CMD)) {
        std::cerr << "[Makcu] Failed to send baud change command" << std::endl;
        close(serial_fd_);
        serial_fd_ = -1;
        return false;
    }

    tcdrain(serial_fd_);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    close(serial_fd_);
    serial_fd_ = -1;

    serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);

    if (serial_fd_ < 0) {
        std::cerr << "[Makcu] Unable to reopen port at high speed: " << strerror(errno) << std::endl;
        return false;
    }

    if (!configureBaudRate(serial_fd_, WORK_BAUD)) {
        close(serial_fd_);
        serial_fd_ = -1;
        return false;
    }

    try {
        startListening();
        is_open_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Makcu] Failed to start listening thread: " << e.what() << std::endl;
        close(serial_fd_);
        serial_fd_ = -1;
        return false;
    }
}
#endif

#ifdef _WIN32
bool MakcuConnection::configureDCB(uint32_t baud_rate) {
    ZeroMemory(&dcb_config_, sizeof(DCB));
    dcb_config_.DCBlength = sizeof(DCB);

    if (!GetCommState(serial_handle_, &dcb_config_)) {
        std::cerr << "[Makcu] Failed to get comm state" << std::endl;
        return false;
    }

    dcb_config_.BaudRate = baud_rate;
    dcb_config_.ByteSize = 8;
    dcb_config_.Parity = NOPARITY;
    dcb_config_.StopBits = ONESTOPBIT;
    dcb_config_.fBinary = TRUE;
    dcb_config_.fParity = FALSE;
    dcb_config_.fOutxCtsFlow = FALSE;
    dcb_config_.fOutxDsrFlow = FALSE;
    // Match pyserial defaults: no hardware handshake but DTR/RTS asserted
    dcb_config_.fDtrControl = DTR_CONTROL_ENABLE;
    dcb_config_.fDsrSensitivity = FALSE;
    dcb_config_.fTXContinueOnXoff = FALSE;
    dcb_config_.fOutX = FALSE;
    dcb_config_.fInX = FALSE;
    dcb_config_.fErrorChar = FALSE;
    dcb_config_.fNull = FALSE;
    dcb_config_.fRtsControl = RTS_CONTROL_ENABLE;
    dcb_config_.fAbortOnError = FALSE;

    if (!SetCommState(serial_handle_, &dcb_config_)) {
        std::cerr << "[Makcu] Failed to set comm state for baud rate: " << baud_rate << std::endl;
        return false;
    }

    return true;
}

bool MakcuConnection::configureTimeouts() {
    timeouts_.ReadIntervalTimeout = 1;
    timeouts_.ReadTotalTimeoutMultiplier = 0;
    timeouts_.ReadTotalTimeoutConstant = 5;
    timeouts_.WriteTotalTimeoutMultiplier = 0;
    timeouts_.WriteTotalTimeoutConstant = 10;

    if (!SetCommTimeouts(serial_handle_, &timeouts_)) {
        std::cerr << "[Makcu] Failed to set timeouts" << std::endl;
        return false;
    }

    return true;
}
#endif

void MakcuConnection::safeMakcuClose() {
#ifdef _WIN32
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) {
        return;
    }

    std::cout << "[Makcu] Starting safe Makcu port closure..." << std::endl;

    try {
        const char* left_release = "LR\n";
        const char* right_release = "RR\n";
        const char* neutral_pos = "M0,0\n";
        const char* stop_cmd = "STOP\n";

        writeAsync(left_release, static_cast<DWORD>(std::strlen(left_release)));
        writeAsync(right_release, static_cast<DWORD>(std::strlen(right_release)));
        writeAsync(neutral_pos, static_cast<DWORD>(std::strlen(neutral_pos)));

        if (write_event_) {
            WaitForSingleObject(write_event_, 10);
        }

        if (!FlushFileBuffers(serial_handle_)) {
            std::cerr << "[Makcu] Warning: FlushFileBuffers failed" << std::endl;
        }

        if (!PurgeComm(serial_handle_, PURGE_RXCLEAR)) {
            std::cerr << "[Makcu] Warning: PURGE_RXCLEAR failed" << std::endl;
        }

        if (!PurgeComm(serial_handle_, PURGE_TXCLEAR)) {
            std::cerr << "[Makcu] Warning: PURGE_TXCLEAR failed" << std::endl;
        }

        writeAsync(stop_cmd, static_cast<DWORD>(std::strlen(stop_cmd)));

        if (write_event_) {
            WaitForSingleObject(write_event_, 50);
        }

        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            int ret = CloseHandle(serial_handle_);
            if (ret == 0) {
                DWORD error = GetLastError();
                std::cerr << "[Makcu] Error closing Makcu port: " << error << std::endl;
            } else {
                serial_handle_ = INVALID_HANDLE_VALUE;
                std::cout << "[Makcu] Makcu port closed successfully." << std::endl;
            }
        }

        is_open_ = false;
        listening_ = false;

    } catch (const std::exception& e) {
        std::cerr << "[Makcu] Error during safe Makcu close: " << e.what() << std::endl;
    }
#else // Linux
    if (!is_open_ || serial_fd_ < 0) {
        return;
    }

    std::cout << "[Makcu] Starting safe Makcu port closure..." << std::endl;

    try {
        const char* left_release = "km.left(0)\r\n";
        const char* right_release = "km.right(0)\r\n";
        const char* neutral_pos = "km.move(0,0)\r\n";

        ::write(serial_fd_, left_release, strlen(left_release));
        ::write(serial_fd_, right_release, strlen(right_release));
        ::write(serial_fd_, neutral_pos, strlen(neutral_pos));

        tcdrain(serial_fd_);
        tcflush(serial_fd_, TCIOFLUSH);

        close(serial_fd_);
        serial_fd_ = -1;
        is_open_ = false;
        listening_ = false;

        std::cout << "[Makcu] Makcu port closed successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Makcu] Error during safe Makcu close: " << e.what() << std::endl;
    }
#endif
}

void MakcuConnection::cleanup() {
    listening_ = false;
    stopButtonPolling();

    safeMakcuClose();

    if (listening_thread_.joinable()) {
        try {
            listening_thread_.join();
        } catch (const std::exception& e) {
            std::cerr << "[Makcu] Error joining listening thread: " << e.what() << std::endl;
        }
    }
}

void MakcuConnection::closeHandle() {
#ifdef _WIN32
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (serial_fd_ >= 0) {
        close(serial_fd_);
        serial_fd_ = -1;
    }
#endif
    is_open_ = false;
}

MakcuConnection::~MakcuConnection()
{
    cleanup();

#ifdef _WIN32
    if (write_event_) {
        CloseHandle(write_event_);
        write_event_ = NULL;
    }
    if (read_event_) {
        CloseHandle(read_event_);
        read_event_ = NULL;
    }
#endif
}

bool MakcuConnection::isOpen() const
{
    return is_open_;
}

void MakcuConnection::write(const std::string& data)
{
#ifdef _WIN32
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE)
        return;

    std::lock_guard<std::mutex> lock(write_mutex_);

    if (!writeAsync(data.c_str(), static_cast<DWORD>(data.length()))) {
        std::cerr << "[Makcu] Write operation failed" << std::endl;
    }
#else
    if (!is_open_ || serial_fd_ < 0)
        return;

    std::lock_guard<std::mutex> lock(write_mutex_);

    ssize_t written = ::write(serial_fd_, data.c_str(), data.length());
    if (written < 0 || static_cast<size_t>(written) != data.length()) {
        std::cerr << "[Makcu] Write operation failed" << std::endl;
    }
#endif
}

std::string MakcuConnection::read()
{
#ifdef _WIN32
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE)
        return "";

    char buffer[256];
    DWORD bytes_read = 0;

    if (readAsync(buffer, sizeof(buffer) - 1, &bytes_read) && bytes_read > 0) {
        buffer[bytes_read] = '\0';
        return std::string(buffer);
    }

    return "";
#else
    if (!is_open_ || serial_fd_ < 0)
        return "";

    char buffer[256];
    ssize_t bytes_read = ::read(serial_fd_, buffer, sizeof(buffer) - 1);

    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        return std::string(buffer);
    }

    return "";
#endif
}

void MakcuConnection::click(int button)
{
    if (button == 1) {
        sendCommand("km.left(1)");
        sendCommand("km.left(0)");
    } else if (button == 2) {
        sendCommand("km.right(1)");
        sendCommand("km.right(0)");
    }
}

void MakcuConnection::press(int button)
{
    if (button == 1) {
        sendCommand("km.left(1)");
    } else if (button == 2) {
        sendCommand("km.right(1)");
    }
}

void MakcuConnection::release(int button)
{
    if (button == 1) {
        sendCommand("km.left(0)");
    } else if (button == 2) {
        sendCommand("km.right(0)");
    }
}

void MakcuConnection::move(int x, int y)
{
    if (x == 0 && y == 0) return;

    char command[64];
    std::snprintf(command, sizeof(command), "km.move(%d,%d)", x, y);
    sendCommand(std::string(command));
}

void MakcuConnection::send_stop()
{
    // No-op for modern Makcu protocol – kept for API compatibility.
}

void MakcuConnection::setStateChangeCallback(StateChangeCallback callback)
{
    state_callback_ = callback;
}

void MakcuConnection::startButtonPolling()
{
    if (polling_enabled_.load()) {
        return;
    }

    polling_enabled_ = true;
    polling_thread_ = std::thread(&MakcuConnection::buttonPollingThreadFunc, this);
    std::cout << "[Makcu] Button polling started" << std::endl;
}

void MakcuConnection::stopButtonPolling()
{
    polling_enabled_ = false;
    if (polling_thread_.joinable()) {
        polling_thread_.join();
    }
}

void MakcuConnection::buttonPollingThreadFunc()
{
    while (polling_enabled_.load() && is_open_) {
        sendCommand("km.left()");
        sendCommand("km.right()");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void MakcuConnection::sendCommand(const std::string& command)
{
    // Makcu Python library sends ASCII commands terminated with CRLF.
    write(command + "\r\n");
}

std::vector<int> MakcuConnection::splitValue(int value)
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

void MakcuConnection::startListening()
{
    listening_ = true;
    listening_thread_ = std::thread(&MakcuConnection::listeningThreadFunc, this);
}

void MakcuConnection::listeningThreadFunc()
{
    std::string buffer;

    while (listening_) {
        if (!is_open_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        std::string data = read();
        if (!data.empty()) {
            buffer += data;

            // Process complete lines
            size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                std::string line = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);

                // Remove trailing CR
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }

                if (!line.empty()) {
                    processIncomingLine(line);
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

void MakcuConnection::processIncomingLine(const std::string& line)
{
    // Skip empty lines and prompts
    if (line.empty() || line == ">>>") {
        return;
    }

    static bool prev_left = false;
    static bool prev_right = false;
    static bool prev_side1 = false;
    static bool prev_side2 = false;
    static std::string last_command;

    // Check for command echo
    if (line.find("km.left()") != std::string::npos) {
        last_command = "left";
        return;
    }
    else if (line.find("km.right()") != std::string::npos) {
        last_command = "right";
        return;
    }
    else if (line.find("km.side1()") != std::string::npos) {
        last_command = "side1";
        return;
    }
    else if (line.find("km.side2()") != std::string::npos) {
        last_command = "side2";
        return;
    }

    // Parse numeric response
    if (!last_command.empty()) {
        try {
            int state = std::stoi(line);
            bool state_changed = false;

            if (last_command == "left") {
                bool new_left = (state > 0);
                if (new_left != prev_left) {
                    left_mouse_active = new_left;
                    prev_left = new_left;
                    state_changed = true;
                }
            }
            else if (last_command == "right") {
                bool new_right = (state > 0);
                if (new_right != prev_right) {
                    right_mouse_active = new_right;
                    prev_right = new_right;
                    state_changed = true;
                }
            }
            else if (last_command == "side1") {
                bool new_side1 = (state > 0);
                if (new_side1 != prev_side1) {
                    side1_active = new_side1;
                    prev_side1 = new_side1;
                    state_changed = true;
                }
            }
            else if (last_command == "side2") {
                bool new_side2 = (state > 0);
                if (new_side2 != prev_side2) {
                    side2_active = new_side2;
                    prev_side2 = new_side2;
                    state_changed = true;
                }
            }

            last_command.clear();

            if (state_changed && state_callback_) {
                state_callback_(left_mouse_active, right_mouse_active, side1_active, side2_active);
            }
        } catch (const std::exception&) {
            last_command.clear();
        }
    }
}

#ifdef _WIN32
bool MakcuConnection::writeAsync(const void* data, DWORD size)
{
    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }

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
            return waitForAsyncOperation(&write_overlapped_, 50);
        }
        return false;
    }

    return true;
}

bool MakcuConnection::readAsync(void* buffer, DWORD size, DWORD* bytesRead)
{
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }

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

bool MakcuConnection::waitForAsyncOperation(OVERLAPPED* overlapped, DWORD timeout_ms)
{
    DWORD result = WaitForSingleObject(overlapped->hEvent, timeout_ms);
    if (result == WAIT_OBJECT_0) {
        DWORD bytes_transferred;
        return GetOverlappedResult(serial_handle_, overlapped, &bytes_transferred, FALSE) != 0;
    }

    if (result == WAIT_TIMEOUT) {
        CancelIo(serial_handle_);
    }

    return false;
}
#endif
