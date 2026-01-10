#include "MakcuConnection.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

/* ---------- Makcu-specific constants ---------------------------- */
static const uint32_t BOOT_BAUD = 115200;      // baud rate after initial connection
static const uint32_t WORK_BAUD = 4000000;     // working baud rate – 4 Mbit/s

/* Baud change command (from a.py): 0xDEAD0500A500093D00 */
static const uint8_t BAUD_CHANGE_CMD[9] =
{ 0xDE,0xAD,0x05,0x00,0xA5,0x00,0x09,0x3D,0x00 };

// ============================================================================
// Platform-specific implementations
// ============================================================================

#ifdef _WIN32
// =========================== WINDOWS IMPLEMENTATION ===========================

MakcuConnection::MakcuConnection(const std::string& port, unsigned int /*baud_rate*/)
    : aiming_active(false),
      shooting_active(false),
      zooming_active(false),
      serial_handle_(INVALID_HANDLE_VALUE),
      is_open_(false),
      listening_(false),
      port_name_(port),
      write_event_(NULL),
      read_event_(NULL)
{
    ZeroMemory(&write_overlapped_, sizeof(write_overlapped_));
    ZeroMemory(&read_overlapped_, sizeof(read_overlapped_));

    write_event_ = CreateEvent(NULL, TRUE, FALSE, NULL);
    read_event_ = CreateEvent(NULL, TRUE, FALSE, NULL);

    if (write_event_ && read_event_) {
        write_overlapped_.hEvent = write_event_;
        read_overlapped_.hEvent = read_event_;
    }

    try {
        if (!initializeMakcuConnection()) {
            throw std::runtime_error("Failed to initialize Makcu connection to " + port);
        }
        std::cout << "[Makcu] Connected at 4Mbps! PORT: " << port
                  << " (Native Windows API - Async I/O)" << std::endl;
    } catch (const std::exception& e) {
        cleanup();
        if (write_event_) CloseHandle(write_event_);
        if (read_event_) CloseHandle(read_event_);
        std::cerr << "[Makcu] Initialization error: " << e.what() << std::endl;
    }
}

bool MakcuConnection::initializeMakcuConnection() {
    std::string full_port = "\\\\.\\" + port_name_;

    serial_handle_ = CreateFileA(
        full_port.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL
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
        0, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL
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
    dcb_config_.fDtrControl = DTR_CONTROL_DISABLE;
    dcb_config_.fDsrSensitivity = FALSE;
    dcb_config_.fTXContinueOnXoff = FALSE;
    dcb_config_.fOutX = FALSE;
    dcb_config_.fInX = FALSE;
    dcb_config_.fErrorChar = FALSE;
    dcb_config_.fNull = FALSE;
    dcb_config_.fRtsControl = RTS_CONTROL_DISABLE;
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

void MakcuConnection::safeMakcuClose() {
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) return;

    std::cout << "[Makcu] Starting safe port closure..." << std::endl;

    try {
        const char* left_release = "LR\n";
        const char* right_release = "RR\n";
        const char* neutral_pos = "M0,0\n";
        const char* stop_cmd = "STOP\n";

        writeAsync(left_release, static_cast<DWORD>(strlen(left_release)));
        writeAsync(right_release, static_cast<DWORD>(strlen(right_release)));
        writeAsync(neutral_pos, static_cast<DWORD>(strlen(neutral_pos)));

        if (write_event_) WaitForSingleObject(write_event_, 10);

        FlushFileBuffers(serial_handle_);
        PurgeComm(serial_handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);

        writeAsync(stop_cmd, static_cast<DWORD>(strlen(stop_cmd)));

        if (write_event_) WaitForSingleObject(write_event_, 50);

        if (serial_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(serial_handle_);
            serial_handle_ = INVALID_HANDLE_VALUE;
            std::cout << "[Makcu] Port closed successfully." << std::endl;
        }

        is_open_ = false;
        listening_ = false;
    } catch (const std::exception& e) {
        std::cerr << "[Makcu] Error during safe close: " << e.what() << std::endl;
    }
}

void MakcuConnection::cleanup() {
    listening_ = false;
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
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
    is_open_ = false;
}

MakcuConnection::~MakcuConnection() {
    cleanup();
    if (write_event_) { CloseHandle(write_event_); write_event_ = NULL; }
    if (read_event_) { CloseHandle(read_event_); read_event_ = NULL; }
}

bool MakcuConnection::isOpen() const { return is_open_; }

void MakcuConnection::write(const std::string& data) {
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) return;
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (!writeAsync(data.c_str(), static_cast<DWORD>(data.length()))) {
        std::cerr << "[Makcu] Write operation failed" << std::endl;
    }
}

std::string MakcuConnection::read() {
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) return "";
    char buffer[256];
    DWORD bytes_read = 0;
    if (readAsync(buffer, sizeof(buffer) - 1, &bytes_read) && bytes_read > 0) {
        buffer[bytes_read] = '\0';
        return std::string(buffer);
    }
    return "";
}

bool MakcuConnection::writeAsync(const void* data, DWORD size) {
    if (serial_handle_ == INVALID_HANDLE_VALUE) return false;
    ResetEvent(write_overlapped_.hEvent);
    DWORD bytes_written = 0;
    BOOL result = WriteFile(serial_handle_, data, size, &bytes_written, &write_overlapped_);
    if (!result) {
        DWORD error = GetLastError();
        if (error == ERROR_IO_PENDING) {
            return waitForAsyncOperation(&write_overlapped_, 50);
        }
        return false;
    }
    return true;
}

bool MakcuConnection::readAsync(void* buffer, DWORD size, DWORD* bytesRead) {
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) return false;
    ResetEvent(read_overlapped_.hEvent);
    BOOL result = ReadFile(serial_handle_, buffer, size, bytesRead, &read_overlapped_);
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

bool MakcuConnection::waitForAsyncOperation(OVERLAPPED* overlapped, DWORD timeout_ms) {
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

#else
// =========================== LINUX IMPLEMENTATION ===========================

MakcuConnection::MakcuConnection(const std::string& port, unsigned int /*baud_rate*/)
    : aiming_active(false),
      shooting_active(false),
      zooming_active(false),
      serial_fd_(-1),
      is_open_(false),
      listening_(false),
      port_name_(port)
{
    memset(&tty_config_, 0, sizeof(tty_config_));

    try {
        if (!initializeMakcuConnection()) {
            throw std::runtime_error("Failed to initialize Makcu connection to " + port);
        }
        std::cout << "[Makcu] Connected at 4Mbps! PORT: " << port
                  << " (Linux termios)" << std::endl;
    } catch (const std::exception& e) {
        cleanup();
        std::cerr << "[Makcu] Initialization error: " << e.what() << std::endl;
    }
}

bool MakcuConnection::initializeMakcuConnection() {
    // Step 1: Open at 115200 baud
    serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ < 0) {
        std::cerr << "[Makcu] Unable to open port: " << port_name_
                  << " (Error: " << strerror(errno) << ")" << std::endl;
        return false;
    }

    if (!configurePort(BOOT_BAUD)) {
        closeHandle();
        return false;
    }

    // Step 2: Send baud change command
    if (writeSerial(BAUD_CHANGE_CMD, sizeof(BAUD_CHANGE_CMD)) != sizeof(BAUD_CHANGE_CMD)) {
        std::cerr << "[Makcu] Failed to send baud change command" << std::endl;
        closeHandle();
        return false;
    }

    // Flush and close
    tcdrain(serial_fd_);
    close(serial_fd_);
    serial_fd_ = -1;

    // Wait for MCU reset
    usleep(100000);  // 100ms

    // Step 3: Reopen at 4Mbps
    serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ < 0) {
        std::cerr << "[Makcu] Unable to reopen port at high speed" << std::endl;
        return false;
    }

    if (!configurePort(WORK_BAUD)) {
        closeHandle();
        return false;
    }

    // Flush buffers
    tcflush(serial_fd_, TCIOFLUSH);

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

bool MakcuConnection::configurePort(int baud_rate) {
    if (tcgetattr(serial_fd_, &tty_config_) != 0) {
        std::cerr << "[Makcu] Failed to get port attributes" << std::endl;
        return false;
    }

    // Set baud rate
    speed_t speed;
    switch (baud_rate) {
        case 115200:  speed = B115200; break;
        case 230400:  speed = B230400; break;
        case 460800:  speed = B460800; break;
        case 500000:  speed = B500000; break;
        case 576000:  speed = B576000; break;
        case 921600:  speed = B921600; break;
        case 1000000: speed = B1000000; break;
        case 1152000: speed = B1152000; break;
        case 1500000: speed = B1500000; break;
        case 2000000: speed = B2000000; break;
        case 2500000: speed = B2500000; break;
        case 3000000: speed = B3000000; break;
        case 3500000: speed = B3500000; break;
        case 4000000: speed = B4000000; break;
        default:
            std::cerr << "[Makcu] Unsupported baud rate: " << baud_rate << std::endl;
            return false;
    }

    cfsetispeed(&tty_config_, speed);
    cfsetospeed(&tty_config_, speed);

    // 8N1, no flow control
    tty_config_.c_cflag &= ~PARENB;        // No parity
    tty_config_.c_cflag &= ~CSTOPB;        // 1 stop bit
    tty_config_.c_cflag &= ~CSIZE;
    tty_config_.c_cflag |= CS8;            // 8 bits
    tty_config_.c_cflag &= ~CRTSCTS;       // No hardware flow control
    tty_config_.c_cflag |= CREAD | CLOCAL; // Enable read, ignore modem control

    // Raw input
    tty_config_.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // Raw output
    tty_config_.c_oflag &= ~OPOST;

    // No software flow control
    tty_config_.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty_config_.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

    // Read settings: return immediately with whatever is available
    tty_config_.c_cc[VMIN] = 0;
    tty_config_.c_cc[VTIME] = 1;  // 100ms timeout

    if (tcsetattr(serial_fd_, TCSANOW, &tty_config_) != 0) {
        std::cerr << "[Makcu] Failed to set port attributes" << std::endl;
        return false;
    }

    return true;
}

ssize_t MakcuConnection::writeSerial(const void* data, size_t size) {
    if (serial_fd_ < 0) return -1;
    return ::write(serial_fd_, data, size);
}

ssize_t MakcuConnection::readSerial(void* buffer, size_t size) {
    if (serial_fd_ < 0) return -1;
    return ::read(serial_fd_, buffer, size);
}

void MakcuConnection::cleanup() {
    listening_ = false;

    if (is_open_ && serial_fd_ >= 0) {
        std::cout << "[Makcu] Starting safe port closure..." << std::endl;

        const char* left_release = "LR\n";
        const char* right_release = "RR\n";
        const char* neutral_pos = "M0,0\n";
        const char* stop_cmd = "STOP\n";

        writeSerial(left_release, strlen(left_release));
        writeSerial(right_release, strlen(right_release));
        writeSerial(neutral_pos, strlen(neutral_pos));
        tcdrain(serial_fd_);

        writeSerial(stop_cmd, strlen(stop_cmd));
        tcdrain(serial_fd_);

        usleep(50000);  // 50ms
    }

    closeHandle();

    if (listening_thread_.joinable()) {
        try {
            listening_thread_.join();
        } catch (const std::exception& e) {
            std::cerr << "[Makcu] Error joining listening thread: " << e.what() << std::endl;
        }
    }
}

void MakcuConnection::closeHandle() {
    if (serial_fd_ >= 0) {
        close(serial_fd_);
        serial_fd_ = -1;
        std::cout << "[Makcu] Port closed successfully." << std::endl;
    }
    is_open_ = false;
}

MakcuConnection::~MakcuConnection() {
    cleanup();
}

bool MakcuConnection::isOpen() const { return is_open_; }

void MakcuConnection::write(const std::string& data) {
    if (!is_open_ || serial_fd_ < 0) return;
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (writeSerial(data.c_str(), data.length()) < 0) {
        std::cerr << "[Makcu] Write operation failed" << std::endl;
    }
}

std::string MakcuConnection::read() {
    if (!is_open_ || serial_fd_ < 0) return "";
    char buffer[256];
    ssize_t bytes_read = readSerial(buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        return std::string(buffer);
    }
    return "";
}

#endif  // _WIN32

// ============================================================================
// Common implementation (both platforms)
// ============================================================================

void MakcuConnection::click(int button) {
    // Click = press then release
    if (button == 1) {
        sendCommand("km.left(1)\r\n");
        sendCommand("km.left(0)\r\n");
    } else if (button == 2) {
        sendCommand("km.right(1)\r\n");
        sendCommand("km.right(0)\r\n");
    }
}

void MakcuConnection::press(int button) {
    if (button == 1) {
        sendCommand("km.left(1)\r\n");
    } else if (button == 2) {
        sendCommand("km.right(1)\r\n");
    }
}

void MakcuConnection::release(int button) {
    if (button == 1) {
        sendCommand("km.left(0)\r\n");
    } else if (button == 2) {
        sendCommand("km.right(0)\r\n");
    }
}

void MakcuConnection::move(int x, int y) {
    if (x == 0 && y == 0) return;
    char command[64];
    snprintf(command, sizeof(command), "km.move(%d,%d)\r\n", x, y);
    sendCommand(std::string(command));
}

void MakcuConnection::send_stop() {
    sendCommand("STOP\n");
}

void MakcuConnection::sendCommand(const std::string& command) {
    write(command);
}

std::vector<int> MakcuConnection::splitValue(int value) {
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

void MakcuConnection::startListening() {
    listening_ = true;
    listening_thread_ = std::thread(&MakcuConnection::listeningThreadFunc, this);
}

void MakcuConnection::listeningThreadFunc() {
    std::string buffer;

    while (listening_) {
        if (!is_open_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
            std::this_thread::sleep_for(std::chrono::microseconds(25));
        }
    }
}

void MakcuConnection::processIncomingLine(const std::string& line) {
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
