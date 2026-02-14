#include "MakcuConnection.h"

#include <iostream>
#include <thread>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cerrno>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <poll.h>
#endif

/* ---------- Makcu-specific constants ---------------------------- */
static const uint32_t BOOT_BAUD = 115200;      // baud rate after initial connection
static const uint32_t WORK_BAUD = 4000000;     // working baud rate – 4 Mbit/s

/* Baud change command (from a.py): 0xDEAD0500A500093D00 */
static const uint8_t BAUD_CHANGE_CMD[9] =
{ 0xDE,0xAD,0x05,0x00,0xA5,0x00,0x09,0x3D,0x00 };

#ifndef _WIN32
namespace {
bool writeAllSerialFd(int fd, const char* data, size_t size, bool failFastOnWouldBlock) {
    if (fd < 0 || !data || size == 0) return false;

    size_t totalWritten = 0;
    while (totalWritten < size) {
        ssize_t written = ::write(fd, data + totalWritten, size - totalWritten);
        if (written > 0) {
            totalWritten += static_cast<size_t>(written);
            continue;
        }
        if (written == 0) {
            return false;
        }

        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            if (failFastOnWouldBlock) {
                return false;
            }
            struct pollfd pfd;
            pfd.fd = fd;
            pfd.events = POLLOUT;
            pfd.revents = 0;
            int pollRet = poll(&pfd, 1, 1);
            if (pollRet < 0 && errno == EINTR) {
                continue;
            }
            if (pollRet <= 0) {
                continue;
            }
            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                return false;
            }
            continue;
        }
        return false;
    }

    return true;
}
} // namespace
#endif

// ============================================================================
// Platform-specific implementations
// ============================================================================

#ifdef _WIN32
// =========================== WINDOWS IMPLEMENTATION ===========================

MakcuConnection::MakcuConnection(const std::string& port, unsigned int /*baud_rate*/)
    : aiming_active(false),
      shooting_active(false),
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

        // Enable button state streaming (event-driven, only emits on button changes)
        // Reference: https://www.makcu.com/en/api#binary-protocol-format
        // ASCII: .buttons(mode, period_ms) - mode: 1=raw, 2=constructed
        // Use mode 2 (mut) = pure event-based, sends data ONLY when button state changes
        // Output: km.<mask_u8>\r\n>>> where mask bits: 0=left, 1=right, 2=middle, 3=side1, 4=side2
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for device to be ready
        write("km.buttons(2,0)\r\n");
        std::cout << "[Makcu] Button streaming enabled (pure event mode)" << std::endl;

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

// Auto-detect Makcu device - prefer CH343 driver device over cdc_acm
static std::string detectMakcuDevice() {
    // Priority order: CH343 driver device > cdc_acm device
    const char* candidates[] = {
        "/dev/ttyCH343USB0",  // CH343 driver (proper full-duplex support)
        "/dev/ttyCH343USB1",
        "/dev/ttyACM0",       // cdc_acm (fallback, may have RX issues)
        "/dev/ttyACM1",
        nullptr
    };

    for (int i = 0; candidates[i] != nullptr; ++i) {
        if (access(candidates[i], F_OK) == 0) {
            return std::string(candidates[i]);
        }
    }
    return "";
}

MakcuConnection::MakcuConnection(const std::string& port, unsigned int baud_rate)
    : aiming_active(false),
      shooting_active(false),
      serial_fd_(-1),
      is_open_(false),
      listening_(false),
      port_name_(port),
      baud_rate_(baud_rate)
{
    memset(&tty_config_, 0, sizeof(tty_config_));

    // Auto-detect Makcu device if port is empty or doesn't exist
    if (port_name_.empty() || access(port_name_.c_str(), F_OK) != 0) {
        std::string detected = detectMakcuDevice();
        if (!detected.empty()) {
            std::cout << "[Makcu] Auto-detected device: " << detected << std::endl;
            port_name_ = detected;
        }
    }

    try {
        if (!initializeMakcuConnection()) {
            throw std::runtime_error("Failed to initialize Makcu connection to " + port_name_);
        }
        std::cout << "[Makcu] Connected at " << baud_rate_ << " baud! PORT: " << port_name_
                  << " (Linux termios)" << std::endl;
    } catch (const std::exception& e) {
        cleanup();
        std::cerr << "[Makcu] Initialization error: " << e.what() << std::endl;
    }
}

bool MakcuConnection::initializeMakcuConnection() {
    // Check device exists
    if (access(port_name_.c_str(), F_OK) != 0) {
        std::cerr << "[Makcu] Device not found: " << port_name_ << std::endl;
        return false;
    }
    std::cout << "[Makcu] Device found: " << port_name_ << std::endl;

    // ========================================================================
    // Step 1: Connect at boot baud rate (115200) and send baud change command
    // ========================================================================
    serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ < 0) {
        std::cerr << "[Makcu] Unable to open port: " << port_name_
                  << " (Error: " << strerror(errno) << ")" << std::endl;
        return false;
    }

    // Configure at boot baud rate first
    if (!configurePort(BOOT_BAUD)) {
        std::cerr << "[Makcu] Failed to configure at boot baud" << std::endl;
        closeHandle();
        return false;
    }
    std::cout << "[Makcu] Opened at " << BOOT_BAUD << " baud (boot mode)" << std::endl;

    // Send baud change command
    tcflush(serial_fd_, TCIOFLUSH);
    ssize_t w = ::write(serial_fd_, BAUD_CHANGE_CMD, sizeof(BAUD_CHANGE_CMD));
    if (w != sizeof(BAUD_CHANGE_CMD)) {
        std::cerr << "[Makcu] Failed to send baud change command" << std::endl;
        closeHandle();
        return false;
    }
    tcdrain(serial_fd_);
    std::cout << "[Makcu] Sent baud change command" << std::endl;

    // Close and reopen at high speed
    close(serial_fd_);
    serial_fd_ = -1;
    usleep(100000);  // 100ms for device to switch

    // ========================================================================
    // Step 2: Reconnect at working baud rate (4Mbps)
    // ========================================================================
    serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ < 0) {
        std::cerr << "[Makcu] Unable to reopen port at high speed" << std::endl;
        return false;
    }

    // Set DTR/RTS
    int modem_bits = TIOCM_DTR | TIOCM_RTS;
    ioctl(serial_fd_, TIOCMSET, &modem_bits);
    usleep(50000);  // 50ms
    tcflush(serial_fd_, TCIOFLUSH);

    if (!configurePort(WORK_BAUD)) {
        closeHandle();
        return false;
    }
    std::cout << "[Makcu] Configured port at " << WORK_BAUD << " baud (fast mode)" << std::endl;

    // CRITICAL: Thoroughly flush any stale data from previous sessions
    // This fixes the "intermittent connection" issue
    tcflush(serial_fd_, TCIOFLUSH);
    usleep(100000);  // 100ms

    // Drain any leftover data by reading until empty
    char drain_buf[256];
    int drain_count = 0;
    while (drain_count < 10) {  // Max 10 attempts
        ssize_t n = ::read(serial_fd_, drain_buf, sizeof(drain_buf));
        if (n <= 0) break;
        drain_count++;
        usleep(10000);  // 10ms between reads
    }
    if (drain_count > 0) {
        std::cout << "[Makcu] Drained " << drain_count << " stale buffer(s)" << std::endl;
    }

    tcflush(serial_fd_, TCIOFLUSH);
    usleep(100000);  // Wait 100ms

    // First, stop any existing streaming mode (from previous session)
    const char* stop_stream = "km.buttons(0)\r";
    const ssize_t stopStreamWrite = ::write(serial_fd_, stop_stream, strlen(stop_stream));
    (void)stopStreamWrite;
    tcdrain(serial_fd_);
    usleep(100000);
    tcflush(serial_fd_, TCIOFLUSH);

    // Test communication with a simple command
    const char* test_cmd = "km.move(0,0)\r\n";
    ssize_t written = ::write(serial_fd_, test_cmd, strlen(test_cmd));
    if (written < 0) {
        std::cerr << "[Makcu] Write test failed: " << strerror(errno) << std::endl;
        closeHandle();
        return false;
    }
    tcdrain(serial_fd_);
    usleep(50000);

    // Read response to verify device is responding
    char resp_buf[64];
    ssize_t resp_len = ::read(serial_fd_, resp_buf, sizeof(resp_buf) - 1);
    if (resp_len > 0) {
        resp_buf[resp_len] = '\0';
        std::cout << "[Makcu] Device responded: " << resp_len << " bytes" << std::endl;
    } else {
        std::cout << "[Makcu] Warning: No response to test command (may still work)" << std::endl;
    }
    tcflush(serial_fd_, TCIOFLUSH);

    // Mark as open BEFORE starting listener thread
    is_open_ = true;

    try {
        startListening();
        std::cout << "[Makcu] Button polling started (" << baud_rate_ << " baud)" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Makcu] Failed to start listening thread: " << e.what() << std::endl;
        is_open_ = false;
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

void MakcuConnection::cleanup() {
    listening_ = false;

    // Stop button streaming before closing - prevents stale data on next connect
    if (is_open_ && serial_fd_ >= 0) {
        std::cout << "[Makcu] Stopping button streaming..." << std::endl;
        const char* stop_stream = "km.buttons(0)\r";
        const ssize_t stopStreamWrite = ::write(serial_fd_, stop_stream, strlen(stop_stream));
        (void)stopStreamWrite;
        tcdrain(serial_fd_);
        usleep(50000);
        tcflush(serial_fd_, TCIOFLUSH);
    }

    if (listening_thread_.joinable()) {
        try {
            listening_thread_.join();
        } catch (const std::exception& e) {
            std::cerr << "[Makcu] Error joining listening thread: " << e.what() << std::endl;
        }
    }

    // Close port properly for clean reconnection
    closeHandle();
    std::cout << "[Makcu] Cleanup complete" << std::endl;
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
    if (!writeAllSerialFd(serial_fd_, data.c_str(), data.length(), false)) {
        std::cerr << "[Makcu] Write operation failed" << std::endl;
    }
}

#endif  // _WIN32

// ============================================================================
// Common implementation (both platforms)
// ============================================================================

void MakcuConnection::move(int x, int y) {
    if (x == 0 && y == 0) return;
    char command[64];
    int len = std::snprintf(command, sizeof(command), "km.move(%d,%d)\r\n", x, y);
    if (len > 0 && static_cast<size_t>(len) < sizeof(command)) {
        const size_t cmdSize = static_cast<size_t>(len);
        if (!sendCommandFast(command, cmdSize)) {
            // Fallback to blocking path to avoid silently dropping movement packets.
            sendCommand(command, cmdSize);
        }
    }
}

void MakcuConnection::sendCommand(const std::string& command) {
    sendCommand(command.c_str(), command.size());
}

void MakcuConnection::sendCommand(const char* command, size_t size) {
    if (!command || size == 0) return;

#ifdef _WIN32
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) return;
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (!writeAsync(command, static_cast<DWORD>(size))) {
        std::cerr << "[Makcu] Write operation failed" << std::endl;
    }
#else
    if (!is_open_ || serial_fd_ < 0) return;
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (!writeAllSerialFd(serial_fd_, command, size, false)) {
        std::cerr << "[Makcu] Write operation failed" << std::endl;
    }
#endif
}

bool MakcuConnection::sendCommandFast(const char* command, size_t size) {
    if (!command || size == 0) return false;

#ifdef _WIN32
    if (!is_open_ || serial_handle_ == INVALID_HANDLE_VALUE) return false;
    std::unique_lock<std::mutex> lock(write_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return false;
    return writeAsync(command, static_cast<DWORD>(size));
#else
    if (!is_open_ || serial_fd_ < 0) return false;
    std::unique_lock<std::mutex> lock(write_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return false;
    return writeAllSerialFd(serial_fd_, command, size, true);
#endif
}

void MakcuConnection::startListening() {
    listening_ = true;
    listening_thread_ = std::thread(&MakcuConnection::listeningThreadFunc, this);
}

void MakcuConnection::listeningThreadFunc() {
    std::cout << "[Makcu] Button streaming thread started" << std::endl;

    // Set high priority for this thread (Linux)
#ifndef _WIN32
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        const int niceRet = nice(-10);
        (void)niceRet;
    }
#endif

    // Enable streaming mode: km.buttons(1) - raw mode, NO period parameter!
    // CRITICAL: Using period (e.g., km.buttons(1,10)) breaks streaming.
    // The correct format is km.buttons(1)\r with ONLY mode parameter.
    // Reference: EVENTURI project uses exactly this format.
    const char* stream_cmd = "km.buttons(1)\r";
    const ssize_t streamWrite = ::write(serial_fd_, stream_cmd, strlen(stream_cmd));
    (void)streamWrite;
    tcdrain(serial_fd_);
    usleep(100000);  // 100ms for device to start streaming

    // Drain the echo response
    char drain_buf[128];
    const ssize_t drainRead = ::read(serial_fd_, drain_buf, sizeof(drain_buf));
    (void)drainRead;

    std::cout << "[Makcu] Button streaming enabled (raw mode)" << std::endl;

    char read_buf[256];
    uint8_t last_mask = 0;

    while (listening_) {
        if (!is_open_ || serial_fd_ < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

#ifndef _WIN32
        struct pollfd pfd;
        pfd.fd = serial_fd_;
        pfd.events = POLLIN | POLLERR | POLLHUP;
        pfd.revents = 0;

        int pollRet = poll(&pfd, 1, 1);
        if (pollRet <= 0) {
            continue;
        }
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            continue;
        }
#endif

        // Read incoming data - button events come as single bytes (mask value)
        ssize_t n = ::read(serial_fd_, read_buf, sizeof(read_buf));
        if (n > 0) {
            // Process each byte - look for button mask values
            for (ssize_t i = 0; i < n; i++) {
                uint8_t byte = static_cast<uint8_t>(read_buf[i]);

                // Skip printable ASCII chars (echo, prompt)
                if (byte >= 0x20 && byte <= 0x7E) continue;
                // Skip CR/LF
                if (byte == 0x0D || byte == 0x0A) continue;

                // Valid button mask: 0x00-0x1F (5 buttons max)
                if (byte <= 0x1F && byte != last_mask) {
                    last_mask = byte;

                    bool left = (byte & 0x01) != 0;
                    bool right = (byte & 0x02) != 0;
                    bool side2 = (byte & 0x10) != 0;

                    // Update button states atomically
                    shooting_active = left;
                    aiming_active = right || side2;
                }
            }
        }
    }

    // Stop streaming on exit
    const char* stop_cmd = "km.buttons(0)\r";
    const ssize_t stopWrite = ::write(serial_fd_, stop_cmd, strlen(stop_cmd));
    (void)stopWrite;
    tcdrain(serial_fd_);

    std::cout << "[Makcu] Button streaming stopped" << std::endl;
}
