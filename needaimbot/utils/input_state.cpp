#include "input_state.h"
#include <iostream>
#include <cstring>

InputStateManager& InputStateManager::getInstance() {
    static InputStateManager instance;
    return instance;
}

InputStateManager::InputStateManager() {
}

InputStateManager::~InputStateManager() {
    stopUDPListener();
}

void InputStateManager::startUDPListener(int port) {
    if (running_.load()) {
        return;
    }

#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[InputState] WSAStartup failed" << std::endl;
        return;
    }

    udp_socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_socket_ == INVALID_SOCKET) {
        std::cerr << "[InputState] socket() failed" << std::endl;
        WSACleanup();
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(udp_socket_, reinterpret_cast<SOCKADDR*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "[InputState] bind() failed: " << WSAGetLastError() << std::endl;
        closesocket(udp_socket_);
        WSACleanup();
        return;
    }

    int timeoutMs = 100;
    setsockopt(udp_socket_, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));
#else
    udp_socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_socket_ < 0) {
        std::cerr << "[InputState] socket() failed" << std::endl;
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(udp_socket_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[InputState] bind() failed" << std::endl;
        close(udp_socket_);
        return;
    }

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000;
    setsockopt(udp_socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    running_.store(true);
    listener_thread_ = std::thread(&InputStateManager::udpListenerThread, this, port);

    std::cout << "[InputState] UDP listener started on port " << port << std::endl;
}

void InputStateManager::stopUDPListener() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }

#ifdef _WIN32
    if (udp_socket_ != INVALID_SOCKET) {
        closesocket(udp_socket_);
        udp_socket_ = INVALID_SOCKET;
    }
    WSACleanup();
#else
    if (udp_socket_ >= 0) {
        close(udp_socket_);
        udp_socket_ = -1;
    }
#endif

    std::cout << "[InputState] UDP listener stopped" << std::endl;
}

void InputStateManager::udpListenerThread(int port) {
    char buffer[256];

    while (running_.load()) {
#ifdef _WIN32
        int ret = recvfrom(udp_socket_, buffer, sizeof(buffer) - 1, 0, nullptr, nullptr);
        if (ret == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) {
                continue;
            }
            if (!running_.load()) {
                break;
            }
            continue;
        }
#else
        ssize_t ret = recvfrom(udp_socket_, buffer, sizeof(buffer) - 1, 0, nullptr, nullptr);
        if (ret < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            if (!running_.load()) {
                break;
            }
            continue;
        }
#endif

        if (ret <= 0) {
            continue;
        }

        buffer[ret] = '\0';
        std::string msg(buffer);

        if (msg.rfind("STATE:", 0) == 0) {
            std::string data = msg.substr(6);
            int left = 0, right = 0;
            if (sscanf(data.c_str(), "%d,%d", &left, &right) == 2) {
                left_button_.store(left != 0);
                right_button_.store(right != 0);
                makcu_mode_.store(true);
            }
        }
    }
}

bool InputStateManager::isLeftButtonPressed() const {
    // Always use GetAsyncKeyState - Makcu UDP state is disabled due to performance
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;

    /*
    if (makcu_mode_.load()) {
        return left_button_.load();
    } else {
        return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    }
    */
}

bool InputStateManager::isRightButtonPressed() const {
    // Always use GetAsyncKeyState - Makcu UDP state is disabled due to performance
    return (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;

    /*
    if (makcu_mode_.load()) {
        return right_button_.load();
    } else {
        return (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    }
    */
}
