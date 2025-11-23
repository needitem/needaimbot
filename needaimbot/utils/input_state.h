#pragma once

#include <atomic>
#include <thread>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

class InputStateManager {
public:
    static InputStateManager& getInstance();

    void startUDPListener(int port = 5006);
    void stopUDPListener();

    bool isLeftButtonPressed() const;
    bool isRightButtonPressed() const;

    bool isMakcuMode() const { return makcu_mode_; }

private:
    InputStateManager();
    ~InputStateManager();
    InputStateManager(const InputStateManager&) = delete;
    InputStateManager& operator=(const InputStateManager&) = delete;

    void udpListenerThread(int port);

    std::atomic<bool> left_button_{false};
    std::atomic<bool> right_button_{false};
    std::atomic<bool> makcu_mode_{false};
    std::atomic<bool> running_{false};

    std::thread listener_thread_;

#ifdef _WIN32
    SOCKET udp_socket_{INVALID_SOCKET};
#else
    int udp_socket_{-1};
#endif
};
