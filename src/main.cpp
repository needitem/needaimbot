#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#endif

#include <atomic>
#include <csignal>
#include <iostream>
#include <string>
#include <mutex>

#include "MakcuConnection.h"

static std::atomic<bool> g_running{ true };

// Global broadcast socket and address for mouse state notifications
#ifdef _WIN32
static SOCKET g_broadcastSock = INVALID_SOCKET;
#else
static int g_broadcastSock = -1;
#endif
static sockaddr_in g_broadcastAddr{};
static std::mutex g_broadcastMutex;

void sendBroadcastEvent(const char* event)
{
    std::lock_guard<std::mutex> lock(g_broadcastMutex);
#ifdef _WIN32
    if (g_broadcastSock != INVALID_SOCKET) {
        sendto(g_broadcastSock, event, static_cast<int>(strlen(event)), 0,
               reinterpret_cast<SOCKADDR*>(&g_broadcastAddr), sizeof(g_broadcastAddr));
    }
#else
    if (g_broadcastSock >= 0) {
        sendto(g_broadcastSock, event, strlen(event), 0,
               reinterpret_cast<struct sockaddr*>(&g_broadcastAddr), sizeof(g_broadcastAddr));
    }
#endif
}

void signalHandler(int)
{
    g_running.store(false);
}

int runRelay(int argc, char** argv)
{
#ifdef _WIN32
    std::string comPort = "COM3";
#else
    std::string comPort = "/dev/ttyUSB0";
#endif

    unsigned short udpPort = 5005;
    std::string broadcastIP = "127.0.0.1";
    unsigned short broadcastPort = 5006;

    if (argc >= 2) {
        comPort = argv[1];
    }
    if (argc >= 3) {
        try {
            udpPort = static_cast<unsigned short>(std::stoi(argv[2]));
        } catch (...) {
            std::cerr << "[MakcuRelay] Invalid UDP port, using default 5005\n";
        }
    }
    if (argc >= 4) {
        broadcastIP = argv[3];
    }
    if (argc >= 5) {
        try {
            broadcastPort = static_cast<unsigned short>(std::stoi(argv[4]));
        } catch (...) {
            std::cerr << "[MakcuRelay] Invalid broadcast port, using default 5006\n";
        }
    }

    std::cout << "[MakcuRelay] COM port: " << comPort
              << ", UDP listen port: " << udpPort
              << ", Broadcast to: " << broadcastIP << ":" << broadcastPort << std::endl;

#ifdef _WIN32
    WSADATA wsaData;
    int wsaErr = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (wsaErr != 0) {
        std::cerr << "[MakcuRelay] WSAStartup failed: " << wsaErr << std::endl;
        return 1;
    }

    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        std::cerr << "[MakcuRelay] socket() failed: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }
#else
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        std::cerr << "[MakcuRelay] socket() failed: " << strerror(errno) << std::endl;
        return 1;
    }
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(udpPort);

#ifdef _WIN32
    if (bind(sock, reinterpret_cast<SOCKADDR*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "[MakcuRelay] bind() failed: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    int timeoutMs = 50;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));
#else
    if (bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[MakcuRelay] bind() failed: " << strerror(errno) << std::endl;
        close(sock);
        return 1;
    }

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 50000; // 50ms
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    MakcuConnection makcu(comPort, 4000000);
    if (!makcu.isOpen()) {
        std::cerr << "[MakcuRelay] Failed to open Makcu on " << comPort << std::endl;
#ifdef _WIN32
        closesocket(sock);
        WSACleanup();
#else
        close(sock);
#endif
        return 1;
    }

    // Setup broadcast socket for sending mouse state events to game PC
#ifdef _WIN32
    g_broadcastSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_broadcastSock == INVALID_SOCKET) {
        std::cerr << "[MakcuRelay] Warning: Failed to create broadcast socket" << std::endl;
    }
#else
    g_broadcastSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_broadcastSock < 0) {
        std::cerr << "[MakcuRelay] Warning: Failed to create broadcast socket" << std::endl;
    }
#endif

    // Configure broadcast destination address
    std::memset(&g_broadcastAddr, 0, sizeof(g_broadcastAddr));
    g_broadcastAddr.sin_family = AF_INET;
    g_broadcastAddr.sin_port = htons(broadcastPort);
    if (inet_pton(AF_INET, broadcastIP.c_str(), &g_broadcastAddr.sin_addr) != 1) {
        std::cerr << "[MakcuRelay] Warning: Invalid broadcast IP: " << broadcastIP << std::endl;
    }

    // Setup mouse button state callback to broadcast events
    makcu.setStateChangeCallback([](bool left_mouse, bool right_mouse, bool side1, bool side2) {
        (void)side1; (void)side2;  // Unused for now
        static bool prev_left = false;
        static bool prev_right = false;

        // Left mouse button for shooting (aim_shoot_offset)
        if (left_mouse != prev_left) {
            if (left_mouse) {
                sendBroadcastEvent("SHOOT:START");
            } else {
                sendBroadcastEvent("SHOOT:STOP");
            }
            prev_left = left_mouse;
        }

        // Right mouse button for aiming
        if (right_mouse != prev_right) {
            if (right_mouse) {
                sendBroadcastEvent("AIM:START");
            } else {
                sendBroadcastEvent("AIM:STOP");
            }
            prev_right = right_mouse;
        }
    });

    // Start button polling thread
    makcu.startButtonPolling();
    std::cout << "[MakcuRelay] Mouse button monitoring enabled -> " << broadcastIP << ":" << broadcastPort << std::endl;

    std::cout << "[MakcuRelay] Listening for UDP commands (MOVE:x,y / CLICK:LEFT|RIGHT)...\n";
    std::cout << "[MakcuRelay] Press Ctrl+C to exit.\n";

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    char buffer[256];

    while (g_running.load()) {
        sockaddr_in from{};
#ifdef _WIN32
        int fromLen = sizeof(from);
        int ret = recvfrom(sock, buffer, sizeof(buffer) - 1, 0,
                           reinterpret_cast<SOCKADDR*>(&from), &fromLen);

        if (ret == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) {
                continue;
            }
            if (!g_running.load()) {
                break;
            }
            std::cerr << "[MakcuRelay] recvfrom() error: " << err << std::endl;
            continue;
        }
#else
        socklen_t fromLen = sizeof(from);
        ssize_t ret = recvfrom(sock, buffer, sizeof(buffer) - 1, 0,
                               reinterpret_cast<struct sockaddr*>(&from), &fromLen);

        if (ret < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            if (!g_running.load()) {
                break;
            }
            std::cerr << "[MakcuRelay] recvfrom() error: " << strerror(errno) << std::endl;
            continue;
        }
#endif

        if (ret <= 0) {
            continue;
        }

        buffer[ret] = '\0';
        std::string msg(buffer);

        while (!msg.empty() && (msg.back() == '\r' || msg.back() == '\n' || msg.back() == ' ')) {
            msg.pop_back();
        }

        // Simple UDP ping support: echo PING back as-is so the sender
        // can measure round-trip latency. No per-packet logging to avoid jitter.
        if (msg.rfind("PING", 0) == 0) {
#ifdef _WIN32
            sendto(sock, buffer, ret, 0,
                   reinterpret_cast<SOCKADDR*>(&from), fromLen);
#else
            sendto(sock, buffer, ret, 0,
                   reinterpret_cast<struct sockaddr*>(&from), fromLen);
#endif
            continue;
        }

        if (msg.rfind("MOVE:", 0) == 0) {
            std::string rest = msg.substr(5);
            auto commaPos = rest.find(',');
            if (commaPos != std::string::npos) {
                try {
                    int dx = std::stoi(rest.substr(0, commaPos));
                    int dy = std::stoi(rest.substr(commaPos + 1));
                    makcu.move(dx, dy);
                } catch (...) {
                    std::cerr << "[MakcuRelay] Invalid MOVE payload: " << msg << std::endl;
                }
            }
        } else if (msg.rfind("PRESS:", 0) == 0) {
            std::string btn = msg.substr(6);
            if (btn == "LEFT") {
                makcu.press(1);
            } else if (btn == "RIGHT") {
                makcu.press(2);
            }
        } else if (msg.rfind("RELEASE:", 0) == 0) {
            std::string btn = msg.substr(8);
            if (btn == "LEFT") {
                makcu.release(1);
            } else if (btn == "RIGHT") {
                makcu.release(2);
            }
        } else if (msg.rfind("CLICK:", 0) == 0) {
            std::string btn = msg.substr(6);
            if (btn == "LEFT") {
                makcu.click(1);
            } else if (btn == "RIGHT") {
                makcu.click(2);
            }
        }
    }

#ifdef _WIN32
    if (g_broadcastSock != INVALID_SOCKET) {
        closesocket(g_broadcastSock);
        g_broadcastSock = INVALID_SOCKET;
    }
    closesocket(sock);
    WSACleanup();
#else
    if (g_broadcastSock >= 0) {
        close(g_broadcastSock);
        g_broadcastSock = -1;
    }
    close(sock);
#endif

    std::cout << "[MakcuRelay] Exiting.\n";
    return 0;
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
    return runRelay(__argc, __argv);
}
#else
int main(int argc, char** argv)
{
    return runRelay(argc, argv);
}
#endif
