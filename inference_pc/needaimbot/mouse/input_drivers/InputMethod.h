#ifndef INPUT_METHOD_H
#define INPUT_METHOD_H

#include "../../core/windows_headers.h"
#include "MakcuConnection.h"
#include "kmboxNet.h"
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <memory>

// Base class for all input methods
class InputMethod
{
public:
    virtual ~InputMethod() = default;
    virtual void move(int x, int y) = 0;
    virtual void press() = 0;
    virtual void release() = 0;
    virtual bool isValid() const = 0;
};

// Direct serial connection to Makcu device (device connected to this PC)
class MakcuSerialInputMethod : public InputMethod
{
public:
    explicit MakcuSerialInputMethod(MakcuConnection* makcu) : makcu_(makcu) {}
    ~MakcuSerialInputMethod() override = default;

    void move(int x, int y) override
    {
        if (makcu_ && makcu_->isOpen())
        {
            makcu_->move(x, y);
        }
    }

    void press() override
    {
        if (makcu_ && makcu_->isOpen())
        {
            makcu_->press(1);  // 1 = left button
        }
    }

    void release() override
    {
        if (makcu_ && makcu_->isOpen())
        {
            makcu_->release(1);  // 1 = left button
        }
    }

    bool isValid() const override
    {
        return makcu_ && makcu_->isOpen();
    }

private:
    MakcuConnection* makcu_;
};

// UDP-based Makcu (for network relay to another PC - legacy support)
class MakcuNetInputMethod : public InputMethod
{
public:
    MakcuNetInputMethod(const std::string& remoteIp, int remotePort)
        : sock_(INVALID_SOCKET), valid_(false)
    {
        // Initialize Winsock locally for this process (idempotent)
        WSADATA wsaData{};
        int wsaErr = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (wsaErr != 0) {
            std::cerr << "[MakcuNet] WSAStartup failed: " << wsaErr << std::endl;
            return;
        }

        // Basic UDP client to second PC running MakcuRelay
        sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET) {
            std::cerr << "[MakcuNet] socket() failed: " << WSAGetLastError() << std::endl;
            return;
        }

        int enable = 1;
        setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&enable), sizeof(enable));

        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(static_cast<u_short>(remotePort));

        if (InetPtonA(AF_INET, remoteIp.c_str(), &addr_.sin_addr) != 1) {
            std::cerr << "[MakcuNet] Invalid remote IP: " << remoteIp << std::endl;
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
            return;
        }

        valid_ = true;
        std::cout << "[MakcuNet] Using remote relay at " << remoteIp << ":" << remotePort << std::endl;
    }

    ~MakcuNetInputMethod() override
    {
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
        }
    }

    void move(int x, int y) override
    {
        if (!valid_ || sock_ == INVALID_SOCKET) return;
        if (x == 0 && y == 0) return;

        char buf[64];
        std::snprintf(buf, sizeof(buf), "MOVE:%d,%d", x, y);
        sendPacket(buf);
    }

    void press() override
    {
        if (!valid_ || sock_ == INVALID_SOCKET) return;
        sendPacket("PRESS:LEFT");
    }

    void release() override
    {
        if (!valid_ || sock_ == INVALID_SOCKET) return;
        sendPacket("RELEASE:LEFT");
    }

    bool isValid() const override
    {
        return valid_ && sock_ != INVALID_SOCKET;
    }

private:
    void sendPacket(const char* msg)
    {
        int len = static_cast<int>(std::strlen(msg));
        sendto(sock_, msg, len, 0,
               reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
    }

    SOCKET sock_;
    sockaddr_in addr_{};
    bool valid_;
};


// KMBox network input method
class KmboxInputMethod : public InputMethod {
public:
    KmboxInputMethod() = default;
    ~KmboxInputMethod() override = default;

    void move(int x, int y) override {
        kmNet_mouse_move(static_cast<short>(x), static_cast<short>(y));
    }

    void press() override {
        kmNet_mouse_left(1);
    }

    void release() override {
        kmNet_mouse_left(0);
    }

    bool isValid() const override {
        return true;
    }
};

#endif // INPUT_METHOD_H
