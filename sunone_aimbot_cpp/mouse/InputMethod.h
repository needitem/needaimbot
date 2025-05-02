#ifndef INPUT_METHOD_H
#define INPUT_METHOD_H

#include "SerialConnection.h"
#include "ghub.h"
#include "kmboxNet.h"

// Interface for mouse input methods
class InputMethod
{
public:
    virtual ~InputMethod() = default;
    virtual void move(int x, int y) = 0;
    virtual void press() = 0;
    virtual void release() = 0;
    virtual bool isValid() const = 0;
};

// Mouse input implementation via serial connection (Arduino)
class SerialInputMethod : public InputMethod
{
public:
    explicit SerialInputMethod(SerialConnection *serial) : serial_(serial) {}
    ~SerialInputMethod() override
    {
        // Only maintaining reference, not deleting in destructor
    }

    void move(int x, int y) override
    {
        if (serial_ && serial_->isOpen())
        {
            serial_->move(x, y);
        }
    }

    void press() override
    {
        if (serial_ && serial_->isOpen())
        {
            serial_->press();
        }
    }

    void release() override
    {
        if (serial_ && serial_->isOpen())
        {
            serial_->release();
        }
    }

    bool isValid() const override
    {
        return serial_ && serial_->isOpen();
    }

private:
    SerialConnection *serial_;
};

// Mouse input implementation via Logitech G HUB
class GHubInputMethod : public InputMethod
{
public:
    explicit GHubInputMethod(GhubMouse *ghub) : ghub_(ghub) {}
    ~GHubInputMethod() override
    {
        // Only maintaining reference, not deleting in destructor
    }

    void move(int x, int y) override
    {
        if (ghub_)
        {
            ghub_->mouse_xy(x, y);
        }
    }

    void press() override
    {
        if (ghub_)
        {
            ghub_->mouse_down();
        }
    }

    void release() override
    {
        if (ghub_)
        {
            ghub_->mouse_up();
        }
    }

    bool isValid() const override
    {
        return ghub_ != nullptr;
    }

private:
    GhubMouse *ghub_;
};

// Default mouse input implementation via Windows API
class Win32InputMethod : public InputMethod
{
public:
    Win32InputMethod() = default;

    void move(int x, int y) override
    {
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        input.mi.dx = x;
        input.mi.dy = y;
        SendInput(1, &input, sizeof(INPUT));
    }

    void press() override
    {
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &input, sizeof(INPUT));
    }

    void release() override
    {
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
    }

    bool isValid() const override
    {
        return true; // Assuming Win32 API is always valid
    }
};


// kmboxNet-based mouse input implementation
class KmboxInputMethod : public InputMethod {
public:
    KmboxInputMethod() = default;
    ~KmboxInputMethod() override = default;

    void move(int x, int y) override {
        // kmNet_mouse_move takes SHORTs
        kmNet_mouse_move(static_cast<short>(x), static_cast<short>(y));
    }

    void press() override {
        // left button down
        kmNet_mouse_left(1);
    }

    void release() override {
        // left button up
        kmNet_mouse_left(0);
    }

    bool isValid() const override {
        // We have no direct “isOpen()” check, so assume init succeeded
        return true;
    }
};

#endif // INPUT_METHOD_H
