#ifndef INPUT_METHOD_H
#define INPUT_METHOD_H

#include "SerialConnection.h"
#include "ghub.h"
#include "kmboxNet.h"
#include "rzctl.h"
#include <filesystem>
#include <iostream>
#include <Windows.h>

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
        // We have no direct "isOpen()" check, so assume init succeeded
        return true;
    }
};

// Razer mouse input implementation
class RZInputMethod : public InputMethod {
private:
    RZControl* rz_control_ = nullptr; // Pointer to the RZControl instance
    bool initialized_ = false;
    HMODULE rz_module_ = NULL; // Store module handle for explicit FreeLibrary

public:
    // Constructor now automatically finds DLL path
    explicit RZInputMethod() {
        char exe_path_buffer[MAX_PATH];
        GetModuleFileNameA(NULL, exe_path_buffer, MAX_PATH);
        std::filesystem::path base_dir = std::filesystem::path(exe_path_buffer).parent_path();
        std::filesystem::path dll_path = base_dir / "rzctl.dll";
        std::wstring w_dll_path = dll_path.wstring(); // RZControl constructor expects wstring

        try {
            // Explicitly load library here to manage handle and check existence
            rz_module_ = LoadLibraryW(w_dll_path.c_str());
            if (rz_module_ == NULL) {
                std::cerr << "[Razer] Failed to load rzctl.dll from: " << dll_path.string() << std::endl;
                throw std::runtime_error("rzctl.dll not found or failed to load.");
            }

            // Pass the path to RZControl (assuming RZControl still takes path)
            // If RZControl is also modified, this needs adjustment
            rz_control_ = new RZControl(w_dll_path); // Pass the found path
            initialized_ = rz_control_->initialize();
            if (!initialized_) {
                 std::cerr << "[Razer] RZControl->initialize() failed!" << std::endl;
                 throw std::runtime_error("RZControl initialization failed.");
            }
            std::cout << "[Razer] rzctl.dll loaded and initialized successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Razer] RZInputMethod initialization failed: " << e.what() 
                      << std::endl;
            delete rz_control_; // Clean up partially created object if any step failed
            rz_control_ = nullptr;
            if (rz_module_) { // Free library if loaded but init failed
                FreeLibrary(rz_module_);
                rz_module_ = NULL;
            }
            initialized_ = false;
            throw; 
        }
    }

    ~RZInputMethod() override {
        delete rz_control_;
        if (rz_module_) { // Ensure library is freed
            FreeLibrary(rz_module_);
        }
    }

    // Disable copy/move semantics for simplicity if managing raw pointer
    RZInputMethod(const RZInputMethod&) = delete;
    RZInputMethod& operator=(const RZInputMethod&) = delete;
    RZInputMethod(RZInputMethod&&) = delete;
    RZInputMethod& operator=(RZInputMethod&&) = delete;

    void move(int x, int y) override {
        if (initialized_ && rz_control_) {
            // Use Razer DLL method
            rz_control_->moveMouse(x, y, true); // Assuming relative movement
        }
    }

    void press() override {
        if (initialized_ && rz_control_) {
            // Use Razer DLL method
            rz_control_->mouseClick(MouseClick::LEFT_DOWN);
        }
    }

    void release() override {
        if (initialized_ && rz_control_) {
            // Use Razer DLL method
            rz_control_->mouseClick(MouseClick::LEFT_UP);
        }
    }

    bool isValid() const override {
        // Method is considered "valid" only if the DLL initialized successfully
        return (initialized_ && rz_control_ != nullptr);
    }
};

#endif // INPUT_METHOD_H
