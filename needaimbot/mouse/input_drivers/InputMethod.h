#ifndef INPUT_METHOD_H
#define INPUT_METHOD_H

#include "SerialConnection.h"
#include "MakcuConnection.h"
#include "ghub.h"
#include "kmboxNet.h"
#include "rzctl.h"
#include <filesystem>
#include <iostream>
#include <Windows.h>


class InputMethod
{
public:
    virtual ~InputMethod() = default;
    virtual void move(int x, int y) = 0;
    virtual void press() = 0;
    virtual void release() = 0;
    virtual bool isValid() const = 0;
};


class SerialInputMethod : public InputMethod
{
public:
    explicit SerialInputMethod(SerialConnection *serial) : serial_(serial) {}
    ~SerialInputMethod() override
    {
        
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


class MakcuInputMethod : public InputMethod
{
public:
    explicit MakcuInputMethod(MakcuConnection *makcu) : makcu_(makcu) {}
    ~MakcuInputMethod() override
    {
        
    }

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
            makcu_->press(1); // Left mouse button
        }
    }

    void release() override
    {
        if (makcu_ && makcu_->isOpen())
        {
            makcu_->release(1); // Left mouse button
        }
    }

    bool isValid() const override
    {
        return makcu_ && makcu_->isOpen();
    }

private:
    MakcuConnection *makcu_;
};


class GHubInputMethod : public InputMethod
{
public:
    explicit GHubInputMethod(GhubMouse *ghub) : ghub_(ghub) {}
    ~GHubInputMethod() override
    {
        
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


class Win32InputMethod : public InputMethod
{
private:
    // Batch multiple inputs for better performance
    static constexpr size_t MAX_BATCH_SIZE = 64;
    std::vector<INPUT> input_batch;
    
    void flushBatch() {
        if (!input_batch.empty()) {
            SendInput(static_cast<UINT>(input_batch.size()), input_batch.data(), sizeof(INPUT));
            input_batch.clear();
        }
    }
    
public:
    Win32InputMethod() {
        input_batch.reserve(MAX_BATCH_SIZE);
    }
    
    ~Win32InputMethod() override {
        flushBatch();
    }

    void move(int x, int y) override
    {
        // For mouse movement, send immediately for responsiveness
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
        return true; 
    }
};



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


class RZInputMethod : public InputMethod {
private:
    RZControl* rz_control_ = nullptr; 
    bool initialized_ = false;
    HMODULE rz_module_ = NULL; 

public:
    
    explicit RZInputMethod() {
        char exe_path_buffer[MAX_PATH];
        GetModuleFileNameA(NULL, exe_path_buffer, MAX_PATH);
        std::filesystem::path base_dir = std::filesystem::path(exe_path_buffer).parent_path();
        std::filesystem::path dll_path = base_dir / "rzctl.dll";
        std::wstring w_dll_path = dll_path.wstring(); 

        try {
            
            rz_module_ = LoadLibraryW(w_dll_path.c_str());
            if (rz_module_ == NULL) {
                std::cerr << "[Razer] Failed to load rzctl.dll from: " << dll_path.string() << std::endl;
                throw std::runtime_error("rzctl.dll not found or failed to load.");
            }

            
            
            rz_control_ = new RZControl(w_dll_path); 
            initialized_ = rz_control_->initialize();
            if (!initialized_) {
                 std::cerr << "[Razer] RZControl->initialize() failed!" << std::endl;
                 throw std::runtime_error("RZControl initialization failed.");
            }
            std::cout << "[Razer] rzctl.dll loaded and initialized successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Razer] RZInputMethod initialization failed: " << e.what() 
                      << std::endl;
            delete rz_control_; 
            rz_control_ = nullptr;
            if (rz_module_) { 
                FreeLibrary(rz_module_);
                rz_module_ = NULL;
            }
            initialized_ = false;
            throw; 
        }
    }

    ~RZInputMethod() override {
        delete rz_control_;
        if (rz_module_) { 
            FreeLibrary(rz_module_);
        }
    }

    
    RZInputMethod(const RZInputMethod&) = delete;
    RZInputMethod& operator=(const RZInputMethod&) = delete;
    RZInputMethod(RZInputMethod&&) = delete;
    RZInputMethod& operator=(RZInputMethod&&) = delete;

    void move(int x, int y) override {
        if (initialized_ && rz_control_) {
            
            rz_control_->moveMouse(x, y, true); 
        }
    }

    void press() override {
        if (initialized_ && rz_control_) {
            
            rz_control_->mouseClick(MouseClick::LEFT_DOWN);
        }
    }

    void release() override {
        if (initialized_ && rz_control_) {
            
            rz_control_->mouseClick(MouseClick::LEFT_UP);
        }
    }

    bool isValid() const override {
        
        return (initialized_ && rz_control_ != nullptr);
    }
};

#endif 

