#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include "windows_headers.h"
#include <thread>
#include <atomic>
#include <functional>
#include <string>
#include <iostream>

class ThreadManager {
public:
    using ThreadFunc = std::function<void()>;
    
    ThreadManager(const std::string& name, ThreadFunc func, int priority = THREAD_PRIORITY_NORMAL)
        : thread_name_(name), running_(false), priority_(priority) {
        thread_func_ = std::move(func);
    }
    
    ~ThreadManager() {
        stop();
    }
    
    // Prevent copying
    ThreadManager(const ThreadManager&) = delete;
    ThreadManager& operator=(const ThreadManager&) = delete;
    
    // Allow moving
    ThreadManager(ThreadManager&& other) noexcept
        : thread_(std::move(other.thread_)),
          thread_name_(std::move(other.thread_name_)),
          thread_func_(std::move(other.thread_func_)),
          running_(other.running_.load()),
          priority_(other.priority_) {
        other.running_ = false;
    }
    
    ThreadManager& operator=(ThreadManager&& other) noexcept {
        if (this != &other) {
            stop();
            thread_ = std::move(other.thread_);
            thread_name_ = std::move(other.thread_name_);
            thread_func_ = std::move(other.thread_func_);
            running_ = other.running_.load();
            priority_ = other.priority_;
            other.running_ = false;
        }
        return *this;
    }
    
    bool start() {
        if (running_) {
            return false;
        }
        
        running_ = true;
        thread_ = std::thread([this]() {
            std::cout << "[Thread] " << thread_name_ << " started." << std::endl;
            
            // Set thread priority
            SetThreadPriority(GetCurrentThread(), priority_);
            
            // Set thread name for debugging
            setThreadName(thread_name_);
            
            try {
                thread_func_();
            } catch (const std::exception& e) {
                std::cerr << "[Thread] " << thread_name_ << " exception: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Thread] " << thread_name_ << " unknown exception." << std::endl;
            }
            
            std::cout << "[Thread] " << thread_name_ << " exiting." << std::endl;
        });
        
        return true;
    }
    
    void stop() {
        if (!running_) {
            return;
        }
        
        running_ = false;
        
        if (thread_.joinable()) {
            std::cout << "[Thread] Waiting for " << thread_name_ << " to finish..." << std::endl;
            thread_.join();
            std::cout << "[Thread] " << thread_name_ << " stopped." << std::endl;
        }
    }
    
    bool isRunning() const {
        return running_;
    }
    
    void setPriority(int priority) {
        priority_ = priority;
        if (thread_.joinable()) {
            SetThreadPriority(thread_.native_handle(), priority);
        }
    }
    
private:
    void setThreadName(const std::string& name) {
        // Windows 10 version 1607+: Use SetThreadDescription
        typedef HRESULT (WINAPI *SetThreadDescriptionFunc)(HANDLE, PCWSTR);
        static auto setThreadDescription = reinterpret_cast<SetThreadDescriptionFunc>(
            GetProcAddress(GetModuleHandle(L"kernel32.dll"), "SetThreadDescription"));
        
        if (setThreadDescription) {
            std::wstring wideName(name.begin(), name.end());
            setThreadDescription(GetCurrentThread(), wideName.c_str());
        }
    }
    
    std::thread thread_;
    std::string thread_name_;
    ThreadFunc thread_func_;
    std::atomic<bool> running_;
    int priority_;
};

#endif // THREAD_MANAGER_H