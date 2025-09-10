#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include "windows_headers.h"
#include <thread>
#include <atomic>
#include <functional>
#include <string>
#include <iostream>
#include <chrono>
#include <condition_variable>
#include <mutex>

class ThreadManager {
public:
    using ThreadFunc = std::function<void()>;
    
    ThreadManager(const std::string& name, ThreadFunc func, int affinity_core = -1)
        : thread_name_(name), running_(false), thread_finished_(false), affinity_core_(affinity_core) {
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
          running_(other.running_.load()) {
        other.running_ = false;
    }
    
    ThreadManager& operator=(ThreadManager&& other) noexcept {
        if (this != &other) {
            stop();
            thread_ = std::move(other.thread_);
            thread_name_ = std::move(other.thread_name_);
            thread_func_ = std::move(other.thread_func_);
            running_ = other.running_.load();
            other.running_ = false;
        }
        return *this;
    }
    
    bool start() {
        if (running_) {
            return false;
        }
        
        running_ = true;
        thread_finished_ = false;
        thread_ = std::thread([this]() {
            
            // Set thread name for debugging
            setThreadName(thread_name_);
            
            // OPTIMIZATION: Set thread affinity for better cache locality
            if (affinity_core_ >= 0) {
                DWORD_PTR mask = 1ULL << affinity_core_;
                SetThreadAffinityMask(GetCurrentThread(), mask);
            }
            
            try {
                thread_func_();
            } catch (const std::exception& e) {
#ifdef _DEBUG
                std::cerr << "[Thread] " << thread_name_ << " exception: " << e.what() << std::endl;
#else
                (void)e;  // Suppress unused variable warning
#endif
            } catch (...) {
#ifdef _DEBUG
                std::cerr << "[Thread] " << thread_name_ << " unknown exception." << std::endl;
#endif
            }
            
            // Signal thread completion
            {
                std::lock_guard<std::mutex> lock(finish_mutex_);
                thread_finished_ = true;
            }
            finish_cv_.notify_all();
        });
        
        return true;
    }
    
    void stop() {
        if (!running_) {
            return;
        }
        
        running_ = false;
        
        if (thread_.joinable()) {
            // Use condition variable for efficient waiting
            std::unique_lock<std::mutex> lock(finish_mutex_);
            if (!finish_cv_.wait_for(lock, std::chrono::milliseconds(500),
                                     [this] { return thread_finished_.load(); })) {
                // Timeout: force termination
#ifdef _DEBUG
                std::cerr << "[Thread] " << thread_name_ << " timeout, forcing termination..." << std::endl;
#endif
                HANDLE hThread = thread_.native_handle();
                if (hThread != INVALID_HANDLE_VALUE && hThread != nullptr) {
                    TerminateThread(hThread, 1);
                }
                if (thread_.joinable()) {
                    thread_.detach();
                }
            } else {
                // Thread finished normally
                if (thread_.joinable()) {
                    thread_.join();
                }
            }
        }
    }
    
    bool isRunning() const {
        return running_;
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
    std::atomic<bool> thread_finished_;
    std::condition_variable finish_cv_;
    std::mutex finish_mutex_;
    int affinity_core_; // -1 for no affinity, >=0 for specific core
};

#endif // THREAD_MANAGER_H