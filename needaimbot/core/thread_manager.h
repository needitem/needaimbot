#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include "windows_headers.h"
#include <thread>
#include <atomic>
#include <functional>
#include <string>
#include <iostream>
#include <chrono>

class ThreadManager {
public:
    using ThreadFunc = std::function<void()>;
    
    ThreadManager(const std::string& name, ThreadFunc func, int affinity_core = -1)
        : thread_name_(name), running_(false), affinity_core_(affinity_core) {
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
        thread_ = std::thread([this]() {
            std::cout << "[Thread] " << thread_name_ << " started." << std::endl;
            
            // Set thread name for debugging
            setThreadName(thread_name_);
            
            // OPTIMIZATION: Set thread affinity for better cache locality
            if (affinity_core_ >= 0) {
                DWORD_PTR mask = 1ULL << affinity_core_;
                if (SetThreadAffinityMask(GetCurrentThread(), mask)) {
                    std::cout << "[Thread] " << thread_name_ << " affinity set to core " << affinity_core_ << std::endl;
                } else {
                    std::cerr << "[Thread] Failed to set affinity for " << thread_name_ << " to core " << affinity_core_ << std::endl;
                }
            }
            
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
            
            // Try graceful shutdown with timeout
            auto start_time = std::chrono::steady_clock::now();
            const auto timeout = std::chrono::milliseconds(500);  // 500ms 타임아웃
            
            // Create a thread to wait for join with timeout
            std::atomic<bool> thread_finished{false};
            std::thread wait_thread([this, &thread_finished]() {
                if (thread_.joinable()) {
                    thread_.join();
                    thread_finished = true;
                }
            });
            
            // Wait for thread to finish or timeout
            while (!thread_finished) {
                if (std::chrono::steady_clock::now() - start_time > timeout) {
                    std::cerr << "[Thread] " << thread_name_ << " did not finish within timeout, forcing termination..." << std::endl;
                    
                    // Force terminate the thread (Windows specific)
                    HANDLE hThread = thread_.native_handle();
                    if (hThread != INVALID_HANDLE_VALUE && hThread != nullptr) {
                        // Attempt to terminate the thread forcefully
                        if (TerminateThread(hThread, 1)) {
                            std::cerr << "[Thread] " << thread_name_ << " forcefully terminated." << std::endl;
                        } else {
                            std::cerr << "[Thread] Failed to terminate " << thread_name_ << " (error: " << GetLastError() << ")" << std::endl;
                        }
                    }
                    
                    // Detach the thread to avoid join issues
                    if (thread_.joinable()) {
                        thread_.detach();
                    }
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Clean up wait thread
            if (wait_thread.joinable()) {
                wait_thread.join();
            }
            
            if (thread_finished) {
                std::cout << "[Thread] " << thread_name_ << " stopped gracefully." << std::endl;
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
    int affinity_core_; // -1 for no affinity, >=0 for specific core
};

#endif // THREAD_MANAGER_H