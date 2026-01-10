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

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

class ThreadManager {
public:
    using ThreadFunc = std::function<void()>;

    // Thread priority levels for better control under high CPU load
    enum class Priority {
#ifdef _WIN32
        IDLE = THREAD_PRIORITY_IDLE,
        LOWEST = THREAD_PRIORITY_LOWEST,
        BELOW_NORMAL = THREAD_PRIORITY_BELOW_NORMAL,
        NORMAL = THREAD_PRIORITY_NORMAL,
        ABOVE_NORMAL = THREAD_PRIORITY_ABOVE_NORMAL,
        HIGHEST = THREAD_PRIORITY_HIGHEST,
        TIME_CRITICAL = THREAD_PRIORITY_TIME_CRITICAL
#else
        IDLE = -15,
        LOWEST = -2,
        BELOW_NORMAL = -1,
        NORMAL = 0,
        ABOVE_NORMAL = 1,
        HIGHEST = 2,
        TIME_CRITICAL = 15
#endif
    };

    ThreadManager(const std::string& name, ThreadFunc func, int affinity_core = -1, Priority priority = Priority::NORMAL)
        : thread_name_(name), running_(false), thread_finished_(false), affinity_core_(affinity_core), priority_(priority) {
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

            // Set thread affinity for better cache locality
            if (affinity_core_ >= 0) {
                setThreadAffinity(affinity_core_);
            }

            // Set thread priority to reduce jitter under high CPU load
            setThreadPriority(priority_);

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
#ifdef _WIN32
                HANDLE hThread = thread_.native_handle();
                if (hThread != INVALID_HANDLE_VALUE && hThread != nullptr) {
                    TerminateThread(hThread, 1);
                }
#else
                pthread_cancel(thread_.native_handle());
#endif
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
#ifdef _WIN32
        // Windows 10 version 1607+: Use SetThreadDescription
        typedef HRESULT (WINAPI *SetThreadDescriptionFunc)(HANDLE, PCWSTR);
        static auto setThreadDescription = reinterpret_cast<SetThreadDescriptionFunc>(
            GetProcAddress(GetModuleHandle(L"kernel32.dll"), "SetThreadDescription"));

        if (setThreadDescription) {
            std::wstring wideName(name.begin(), name.end());
            setThreadDescription(GetCurrentThread(), wideName.c_str());
        }
#else
        // Linux: Use pthread_setname_np (max 16 chars including null)
        std::string truncated = name.substr(0, 15);
        pthread_setname_np(pthread_self(), truncated.c_str());
#endif
    }

    void setThreadAffinity(int core) {
#ifdef _WIN32
        DWORD_PTR mask = 1ULL << core;
        SetThreadAffinityMask(GetCurrentThread(), mask);
#else
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
    }

    void setThreadPriority(Priority priority) {
#ifdef _WIN32
        SetThreadPriority(GetCurrentThread(), static_cast<int>(priority));
#else
        // Linux: Use nice value or SCHED_RR for real-time
        int nice_val = 0;
        switch (priority) {
            case Priority::IDLE: nice_val = 19; break;
            case Priority::LOWEST: nice_val = 10; break;
            case Priority::BELOW_NORMAL: nice_val = 5; break;
            case Priority::NORMAL: nice_val = 0; break;
            case Priority::ABOVE_NORMAL: nice_val = -5; break;
            case Priority::HIGHEST: nice_val = -10; break;
            case Priority::TIME_CRITICAL: nice_val = -20; break;
        }
        // Note: nice() requires root for negative values
        (void)nice_val;  // Silently ignore on Linux unless we implement setpriority
#endif
    }

    std::thread thread_;
    std::string thread_name_;
    ThreadFunc thread_func_;
    std::atomic<bool> running_;
    std::atomic<bool> thread_finished_;
    std::condition_variable finish_cv_;
    std::mutex finish_mutex_;
    int affinity_core_; // -1 for no affinity, >=0 for specific core
    Priority priority_;  // Thread priority for scheduling
};

#endif // THREAD_MANAGER_H
