#ifndef SYNC_MANAGER_H
#define SYNC_MANAGER_H

#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <functional>

class SyncManager {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;
    
    // Thread-safe event notification system
    class Event {
    public:
        void notify() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                signaled_ = true;
                version_++;
            }
            cv_.notify_all();
        }
        
        void reset() {
            std::lock_guard<std::mutex> lock(mutex_);
            signaled_ = false;
        }
        
        bool wait() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return signaled_; });
            return true;
        }
        
        template<typename Rep, typename Period>
        bool wait_for(const std::chrono::duration<Rep, Period>& timeout) {
            std::unique_lock<std::mutex> lock(mutex_);
            return cv_.wait_for(lock, timeout, [this] { return signaled_; });
        }
        
        int version() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return version_;
        }
        
    private:
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        bool signaled_ = false;
        int version_ = 0;
    };
    
    // Scoped performance timer
    class ScopedTimer {
    public:
        ScopedTimer(std::atomic<float>& result_ms, const std::string& name = "")
            : result_(result_ms), name_(name), start_(Clock::now()) {}
        
        ~ScopedTimer() {
            auto end = Clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            result_ = duration.count() / 1000.0f;
            
            if (!name_.empty() && result_ > 10.0f) {  // Log if takes more than 10ms
                std::cout << "[Performance] " << name_ << " took " << result_.load() << "ms" << std::endl;
            }
        }
        
    private:
        std::atomic<float>& result_;
        std::string name_;
        TimePoint start_;
    };
    
    // Thread-safe data exchange
    template<typename T>
    class DataExchange {
    public:
        void write(T&& data) {
            std::lock_guard<std::mutex> lock(mutex_);
            data_ = std::forward<T>(data);
            version_++;
            has_data_ = true;
        }
        
        bool read(T& out_data, int& out_version) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!has_data_) {
                return false;
            }
            out_data = data_;
            out_version = version_;
            return true;
        }
        
        bool hasNewData(int last_version) const {
            std::lock_guard<std::mutex> lock(mutex_);
            return has_data_ && version_ > last_version;
        }
        
        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            data_ = T{};
            has_data_ = false;
        }
        
    private:
        mutable std::mutex mutex_;
        T data_{};
        int version_ = 0;
        bool has_data_ = false;
    };
    
    // Rate limiter for preventing excessive operations
    class RateLimiter {
    public:
        RateLimiter(std::chrono::milliseconds min_interval)
            : min_interval_(min_interval) {}
        
        bool shouldAllow() {
            auto now = Clock::now();
            std::lock_guard<std::mutex> lock(mutex_);
            
            if (now - last_time_ >= min_interval_) {
                last_time_ = now;
                return true;
            }
            return false;
        }
        
        void setInterval(std::chrono::milliseconds interval) {
            std::lock_guard<std::mutex> lock(mutex_);
            min_interval_ = interval;
        }
        
    private:
        std::mutex mutex_;
        std::chrono::milliseconds min_interval_;
        TimePoint last_time_{};
    };
};

#endif // SYNC_MANAGER_H