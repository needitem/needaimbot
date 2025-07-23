#ifndef THREAD_UTILS_H
#define THREAD_UTILS_H

#pragma warning(push)
#pragma warning(disable: 4996 4267 4244 4305 4018 4101 4800)

#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <vector>
#include <functional>
#include <queue>
#include <future>
#include <array>

namespace ThreadUtils {
    
    // High-performance reader-writer lock
    class ReadWriteLock {
    private:
        mutable std::shared_mutex mutex_;
        
    public:
        // RAII lock guards
        class ReadLock {
        private:
            std::shared_lock<std::shared_mutex> lock_;
        public:
            explicit ReadLock(const ReadWriteLock& rwlock) 
                : lock_(rwlock.mutex_) {}
        };
        
        class WriteLock {
        private:
            std::unique_lock<std::shared_mutex> lock_;
        public:
            explicit WriteLock(const ReadWriteLock& rwlock) 
                : lock_(rwlock.mutex_) {}
        };
        
        ReadLock acquireRead() const { return ReadLock(*this); }
        WriteLock acquireWrite() const { return WriteLock(*this); }
    };
    
    // Lock-free circular buffer for high-frequency data
    template<typename T, size_t Size>
    class LockFreeCircularBuffer {
    private:
        alignas(64) std::atomic<size_t> head_{0};
        alignas(64) std::atomic<size_t> tail_{0};
        alignas(64) std::array<T, Size> buffer_;
        
        static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
        static constexpr size_t MASK = Size - 1;
        
    public:
        bool push(const T& item) {
            const size_t current_tail = tail_.load(std::memory_order_relaxed);
            const size_t next_tail = (current_tail + 1) & MASK;
            
            if (next_tail == head_.load(std::memory_order_acquire)) {
                return false; // Buffer full
            }
            
            buffer_[current_tail] = item;
            tail_.store(next_tail, std::memory_order_release);
            return true;
        }
        
        bool pop(T& item) {
            const size_t current_head = head_.load(std::memory_order_relaxed);
            
            if (current_head == tail_.load(std::memory_order_acquire)) {
                return false; // Buffer empty
            }
            
            item = buffer_[current_head];
            head_.store((current_head + 1) & MASK, std::memory_order_release);
            return true;
        }
        
        size_t size() const {
            const size_t current_tail = tail_.load(std::memory_order_acquire);
            const size_t current_head = head_.load(std::memory_order_acquire);
            return (current_tail - current_head) & MASK;
        }
        
        bool empty() const {
            return head_.load(std::memory_order_acquire) == 
                   tail_.load(std::memory_order_acquire);
        }
        
        bool full() const {
            const size_t current_tail = tail_.load(std::memory_order_acquire);
            const size_t next_tail = (current_tail + 1) & MASK;
            return next_tail == head_.load(std::memory_order_acquire);
        }
    };
    
    // Thread pool for parallel processing
    class ThreadPool {
    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        mutable std::mutex queue_mutex_;
        std::condition_variable condition_;
        std::atomic<bool> stop_{false};
        
    public:
        explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
            for (size_t i = 0; i < num_threads; ++i) {
                workers_.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex_);
                            condition_.wait(lock, [this] { 
                                return stop_.load() || !tasks_.empty(); 
                            });
                            
                            if (stop_.load() && tasks_.empty()) {
                                return;
                            }
                            
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        
                        task();
                    }
                });
            }
        }
        
        ~ThreadPool() {
            stop_.store(true);
            condition_.notify_all();
            
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }
        
        template<typename F, typename... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
            using return_type = typename std::result_of<F(Args...)>::type;
            
            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );
            
            auto result = task->get_future();
            
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (stop_.load()) {
                    throw std::runtime_error("enqueue on stopped ThreadPool");
                }
                
                tasks_.emplace([task] { (*task)(); });
            }
            
            condition_.notify_one();
            return result;
        }
        
        size_t getThreadCount() const { return workers_.size(); }
        size_t getPendingTasks() const {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            return tasks_.size();
        }
    };
    
    // Spinlock for very short critical sections
    class SpinLock {
    private:
        std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
        
    public:
        void lock() {
            while (flag_.test_and_set(std::memory_order_acquire)) {
                // Busy wait with pause for better performance
                #if defined(_MSC_VER)
                _mm_pause();
                #elif defined(__GNUC__)
                __builtin_ia32_pause();
                #endif
            }
        }
        
        void unlock() {
            flag_.clear(std::memory_order_release);
        }
        
        bool try_lock() {
            return !flag_.test_and_set(std::memory_order_acquire);
        }
    };
    
    // Performance counter for monitoring
    class PerformanceCounter {
    private:
        std::atomic<uint64_t> count_{0};
        std::atomic<uint64_t> total_time_ns_{0};
        mutable std::mutex stats_mutex_;
        
    public:
        class Timer {
        private:
            PerformanceCounter* counter_;
            std::chrono::high_resolution_clock::time_point start_;
            
        public:
            explicit Timer(PerformanceCounter* counter) 
                : counter_(counter), start_(std::chrono::high_resolution_clock::now()) {}
            
            ~Timer() {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
                counter_->record(duration.count());
            }
        };
        
        void record(uint64_t duration_ns) {
            count_.fetch_add(1, std::memory_order_relaxed);
            total_time_ns_.fetch_add(duration_ns, std::memory_order_relaxed);
        }
        
        Timer startTimer() { return Timer(this); }
        
        uint64_t getCount() const { return count_.load(std::memory_order_relaxed); }
        uint64_t getTotalTimeNs() const { return total_time_ns_.load(std::memory_order_relaxed); }
        
        double getAverageTimeMs() const {
            uint64_t count = getCount();
            if (count == 0) return 0.0;
            return (getTotalTimeNs() / static_cast<double>(count)) / 1000000.0;
        }
        
        void reset() {
            count_.store(0, std::memory_order_relaxed);
            total_time_ns_.store(0, std::memory_order_relaxed);
        }
    };
}

#pragma warning(pop)

#endif // THREAD_UTILS_H