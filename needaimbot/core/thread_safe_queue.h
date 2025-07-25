#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>

namespace NeedAimbot {

template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;

    // Delete copy operations
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // Allow move operations
    ThreadSafeQueue(ThreadSafeQueue&&) = default;
    ThreadSafeQueue& operator=(ThreadSafeQueue&&) = default;

    // Push item to queue
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }  // Lock released here
        cv_.notify_one();  // Notify without holding lock - prevents unnecessary contention
    }

    // Try to push with size limit
    bool try_push(T item, size_t max_size) {
        bool pushed = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.size() < max_size) {
                queue_.push(std::move(item));
                pushed = true;
            }
        }  // Lock released here
        if (pushed) {
            cv_.notify_one();  // Notify without holding lock
        }
        return pushed;
    }

    // Pop item with timeout
    std::optional<T> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || stop_; })) {
            return std::nullopt;  // Timeout
        }
        
        if (stop_ || queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Try to pop without blocking
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Get size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // Check if empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // Clear all items
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        // More efficient than creating empty queue and swapping
        while (!queue_.empty()) {
            queue_.pop();
        }
    }

    // Stop the queue (unblock all waiting threads)
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    // Reset stop flag
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = false;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    bool stop_ = false;
};

// Specialized version for pointer types with automatic cleanup
template<typename T>
class ThreadSafeQueue<std::unique_ptr<T>> {
public:
    using PtrType = std::unique_ptr<T>;

    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() {
        clear();  // Ensure all pointers are properly deleted
    }

    // Delete copy operations
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    void push(PtrType item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    PtrType pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || stop_; })) {
            return nullptr;  // Timeout
        }
        
        if (stop_ || queue_.empty()) {
            return nullptr;
        }

        PtrType item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    PtrType try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return nullptr;
        }

        PtrType item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();  // unique_ptr will automatically delete
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = false;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<PtrType> queue_;
    bool stop_ = false;
};

} // namespace NeedAimbot

#endif // THREAD_SAFE_QUEUE_H