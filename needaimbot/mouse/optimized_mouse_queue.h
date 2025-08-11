#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <thread>

// Lock-free ring buffer optimized for cache-line performance
// Uses cache-line padding to prevent false sharing
template<typename T, size_t Size = 512>
class OptimizedMouseQueue {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t BUFFER_SIZE = Size;
    static_assert((BUFFER_SIZE & (BUFFER_SIZE - 1)) == 0, "Size must be power of 2");
    
    struct alignas(CACHE_LINE_SIZE) CachePaddedAtomic {
        std::atomic<size_t> value{0};
        char padding[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)];
    };
    
    // Cache-line aligned data
    alignas(CACHE_LINE_SIZE) T buffer[BUFFER_SIZE];
    CachePaddedAtomic head;  // Producer position
    CachePaddedAtomic tail;  // Consumer position
    
    // Pre-computed mask for fast modulo
    static constexpr size_t MASK = BUFFER_SIZE - 1;
    
public:
    OptimizedMouseQueue() = default;
    
    // Non-blocking enqueue
    bool tryEnqueue(const T& item) {
        size_t current_head = head.value.load(std::memory_order_relaxed);
        size_t next_head = (current_head + 1) & MASK;
        
        // Check if queue is full
        if (next_head == tail.value.load(std::memory_order_acquire)) {
            return false;  // Queue full
        }
        
        // Write item
        buffer[current_head] = item;
        
        // Update head with release semantics
        head.value.store(next_head, std::memory_order_release);
        return true;
    }
    
    // Non-blocking dequeue
    bool tryDequeue(T& item) {
        size_t current_tail = tail.value.load(std::memory_order_relaxed);
        
        // Check if queue is empty
        if (current_tail == head.value.load(std::memory_order_acquire)) {
            return false;  // Queue empty
        }
        
        // Read item
        item = buffer[current_tail];
        
        // Update tail with release semantics
        tail.value.store((current_tail + 1) & MASK, std::memory_order_release);
        return true;
    }
    
    // Batch enqueue for better throughput
    size_t tryEnqueueBatch(const T* items, size_t count) {
        size_t current_head = head.value.load(std::memory_order_relaxed);
        size_t current_tail = tail.value.load(std::memory_order_acquire);
        
        // Calculate available space
        size_t available = (current_tail - current_head - 1) & MASK;
        size_t to_write = (count < available) ? count : available;
        
        if (to_write == 0) return 0;
        
        // Copy items in batch (optimized for contiguous memory)
        for (size_t i = 0; i < to_write; ++i) {
            buffer[(current_head + i) & MASK] = items[i];
        }
        
        // Update head once
        head.value.store((current_head + to_write) & MASK, std::memory_order_release);
        return to_write;
    }
    
    // Check if empty without modifying state
    bool empty() const {
        return tail.value.load(std::memory_order_relaxed) == 
               head.value.load(std::memory_order_relaxed);
    }
    
    // Approximate size (may be slightly off due to concurrent access)
    size_t size() const {
        size_t h = head.value.load(std::memory_order_relaxed);
        size_t t = tail.value.load(std::memory_order_relaxed);
        return (h - t) & MASK;
    }
    
    // Wrapper methods for compatibility with MouseCommand
    bool enqueue(const T& item) {
        return tryEnqueue(item);
    }
    
    bool enqueue(T&& item) {
        return tryEnqueue(item);
    }
    
    bool tryDequeue(T& item, int timeout_ms) {
        // Simple busy-wait implementation for timeout
        auto start = std::chrono::steady_clock::now();
        auto timeout = std::chrono::milliseconds(timeout_ms);
        
        while (!tryDequeue(item)) {
            if (std::chrono::steady_clock::now() - start >= timeout) {
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    }
};

// Use the MouseCommand from lockless_queue.h
#include "lockless_queue.h"

// Type alias for the optimized queue using MouseCommand
using OptimizedMouseCommandQueue = OptimizedMouseQueue<MouseCommand, 512>;