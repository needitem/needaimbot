#pragma once
#include <array>
#include <atomic>
#include <algorithm>

// Cache line size for most modern CPUs
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template<typename T, size_t N>
class CircularBuffer {
private:
    // Align buffer to cache line to avoid false sharing
    alignas(CACHE_LINE_SIZE) std::array<T, N> buffer;
    
    // Separate atomics with padding to prevent false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> size{0};

public:
    CircularBuffer() = default;

    void push(const T& value) {
        size_t current_head = head.load(std::memory_order_relaxed);
        buffer[current_head] = value;
        
        size_t new_head = (current_head + 1) % N;
        head.store(new_head, std::memory_order_relaxed);
        
        size_t current_size = size.load(std::memory_order_relaxed);
        if (current_size < N) {
            size.store(current_size + 1, std::memory_order_relaxed);
        }
    }

    float average() const {
        size_t current_size = size.load(std::memory_order_relaxed);
        if (current_size == 0) return 0.0f;
        
        float sum = 0.0f;
        size_t current_head = head.load(std::memory_order_relaxed);
        size_t start = (current_head + N - current_size) % N;
        
        for (size_t i = 0; i < current_size; ++i) {
            sum += buffer[(start + i) % N];
        }
        
        return sum / current_size;
    }

    void clear() {
        head.store(0, std::memory_order_relaxed);
        size.store(0, std::memory_order_relaxed);
    }
};