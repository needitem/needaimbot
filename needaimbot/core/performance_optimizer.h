#ifndef PERFORMANCE_OPTIMIZER_H
#define PERFORMANCE_OPTIMIZER_H

#pragma warning(push)
#pragma warning(disable: 4996 4267 4244 4305 4018 4101 4800)

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>

namespace PerformanceOptimizer {
    
    // Fast string stream replacement for frequent logging
    class FastStringBuilder {
    private:
        std::string buffer_;
        
    public:
        FastStringBuilder() { buffer_.reserve(256); }
        
        template<typename T>
        FastStringBuilder& operator<<(const T& value) {
            if constexpr (std::is_arithmetic_v<T>) {
                if constexpr (std::is_integral_v<T>) {
                    buffer_ += std::to_string(value);
                } else {
                    buffer_ += std::to_string(value);
                }
            } else if constexpr (std::is_same_v<T, std::string>) {
                buffer_ += value;
            } else if constexpr (std::is_same_v<T, const char*>) {
                buffer_ += value;
            } else {
                std::ostringstream oss;
                oss << value;
                buffer_ += oss.str();
            }
            return *this;
        }
        
        const std::string& str() const { return buffer_; }
        void clear() { buffer_.clear(); }
        void reserve(size_t size) { buffer_.reserve(size); }
    };
    
    // Optimized output without flush
    class FastOutput {
    public:
        template<typename... Args>
        static void print(Args&&... args) {
            FastStringBuilder builder;
            (builder << ... << args);
            builder << '\n';  // Use '\n' instead of std::endl
            std::cout << builder.str();
        }
        
        template<typename... Args>
        static void error(Args&&... args) {
            FastStringBuilder builder;
            (builder << ... << args);
            builder << '\n';
            std::cerr << builder.str();
        }
    };
    
    // Memory copy optimization
    template<typename T>
    inline void fast_copy(T* dest, const T* src, size_t count) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(dest, src, count * sizeof(T));
        } else {
            std::copy(src, src + count, dest);
        }
    }
    
    template<typename T>
    inline void fast_fill(T* dest, const T& value, size_t count) {
        if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 1) {
            std::memset(dest, static_cast<int>(value), count);
        } else {
            std::fill(dest, dest + count, value);
        }
    }
    
    // Loop optimization helpers
    template<typename Container, typename Func>
    inline void parallel_for_each(Container& container, Func&& func) {
        if (container.size() > 1000) {  // Use parallel execution for large containers
            std::for_each(std::execution::par_unseq, 
                         container.begin(), container.end(), std::forward<Func>(func));
        } else {
            std::for_each(container.begin(), container.end(), std::forward<Func>(func));
        }
    }
    
    // RAII performance timer
    class ScopedTimer {
    private:
        std::chrono::high_resolution_clock::time_point start_;
        std::string name_;
        bool print_on_destroy_;
        
    public:
        explicit ScopedTimer(const std::string& name, bool print = true) 
            : start_(std::chrono::high_resolution_clock::now())
            , name_(name)
            , print_on_destroy_(print) {}
        
        ~ScopedTimer() {
            if (print_on_destroy_) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double, std::milli>(end - start_);
                FastOutput::print("[PERF] ", name_, " took ", duration.count(), "ms");
            }
        }
        
        double elapsed_ms() const {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - start_).count();
        }
    };
    
    // Efficient assertion with compile-time optimization
    #ifdef NDEBUG
        #define FAST_ASSERT(condition) ((void)0)
        #define FAST_ASSERT_MSG(condition, message) ((void)0)
    #else
        #define FAST_ASSERT(condition) \
            do { \
                if (!(condition)) { \
                    PerformanceOptimizer::FastOutput::error( \
                        "Assertion failed: ", #condition, " at ", __FILE__, ":", __LINE__); \
                    std::abort(); \
                } \
            } while(0)
            
        #define FAST_ASSERT_MSG(condition, message) \
            do { \
                if (!(condition)) { \
                    PerformanceOptimizer::FastOutput::error( \
                        "Assertion failed: ", #condition, " - ", message, \
                        " at ", __FILE__, ":", __LINE__); \
                    std::abort(); \
                } \
            } while(0)
    #endif
    
    // Cache-friendly data structures
    template<typename T, size_t Alignment = 64>
    class AlignedVector {
    private:
        alignas(Alignment) std::vector<T> data_;
        
    public:
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;
        
        // Forward vector interface
        void push_back(const T& value) { data_.push_back(value); }
        void push_back(T&& value) { data_.push_back(std::move(value)); }
        
        template<typename... Args>
        void emplace_back(Args&&... args) { data_.emplace_back(std::forward<Args>(args)...); }
        
        T& operator[](size_t index) { return data_[index]; }
        const T& operator[](size_t index) const { return data_[index]; }
        
        size_t size() const { return data_.size(); }
        bool empty() const { return data_.empty(); }
        void clear() { data_.clear(); }
        void reserve(size_t size) { data_.reserve(size); }
        
        iterator begin() { return data_.begin(); }
        iterator end() { return data_.end(); }
        const_iterator begin() const { return data_.begin(); }
        const_iterator end() const { return data_.end(); }
        
        T* data() { return data_.data(); }
        const T* data() const { return data_.data(); }
    };
    
    // Branch prediction hints
    #ifdef __GNUC__
        #define LIKELY(x) __builtin_expect(!!(x), 1)
        #define UNLIKELY(x) __builtin_expect(!!(x), 0)
    #else
        #define LIKELY(x) (x)
        #define UNLIKELY(x) (x)
    #endif
    
    // Hot path optimization marker
    #ifdef __GNUC__
        #define HOT_PATH __attribute__((hot))
        #define COLD_PATH __attribute__((cold))
    #else
        #define HOT_PATH
        #define COLD_PATH
    #endif
}

// Convenience macros
#define PERF_TIMER(name) PerformanceOptimizer::ScopedTimer _timer(name)
#define PERF_TIMER_QUIET(name) PerformanceOptimizer::ScopedTimer _timer(name, false)

#pragma warning(pop)

#endif // PERFORMANCE_OPTIMIZER_H