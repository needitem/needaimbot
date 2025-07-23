#ifndef SAFE_OPERATIONS_H
#define SAFE_OPERATIONS_H

#pragma warning(push)
#pragma warning(disable: 4996 4267 4244 4305 4018 4101 4800)

#include <functional>
#include <stdexcept>
#include <type_traits>
#include <thread>
#include <chrono>
#include "error_handler.h"
#include "performance_optimizer.h"

namespace SafeOperations {
    
    // Exception-safe operation wrapper with logging
    template<typename T>
    T safely_execute(const std::string& operation_name, std::function<T()> operation, T default_value = T{}) {
        try {
            PERF_TIMER_QUIET(operation_name.c_str());
            return operation();
        }
        catch (const std::exception& e) {
            ErrorHandler::Logger::error("SafeOps", "Exception in ", operation_name, ": ", e.what());
            return default_value;
        }
        catch (...) {
            ErrorHandler::Logger::error("SafeOps", "Unknown exception in ", operation_name);
            return default_value;
        }
    }
    
    // Void operation wrapper
    bool safely_execute_void(const std::string& operation_name, std::function<void()> operation) {
        try {
            PERF_TIMER_QUIET(operation_name.c_str());
            operation();
            return true;
        }
        catch (const std::exception& e) {
            ErrorHandler::Logger::error("SafeOps", "Exception in ", operation_name, ": ", e.what());
            return false;
        }
        catch (...) {
            ErrorHandler::Logger::error("SafeOps", "Unknown exception in ", operation_name);
            return false;
        }
    }
    
    // Resource-safe operation with cleanup
    template<typename T, typename CleanupFunc>
    T safely_execute_with_cleanup(const std::string& operation_name, std::function<T()> operation, 
                                  CleanupFunc cleanup, T default_value = T{}) {
        try {
            PERF_TIMER_QUIET(operation_name.c_str());
            auto result = operation();
            cleanup();
            return result;
        }
        catch (const std::exception& e) {
            ErrorHandler::Logger::error("SafeOps", "Exception in ", operation_name, ": ", e.what());
            try {
                cleanup();
            }
            catch (...) {
                ErrorHandler::Logger::error("SafeOps", "Cleanup failed for ", operation_name);
            }
            return default_value;
        }
        catch (...) {
            ErrorHandler::Logger::error("SafeOps", "Unknown exception in ", operation_name);
            try {
                cleanup();
            }
            catch (...) {
                ErrorHandler::Logger::error("SafeOps", "Cleanup failed for ", operation_name);
            }
            return default_value;
        }
    }
    
    // Retry mechanism with exponential backoff
    template<typename T>
    T safely_execute_with_retry(const std::string& operation_name, std::function<T()> operation, 
                               int max_retries = 3, int base_delay_ms = 10, T default_value = T{}) {
        int attempt = 0;
        int delay = base_delay_ms;
        
        while (attempt < max_retries) {
            try {
                PERF_TIMER_QUIET((operation_name + "_attempt_" + std::to_string(attempt)).c_str());
                return operation();
            }
            catch (const std::exception& e) {
                attempt++;
                if (attempt >= max_retries) {
                    ErrorHandler::Logger::error("SafeOps", "Final attempt failed for ", operation_name, ": ", e.what());
                    return default_value;
                }
                
                ErrorHandler::Logger::warn("SafeOps", "Attempt ", attempt, " failed for ", operation_name, 
                                         ": ", e.what(), ". Retrying in ", delay, "ms");
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                delay *= 2; // Exponential backoff
            }
            catch (...) {
                attempt++;
                if (attempt >= max_retries) {
                    ErrorHandler::Logger::error("SafeOps", "Final attempt failed for ", operation_name, " (unknown exception)");
                    return default_value;
                }
                
                ErrorHandler::Logger::warn("SafeOps", "Attempt ", attempt, " failed for ", operation_name, 
                                         " (unknown exception). Retrying in ", delay, "ms");
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                delay *= 2;
            }
        }
        
        return default_value;
    }
    
    // Bounds-checked array access
    template<typename Container, typename IndexType>
    auto safe_access(const Container& container, IndexType index, 
                    const std::string& context = "") -> typename Container::value_type {
        if (index >= 0 && static_cast<size_t>(index) < container.size()) {
            return container[index];
        }
        
        ErrorHandler::Logger::error("SafeOps", "Index ", index, " out of bounds (size: ", 
                                   container.size(), ") in ", context);
        return typename Container::value_type{};
    }
    
    // Null pointer check wrapper
    template<typename T>
    T* safe_pointer_access(T* ptr, const std::string& context = "") {
        if (ptr != nullptr) {
            return ptr;
        }
        
        ErrorHandler::Logger::error("SafeOps", "Null pointer access in ", context);
        return nullptr;
    }
    
    // Division by zero protection
    template<typename T>
    T safe_divide(T numerator, T denominator, T default_value = T{}, const std::string& context = "") {
        static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type");
        
        if (denominator != T{}) {
            return numerator / denominator;
        }
        
        ErrorHandler::Logger::error("SafeOps", "Division by zero in ", context);
        return default_value;
    }
    
    // Memory allocation wrapper
    template<typename T>
    std::unique_ptr<T> safe_allocate(size_t count = 1, const std::string& context = "") {
        try {
            if (count == 1) {
                return std::make_unique<T>();
            } else {
                return std::unique_ptr<T>(new T[count]);
            }
        }
        catch (const std::bad_alloc& e) {
            ErrorHandler::Logger::error("SafeOps", "Memory allocation failed for ", count, 
                                       " objects of size ", sizeof(T), " in ", context, ": ", e.what());
            return nullptr;
        }
    }
}

// Convenience macros
#define SAFE_CALL(name, operation) SafeOperations::safely_execute_void(name, [&]() { operation; })
#define SAFE_CALL_RETURN(name, operation, default_val) SafeOperations::safely_execute<decltype(operation)>(name, [&]() { return operation; }, default_val)

#pragma warning(pop)

#endif // SAFE_OPERATIONS_H