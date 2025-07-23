#ifndef NEEDAIMBOT_CORE_ERROR_HANDLER_H
#define NEEDAIMBOT_CORE_ERROR_HANDLER_H

// Prevent Windows ERROR macro from interfering
#ifdef ERROR
#undef ERROR
#endif

#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

namespace ErrorHandler {
    
    enum class ErrorLevel {
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };
    
    /**
     * @brief Check CUDA error with detailed context information
     * @param error CUDA error code to check
     * @param context Context string describing where the error occurred
     * @param level Error severity level
     * @return true if no error, false if error occurred
     */
    inline bool checkCudaError(cudaError_t error, const std::string& context, ErrorLevel level = ErrorLevel::ERROR) {
        if (error != cudaSuccess) {
            const char* errorStr = cudaGetErrorString(error);
            const char* errorName = cudaGetErrorName(error);
            
            std::string prefix;
            switch (level) {
                case ErrorLevel::INFO: prefix = "[INFO]"; break;
                case ErrorLevel::WARNING: prefix = "[WARNING]"; break;
                case ErrorLevel::ERROR: prefix = "[ERROR]"; break;
                case ErrorLevel::CRITICAL: prefix = "[CRITICAL]"; break;
            }
            
            std::cerr << prefix << " CUDA error in " << context << ": " 
                      << errorName << " - " << errorStr << std::endl;
            
            if (level == ErrorLevel::CRITICAL) {
                std::cerr << "Critical error detected. The application may not function correctly." << std::endl;
            }
            
            return false;
        }
        return true;
    }
    
    // Generic error logging
    inline void logError(const std::string& component, const std::string& message, ErrorLevel level = ErrorLevel::ERROR) {
        std::string prefix;
        switch (level) {
            case ErrorLevel::INFO: prefix = "[INFO]"; break;
            case ErrorLevel::WARNING: prefix = "[WARNING]"; break;
            case ErrorLevel::ERROR: prefix = "[ERROR]"; break;
            case ErrorLevel::CRITICAL: prefix = "[CRITICAL]"; break;
        }
        
        std::cerr << prefix << " [" << component << "] " << message << std::endl;
    }
    
    // RAII cleanup helper
    template<typename T>
    class ScopedCleanup {
    private:
        std::function<void(T*)> cleanup;
        T* resource;
        bool released = false;
        
    public:
        ScopedCleanup(T* res, std::function<void(T*)> cleanupFunc) 
            : resource(res), cleanup(cleanupFunc) {}
        
        ~ScopedCleanup() {
            if (!released && resource) {
                cleanup(resource);
            }
        }
        
        void release() { released = true; }
        T* get() { return resource; }
    };
    
    // Retry mechanism for transient failures
    template<typename Func>
    bool retryOperation(Func operation, int maxRetries = 3, int delayMs = 100) {
        for (int i = 0; i < maxRetries; ++i) {
            if (operation()) {
                return true;
            }
            if (i < maxRetries - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
            }
        }
        return false;
    }
    
    // Performance-aware logging (replaces direct cout/cerr usage)
    class Logger {
    private:
        static std::atomic<bool> verbose_mode_;
        static std::mutex log_mutex_;
        
    public:
        static void setVerbose(bool enabled) { verbose_mode_.store(enabled); }
        
        template<typename... Args>
        static void info(const std::string& component, Args&&... args) {
            if (!verbose_mode_.load()) return;
            
            std::lock_guard<std::mutex> lock(log_mutex_);
            std::cout << "[INFO][" << component << "] ";
            (std::cout << ... << args) << std::endl;
        }
        
        template<typename... Args>  
        static void error(const std::string& component, Args&&... args) {
            std::lock_guard<std::mutex> lock(log_mutex_);
            std::cerr << "[ERROR][" << component << "] ";
            (std::cerr << ... << args) << std::endl;
        }
        
        template<typename... Args>
        static void debug(const std::string& component, Args&&... args) {
            #ifdef _DEBUG
            if (!verbose_mode_.load()) return;
            
            std::lock_guard<std::mutex> lock(log_mutex_);
            std::cout << "[DEBUG][" << component << "] ";
            (std::cout << ... << args) << std::endl;
            #endif
        }
    };
    
    // Static member definitions (should be in .cpp file in real implementation)
    inline std::atomic<bool> Logger::verbose_mode_{false};
    inline std::mutex Logger::log_mutex_;
}

#endif // NEEDAIMBOT_CORE_ERROR_HANDLER_H