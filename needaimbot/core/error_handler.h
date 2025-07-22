#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <functional>

namespace ErrorHandler {
    
    enum class ErrorLevel {
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };
    
    // CUDA error checking with detailed context
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
}

#endif // ERROR_HANDLER_H