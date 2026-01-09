#ifndef ERROR_MANAGER_H
#define ERROR_MANAGER_H

#include <string>
#include <functional>
#include <vector>
#include <mutex>
#include <iostream>
#include <chrono>
#include <atomic>

class ErrorManager {
public:
    enum class ErrorLevel {
        Info,
        Warning,
        Error,
        Critical
    };
    
    struct ErrorEntry {
        ErrorLevel level;
        std::string message;
        std::string component;
        std::chrono::system_clock::time_point timestamp;
    };
    
    using ErrorHandler = std::function<void(const ErrorEntry&)>;
    
    static ErrorManager& getInstance() {
        static ErrorManager instance;
        return instance;
    }
    
    void logError(ErrorLevel level, const std::string& component, const std::string& message) {
        ErrorEntry entry{
            level,
            message,
            component,
            std::chrono::system_clock::now()
        };
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            error_history_.push_back(entry);
            
            // Keep only last 100 errors
            if (error_history_.size() > 100) {
                error_history_.erase(error_history_.begin());
            }
        }
        
        // Console output
        const char* level_str = getLevelString(level);
        std::cerr << "[" << level_str << "] " << component << ": " << message << std::endl;
        
        // Call handlers
        for (const auto& handler : handlers_) {
            handler(entry);
        }
        
        // Update error counts
        switch (level) {
            case ErrorLevel::Warning:
                warning_count_++;
                break;
            case ErrorLevel::Error:
                error_count_++;
                break;
            case ErrorLevel::Critical:
                critical_count_++;
                // Critical errors might need immediate action
                if (critical_handler_) {
                    critical_handler_(entry);
                }
                break;
            default:
                break;
        }
    }
    
    void registerHandler(ErrorHandler handler) {
        handlers_.push_back(handler);
    }
    
    void setCriticalHandler(ErrorHandler handler) {
        critical_handler_ = handler;
    }
    
    std::vector<ErrorEntry> getRecentErrors(size_t count = 10) const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t start = error_history_.size() > count ? error_history_.size() - count : 0;
        return std::vector<ErrorEntry>(error_history_.begin() + start, error_history_.end());
    }
    
    void clearErrorCounts() {
        warning_count_ = 0;
        error_count_ = 0;
        critical_count_ = 0;
    }
    
    int getWarningCount() const { return warning_count_; }
    int getErrorCount() const { return error_count_; }
    int getCriticalCount() const { return critical_count_; }
    
    // Convenience macros
    void info(const std::string& component, const std::string& message) {
        logError(ErrorLevel::Info, component, message);
    }
    
    void warning(const std::string& component, const std::string& message) {
        logError(ErrorLevel::Warning, component, message);
    }
    
    void error(const std::string& component, const std::string& message) {
        logError(ErrorLevel::Error, component, message);
    }
    
    void critical(const std::string& component, const std::string& message) {
        logError(ErrorLevel::Critical, component, message);
    }
    
private:
    ErrorManager() = default;
    
    const char* getLevelString(ErrorLevel level) const {
        switch (level) {
            case ErrorLevel::Info: return "INFO";
            case ErrorLevel::Warning: return "WARNING";
            case ErrorLevel::Error: return "ERROR";
            case ErrorLevel::Critical: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }
    
    mutable std::mutex mutex_;
    std::vector<ErrorEntry> error_history_;
    std::vector<ErrorHandler> handlers_;
    ErrorHandler critical_handler_;
    
    std::atomic<int> warning_count_{0};
    std::atomic<int> error_count_{0};
    std::atomic<int> critical_count_{0};
};

// Convenience macros
#define LOG_INFO(component, message) ErrorManager::getInstance().info(component, message)
#define LOG_WARNING(component, message) ErrorManager::getInstance().warning(component, message)
#define LOG_ERROR(component, message) ErrorManager::getInstance().error(component, message)
#define LOG_CRITICAL(component, message) ErrorManager::getInstance().critical(component, message)

#endif // ERROR_MANAGER_H