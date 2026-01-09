#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <atomic>

enum class LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3,
    LOG_CRITICAL = 4
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLogLevel(LogLevel level) {
        currentLevel_ = level;
    }

    void setFileLogging(bool enable, const std::string& filename = "needaimbot.log") {
        std::lock_guard<std::mutex> lock(mutex_);
        fileLoggingEnabled_ = enable;
        if (enable && !fileStream_.is_open()) {
            fileStream_.open(filename, std::ios::app);
        } else if (!enable && fileStream_.is_open()) {
            fileStream_.close();
        }
    }

    void setConsoleLogging(bool enable) {
        consoleLoggingEnabled_ = enable;
    }

    template<typename... Args>
    void log(LogLevel level, const std::string& component, Args... args) {
        // Early return for performance
        if (level < currentLevel_) return;
        
#ifndef _DEBUG
        // In release builds, skip debug/info/warning logs
        if (level < LogLevel::LOG_ERROR) return;
#endif

        std::stringstream ss;
        ss << "[" << getCurrentTime() << "] ";
        ss << "[" << getLevelString(level) << "] ";
        ss << "[" << component << "] ";
        ((ss << args), ...);

        output(ss.str(), level);
    }

    // Convenience methods
    template<typename... Args>
    void debug(const std::string& component, Args... args) {
        log(LogLevel::LOG_DEBUG, component, args...);
    }

    template<typename... Args>
    void info(const std::string& component, Args... args) {
        log(LogLevel::LOG_INFO, component, args...);
    }

    template<typename... Args>
    void warning(const std::string& component, Args... args) {
        log(LogLevel::LOG_WARNING, component, args...);
    }

    template<typename... Args>
    void error(const std::string& component, Args... args) {
        log(LogLevel::LOG_ERROR, component, args...);
    }

    template<typename... Args>
    void critical(const std::string& component, Args... args) {
        log(LogLevel::LOG_CRITICAL, component, args...);
    }

private:
    Logger() : currentLevel_(LogLevel::LOG_ERROR),  // Default to ERROR level for performance
               consoleLoggingEnabled_(false),  // Disable console by default for performance
               fileLoggingEnabled_(false) {}
    
    ~Logger() {
        if (fileStream_.is_open()) {
            fileStream_.close();
        }
    }

    std::string getCurrentTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::string getLevelString(LogLevel level) {
        switch (level) {
            case LogLevel::LOG_DEBUG: return "DEBUG";
            case LogLevel::LOG_INFO: return "INFO ";
            case LogLevel::LOG_WARNING: return "WARN ";
            case LogLevel::LOG_ERROR: return "ERROR";
            case LogLevel::LOG_CRITICAL: return "CRIT ";
            default: return "UNKN ";
        }
    }

    void output(const std::string& message, LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (consoleLoggingEnabled_) {
            if (level >= LogLevel::LOG_ERROR) {
                std::cerr << message << std::endl;
            } else {
                std::cout << message << std::endl;
            }
        }
        
        if (fileLoggingEnabled_ && fileStream_.is_open()) {
            fileStream_ << message << std::endl;
            fileStream_.flush();
        }
    }

    std::atomic<LogLevel> currentLevel_;
    std::atomic<bool> consoleLoggingEnabled_;
    std::atomic<bool> fileLoggingEnabled_;
    std::ofstream fileStream_;
    std::mutex mutex_;
};

// Convenience macros - Renamed to avoid conflict with error_manager.h
#ifdef _DEBUG
    #define LOGGER_DEBUG(component, ...) Logger::getInstance().debug(component, __VA_ARGS__)
    #define LOGGER_INFO(component, ...) Logger::getInstance().info(component, __VA_ARGS__)
    #define LOGGER_WARNING(component, ...) Logger::getInstance().warning(component, __VA_ARGS__)
#else
    // In release builds, disable debug/info/warning logs for performance
    #define LOGGER_DEBUG(component, ...) ((void)0)
    #define LOGGER_INFO(component, ...) ((void)0)
    #define LOGGER_WARNING(component, ...) ((void)0)
#endif

// Always keep error and critical logs even in release
#define LOGGER_ERROR(component, ...) Logger::getInstance().error(component, __VA_ARGS__)
#define LOGGER_CRITICAL(component, ...) Logger::getInstance().critical(component, __VA_ARGS__)

// Performance logging helper
class PerformanceTimer {
public:
    PerformanceTimer(const std::string& name, const std::string& component) 
        : name_(name), component_(component), start_(std::chrono::high_resolution_clock::now()) {}
    
    ~PerformanceTimer() {
#ifdef _DEBUG
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        LOG_DEBUG(component_, name_, " took ", duration, " microseconds");
#endif
    }

private:
    std::string name_;
    std::string component_;
    std::chrono::high_resolution_clock::time_point start_;
};

