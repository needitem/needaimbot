#pragma once

#include <NvInfer.h>
#include <iostream>
#include <string>

class TensorRTLogger : public nvinfer1::ILogger {
public:
    explicit TensorRTLogger(bool verbose = false) : m_verbose(verbose) {}
    
    void log(Severity severity, const char* msg) noexcept override {
        // Skip internal TensorRT messages unless in verbose mode
        if (!m_verbose && severity < Severity::kERROR) {
            // Filter out common internal messages
            std::string message(msg);
            if (message.find("defaultAllocator.cpp") != std::string::npos ||
                message.find("enqueueV3") != std::string::npos) {
                return;
            }
        }
        
        // Print messages based on severity
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[TRT INTERNAL_ERROR] " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "[TRT ERROR] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                if (m_verbose) {
                    std::cout << "[TRT WARNING] " << msg << std::endl;
                }
                break;
            case Severity::kINFO:
                if (m_verbose) {
                    std::cout << "[TRT INFO] " << msg << std::endl;
                }
                break;
            case Severity::kVERBOSE:
                if (m_verbose) {
                    std::cout << "[TRT VERBOSE] " << msg << std::endl;
                }
                break;
        }
    }
    
    void setVerbose(bool verbose) {
        m_verbose = verbose;
    }
    
private:
    bool m_verbose;
};