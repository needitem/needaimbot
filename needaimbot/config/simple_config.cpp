#include "simple_config.h"
#include "config.h"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <functional>

bool SimpleConfig::loadConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        if (verbose) {
            std::cout << "[SimpleConfig] Config file not found, using defaults: " << filename << std::endl;
        }
        return false;
    }
    
    // 최적화: 성능을 위한 해시맵 기반 파싱
    static const std::unordered_map<std::string, std::function<void(const std::string&)>> config_parsers = {
        {"detection_resolution", [this](const std::string& v) { detection_resolution = std::stoi(v); }},
        {"confidence_threshold", [this](const std::string& v) { confidence_threshold = std::stof(v); }},
        {"nms_threshold", [this](const std::string& v) { nms_threshold = std::stof(v); }},
        {"ai_model", [this](const std::string& v) { ai_model = v; }},
        {"sensitivity_x", [this](const std::string& v) { sensitivity_x = std::stof(v); }},
        {"sensitivity_y", [this](const std::string& v) { sensitivity_y = std::stof(v); }},
        {"smoothing", [this](const std::string& v) { smoothing = std::stof(v); }},
        {"auto_aim", [this](const std::string& v) { auto_aim = (v == "true" || v == "1"); }},
        {"capture_fps", [this](const std::string& v) { capture_fps = std::stoi(v); }},
        {"capture_use_cuda", [this](const std::string& v) { capture_use_cuda = (v == "true" || v == "1"); }},
        {"show_window", [this](const std::string& v) { show_window = (v == "true" || v == "1"); }},
        {"show_fps", [this](const std::string& v) { show_fps = (v == "true" || v == "1"); }},
        {"verbose", [this](const std::string& v) { verbose = (v == "true" || v == "1"); }},
        {"input_method", [this](const std::string& v) { input_method = v; }}
    };
    
    std::string line;
    line.reserve(256); // 최적화: 메모리 재할당 방지
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        
        size_t equals = line.find('=');
        if (equals == std::string::npos) continue;
        
        // 최적화: 문자열 뷰 사용하여 복사 최소화
        std::string key = line.substr(0, equals);
        std::string value = line.substr(equals + 1);
        
        // 최적화: 해시맵 기반 O(1) 룩업
        auto it = config_parsers.find(key);
        if (it != config_parsers.end()) {
            try {
                it->second(value);
            } catch (const std::exception& e) {
                std::cerr << "[SimpleConfig] Error parsing " << key << "=" << value << ": " << e.what() << std::endl;
            }
        }
    }
    
    if (verbose) {
        std::cout << "[SimpleConfig] Loaded config from: " << filename << std::endl;
    }
    return true;
}

bool SimpleConfig::saveConfig(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[SimpleConfig] Failed to save config: " << filename << std::endl;
        return false;
    }
    
    file << "# Simple Aimbot Configuration\n";
    file << "# Detection Settings\n";
    file << "detection_resolution=" << detection_resolution << "\n";
    file << "confidence_threshold=" << confidence_threshold << "\n";
    file << "nms_threshold=" << nms_threshold << "\n";
    file << "ai_model=" << ai_model << "\n\n";
    
    file << "# Mouse Control\n";
    file << "sensitivity_x=" << sensitivity_x << "\n";
    file << "sensitivity_y=" << sensitivity_y << "\n";
    file << "smoothing=" << smoothing << "\n";
    file << "auto_aim=" << (auto_aim ? "true" : "false") << "\n\n";
    
    file << "# Capture Settings\n";
    file << "capture_fps=" << capture_fps << "\n";
    file << "capture_use_cuda=" << (capture_use_cuda ? "true" : "false") << "\n\n";
    
    file << "# UI Settings\n";
    file << "show_window=" << (show_window ? "true" : "false") << "\n";
    file << "show_fps=" << (show_fps ? "true" : "false") << "\n";
    file << "verbose=" << (verbose ? "true" : "false") << "\n\n";
    
    file << "# Input Method\n";
    file << "input_method=" << input_method << "\n";
    
    if (verbose) {
        std::cout << "[SimpleConfig] Saved config to: " << filename << std::endl;
    }
    return true;
}

void SimpleConfig::convertFromComplexConfig(const Config& complex_config) {
    // 복잡한 설정에서 핵심 값들만 추출
    detection_resolution = complex_config.detection_resolution;
    confidence_threshold = complex_config.confidence_threshold;
    nms_threshold = complex_config.nms_threshold;
    ai_model = complex_config.ai_model;
    
    // PID 매개변수를 간단한 감도로 변환
    sensitivity_x = static_cast<float>(complex_config.kp_x);
    sensitivity_y = static_cast<float>(complex_config.kp_y);
    smoothing = complex_config.movement_smoothing;
    
    auto_aim = complex_config.auto_aim;
    capture_fps = complex_config.capture_fps;
    capture_use_cuda = complex_config.capture_use_cuda;
    show_window = complex_config.show_window;
    show_fps = complex_config.show_fps;
    verbose = complex_config.verbose;
    input_method = complex_config.input_method;
    
    if (!complex_config.button_targeting.empty()) {
        button_targeting = complex_config.button_targeting;
    }
    if (!complex_config.button_exit.empty()) {
        button_exit = complex_config.button_exit;
    }
    if (!complex_config.button_pause.empty()) {
        button_pause = complex_config.button_pause;
    }
}

void SimpleConfig::applyToComplexConfig(Config& complex_config) const {
    // 간단한 설정을 복잡한 설정에 적용
    complex_config.detection_resolution = detection_resolution;
    complex_config.confidence_threshold = confidence_threshold;
    complex_config.nms_threshold = nms_threshold;
    complex_config.ai_model = ai_model;
    
    // 간단한 감도를 PID 매개변수로 변환
    complex_config.kp_x = sensitivity_x;
    complex_config.kp_y = sensitivity_y;
    complex_config.ki_x = 0.0; // I, D 게인은 0으로 설정 (비례 제어만)
    complex_config.ki_y = 0.0;
    complex_config.kd_x = 0.0;
    complex_config.kd_y = 0.0;
    complex_config.movement_smoothing = smoothing;
    
    complex_config.auto_aim = auto_aim;
    complex_config.capture_fps = capture_fps;
    complex_config.capture_use_cuda = capture_use_cuda;
    complex_config.show_window = show_window;
    complex_config.show_fps = show_fps;
    complex_config.verbose = verbose;
    complex_config.input_method = input_method;
    
    complex_config.button_targeting = button_targeting;
    complex_config.button_exit = button_exit;
    complex_config.button_pause = button_pause;
}