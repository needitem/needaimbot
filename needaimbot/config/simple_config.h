#ifndef SIMPLE_CONFIG_H
#define SIMPLE_CONFIG_H

#include <string>
#include <vector>

// 핵심적인 클래스 설정만
struct SimpleClassSetting {
    int id;
    std::string name;
    bool ignore;
    
    SimpleClassSetting(int i = 0, std::string n = "", bool ign = false) 
        : id(i), name(std::move(n)), ignore(ign) {}
};

// 간소화된 설정 구조체
class SimpleConfig {
public:
    // === 핵심 검출 설정 ===
    int detection_resolution = 640;
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    std::string ai_model = "yolo11n.engine";
    
    // === 기본 마우스 제어 ===
    float sensitivity_x = 1.0f;
    float sensitivity_y = 1.0f;
    float smoothing = 0.7f;
    bool auto_aim = true;
    
    // === 화면 캡처 ===
    int capture_fps = 60;
    bool capture_use_cuda = true;
    int monitor_idx = 0;
    
    // === 타겟 우선순위 ===
    float distance_weight = 1.0f;
    float confidence_weight = 0.5f;
    std::string target_preference = "closest"; // "closest", "highest_confidence", "center"
    
    // === 입력 방식 ===
    std::string input_method = "win32";
    
    // === 기본 키바인딩 ===
    std::vector<std::string> button_targeting = {"X"};
    std::vector<std::string> button_exit = {"F9"};
    std::vector<std::string> button_pause = {"F8"};
    
    // === UI 설정 ===
    bool show_window = false;
    bool show_fps = false;
    bool verbose = false;
    
    // === 클래스 설정 ===
    std::vector<SimpleClassSetting> class_settings;
    
    // === 생성자 ===
    SimpleConfig() {
        // 기본 클래스 설정 초기화
        class_settings.push_back(SimpleClassSetting(0, "Person", false));
        class_settings.push_back(SimpleClassSetting(1, "Head", false));
    }
    
    // === 간단한 저장/로드 함수 ===
    bool loadConfig(const std::string& filename = "simple_config.ini");
    bool saveConfig(const std::string& filename = "simple_config.ini");
    
    // === 기존 Config와의 호환성을 위한 변환 함수 ===
    void convertFromComplexConfig(const class Config& complex_config);
    void applyToComplexConfig(class Config& complex_config) const;
};

#endif // SIMPLE_CONFIG_H