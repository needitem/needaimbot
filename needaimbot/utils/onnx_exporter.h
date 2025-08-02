#pragma once
#include <string>
#include <vector>
#include <Windows.h>

class OnnxExporter {
public:
    struct ExportConfig {
        std::string inputPath;
        std::string outputPath;
        int imgSize = 640;
        int batchSize = 1;
        bool simplify = true;
        bool fp16 = false;
    };

    // Python 스크립트를 호출하여 ONNX 변환
    static bool exportToOnnx(const ExportConfig& config);
    
    // 디렉토리의 모든 .pt 파일 변환
    static bool batchExport(const std::string& inputDir, const std::string& outputDir = "");
    
    // Python 환경 확인
    static bool checkPythonEnvironment();

private:
    static std::string buildCommand(const ExportConfig& config);
    static bool executeCommand(const std::string& command);
};