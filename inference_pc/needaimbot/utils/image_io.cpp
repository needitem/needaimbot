#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../modules/stb/stb_image_write.h"
#include "image_io.h"
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace ImageIO {

bool saveImage(const SimpleMat& image, const std::string& filename, int quality) {
    if (image.empty()) {
        std::cerr << "[ImageIO] Cannot save empty image" << std::endl;
        return false;
    }
    
    std::string ext = std::filesystem::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    const uint8_t* data = image.data();
    int width = image.cols();
    int height = image.rows();
    int channels = image.channels();
    int stride = static_cast<int>(image.step());
    
    int result = 0;
    
    if (ext == ".png") {
        result = stbi_write_png(filename.c_str(), width, height, channels, data, stride);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        result = stbi_write_jpg(filename.c_str(), width, height, channels, data, quality);
    } else if (ext == ".bmp") {
        result = stbi_write_bmp(filename.c_str(), width, height, channels, data);
    } else if (ext == ".tga") {
        result = stbi_write_tga(filename.c_str(), width, height, channels, data);
    } else {
        std::cerr << "[ImageIO] Unsupported file format: " << ext << std::endl;
        return false;
    }
    
    return result != 0;
}

void saveScreenshot(const SimpleMat& frame, const std::string& directory) {
    static int screenshotCount = 0;
    
    if (frame.empty()) {
        std::cerr << "[Screenshot] Cannot save empty frame" << std::endl;
        return;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::create_directories(directory);
    
    // Generate timestamp-based filename
    auto now = std::chrono::system_clock::now();
    auto time_point = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_s(&tm, &time_point);
    
    std::ostringstream oss;
    oss << directory << "/screenshot_"
        << std::put_time(&tm, "%Y%m%d_%H%M%S_")
        << std::setfill('0') << std::setw(3) << screenshotCount++
        << ".png";
    
    std::string filepath = oss.str();
    
    if (!saveImage(frame, filepath)) {
        std::cerr << "[Screenshot] Failed to save: " << filepath << std::endl;
    }
}

} // namespace ImageIO