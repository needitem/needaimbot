// Test simple inference with debug frame images
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>

#include "needaimbot/cuda/simple_inference.h"
#include "needaimbot/cuda/simple_postprocess.h"

#define STB_IMAGE_IMPLEMENTATION
#include "needaimbot/modules/stb_image.h"

int main(int argc, char* argv[]) {
    std::string enginePath = "/home/hwan/needaimbot/sunxds_0.8.2_TRT_320_fp16.engine";
    std::string imagePath = "/home/hwan/needaimbot/debug_frames/frame_0.jpg";

    if (argc > 1) enginePath = argv[1];
    if (argc > 2) imagePath = argv[2];

    std::cout << "=== Simple Inference Test (Real Image) ===" << std::endl;
    std::cout << "Engine: " << enginePath << std::endl;
    std::cout << "Image: " << imagePath << std::endl;

    // Load engine
    gpa::SimpleInference inference;
    if (!inference.loadEngine(enginePath)) {
        std::cerr << "Failed to load engine" << std::endl;
        return 1;
    }

    int modelRes = inference.getModelResolution();
    std::cout << "Model resolution: " << modelRes << "x" << modelRes << std::endl;

    // Load image with stb_image
    int imgW, imgH, channels;
    unsigned char* imgData = stbi_load(imagePath.c_str(), &imgW, &imgH, &channels, 3);
    if (!imgData) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << imgW << "x" << imgH << " (" << channels << " channels)" << std::endl;

    // Resize to model input (simple nearest neighbor)
    std::vector<uint8_t> resizedImage(modelRes * modelRes * 3);
    for (int y = 0; y < modelRes; y++) {
        for (int x = 0; x < modelRes; x++) {
            int srcX = x * imgW / modelRes;
            int srcY = y * imgH / modelRes;
            int srcIdx = (srcY * imgW + srcX) * 3;
            int dstIdx = (y * modelRes + x) * 3;
            resizedImage[dstIdx + 0] = imgData[srcIdx + 0];
            resizedImage[dstIdx + 1] = imgData[srcIdx + 1];
            resizedImage[dstIdx + 2] = imgData[srcIdx + 2];
        }
    }
    stbi_image_free(imgData);
    std::cout << "Resized to " << modelRes << "x" << modelRes << std::endl;

    // Test parameters
    float confThreshold = 0.25f;
    int headClassId = 1;
    float headBonus = 0.15f;

    gpa::PIDConfig pidConfig;
    pidConfig.kp_x = 0.5f;
    pidConfig.kp_y = 0.5f;
    pidConfig.ki_x = 0.0f;
    pidConfig.ki_y = 0.0f;
    pidConfig.kd_x = 0.3f;
    pidConfig.kd_y = 0.3f;

    float iouStickiness = 0.3f;
    float headAimPoint = 1.0f;
    float bodyAimPoint = 0.15f;

    gpa::MouseMovement movement;
    gpa::Detection bestTarget;

    // Warmup
    std::cout << "\nWarmup (10 iterations)..." << std::endl;
    for (int i = 0; i < 10; i++) {
        inference.runInferenceFused(resizedImage.data(), modelRes, modelRes,
                                     confThreshold, headClassId, headBonus,
                                     pidConfig, iouStickiness,
                                     headAimPoint, bodyAimPoint,
                                     movement, &bestTarget);
    }

    // Single inference to check detection
    std::cout << "\n=== Single Inference Result ===" << std::endl;
    bool hasTarget = inference.runInferenceFused(resizedImage.data(), modelRes, modelRes,
                                                  confThreshold, headClassId, headBonus,
                                                  pidConfig, iouStickiness,
                                                  headAimPoint, bodyAimPoint,
                                                  movement, &bestTarget);

    if (hasTarget) {
        std::cout << "Detection FOUND!" << std::endl;
        std::cout << "  BBox: (" << bestTarget.x1 << ", " << bestTarget.y1 << ") - ("
                  << bestTarget.x2 << ", " << bestTarget.y2 << ")" << std::endl;
        float w = bestTarget.x2 - bestTarget.x1;
        float h = bestTarget.y2 - bestTarget.y1;
        std::cout << "  Size: " << w << " x " << h << std::endl;
        std::cout << "  Confidence: " << bestTarget.confidence << std::endl;
        std::cout << "  Class: " << bestTarget.classId << (bestTarget.classId == headClassId ? " (Head)" : " (Body)") << std::endl;
        std::cout << "  Mouse movement: dx=" << movement.dx << ", dy=" << movement.dy << std::endl;
    } else {
        std::cout << "No detection found" << std::endl;
    }

    // Benchmark
    const int NUM_ITERATIONS = 100;
    std::cout << "\n=== Benchmark (" << NUM_ITERATIONS << " iterations) ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    int detectionCount = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        bool found = inference.runInferenceFused(resizedImage.data(), modelRes, modelRes,
                                                  confThreshold, headClassId, headBonus,
                                                  pidConfig, iouStickiness,
                                                  headAimPoint, bodyAimPoint,
                                                  movement, &bestTarget);
        if (found) detectionCount++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double avgMs = totalMs / NUM_ITERATIONS;
    double fps = 1000.0 / avgMs;

    std::cout << "Total time: " << totalMs << " ms" << std::endl;
    std::cout << "Average: " << avgMs << " ms/frame" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Detections: " << detectionCount << "/" << NUM_ITERATIONS << " frames" << std::endl;

    // Test all debug frames
    std::cout << "\n=== Testing all debug frames ===" << std::endl;
    for (int frameNum = 0; frameNum <= 9; frameNum++) {
        char framePath[256];
        snprintf(framePath, sizeof(framePath), "/home/hwan/needaimbot/debug_frames/frame_%d.jpg", frameNum);

        int w, h, c;
        unsigned char* data = stbi_load(framePath, &w, &h, &c, 3);
        if (!data) continue;

        // Resize
        for (int y = 0; y < modelRes; y++) {
            for (int x = 0; x < modelRes; x++) {
                int srcX = x * w / modelRes;
                int srcY = y * h / modelRes;
                int srcIdx = (srcY * w + srcX) * 3;
                int dstIdx = (y * modelRes + x) * 3;
                resizedImage[dstIdx + 0] = data[srcIdx + 0];
                resizedImage[dstIdx + 1] = data[srcIdx + 1];
                resizedImage[dstIdx + 2] = data[srcIdx + 2];
            }
        }
        stbi_image_free(data);

        bool found = inference.runInferenceFused(resizedImage.data(), modelRes, modelRes,
                                                  confThreshold, headClassId, headBonus,
                                                  pidConfig, iouStickiness,
                                                  headAimPoint, bodyAimPoint,
                                                  movement, &bestTarget);

        std::cout << "frame_" << frameNum << ".jpg: ";
        if (found) {
            std::cout << "DETECTED - class=" << bestTarget.classId
                      << " conf=" << bestTarget.confidence
                      << " move=(" << movement.dx << "," << movement.dy << ")" << std::endl;
        } else {
            std::cout << "No detection" << std::endl;
        }
    }

    std::cout << "\nTest completed!" << std::endl;
    return 0;
}
