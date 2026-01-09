#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <string>
#include <cstdint>
#include "config.h"

// STB Image libraries for image loading/saving
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Include GLFW and ImGui for display
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/gl.h>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

struct Detection {
    float x, y, w, h;
    float confidence;
    int classId;
};

class TRTEngine {
private:
    Logger gLogger;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    void* buffers[2];
    cudaStream_t stream;

    int inputIndex;
    int outputIndex;
    size_t inputSize;
    size_t outputSize;

    // Host pinned buffers for faster transfers
    float* hostInput;
    float* hostOutput;

    std::string inputTensorName;
    std::string outputTensorName;

    std::vector<int> resizeXIndices;
    std::vector<int> resizeYIndices;
    int cachedSrcWidth = -1;
    int cachedSrcHeight = -1;

    // Model dimensions
    int inputC = 3;
    int inputH = 640;
    int inputW = 640;
    int numClasses = 80;
    int maxDetections = 8400;

    bool checkCuda(cudaError_t status, const char* msg) {
        if (status != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(status) << std::endl;
            return false;
        }
        return true;
    }

    void releaseBuffers() {
        if (buffers[0]) {
            cudaFree(buffers[0]);
            buffers[0] = nullptr;
        }
        if (buffers[1]) {
            cudaFree(buffers[1]);
            buffers[1] = nullptr;
        }
        if (hostInput) {
            cudaFreeHost(hostInput);
            hostInput = nullptr;
        }
        if (hostOutput) {
            cudaFreeHost(hostOutput);
            hostOutput = nullptr;
        }
    }
    
public:
    TRTEngine()
        : buffers{nullptr, nullptr},
          stream(nullptr),
          hostInput(nullptr),
          hostOutput(nullptr),
          inputIndex(-1),
          outputIndex(-1),
          inputSize(0),
          outputSize(0) {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            stream = nullptr;
            std::cerr << "Failed to create CUDA stream" << std::endl;
        }
    }

    ~TRTEngine() {
        releaseBuffers();
        if (stream) cudaStreamDestroy(stream);
    }

    bool loadEngine(const std::string& enginePath) {
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Cannot open engine file: " << enginePath << std::endl;
            return false;
        }
        
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        std::unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        if (!engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }
        
        context.reset(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // TensorRT 10 API - find tensor indices
        inputIndex = -1;
        outputIndex = -1;
        inputTensorName.clear();
        outputTensorName.clear();
        
        for (int i = 0; i < engine->getNbIOTensors(); i++) {
            const char* tensorName = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(tensorName);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (inputIndex == -1 || strcmp(tensorName, "images") == 0) {
                    inputIndex = i;
                    inputTensorName = tensorName;
                }
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                if (outputIndex == -1 || strcmp(tensorName, "output0") == 0) {
                    outputIndex = i;
                    outputTensorName = tensorName;
                }
            }
        }

        if (inputIndex == -1 || outputIndex == -1) {
            std::cerr << "Failed to find input/output tensors" << std::endl;
            return false;
        }
        if (inputTensorName.empty() || outputTensorName.empty()) {
            std::cerr << "Input or output tensor name not resolved" << std::endl;
            return false;
        }
        
        auto inputDims = engine->getTensorShape(engine->getIOTensorName(inputIndex));
        auto outputDims = engine->getTensorShape(engine->getIOTensorName(outputIndex));
        
        if (inputDims.nbDims >= 4) {
            inputC = std::max(1, static_cast<int>(inputDims.d[1]));
            inputH = std::max(1, static_cast<int>(inputDims.d[2]));
            inputW = std::max(1, static_cast<int>(inputDims.d[3]));
        }

        if (outputDims.nbDims >= 3) {
            maxDetections = std::max(1, static_cast<int>(outputDims.d[1]));
            numClasses = std::max(0, static_cast<int>(outputDims.d[2]) - 4);
        }

        inputSize = static_cast<size_t>(inputC) * inputH * inputW * sizeof(float);
        outputSize = static_cast<size_t>(maxDetections) * (numClasses + 4) * sizeof(float);

        if (!stream) {
            if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
                std::cerr << "Failed to create CUDA stream" << std::endl;
                return false;
            }
        }

        cachedSrcWidth = -1;
        cachedSrcHeight = -1;
        resizeXIndices.clear();
        resizeYIndices.clear();

        releaseBuffers();

        if (!checkCuda(cudaMalloc(&buffers[0], inputSize), "Failed to allocate device input buffer")) {
            releaseBuffers();
            return false;
        }
        if (!checkCuda(cudaMalloc(&buffers[1], outputSize), "Failed to allocate device output buffer")) {
            releaseBuffers();
            return false;
        }

        if (!checkCuda(cudaMallocHost(&hostInput, inputSize), "Failed to allocate host input buffer")) {
            releaseBuffers();
            return false;
        }
        if (!checkCuda(cudaMallocHost(&hostOutput, outputSize), "Failed to allocate host output buffer")) {
            releaseBuffers();
            return false;
        }

        nvinfer1::Dims inputShape;
        inputShape.nbDims = 4;
        inputShape.d[0] = 1;
        inputShape.d[1] = inputC;
        inputShape.d[2] = inputH;
        inputShape.d[3] = inputW;
        if (!context->setInputShape(inputTensorName.c_str(), inputShape)) {
            std::cerr << "Failed to set input shape on execution context" << std::endl;
            return false;
        }

        std::cout << "Engine loaded successfully!" << std::endl;
        std::cout << "Input shape: " << inputDims.d[0] << "x" << inputDims.d[1]
                  << "x" << inputDims.d[2] << "x" << inputDims.d[3] << std::endl;
        std::cout << "Output shape: " << outputDims.d[0] << "x" << outputDims.d[1]
                  << "x" << outputDims.d[2] << std::endl;
        
        return true;
    }
    
    std::vector<Detection> infer(unsigned char* imageData, int width, int height, int channels) {
        // Resize and preprocess image
        if (!context) {
            std::cerr << "Execution context is not initialized" << std::endl;
            return {};
        }
        if (!stream) {
            std::cerr << "CUDA stream is not available" << std::endl;
            return {};
        }
        const int planeSize = inputH * inputW;
        const int copyChannels = std::min(channels, inputC);
        const float inv255 = 1.0f / 255.0f;
        if (channels < inputC) {
            std::fill(hostInput + copyChannels * planeSize, hostInput + inputC * planeSize, 0.0f);
        }

        if (width != cachedSrcWidth) {
            resizeXIndices.resize(inputW);
            float scaleX = static_cast<float>(width) / static_cast<float>(inputW);
            int maxX = std::max(width - 1, 0);
            for (int x = 0; x < inputW; ++x) {
                int srcX = static_cast<int>(x * scaleX);
                resizeXIndices[x] = std::min(srcX, maxX);
            }
            cachedSrcWidth = width;
        }
        if (height != cachedSrcHeight) {
            resizeYIndices.resize(inputH);
            float scaleY = static_cast<float>(height) / static_cast<float>(inputH);
            int maxY = std::max(height - 1, 0);
            for (int y = 0; y < inputH; ++y) {
                int srcY = static_cast<int>(y * scaleY);
                resizeYIndices[y] = std::min(srcY, maxY);
            }
            cachedSrcHeight = height;
        }

        if (resizeXIndices.size() != static_cast<size_t>(inputW)) {
            resizeXIndices.assign(inputW, 0);
        }
        if (resizeYIndices.size() != static_cast<size_t>(inputH)) {
            resizeYIndices.assign(inputH, 0);
        }

        for (int y = 0; y < inputH; y++) {
            int srcY = resizeYIndices[y];
            int rowBase = (srcY * width) * channels;
            int dstRowBase = y * inputW;
            for (int x = 0; x < inputW; x++) {
                int srcX = resizeXIndices[x];
                int srcIdx = rowBase + srcX * channels;
                int dstIdx = dstRowBase + x;

                for (int c = 0; c < copyChannels; ++c) {
                    hostInput[c * planeSize + dstIdx] = imageData[srcIdx + c] * inv255;
                }
            }
        }

        // Copy to GPU
        if (!checkCuda(cudaMemcpyAsync(buffers[0], hostInput, inputSize,
                                       cudaMemcpyHostToDevice, stream),
                       "Failed to copy input to device")) {
            return {};
        }
        
        // Run inference - TensorRT 10 API
        if (!context->setTensorAddress(inputTensorName.c_str(), buffers[0])) {
            std::cerr << "Failed to set input tensor address" << std::endl;
            return {};
        }
        if (!context->setTensorAddress(outputTensorName.c_str(), buffers[1])) {
            std::cerr << "Failed to set output tensor address" << std::endl;
            return {};
        }
        
        bool status = context->enqueueV3(stream);
        if (!status) {
            std::cerr << "Failed to run inference" << std::endl;
            return {};
        }
        
        // Copy output back
        if (!checkCuda(cudaMemcpyAsync(hostOutput, buffers[1], outputSize,
                                       cudaMemcpyDeviceToHost, stream),
                       "Failed to copy output to host")) {
            return {};
        }
        if (!checkCuda(cudaStreamSynchronize(stream), "Failed to synchronize CUDA stream")) {
            return {};
        }

        // Postprocess
        return postprocess(hostOutput);
    }

private:
    std::vector<Detection> postprocess(float* output, float confThreshold = 0.25f) {
        std::vector<Detection> detections;
        detections.reserve(maxDetections);

        for (int i = 0; i < maxDetections; i++) {
            float* ptr = output + i * (numClasses + 4);
            
            float maxScore = 0;
            int maxClassId = 0;
            for (int j = 4; j < numClasses + 4; j++) {
                if (ptr[j] > maxScore) {
                    maxScore = ptr[j];
                    maxClassId = j - 4;
                }
            }
            
            if (maxScore > confThreshold) {
                Detection det;
                det.x = ptr[0];
                det.y = ptr[1];
                det.w = ptr[2];
                det.h = ptr[3];
                det.confidence = maxScore;
                det.classId = maxClassId;
                detections.push_back(det);
            }
        }
        
        // Simple NMS
        std::vector<Detection> nmsResult;
        nmsResult.reserve(detections.size());
        std::vector<uint8_t> suppressed(detections.size(), 0);

        for (size_t i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;

            nmsResult.push_back(detections[i]);

            for (size_t j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;
                if (detections[i].classId != detections[j].classId) continue;

                float iou = calculateIoU(detections[i], detections[j]);
                if (iou > 0.45f) {
                    suppressed[j] = 1;
                }
            }
        }
        
        return nmsResult;
    }
    
    float calculateIoU(const Detection& a, const Detection& b) {
        float x1_a = a.x - a.w / 2;
        float y1_a = a.y - a.h / 2;
        float x2_a = a.x + a.w / 2;
        float y2_a = a.y + a.h / 2;
        
        float x1_b = b.x - b.w / 2;
        float y1_b = b.y - b.h / 2;
        float x2_b = b.x + b.w / 2;
        float y2_b = b.y + b.h / 2;
        
        float x1_i = std::max(x1_a, x1_b);
        float y1_i = std::max(y1_a, y1_b);
        float x2_i = std::min(x2_a, x2_b);
        float y2_i = std::min(y2_a, y2_b);
        
        float intersection = std::max(0.0f, x2_i - x1_i) * std::max(0.0f, y2_i - y1_i);
        float area_a = a.w * a.h;
        float area_b = b.w * b.h;
        float unionArea = area_a + area_b - intersection;
        
        return intersection / (unionArea + 1e-6f);
    }
};

// Simple video decoder using FFmpeg (command line)
class SimpleVideoDecoder {
private:
    std::string videoPath;
    std::string tempDir;
    int frameCount;
    int currentFrame;
    
public:
    SimpleVideoDecoder(const std::string& path) : videoPath(path), currentFrame(0) {
        tempDir = "temp_frames";
        extractFrames();
    }
    
    ~SimpleVideoDecoder() {
        cleanup();
    }
    
    void extractFrames() {
        // Create temp directory
        std::string mkdirCmd = "mkdir " + tempDir + " 2>nul";
        system(mkdirCmd.c_str());
        
        // Extract frames using ffmpeg
        std::string cmd = "ffmpeg -i \"" + videoPath + "\" -q:v 2 " + tempDir + "/frame_%04d.jpg -y 2>nul";
        std::cout << "Extracting frames from video..." << std::endl;
        system(cmd.c_str());
        
        // Count frames
        frameCount = 0;
        while (true) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/frame_%04d.jpg", tempDir.c_str(), frameCount + 1);
            std::ifstream test(filename);
            if (!test.good()) break;
            frameCount++;
        }
        
        std::cout << "Extracted " << frameCount << " frames" << std::endl;
    }
    
    unsigned char* getNextFrame(int& width, int& height, int& channels) {
        if (currentFrame >= frameCount) return nullptr;
        
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_%04d.jpg", tempDir.c_str(), currentFrame + 1);
        currentFrame++;
        
        return stbi_load(filename, &width, &height, &channels, 0);
    }
    
    void reset() {
        currentFrame = 0;
    }
    
    int getTotalFrames() const { return frameCount; }
    int getCurrentFrame() const { return currentFrame; }
    
    void cleanup() {
        std::string cmd = "rmdir /s /q " + tempDir + " 2>nul";
        system(cmd.c_str());
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <engine_file> <video_file>" << std::endl;
        std::cout << "Example: " << argv[0] << " model.engine test/test_det.mp4" << std::endl;
        return -1;
    }
    
    std::string enginePath = argv[1];
    std::string videoPath = argv[2];
    
    // Initialize TensorRT engine
    TRTEngine engine;
    if (!engine.loadEngine(enginePath)) {
        std::cerr << "Failed to load engine" << std::endl;
        return -1;
    }
    
    // Initialize video decoder
    SimpleVideoDecoder decoder(videoPath);
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(1280, 720, "TensorRT Engine Test", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    
    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    
    // Performance metrics
    std::vector<double> inferTimes;
    const int avgWindow = 30;
    bool isPaused = false;
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        if (!isPaused) {
            int width, height, channels;
            unsigned char* frameData = decoder.getNextFrame(width, height, channels);
            
            if (frameData) {
                auto startTime = std::chrono::high_resolution_clock::now();
                
                // Run inference
                auto detections = engine.infer(frameData, width, height, channels);
                
                auto endTime = std::chrono::high_resolution_clock::now();
                double inferTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
                
                inferTimes.push_back(inferTime);
                if (inferTimes.size() > avgWindow) {
                    inferTimes.erase(inferTimes.begin());
                }
                
                // Draw frame with detections
                GLuint texture;
                glGenTextures(1, &texture);
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, 
                           channels == 3 ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, frameData);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                
                // Display in ImGui window
                ImGui::SetNextWindowPos(ImVec2(0, 0));
                ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
                ImGui::Begin("Video", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
                
                // Draw performance metrics
                double avgInferTime = 0;
                if (!inferTimes.empty()) {
                    avgInferTime = std::accumulate(inferTimes.begin(), inferTimes.end(), 0.0) / inferTimes.size();
                }
                double avgFPS = avgInferTime > 0 ? 1000.0 / avgInferTime : 0;
                
                ImGui::SetCursorPos(ImVec2(10, 10));
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0.7f));
                ImGui::BeginChild("Stats", ImVec2(200, 80), true);
                ImGui::Text("FPS: %.1f", avgFPS);
                ImGui::Text("Latency: %.1f ms", avgInferTime);
                ImGui::Text("Frame: %d/%d", decoder.getCurrentFrame(), decoder.getTotalFrames());
                ImGui::Text("Detections: %zu", detections.size());
                ImGui::EndChild();
                ImGui::PopStyleColor();
                
                // Draw video frame
                ImVec2 imageSize(width, height);
                ImGui::SetCursorPos(ImVec2((ImGui::GetWindowWidth() - imageSize.x) * 0.5f,
                                          (ImGui::GetWindowHeight() - imageSize.y) * 0.5f));
                ImGui::Image((void*)(intptr_t)texture, imageSize);
                
                // Draw detections overlay
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                ImVec2 imagePos = ImGui::GetCursorScreenPos();
                imagePos.y -= imageSize.y;
                
                for (const auto& det : detections) {
                    float x1 = (det.x - det.w/2) * width;
                    float y1 = (det.y - det.h/2) * height;
                    float x2 = (det.x + det.w/2) * width;
                    float y2 = (det.y + det.h/2) * height;
                    
                    ImVec2 p1(imagePos.x + x1, imagePos.y + y1);
                    ImVec2 p2(imagePos.x + x2, imagePos.y + y2);
                    
                    drawList->AddRect(p1, p2, IM_COL32(0, 255, 0, 255), 0.0f, 0, 2.0f);
                    
                    char label[64];
                    snprintf(label, sizeof(label), "Class %d: %.0f%%", 
                            det.classId, det.confidence * 100);
                    drawList->AddText(p1, IM_COL32(255, 255, 0, 255), label);
                }
                
                ImGui::End();
                
                // Controls
                ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 60));
                ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize);
                if (ImGui::Button(isPaused ? "Resume" : "Pause")) {
                    isPaused = !isPaused;
                }
                ImGui::SameLine();
                if (ImGui::Button("Restart")) {
                    decoder.reset();
                }
                ImGui::SameLine();
                if (ImGui::Button("Quit")) {
                    glfwSetWindowShouldClose(window, true);
                }
                ImGui::End();
                
                glDeleteTextures(1, &texture);
                stbi_image_free(frameData);
            } else {
                // End of video, restart
                decoder.reset();
            }
        }
        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "\nTest completed!" << std::endl;
    
    return 0;
}
