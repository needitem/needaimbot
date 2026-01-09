// UDP Frame Receive + TensorRT Inference Test
// Usage: test_udp_inference.exe [model.engine] [port] [game_pc_ip]

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <atomic>
#include <csignal>
#include <iomanip>
#include <algorithm>

#include "needaimbot/capture/udp_capture.h"
#include "needaimbot/capture/lz4.h"
#include "display_window.h"

// TensorRT
#include <NvInfer.h>
#include <NvInferRuntime.h>

// Simple logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;
static std::atomic<bool> g_running{true};

void signalHandler(int sig) {
    std::cout << "\nShutting down..." << std::endl;
    g_running = false;
}

// Load TensorRT engine from file
nvinfer1::ICudaEngine* loadEngine(const std::string& enginePath, nvinfer1::IRuntime* runtime) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Cannot open engine file: " << enginePath << std::endl;
        return nullptr;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    
    return runtime->deserializeCudaEngine(engineData.data(), size);
}

// Preprocess RGB to normalized float (NCHW format) - declared in preprocessing_simple.cu
extern "C" void launchPreprocessKernel(const uint8_t* rgb, float* output,
                                       int srcWidth, int srcHeight,
                                       int dstWidth, int dstHeight,
                                       cudaStream_t stream);

int main(int argc, char* argv[]) {
    std::cout << "=== UDP Frame + TensorRT Inference Test ===" << std::endl;
    
    // Parse arguments
    std::string modelPath = "models/best.engine";
    unsigned short port = 5007;
    std::string gamePcIp = "";
    
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) port = static_cast<unsigned short>(std::stoi(argv[2]));
    if (argc > 3) gamePcIp = argv[3];
    
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Listen port: " << port << std::endl;
    if (!gamePcIp.empty()) {
        std::cout << "Game PC IP: " << gamePcIp << std::endl;
    }
    
    // Signal handler
    std::signal(SIGINT, signalHandler);
    
    // Initialize CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Load TensorRT engine
    std::cout << "Loading TensorRT engine..." << std::endl;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return 1;
    }
    
    nvinfer1::ICudaEngine* engine = loadEngine(modelPath, runtime);
    if (!engine) {
        std::cerr << "Failed to load engine" << std::endl;
        return 1;
    }
    
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return 1;
    }
    
    // Get input/output info
    int numIOTensors = engine->getNbIOTensors();
    std::cout << "Engine has " << numIOTensors << " IO tensors" << std::endl;
    
    std::string inputName, outputName;
    nvinfer1::Dims inputDims, outputDims;
    
    for (int i = 0; i < numIOTensors; i++) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        
        std::cout << "  Tensor[" << i << "]: " << name 
                  << (mode == nvinfer1::TensorIOMode::kINPUT ? " (INPUT)" : " (OUTPUT)")
                  << " shape: [";
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d];
            if (d < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputName = name;
            inputDims = dims;
        } else {
            outputName = name;
            outputDims = dims;
        }
    }
    
    // Calculate buffer sizes
    int inputH = inputDims.d[2];
    int inputW = inputDims.d[3];
    size_t inputSize = 1 * 3 * inputH * inputW * sizeof(float);
    
    size_t outputElements = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        outputElements *= outputDims.d[i];
    }
    size_t outputSize = outputElements * sizeof(float);
    
    std::cout << "Input size: " << inputW << "x" << inputH << " (" << inputSize / 1024 << " KB)" << std::endl;
    std::cout << "Output elements: " << outputElements << " (" << outputSize / 1024 << " KB)" << std::endl;
    
    // Allocate GPU buffers
    void* d_input = nullptr;
    void* d_output = nullptr;
    void* d_rgb = nullptr;  // For incoming RGB frame
    
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_rgb, 640 * 640 * 3);  // Max expected frame size
    
    // Set tensor addresses
    context->setTensorAddress(inputName.c_str(), d_input);
    context->setTensorAddress(outputName.c_str(), d_output);
    
    // Host output buffer for result checking
    std::vector<float> h_output(outputElements);
    
    // Initialize UDP capture
    std::cout << "Initializing UDP capture on port " << port << "..." << std::endl;
    UDPCapture capture;
    if (!capture.Initialize(port, gamePcIp)) {
        std::cerr << "Failed to initialize UDP capture" << std::endl;
        return 1;
    }
    
    if (!capture.StartCapture()) {
        std::cerr << "Failed to start capture" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Waiting for frames from Game PC ===" << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;

    // Initialize display window
    DisplayWindow display;
    if (!display.initialize(640, 640, "Inference Viewer - 320x320")) {
        std::cerr << "Failed to initialize display window" << std::endl;
    }
    
    // Performance tracking
    uint64_t frameCount = 0;
    uint64_t totalInferenceUs = 0;
    uint64_t totalE2EUs = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto lastPrintTime = startTime;
    
    // Main loop
    while (g_running) {
        auto frameStart = std::chrono::steady_clock::now();
        
        // Wait for frame
        void* rgbData = nullptr;
        unsigned int width = 0, height = 0;
        uint64_t frameId = 0;
        
        if (!capture.AcquireFrameSync(&rgbData, &width, &height, &frameId, 100)) {
            // Timeout - no frame received
            continue;
        }
        
        auto recvTime = std::chrono::steady_clock::now();
        
        // Copy to GPU
        size_t frameSize = width * height * 3;
        cudaMemcpyAsync(d_rgb, rgbData, frameSize, cudaMemcpyHostToDevice, stream);

        // Preprocess (resize + normalize)
        launchPreprocessKernel(
            static_cast<uint8_t*>(d_rgb),
            static_cast<float*>(d_input),
            width, height, inputW, inputH,
            stream
        );

        // Run inference
        auto inferStart = std::chrono::steady_clock::now();
        
        if (!context->enqueueV3(stream)) {
            std::cerr << "Inference failed!" << std::endl;
            continue;
        }
        
        // Sync and measure
        cudaStreamSynchronize(stream);

        // Update display window
        display.updateFrame(static_cast<unsigned char*>(rgbData), width, height);

        auto inferEnd = std::chrono::steady_clock::now();
        auto frameEnd = inferEnd;
        
        // Calculate timings
        auto inferUs = std::chrono::duration_cast<std::chrono::microseconds>(inferEnd - inferStart).count();
        auto e2eUs = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart).count();
        auto recvUs = std::chrono::duration_cast<std::chrono::microseconds>(recvTime - frameStart).count();
        
        totalInferenceUs += inferUs;
        totalE2EUs += e2eUs;
        frameCount++;
        
        // Copy output to check results (optional - for debugging)
        if (frameCount == 1 || frameCount % 100 == 0) {
            cudaMemcpy(h_output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);
            
            // Find detections with confidence > threshold
            const float confThreshold = 0.25f;
            std::vector<std::pair<float, int>> detections;
            
            int numBoxes = outputDims.d[1];
            int boxSize = outputDims.d[2];
            
            for (int i = 0; i < numBoxes; i++) {
                float conf = h_output[i * boxSize + 4];
                if (conf > confThreshold) {
                    detections.emplace_back(conf, i);
                }
            }
            
            // Sort by confidence (highest first)
            std::sort(detections.begin(), detections.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
            
            std::cout << "Frame " << frameCount << " [" << frameId << "]: "
                      << width << "x" << height 
                      << " | recv=" << recvUs << "us"
                      << " | infer=" << inferUs << "us"
                      << " | e2e=" << e2eUs << "us"
                      << " | detections=" << detections.size() << std::endl;
            
            // Show top 5 detections
            int showCount = std::min(5, (int)detections.size());
            for (int i = 0; i < showCount; i++) {
                int idx = detections[i].second;
                float x = h_output[idx * boxSize + 0];
                float y = h_output[idx * boxSize + 1];
                float w = h_output[idx * boxSize + 2];
                float h = h_output[idx * boxSize + 3];
                float conf = h_output[idx * boxSize + 4];
                int cls = (int)h_output[idx * boxSize + 5];
                
                std::cout << "  [" << i << "] cls=" << cls
                          << " conf=" << std::fixed << std::setprecision(3) << conf
                          << " box=[" << x << "," << y << "," << w << "," << h << "]"
                          << std::endl;
            }
        }
        
        // Print stats every second
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPrintTime).count();
        if (elapsed >= 1000) {
            double fps = frameCount * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            double avgInferMs = (frameCount > 0) ? totalInferenceUs / 1000.0 / frameCount : 0;
            double avgE2EMs = (frameCount > 0) ? totalE2EUs / 1000.0 / frameCount : 0;
            
            std::cout << "\n[STATS] Frames: " << frameCount 
                      << " | FPS: " << std::fixed << std::setprecision(1) << fps
                      << " | Avg Inference: " << std::setprecision(2) << avgInferMs << "ms"
                      << " | Avg E2E: " << avgE2EMs << "ms"
                      << " | Received: " << capture.GetReceivedFrameCount()
                      << " | Dropped: " << capture.GetDroppedFrameCount()
                      << "\n" << std::endl;
            
            lastPrintTime = now;
        }

        // Process window messages
        if (!display.processMessages()) {
            break;
        }
    }
    
    // Cleanup
    std::cout << "\n=== Final Statistics ===" << std::endl;
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();
    
    if (frameCount > 0) {
        std::cout << "Total frames processed: " << frameCount << std::endl;
        std::cout << "Total time: " << totalTime / 1000.0 << " seconds" << std::endl;
        std::cout << "Average FPS: " << frameCount * 1000.0 / totalTime << std::endl;
        std::cout << "Average inference time: " << totalInferenceUs / 1000.0 / frameCount << " ms" << std::endl;
        std::cout << "Average E2E latency: " << totalE2EUs / 1000.0 / frameCount << " ms" << std::endl;
    }
    
    capture.StopCapture();
    capture.Shutdown();
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_rgb);
    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;

    display.shutdown();

    std::cout << "Done." << std::endl;
    return 0;
}
