// UDP Frame Receive + TensorRT Inference Test with Bbox Visualization
// Usage: test_udp_inference [model.engine] [port] [game_pc_ip]

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "needaimbot/modules/stb/stb_image_write.h"

// TensorRT
#include <NvInfer.h>
#include <NvInferRuntime.h>

// TensorRT version compatibility
#define TRT_VERSION (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH)
#define TRT_VERSION_10_0 10000  // TensorRT 10.0.0

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

// Preprocess RGB to normalized FP16 (NCHW format) - declared in preprocessing_simple.cu
extern "C" void launchPreprocessKernel(const uint8_t* rgb, void* output,
                                       int srcWidth, int srcHeight,
                                       int dstWidth, int dstHeight,
                                       cudaStream_t stream);

// Draw bounding box on RGB image (will be displayed as BGR by Windows)
// Simple INI parser for config.ini
bool loadConfig(const char* filename, std::string& modelPath, unsigned short& port, std::string& gamePcIp) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Config file '" << filename << "' not found, using defaults\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        if (!line.empty() && line.back() == '\r') line.pop_back();
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == ';' || line[0] == '#' || line[0] == '[') {
            continue;
        }

        // Parse key=value
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        // Trim key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // Set config values
        if (key == "ModelPath") {
            modelPath = value;
        } else if (key == "Port") {
            port = (unsigned short)std::stoi(value);
        } else if (key == "GamePcIP") {
            gamePcIp = value;
        }
    }

    file.close();
    std::cout << "Loaded config from '" << filename << "'\n";
    return true;
}

void drawBox(uint8_t* img, int imgW, int imgH, int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
    // Clamp coordinates
    x1 = std::max(0, std::min(x1, imgW - 1));
    y1 = std::max(0, std::min(y1, imgH - 1));
    x2 = std::max(0, std::min(x2, imgW - 1));
    y2 = std::max(0, std::min(y2, imgH - 1));

    // Draw horizontal lines
    for (int t = 0; t < thickness; t++) {
        int top = std::max(0, y1 - t);
        int bottom = std::min(imgH - 1, y2 + t);

        for (int x = x1; x <= x2; x++) {
            // Top line (BGR order for Windows display)
            int idx_top = (top * imgW + x) * 3;
            img[idx_top + 0] = b;
            img[idx_top + 1] = g;
            img[idx_top + 2] = r;

            // Bottom line (BGR order for Windows display)
            int idx_bottom = (bottom * imgW + x) * 3;
            img[idx_bottom + 0] = b;
            img[idx_bottom + 1] = g;
            img[idx_bottom + 2] = r;
        }
    }

    // Draw vertical lines
    for (int t = 0; t < thickness; t++) {
        int left = std::max(0, x1 - t);
        int right = std::min(imgW - 1, x2 + t);

        for (int y = y1; y <= y2; y++) {
            // Left line (BGR order for Windows display)
            int idx_left = (y * imgW + left) * 3;
            img[idx_left + 0] = b;
            img[idx_left + 1] = g;
            img[idx_left + 2] = r;

            // Right line (BGR order for Windows display)
            int idx_right = (y * imgW + right) * 3;
            img[idx_right + 0] = b;
            img[idx_right + 1] = g;
            img[idx_right + 2] = r;
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== UDP Frame + TensorRT Inference Test (with Bbox) ===" << std::endl;

    // Default values
    std::string modelPath = "models/best.engine";
    unsigned short port = 5007;
    std::string gamePcIp = "";

    // Load config from file first
    loadConfig("config.ini", modelPath, port, gamePcIp);

    // Command line args override config file
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
    std::string inputName, outputName;
    nvinfer1::Dims inputDims, outputDims;

#if TRT_VERSION >= TRT_VERSION_10_0
    int numIOTensors = engine->getNbIOTensors();
    std::cout << "Engine has " << numIOTensors << " IO tensors" << std::endl;

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
#else
    // TensorRT 8.x API
    int numBindings = engine->getNbBindings();
    std::cout << "Engine has " << numBindings << " bindings" << std::endl;

    for (int i = 0; i < numBindings; i++) {
        const char* name = engine->getBindingName(i);
        bool isInput = engine->bindingIsInput(i);
        nvinfer1::Dims dims = engine->getBindingDimensions(i);

        std::cout << "  Binding[" << i << "]: " << name
                  << (isInput ? " (INPUT)" : " (OUTPUT)")
                  << " shape: [";
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d];
            if (d < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        if (isInput) {
            inputName = name;
            inputDims = dims;
        } else {
            outputName = name;
            outputDims = dims;
        }
    }
#endif

    // Calculate buffer sizes (FP16)
    int inputH = inputDims.d[2];
    int inputW = inputDims.d[3];
    size_t inputSize = 1 * 3 * inputH * inputW * sizeof(__half);  // FP16

    size_t outputElements = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        outputElements *= outputDims.d[i];
    }
    size_t outputSize = outputElements * sizeof(__half);  // FP16

    std::cout << "Input size: " << inputW << "x" << inputH << " (" << inputSize / 1024 << " KB) FP16" << std::endl;
    std::cout << "Output elements: " << outputElements << " (" << outputSize / 1024 << " KB) FP16" << std::endl;

    // Allocate GPU buffers
    void* d_input = nullptr;
    void* d_output = nullptr;
    void* d_rgb = nullptr;  // For incoming RGB frame

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_rgb, 640 * 640 * 3);  // Max expected frame size

#if TRT_VERSION >= TRT_VERSION_10_0
    // TensorRT 10.x: Set tensor addresses
    context->setTensorAddress(inputName.c_str(), d_input);
    context->setTensorAddress(outputName.c_str(), d_output);
#else
    // TensorRT 8.x: Use bindings array for enqueueV2
    void* bindings[2];
    int inputIndex = engine->getBindingIndex(inputName.c_str());
    int outputIndex = engine->getBindingIndex(outputName.c_str());
    std::cout << "Binding indices: input=" << inputIndex << ", output=" << outputIndex << std::endl;
    bindings[inputIndex] = d_input;
    bindings[outputIndex] = d_output;
#endif

    // Host output buffer for result checking (FP16)
    std::vector<__half> h_output_fp16(outputElements);
    std::vector<float> h_output(outputElements);  // For processing

    // Frame buffer for drawing bboxes
    std::vector<uint8_t> displayFrame;

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

    // JPG output settings
    const char* outputDir = "output_frames";
    system("mkdir -p output_frames");
    uint64_t savedFrameCount = 0;
    const int saveInterval = 30;  // Save every 30 frames

    // Performance tracking
    uint64_t frameCount = 0;
    uint64_t totalInferenceUs = 0;
    uint64_t totalE2EUs = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto lastPrintTime = startTime;

    const float confThreshold = 0.25f;
    const uint8_t colors[][3] = {
        {0, 255, 0},    // Green - class 0
        {255, 0, 0},    // Red - class 1
        {0, 0, 255},    // Blue - class 2
        {255, 255, 0},  // Yellow - class 3
        {255, 0, 255},  // Magenta - class 4
        {0, 255, 255}   // Cyan - class 5
    };

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

        // Preprocess (resize + normalize to FP16)
        launchPreprocessKernel(
            static_cast<uint8_t*>(d_rgb),
            d_input,  // FP16 output
            width, height, inputW, inputH,
            stream
        );

        // Debug: verify input data after preprocessing (FP16)
        if (frameCount < 3) {
            size_t inputElements = inputSize / sizeof(__half);
            std::vector<__half> h_input_debug(inputElements);
            cudaMemcpy(h_input_debug.data(), d_input, inputSize, cudaMemcpyDeviceToHost);
            float minIn = __half2float(h_input_debug[0]), maxIn = __half2float(h_input_debug[0]);
            for (size_t i = 0; i < h_input_debug.size(); i++) {
                float val = __half2float(h_input_debug[i]);
                minIn = std::min(minIn, val);
                maxIn = std::max(maxIn, val);
            }
            std::cout << "[DEBUG] Frame " << frameCount << " input range after preprocess (FP16): [" << minIn << ", " << maxIn << "]" << std::endl;
        }

        // Run inference
        auto inferStart = std::chrono::steady_clock::now();

#if TRT_VERSION >= TRT_VERSION_10_0
        if (!context->enqueueV3(stream)) {
#else
        if (!context->enqueueV2(bindings, stream, nullptr)) {
#endif
            std::cerr << "Inference failed!" << std::endl;
            continue;
        }

        // Sync after inference and check for CUDA errors
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after inference: " << cudaGetErrorString(err) << std::endl;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA last error: " << cudaGetErrorString(err) << std::endl;
        }

        // Copy FP16 output and convert to FP32
        cudaMemcpyAsync(h_output_fp16.data(), d_output, outputSize, cudaMemcpyDeviceToHost, stream);

        // Sync and measure
        cudaStreamSynchronize(stream);

        // Convert FP16 to FP32
        for (size_t i = 0; i < outputElements; i++) {
            h_output[i] = __half2float(h_output_fp16[i]);
        }

        auto inferEnd = std::chrono::steady_clock::now();

        // Parse detections
        // Output format: [1, 6, numBoxes] = [batch, features, num_boxes] (transposed)
        // Features: [x, y, w, h, cls0_prob, cls1_prob]
        std::vector<std::pair<float, int>> detections;
        int numFeatures = outputDims.d[1];  // 6
        int numBoxes = outputDims.d[2];     // 2100 for 320x320 model

        // Debug: print output range for first few frames
        if (frameCount < 3) {
            float minVal = h_output[0], maxVal = h_output[0];
            float maxCls0 = 0, maxCls1 = 0;
            for (int i = 0; i < numBoxes; i++) {
                maxCls0 = std::max(maxCls0, h_output[4 * numBoxes + i]);
                maxCls1 = std::max(maxCls1, h_output[5 * numBoxes + i]);
            }
            for (size_t i = 0; i < h_output.size(); i++) {
                minVal = std::min(minVal, h_output[i]);
                maxVal = std::max(maxVal, h_output[i]);
            }
            std::cout << "[DEBUG] Frame " << frameCount << " output range: [" << minVal << ", " << maxVal << "]"
                      << " cls0_max=" << maxCls0 << " cls1_max=" << maxCls1 << std::endl;
        }

        for (int i = 0; i < numBoxes; i++) {
            // Each feature is a row, each box is a column
            float cls0_prob = h_output[4 * numBoxes + i];
            float cls1_prob = h_output[5 * numBoxes + i];

            // Get max class probability as confidence
            float conf = std::max(cls0_prob, cls1_prob);

            if (conf > confThreshold) {
                detections.emplace_back(conf, i);
            }
        }

        // Sort by confidence
        std::sort(detections.begin(), detections.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        // Save frame with bboxes as JPG periodically
        if (frameCount % saveInterval == 0) {
            // Copy frame for drawing
            displayFrame.resize(frameSize);
            memcpy(displayFrame.data(), rgbData, frameSize);

            // Draw bboxes on frame (RGB order for JPG)
            for (size_t i = 0; i < std::min((size_t)20, detections.size()); i++) {
                int boxIdx = detections[i].second;
                float x_center = h_output[0 * numBoxes + boxIdx];
                float y_center = h_output[1 * numBoxes + boxIdx];
                float bw = h_output[2 * numBoxes + boxIdx];
                float bh = h_output[3 * numBoxes + boxIdx];
                float cls0_prob = h_output[4 * numBoxes + boxIdx];
                float cls1_prob = h_output[5 * numBoxes + boxIdx];
                int cls = (cls1_prob > cls0_prob) ? 1 : 0;

                // Convert to pixel coordinates (scale from model input to original frame)
                float scaleX = (float)width / inputW;
                float scaleY = (float)height / inputH;
                int x1 = (int)((x_center - bw/2) * scaleX);
                int y1 = (int)((y_center - bh/2) * scaleY);
                int x2 = (int)((x_center + bw/2) * scaleX);
                int y2 = (int)((y_center + bh/2) * scaleY);

                // Select color based on class (RGB)
                int colorIdx = cls % 6;
                uint8_t r = colors[colorIdx][0];
                uint8_t g = colors[colorIdx][1];
                uint8_t b = colors[colorIdx][2];

                // Draw bbox
                drawBox(displayFrame.data(), width, height, x1, y1, x2, y2, r, g, b, 2);
            }

            // Save as JPG
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/frame_%06lu.jpg", outputDir, savedFrameCount);
            stbi_write_jpg(filename, width, height, 3, displayFrame.data(), 90);
            savedFrameCount++;

            if (savedFrameCount <= 5) {
                std::cout << "[SAVE] Saved " << filename << " (detections: " << detections.size() << ")" << std::endl;
            }
        }

        auto frameEnd = std::chrono::steady_clock::now();

        // Calculate timings
        auto inferUs = std::chrono::duration_cast<std::chrono::microseconds>(inferEnd - inferStart).count();
        auto e2eUs = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart).count();
        auto recvUs = std::chrono::duration_cast<std::chrono::microseconds>(recvTime - frameStart).count();

        totalInferenceUs += inferUs;
        totalE2EUs += e2eUs;
        frameCount++;

        // Print stats every second
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPrintTime).count();
        if (elapsed >= 1000) {
            double fps = frameCount * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            double avgInferMs = (frameCount > 0) ? totalInferenceUs / 1000.0 / frameCount : 0;
            double avgE2EMs = (frameCount > 0) ? totalE2EUs / 1000.0 / frameCount : 0;

            std::cout << "[STATS] Frames: " << frameCount
                      << " | FPS: " << std::fixed << std::setprecision(1) << fps
                      << " | Inference: " << std::setprecision(2) << avgInferMs << "ms"
                      << " | E2E: " << avgE2EMs << "ms"
                      << " | Detections: " << detections.size()
                      << " | Received: " << capture.GetReceivedFrameCount()
                      << " | Dropped: " << capture.GetDroppedFrameCount()
                      << std::endl;

            // Print top 5 detections with class info
            if (detections.size() > 0) {
                std::cout << "  Top detections: ";
                for (size_t i = 0; i < std::min((size_t)5, detections.size()); i++) {
                    int boxIdx = detections[i].second;
                    float cls0_prob = h_output[4 * numBoxes + boxIdx];
                    float cls1_prob = h_output[5 * numBoxes + boxIdx];
                    float conf = std::max(cls0_prob, cls1_prob);
                    int cls = (cls1_prob > cls0_prob) ? 1 : 0;
                    std::cout << "[cls=" << cls << " (c0=" << std::setprecision(4) << cls0_prob
                             << " c1=" << cls1_prob << ") conf=" << std::setprecision(2) << conf << "] ";
                }
                std::cout << std::endl;
            }

            lastPrintTime = now;
        }
    }

    // Cleanup
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Saved " << savedFrameCount << " frames to " << outputDir << "/" << std::endl;
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

    std::cout << "Done." << std::endl;
    return 0;
}
