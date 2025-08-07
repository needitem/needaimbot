#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

// Async TensorRT inference with multiple execution contexts
class AsyncTensorRTInference {
private:
    // TensorRT resources
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    
    // Multiple execution contexts for overlapped inference
    struct ExecutionContext {
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        cudaStream_t stream;
        cudaEvent_t inputReady;
        cudaEvent_t outputReady;
        std::vector<void*> bindings;
        bool inUse;
        int id;
    };
    
    static constexpr int MAX_CONTEXTS = 3;
    std::vector<ExecutionContext> contexts_;
    
    // Circular buffer for context rotation
    std::atomic<int> currentContext_{0};
    
    // Input/output dimensions
    struct TensorInfo {
        std::string name;
        nvinfer1::Dims dims;
        size_t size;
        int bindingIndex;
    };
    
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
    
    // Memory pools for zero-allocation inference
    struct MemoryPool {
        std::vector<void*> inputBuffers;
        std::vector<void*> outputBuffers;
        std::mutex mutex;
        
        void* getInputBuffer() {
            std::lock_guard<std::mutex> lock(mutex);
            if (!inputBuffers.empty()) {
                void* buf = inputBuffers.back();
                inputBuffers.pop_back();
                return buf;
            }
            return nullptr;
        }
        
        void returnInputBuffer(void* buf) {
            std::lock_guard<std::mutex> lock(mutex);
            inputBuffers.push_back(buf);
        }
        
        void* getOutputBuffer() {
            std::lock_guard<std::mutex> lock(mutex);
            if (!outputBuffers.empty()) {
                void* buf = outputBuffers.back();
                outputBuffers.pop_back();
                return buf;
            }
            return nullptr;
        }
        
        void returnOutputBuffer(void* buf) {
            std::lock_guard<std::mutex> lock(mutex);
            outputBuffers.push_back(buf);
        }
    };
    
    MemoryPool memoryPool_;
    
    // Performance metrics
    std::atomic<float> avgInferenceTime_{0.0f};
    std::atomic<int> inferenceCount_{0};
    
public:
    AsyncTensorRTInference(const std::string& enginePath, int batchSize = 1) {
        initializeEngine(enginePath);
        initializeContexts(batchSize);
        initializeMemoryPool();
    }
    
    ~AsyncTensorRTInference() {
        // Clean up contexts
        for (auto& ctx : contexts_) {
            if (ctx.stream) cudaStreamDestroy(ctx.stream);
            if (ctx.inputReady) cudaEventDestroy(ctx.inputReady);
            if (ctx.outputReady) cudaEventDestroy(ctx.outputReady);
            for (auto ptr : ctx.bindings) {
                if (ptr) cudaFree(ptr);
            }
        }
        
        // Clean up memory pool
        for (auto ptr : memoryPool_.inputBuffers) {
            cudaFree(ptr);
        }
        for (auto ptr : memoryPool_.outputBuffers) {
            cudaFree(ptr);
        }
    }
    
    // Async inference submission
    int submitInference(void* inputData, cudaStream_t inputStream) {
        // Get next available context
        int contextId = getNextContext();
        if (contextId < 0) return -1; // All contexts busy
        
        auto& ctx = contexts_[contextId];
        
        // Wait for previous inference on this context to complete
        if (ctx.inUse) {
            cudaStreamWaitEvent(inputStream, ctx.outputReady, 0);
        }
        
        // Copy input data to context bindings
        for (const auto& input : inputs_) {
            cudaMemcpyAsync(ctx.bindings[input.bindingIndex],
                           inputData,
                           input.size,
                           cudaMemcpyDeviceToDevice,
                           ctx.stream);
        }
        
        // Record input ready
        cudaEventRecord(ctx.inputReady, ctx.stream);
        
        // Launch async inference
        bool success = ctx.context->enqueueV3(ctx.stream);
        if (!success) {
            ctx.inUse = false;
            return -1;
        }
        
        // Record output ready
        cudaEventRecord(ctx.outputReady, ctx.stream);
        ctx.inUse = true;
        
        return contextId;
    }
    
    // Get inference results
    bool getInferenceResults(int contextId, void* outputData, cudaStream_t outputStream) {
        if (contextId < 0 || contextId >= MAX_CONTEXTS) return false;
        
        auto& ctx = contexts_[contextId];
        if (!ctx.inUse) return false;
        
        // Wait for inference to complete
        cudaStreamWaitEvent(outputStream, ctx.outputReady, 0);
        
        // Copy output data
        for (const auto& output : outputs_) {
            cudaMemcpyAsync(outputData,
                           ctx.bindings[output.bindingIndex],
                           output.size,
                           cudaMemcpyDeviceToDevice,
                           outputStream);
        }
        
        // Mark context as available
        ctx.inUse = false;
        
        // Update metrics
        updateMetrics(ctx.stream);
        
        return true;
    }
    
    // Pipelined inference for maximum throughput
    void pipelinedInference(std::vector<void*>& inputBatch,
                           std::vector<void*>& outputBatch,
                           cudaStream_t stream) {
        const int batchSize = inputBatch.size();
        std::vector<int> contextIds(batchSize);
        
        // Submit all inferences
        for (int i = 0; i < batchSize; ++i) {
            contextIds[i] = submitInference(inputBatch[i], stream);
        }
        
        // Collect all results
        for (int i = 0; i < batchSize; ++i) {
            if (contextIds[i] >= 0) {
                getInferenceResults(contextIds[i], outputBatch[i], stream);
            }
        }
    }
    
    float getAverageInferenceTime() const {
        return avgInferenceTime_.load();
    }
    
private:
    void initializeEngine(const std::string& enginePath) {
        // Load serialized engine
        std::ifstream file(enginePath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open engine file");
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        
        // Create runtime and deserialize engine
        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
        
        // Parse input/output bindings
        int numBindings = engine_->getNbIOTensors();
        for (int i = 0; i < numBindings; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(name);
            auto dims = engine_->getTensorShape(name);
            auto dtype = engine_->getTensorDataType(name);
            
            size_t size = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                size *= dims.d[d];
            }
            size *= getDataTypeSize(dtype);
            
            TensorInfo info{name, dims, size, i};
            
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                inputs_.push_back(info);
            } else {
                outputs_.push_back(info);
            }
        }
    }
    
    void initializeContexts(int batchSize) {
        contexts_.resize(MAX_CONTEXTS);
        
        for (int i = 0; i < MAX_CONTEXTS; ++i) {
            auto& ctx = contexts_[i];
            ctx.id = i;
            ctx.inUse = false;
            
            // Create execution context
            ctx.context.reset(engine_->createExecutionContext());
            
            // Set optimization profile if needed
            if (engine_->getNbOptimizationProfiles() > 0) {
                ctx.context->setOptimizationProfileAsync(0, ctx.stream);
            }
            
            // Create stream and events
            cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&ctx.inputReady, 
                cudaEventDisableTiming | cudaEventBlockingSync);
            cudaEventCreateWithFlags(&ctx.outputReady,
                cudaEventDisableTiming | cudaEventBlockingSync);
            
            // Allocate bindings
            ctx.bindings.resize(engine_->getNbIOTensors());
            for (const auto& input : inputs_) {
                cudaMalloc(&ctx.bindings[input.bindingIndex], input.size);
            }
            for (const auto& output : outputs_) {
                cudaMalloc(&ctx.bindings[output.bindingIndex], output.size);
            }
        }
    }
    
    void initializeMemoryPool() {
        // Pre-allocate buffers for zero-allocation operation
        const int POOL_SIZE = MAX_CONTEXTS * 2;
        
        for (int i = 0; i < POOL_SIZE; ++i) {
            for (const auto& input : inputs_) {
                void* buffer;
                cudaMalloc(&buffer, input.size);
                memoryPool_.inputBuffers.push_back(buffer);
            }
            
            for (const auto& output : outputs_) {
                void* buffer;
                cudaMalloc(&buffer, output.size);
                memoryPool_.outputBuffers.push_back(buffer);
            }
        }
    }
    
    int getNextContext() {
        // Round-robin context selection
        int startIdx = currentContext_.load();
        for (int i = 0; i < MAX_CONTEXTS; ++i) {
            int idx = (startIdx + i) % MAX_CONTEXTS;
            if (!contexts_[idx].inUse) {
                currentContext_ = (idx + 1) % MAX_CONTEXTS;
                return idx;
            }
        }
        return -1; // All contexts busy
    }
    
    void updateMetrics(cudaStream_t stream) {
        // Update inference metrics
        float elapsed = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, stream);
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&elapsed, start, stop);
        
        // Update rolling average
        int count = inferenceCount_.fetch_add(1);
        float avg = avgInferenceTime_.load();
        avgInferenceTime_ = (avg * count + elapsed) / (count + 1);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    size_t getDataTypeSize(nvinfer1::DataType dtype) {
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF: return 2;
            case nvinfer1::DataType::kINT8: return 1;
            case nvinfer1::DataType::kINT32: return 4;
            case nvinfer1::DataType::kBOOL: return 1;
            default: return 4;
        }
    }
};