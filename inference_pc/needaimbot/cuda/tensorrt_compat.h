#pragma once

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <vector>
#include <string>
#include <map>
#include <iostream>

// TensorRT version compatibility layer
// TRT 10.x uses new API (enqueueV3, setTensorAddress, getTensorShape, etc.)
// TRT 8.x uses old API (enqueueV2, bindings array, getBindingDimensions, etc.)

#if NV_TENSORRT_MAJOR >= 10

#define TRT_USE_V3_API 1
#define TRT_ENQUEUE(ctx, stream) ((ctx)->enqueueV3(stream))

// Helper class for TRT 10.x - just wraps the native API
class TRTBindingsHelper {
public:
    TRTBindingsHelper(nvinfer1::ICudaEngine* engine) : engine_(engine) {}

    void setBinding(const char* name, void* address) {
        // In TRT 10.x, we use setTensorAddress directly on context
        bindings_[name] = address;
    }

    bool applyBindings(nvinfer1::IExecutionContext* context) {
        for (const auto& binding : bindings_) {
            if (!context->setTensorAddress(binding.first.c_str(), binding.second)) {
                return false;
            }
        }
        return true;
    }

    nvinfer1::Dims getTensorShape(const char* name) {
        return engine_->getTensorShape(name);
    }

    nvinfer1::DataType getTensorDataType(const char* name) {
        return engine_->getTensorDataType(name);
    }

    int getNbIOTensors() {
        return engine_->getNbIOTensors();
    }

    const char* getIOTensorName(int index) {
        return engine_->getIOTensorName(index);
    }

    bool isInputTensor(const char* name) {
        return engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    }

    bool isOutputTensor(const char* name) {
        return engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT;
    }

private:
    nvinfer1::ICudaEngine* engine_;
    std::map<std::string, void*> bindings_;
};

#else // TRT 8.x

#define TRT_USE_V3_API 0

// Helper class for TRT 8.x bindings management
class TRTBindingsHelper {
public:
    TRTBindingsHelper(nvinfer1::ICudaEngine* engine) : engine_(engine) {
        int nbBindings = engine_->getNbBindings();
        bindings_.resize(nbBindings, nullptr);
    }

    void setBinding(const char* name, void* address) {
        int index = engine_->getBindingIndex(name);
        if (index >= 0 && index < static_cast<int>(bindings_.size())) {
            bindings_[index] = address;
        }
    }

    bool applyBindings(nvinfer1::IExecutionContext* context) {
        // TRT 8.x: Bindings are passed to enqueueV2, not set separately
        (void)context;
        return true;
    }

    void* const* getBindingsArray() const {
        return bindings_.data();
    }

    nvinfer1::Dims getTensorShape(const char* name) {
        int index = engine_->getBindingIndex(name);
        if (index >= 0) {
            return engine_->getBindingDimensions(index);
        }
        return nvinfer1::Dims{};
    }

    nvinfer1::DataType getTensorDataType(const char* name) {
        int index = engine_->getBindingIndex(name);
        if (index >= 0) {
            return engine_->getBindingDataType(index);
        }
        return nvinfer1::DataType::kFLOAT;
    }

    int getNbIOTensors() {
        return engine_->getNbBindings();
    }

    const char* getIOTensorName(int index) {
        return engine_->getBindingName(index);
    }

    bool isInputTensor(const char* name) {
        int index = engine_->getBindingIndex(name);
        return index >= 0 && engine_->bindingIsInput(index);
    }

    bool isOutputTensor(const char* name) {
        int index = engine_->getBindingIndex(name);
        return index >= 0 && !engine_->bindingIsInput(index);
    }

    // TRT 8.x: enqueue with bindings array
    bool enqueue(nvinfer1::IExecutionContext* context, cudaStream_t stream) {
        return context->enqueueV2(bindings_.data(), stream, nullptr);
    }

private:
    nvinfer1::ICudaEngine* engine_;
    std::vector<void*> bindings_;
};

// Macro that uses the helper for TRT 8.x
#define TRT_ENQUEUE(ctx, stream) (bindingsHelper_->enqueue(ctx, stream))

#endif // NV_TENSORRT_MAJOR >= 10
