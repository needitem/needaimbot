#pragma once

#include "capture.h"
#include "../cuda/simple_cuda_mat.h"
#include <d3d11.h>
#include <dxgi1_2.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <memory>
#include <atomic>

class ZeroCopyCapture : public IScreenCapture {
private:
    // D3D11 resources
    ID3D11Device* d3dDevice_ = nullptr;
    ID3D11DeviceContext* d3dContext_ = nullptr;
    IDXGIOutputDuplication* duplication_ = nullptr;
    
    // CUDA-D3D11 interop resources
    cudaGraphicsResource_t cudaResource_ = nullptr;
    ID3D11Texture2D* stagingTexture_ = nullptr;
    
    // Frame management
    SimpleCudaMat gpuFrame_;
    cudaEvent_t captureDoneEvent_;
    cudaStream_t captureStream_;
    
    // State
    std::atomic<bool> initialized_{false};
    int width_;
    int height_;
    float offsetX_ = 0.0f;
    float offsetY_ = 0.0f;
    
    // Performance optimization
    bool useNvLink_ = false;
    
public:
    ZeroCopyCapture(int width, int height);
    ~ZeroCopyCapture();
    
    // IScreenCapture interface
    SimpleCudaMat GetNextFrameGpu() override;
    cudaEvent_t GetCaptureDoneEvent() const override { return captureDoneEvent_; }
    bool IsInitialized() const override { return initialized_.load(); }
    void UpdateCaptureRegion(float offsetX, float offsetY) override;
    
private:
    bool InitializeD3D11();
    bool InitializeDuplication();
    bool InitializeCudaInterop();
    void ReleaseResources();
    bool CaptureFrameDirect();
};