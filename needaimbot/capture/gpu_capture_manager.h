#pragma once

// Windows SDK 10.0.26100.0 compatible
#include <windows.h>
#include <d3d11_4.h>  // D3D11.4 for ID3D11Device5 and ID3D11Fence
#include <dxgi1_5.h>  // DXGI 1.5
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <atomic>
#include <thread>
#include <memory>
#include "../cuda/simple_cuda_mat.h"
#include "../cuda/gpu_mouse_controller.h"

// SimpleCudaMat typedef for compatibility
using SimpleCudaMat = SimpleCudaMat;

using Microsoft::WRL::ComPtr;

class GPUCaptureManager {
public:
    GPUCaptureManager(int width, int height);
    ~GPUCaptureManager();
    
    bool Initialize();
    void StartCapture();
    void StopCapture();
    
    // GPU 이벤트 기반 - CPU 대기 없음
    SimpleCudaMat& WaitForNextFrame();
    
    // GPU 마우스 컨트롤러 접근
    needaimbot::cuda::GPUMouseController* GetMouseController() { return m_mouseController.get(); }
    
    // YOLO 감지 결과 처리 및 마우스 이동량 계산 (GPU에서 직접)
    bool ProcessDetectionsGPU(needaimbot::cuda::Detection* d_detections, int numDetections, 
                              needaimbot::cuda::MouseMovement& movement);
    
private:
    // GPU 동기화를 위한 Fence (Windows 10/11)
    ComPtr<ID3D11Fence> m_fence;
    ComPtr<ID3D11Device5> m_device5;  // D3D11.4 device (CreateFence support)
    ComPtr<ID3D11DeviceContext4> m_context4;  // D3D11.3 context  
    HANDLE m_fenceEvent;
    UINT64 m_fenceValue;
    
    // DXGI Desktop Duplication
    ComPtr<IDXGIOutputDuplication> m_duplication;
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<ID3D11Texture2D> m_stagingTexture;
    
    // CUDA interop
    cudaGraphicsResource_t m_cudaResource;
    cudaStream_t m_captureStream;
    cudaEvent_t m_frameReadyEvent;
    
    // 캡처 영역
    int m_width;
    int m_height;
    
    // VSync 동기화
    bool m_useVSync;
    std::atomic<bool> m_isCapturing;
    
    // GPU 전용 프레임 버퍼
    SimpleCudaMat m_gpuFrameBuffer;
    
    // GPU 마우스 컨트롤러
    std::unique_ptr<needaimbot::cuda::GPUMouseController> m_mouseController;
    
    void InitializeDXGI();
    void InitializeCUDAInterop();
    void ProcessGPUFrame();
};