#include "gpu_capture_manager.h"
#include <iostream>
#include <chrono>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

GPUCaptureManager::GPUCaptureManager(int width, int height) 
    : m_width(width)
    , m_height(height)
    , m_fenceValue(0)
    , m_useVSync(true)
    , m_isCapturing(false)
    , m_cudaResource(nullptr)
    , m_captureStream(nullptr)
    , m_frameReadyEvent(nullptr)
    , m_fenceEvent(nullptr) {
}

GPUCaptureManager::~GPUCaptureManager() {
    StopCapture();
    
    if (m_fenceEvent) {
        CloseHandle(m_fenceEvent);
    }
    
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
    }
    
    if (m_captureStream) {
        cudaStreamDestroy(m_captureStream);
    }
    
    if (m_frameReadyEvent) {
        cudaEventDestroy(m_frameReadyEvent);
    }
}

bool GPUCaptureManager::Initialize() {
    std::cout << "[GPUCapture] Initialize: Starting initialization..." << std::endl;
    
    // 1. DXGI/D3D11 초기화
    std::cout << "[GPUCapture] Initialize: Calling InitializeDXGI..." << std::endl;
    InitializeDXGI();
    
    // 2. D3D11.4 인터페이스 획득 (Windows 10/11)
    HRESULT hr = m_device->QueryInterface(__uuidof(ID3D11Device5), (void**)&m_device5);
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] Failed to get ID3D11Device5 interface" << std::endl;
        return false;
    }
    
    hr = m_context->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&m_context4);
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] Failed to get ID3D11DeviceContext4 interface" << std::endl;
        return false;
    }
    
    // 3. GPU Fence 생성 (최신 동기화 메커니즘)
    hr = m_device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] Failed to create D3D11 Fence" << std::endl;
        return false;
    }
    
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) {
        std::cerr << "[GPUCapture] Failed to create fence event" << std::endl;
        return false;
    }
    
    // 3. CUDA Interop 초기화
    InitializeCUDAInterop();
    
    // 4. GPU 프레임 버퍼 할당
    m_gpuFrameBuffer.create(m_height, m_width, 4); // BGRA
    
    return true;
}

void GPUCaptureManager::InitializeDXGI() {
    std::cout << "[GPUCapture] InitializeDXGI: Creating D3D11 device..." << std::endl;
    
    // D3D11 디바이스 생성
    D3D_FEATURE_LEVEL featureLevel;
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        nullptr, 0,
        D3D11_SDK_VERSION,
        &m_device,
        &featureLevel,
        &m_context
    );
    
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] ERROR: Failed to create D3D11 device! HRESULT=" << std::hex << hr << std::dec << std::endl;
        return;
    }
    std::cout << "[GPUCapture] D3D11 device created successfully" << std::endl;
    
    // DXGI Output Duplication 초기화
    std::cout << "[GPUCapture] Getting DXGI interfaces..." << std::endl;
    ComPtr<IDXGIDevice> dxgiDevice;
    hr = m_device.As(&dxgiDevice);
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] ERROR: Failed to get IDXGIDevice! HRESULT=" << std::hex << hr << std::dec << std::endl;
        return;
    }
    
    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] ERROR: Failed to get adapter! HRESULT=" << std::hex << hr << std::dec << std::endl;
        return;
    }
    
    ComPtr<IDXGIOutput> output;
    hr = adapter->EnumOutputs(0, &output);
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] ERROR: Failed to enumerate outputs! HRESULT=" << std::hex << hr << std::dec << std::endl;
        return;
    }
    
    ComPtr<IDXGIOutput1> output1;
    hr = output.As(&output1);
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] ERROR: Failed to get IDXGIOutput1! HRESULT=" << std::hex << hr << std::dec << std::endl;
        return;
    }
    
    // Desktop Duplication API
    std::cout << "[GPUCapture] Initializing Desktop Duplication..." << std::endl;
    hr = output1->DuplicateOutput(m_device.Get(), &m_duplication);
    if (FAILED(hr)) {
        std::cerr << "[GPUCapture] ERROR: Failed to duplicate output! HRESULT=" << std::hex << hr << std::dec << std::endl;
        if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE) {
            std::cerr << "[GPUCapture] Desktop Duplication is not available (another app may be using it)" << std::endl;
        } else if (hr == E_ACCESSDENIED) {
            std::cerr << "[GPUCapture] Access denied - need to run as administrator or check display settings" << std::endl;
        }
        return;
    }
    std::cout << "[GPUCapture] Desktop Duplication initialized successfully" << std::endl;
    
    // Staging texture for GPU-GPU copy
    std::cout << "[GPUCapture] Creating staging texture: " << m_width << "x" << m_height << std::endl;
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = m_width;
    desc.Height = m_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    m_device->CreateTexture2D(&desc, nullptr, &m_stagingTexture);
}

void GPUCaptureManager::InitializeCUDAInterop() {
    // CUDA stream 생성 (비동기 처리용)
    cudaStreamCreateWithFlags(&m_captureStream, cudaStreamNonBlocking);
    
    // CUDA event 생성 (GPU 동기화용)
    cudaEventCreateWithFlags(&m_frameReadyEvent, cudaEventDisableTiming);
    
    // D3D11 texture를 CUDA에 등록
    cudaGraphicsD3D11RegisterResource(
        &m_cudaResource,
        m_stagingTexture.Get(),
        cudaGraphicsRegisterFlagsNone
    );
    
    // CUDA 디바이스 설정
    cudaSetDevice(0);
}

SimpleCudaMat& GPUCaptureManager::WaitForNextFrame() {
    // VSync 또는 특정 타이밍까지 GPU가 대기
    // CPU는 여기서 블록되지만, 이벤트 기반이므로 CPU 사용률 0%
    
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    ComPtr<IDXGIResource> resource;
    
    // INFINITE 대기 - 새 프레임이 올 때까지 CPU 대기 (이벤트 기반)
    // 또는 16ms (60fps) 대기
    HRESULT hr = m_duplication->AcquireNextFrame(16, &frameInfo, &resource);
    
    if (FAILED(hr)) {
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            // 타임아웃은 정상 - 기존 프레임 반환
            return m_gpuFrameBuffer;
        }
        
        // 에러 처리
        m_duplication->ReleaseFrame();
        return m_gpuFrameBuffer;
    }
    
    // GPU에서 GPU로 직접 복사 (CPU 관여 없음)
    ComPtr<ID3D11Texture2D> frameTex;
    resource.As(&frameTex);
    
    // 화면 중앙에서 원하는 크기만큼 크롭
    D3D11_TEXTURE2D_DESC fullDesc;
    frameTex->GetDesc(&fullDesc);
    
    
    
    // 캡처할 영역 계산 (화면 중앙)
    int srcX = (fullDesc.Width - m_width) / 2;
    int srcY = (fullDesc.Height - m_height) / 2;
    
    // 영역이 화면을 벗어나지 않도록 보정
    srcX = max(0, min(srcX, (int)fullDesc.Width - m_width));
    srcY = max(0, min(srcY, (int)fullDesc.Height - m_height));
    
    
    
    // 특정 영역만 복사
    D3D11_BOX srcBox;
    srcBox.left = srcX;
    srcBox.right = srcX + m_width;
    srcBox.top = srcY;
    srcBox.bottom = srcY + m_height;
    srcBox.front = 0;
    srcBox.back = 1;
    
    // GPU command를 큐에 넣기만 함 (비동기)
    m_context->CopySubresourceRegion(
        m_stagingTexture.Get(), 0,  // Destination
        0, 0, 0,                     // Destination position
        frameTex.Get(), 0,           // Source 
        &srcBox                      // Source region
    );
    
    // GPU Fence 신호 - GPU가 작업 완료시 신호
    m_context4->Signal(m_fence.Get(), ++m_fenceValue);
    
    // CPU가 GPU 완료를 기다림 (이벤트 기반, CPU 사용 최소)
    m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent);
    
    // 비동기로 대기 (다른 작업 가능)
    // WaitForSingleObject(m_fenceEvent, INFINITE);
    
    // 프레임 해제
    m_duplication->ReleaseFrame();
    
    // CUDA로 비동기 복사
    ProcessGPUFrame();
    
    return m_gpuFrameBuffer;
}

void GPUCaptureManager::ProcessGPUFrame() {
    // CUDA 리소스 맵핑 (GPU-GPU)
    cudaGraphicsMapResources(1, &m_cudaResource, m_captureStream);
    
    cudaArray_t cudaArray;
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, m_cudaResource, 0, 0);
    
    // GPU에서 GPU로 직접 복사 (비동기)
    cudaMemcpy2DFromArrayAsync(
        m_gpuFrameBuffer.data(),
        m_gpuFrameBuffer.step(),
        cudaArray,
        0, 0,
        m_width * 4,
        m_height,
        cudaMemcpyDeviceToDevice,
        m_captureStream
    );
    
    // 리소스 언맵
    cudaGraphicsUnmapResources(1, &m_cudaResource, m_captureStream);
    
    // GPU 이벤트 기록 (다른 스트림에서 동기화 가능)
    cudaEventRecord(m_frameReadyEvent, m_captureStream);
}

void GPUCaptureManager::StartCapture() {
    m_isCapturing = true;
}

void GPUCaptureManager::StopCapture() {
    m_isCapturing = false;
    
    // GPU 작업 완료 대기
    if (m_captureStream) {
        cudaStreamSynchronize(m_captureStream);
    }
}