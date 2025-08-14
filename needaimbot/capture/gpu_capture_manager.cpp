#include "gpu_capture_manager.h"
#include "../AppContext.h"
#include "../keyboard/keyboard_listener.h"
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
    
    
    // 1. DXGI/D3D11 초기화
    
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
    if (!InitializeCUDAInterop()) {
        std::cerr << "[GPUCapture] Failed to initialize CUDA interop" << std::endl;
        return false;
    }
    
    // Clear any previous CUDA errors before allocating buffers
    cudaError_t prevErr = cudaGetLastError();
    if (prevErr != cudaSuccess) {
        std::cerr << "[GPUCapture] Clearing previous CUDA error: " << cudaGetErrorString(prevErr) << std::endl;
    }
    
    // 4. GPU 프레임 버퍼 할당 - 제거 (불필요한 중간 버퍼)
    
    // Mouse controller initialization removed - handled by unified pipeline
    
    std::cout << "[GPUCapture] GPU Mouse Controller initialized successfully" << std::endl;
    
    return true;
}

void GPUCaptureManager::InitializeDXGI() {
    
    
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
    
    
    // DXGI Output Duplication 초기화
    
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
    
    
    // Staging texture for GPU-GPU copy
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

bool GPUCaptureManager::InitializeCUDAInterop() {
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    // CUDA 디바이스 설정
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "[GPUCapture] Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // CUDA stream 생성 (비동기 처리용)
    err = cudaStreamCreateWithFlags(&m_captureStream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "[GPUCapture] Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // CUDA event 생성 (GPU 동기화용)
    err = cudaEventCreateWithFlags(&m_frameReadyEvent, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "[GPUCapture] Failed to create CUDA event: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // D3D11 texture를 CUDA에 등록
    if (m_stagingTexture) {
        err = cudaGraphicsD3D11RegisterResource(
            &m_cudaResource,
            m_stagingTexture.Get(),
            cudaGraphicsRegisterFlagsNone
        );
        if (err != cudaSuccess) {
            std::cerr << "[GPUCapture] Failed to register D3D11 resource: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
    }
    
    return true;
}

bool GPUCaptureManager::WaitForNextFrame() {
    // VSync 또는 특정 타이밍까지 GPU가 대기
    // CPU는 여기서 블록되지만, 이벤트 기반이므로 CPU 사용률 0%
    
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    ComPtr<IDXGIResource> resource;
    
    // 더 빠른 폴링을 위해 타임아웃을 1ms로 설정 (1000 FPS까지 가능)
    HRESULT hr = m_duplication->AcquireNextFrame(1, &frameInfo, &resource);
    
    if (FAILED(hr)) {
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            // 타임아웃은 정상 - 새 프레임 없음
            return false;
        }
        
        // 에러 처리
        m_duplication->ReleaseFrame();
        return false;
    }
    
    // GPU에서 GPU로 직접 복사 (CPU 관여 없음)
    ComPtr<ID3D11Texture2D> frameTex;
    resource.As(&frameTex);
    
    // 화면 중앙에서 원하는 크기만큼 크롭
    D3D11_TEXTURE2D_DESC fullDesc;
    frameTex->GetDesc(&fullDesc);
    
    
    
    // 캡처할 영역 계산 (화면 중앙 + 오프셋)
    auto& ctx = AppContext::getInstance();
    int centerX = fullDesc.Width / 2;
    int centerY = fullDesc.Height / 2;
    
    // 오프셋 적용 (config에서 가져옴)
    // 조준+사격 중이면 aim_shoot_offset 사용, 아니면 일반 offset 사용
    bool useAimShootOffset = false;
    if (ctx.config.enable_aim_shoot_offset) {
        // 조준 버튼과 사격 버튼이 모두 눌렸는지 확인
        bool aimPressed = !ctx.config.button_targeting.empty() && 
                         isAnyKeyPressed(ctx.config.button_targeting);
        bool shootPressed = !ctx.config.button_auto_shoot.empty() && 
                           isAnyKeyPressed(ctx.config.button_auto_shoot);
        useAimShootOffset = aimPressed && shootPressed;
    }
    
    int offsetX = useAimShootOffset ? 
                  static_cast<int>(ctx.config.aim_shoot_offset_x) : 
                  static_cast<int>(ctx.config.crosshair_offset_x);
    int offsetY = useAimShootOffset ? 
                  static_cast<int>(ctx.config.aim_shoot_offset_y) : 
                  static_cast<int>(ctx.config.crosshair_offset_y);
    
    // 캡처 영역의 좌상단 계산
    int srcX = centerX + offsetX - (m_width / 2);
    int srcY = centerY + offsetY - (m_height / 2);
    
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
    
    // ProcessGPUFrame() 호출 제거 - UnifiedGraphPipeline이 처리
    // 실제 데이터는 m_stagingTexture에 있고, GetCudaResource()를 통해 접근
    
    // 캡처 성공
    return true;
}

// ProcessGPUFrame 함수 제거 - UnifiedGraphPipeline이 리소스 관리를 담당

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

// ProcessDetectionsGPU removed - mouse movement handled by unified pipeline