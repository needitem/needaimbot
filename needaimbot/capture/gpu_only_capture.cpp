#include "../AppContext.h"
#include "gpu_capture_manager.h"
#include "global_gpu_buffer.h"
#include "../detector/detector.h"
#include "../cuda/unified_graph_pipeline.h"
#include <thread>
#include <chrono>
#include "../core/windows_headers.h"
#include <iostream>

// GPU 전용 캡처 스레드 - CPU 사용 최소화
void gpuOnlyCaptureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT) {
    auto& ctx = AppContext::getInstance();
    
    // GPU 캡처 매니저 초기화
    GPUCaptureManager gpuCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
    if (!gpuCapture.Initialize()) {
        return;
    }
    
    // 스레드 우선순위는 NORMAL로 설정 (CPU를 거의 안 쓰므로)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
    
    // FPS 카운터 (GPU 기반)
    int frameCount = 0;
    auto lastFpsTime = std::chrono::steady_clock::now();
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    gpuCapture.StartCapture();
    
    // Get pipeline instance initialized in main
    auto& pipelineManager = needaimbot::PipelineManager::getInstance();
    auto* pipeline = pipelineManager.getPipeline();
    
    if (!pipeline) {
        gpuCapture.StopCapture();
        return;
    }
    
    // Pipeline에 CUDA 리소스 설정
    pipeline->setInputTexture(gpuCapture.GetCudaResource());
    
    // GPU 캡처 루프 - 오직 캡처만 수행
    int loopCount = 0;
    
    while (!ctx.should_exit) {
        // GPU가 새 프레임을 기다림 (CPU는 이벤트 대기 상태로 휴면)
        bool frameAvailable = gpuCapture.WaitForNextFrame();
        
        if (frameAvailable) {
            frameCount++;
            loopCount++;
            
            
            
            // 프레임 간격 측정
            auto currentTime = std::chrono::steady_clock::now();
            auto frameDelta = std::chrono::duration<float, std::milli>(currentTime - lastFrameTime).count();
            lastFrameTime = currentTime;
            
            // setInputFrame 제거 - UnifiedGraphPipeline이 D3D11 텍스처에서 직접 데이터 가져옴
            
            // UnifiedGraphPipeline에 프레임 전달 및 전체 파이프라인 실행
            // Pipeline이 Detection, Tracking, Mouse 이동까지 모두 처리
            if (ctx.use_cuda_graph && pipeline && pipeline->isGraphReady()) {
                // CUDA Graph 실행 (최적화된 경로)
                pipeline->executeGraph();
            } else if (pipeline) {
                // Direct 실행 (fallback)
                pipeline->executeDirect();
            }
            
            // FPS 계산 (1초마다)
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - lastFpsTime).count();
            if (elapsed >= 1.0f) {
                float fps = frameCount / elapsed;
                ctx.g_current_capture_fps.store(fps);
                frameCount = 0;
                lastFpsTime = now;
            }
        }
        
        // 설정 변경 체크 (최소한의 CPU 사용)
        if (ctx.capture_method_changed.load()) {
            ctx.capture_method_changed.store(false);
        }
        
        // 종료 신호 체크
        if (ctx.should_exit) {
            break;
        }
    }
    
    // 정리
    gpuCapture.StopCapture();
}