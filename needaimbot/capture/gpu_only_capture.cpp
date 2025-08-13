#include "../AppContext.h"
#include "gpu_capture_manager.h"
#include "global_gpu_buffer.h"
#include "../detector/detector.h"
#include <thread>
#include <chrono>
#include <windows.h>

// GPU 전용 캡처 스레드 - CPU 사용 최소화
void gpuOnlyCaptureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT) {
    auto& ctx = AppContext::getInstance();
    
    // GPU 캡처 매니저 초기화
    
    GPUCaptureManager gpuCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
    if (!gpuCapture.Initialize()) {
        std::cerr << "[GPUCapture] Failed to initialize GPU capture" << std::endl;
        return;
    }
    
    
    // 스레드 우선순위는 NORMAL로 설정 (CPU를 거의 안 쓰므로)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
    
    // GPU 이벤트 핸들
    HANDLE gpuFrameEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    
    // FPS 카운터 (GPU 기반)
    int frameCount = 0;
    auto lastFpsTime = std::chrono::steady_clock::now();
    
    gpuCapture.StartCapture();
    
    int debugFrameCount = 0;
    while (!ctx.should_exit) {
        // GPU가 새 프레임을 기다림 (CPU는 이벤트 대기 상태로 휴면)
        // AcquireNextFrame(16)은 최대 16ms 대기 = 60 FPS
        // 새 프레임이 있으면 즉시 반환, 없으면 타임아웃까지 대기
        SimpleCudaMat& gpuFrame = gpuCapture.WaitForNextFrame();
        
        if (!gpuFrame.empty()) {
            frameCount++;
            debugFrameCount++;
            
            
            
            // 전역 GPU 버퍼 업데이트 (프리뷰용)
            latestFrameGpu.copyFrom(gpuFrame);
            
            
            
            // GPU에서 직접 Detector로 전달 (CPU 관여 없음)
            if (ctx.detector) {
                
                ctx.detector->processFrame(gpuFrame);
            } else {
                
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
            // GPU 캡처는 하나의 방식만 사용
            ctx.capture_method_changed.store(false);
        }
        
        // 종료 신호 체크
        if (ctx.should_exit) {
            break;
        }
    }
    
    gpuCapture.StopCapture();
    CloseHandle(gpuFrameEvent);
    
    
}