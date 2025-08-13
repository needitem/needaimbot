#include "../AppContext.h"
#include "gpu_capture_manager.h"
#include "global_gpu_buffer.h"
#include "../detector/detector.h"
#include "../cuda/gpu_mouse_controller.h"
#include "../mouse/mouse.h"
#include <thread>
#include <chrono>
#include <windows.h>
#include <iostream>

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
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    gpuCapture.StartCapture();
    
    int debugFrameCount = 0;
    
    // GPU 마우스 이동량 결과 버퍼
    needaimbot::cuda::MouseMovement gpuMovement;
    
    while (!ctx.should_exit) {
        // GPU가 새 프레임을 기다림 (CPU는 이벤트 대기 상태로 휴면)
        // AcquireNextFrame(16)은 최대 16ms 대기 = 60 FPS
        // 새 프레임이 있으면 즉시 반환, 없으면 타임아웃까지 대기
        SimpleCudaMat& gpuFrame = gpuCapture.WaitForNextFrame();
        
        if (!gpuFrame.empty()) {
            frameCount++;
            debugFrameCount++;
            
            // 프레임 간격 측정
            auto currentTime = std::chrono::steady_clock::now();
            auto frameDelta = std::chrono::duration<float, std::milli>(currentTime - lastFrameTime).count();
            lastFrameTime = currentTime;
            
            // 로그 제거 - 성능 향상
            // if (debugFrameCount % 60 == 0) {
            //     std::cout << "[GPU Capture] Frame interval: " << frameDelta << "ms (" << (1000.0f/frameDelta) << " FPS)" << std::endl;
            // }
            
            // 전역 GPU 버퍼 업데이트 (프리뷰용)
            latestFrameGpu.copyFrom(gpuFrame);
            
            // GPU에서 직접 Detector로 전달 (CPU 관여 없음)
            if (ctx.detector) {
                auto detectStart = std::chrono::steady_clock::now();
                ctx.detector->processFrame(gpuFrame);
                auto detectEnd = std::chrono::steady_clock::now();
                
                // 로그 제거 - 성능 향상
                // if (debugFrameCount % 60 == 0) {
                //     auto detectTime = std::chrono::duration<float, std::milli>(detectEnd - detectStart).count();
                //     std::cout << "[GPU Capture] YOLO inference time: " << detectTime << "ms" << std::endl;
                // }
                
                // YOLO 감지 결과가 있으면 GPU에서 직접 마우스 이동량 계산
                auto detections = ctx.detector->getLatestDetectionsGPU();
                if (detections.first && detections.second > 0) {
                    // Target과 Detection이 이제 동일한 구조이므로 직접 캐스팅 가능
                    if (gpuCapture.ProcessDetectionsGPU(
                        (needaimbot::cuda::Detection*)detections.first,
                        detections.second, 
                        gpuMovement)) {
                        
                        // 마우스 이동이 필요한 경우만 CPU로 전달
                        if (gpuMovement.shouldMove && ctx.aiming) {  // aiming 상태 체크 추가
                            // 마우스 스레드로 이동량 전달 (최소한의 데이터만)
                            ctx.latestMouseMovement.dx = static_cast<int>(gpuMovement.dx);
                            ctx.latestMouseMovement.dy = static_cast<int>(gpuMovement.dy);
                            ctx.latestMouseMovement.confidence = gpuMovement.confidence;
                            ctx.latestMouseMovement.hasTarget = true;
                            
                            // 마우스 스레드에 신호
                            ctx.mouseDataReady.store(true);
                            ctx.mouseDataCV.notify_one();
                        }
                    }
                }
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