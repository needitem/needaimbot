#pragma once
#include <Windows.h>
#include <cuda_runtime.h>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <queue>

// 진짜 이벤트 기반 GPU 파이프라인
class GpuEventPipeline {
private:
    // 이벤트 핸들들
    HANDLE captureReadyEvent;      // 새 프레임 준비됨
    HANDLE targetDetectedEvent;     // 타겟 탐지됨
    HANDLE mouseMovedEvent;         // 마우스 이동 완료
    
    // 조건 변수 (크로스 플랫폼)
    std::condition_variable captureCV;
    std::condition_variable detectionCV;
    std::condition_variable mouseCV;
    
    std::mutex captureMutex;
    std::mutex detectionMutex;
    std::mutex mouseMutex;
    
    // 데이터 큐
    struct FrameData {
        void* gpuData;
        size_t size;
        cudaStream_t stream;
    };
    
    struct TargetData {
        float x, y, w, h;
        float confidence;
        bool valid;
    };
    
    std::queue<FrameData> frameQueue;
    std::queue<TargetData> targetQueue;
    
    // 상태
    std::atomic<bool> running{true};
    std::atomic<bool> targetAvailable{false};
    
    // CUDA 스트림 (비동기 실행)
    cudaStream_t captureStream;
    cudaStream_t inferenceStream;
    cudaStream_t mouseStream;
    
    // CUDA 이벤트 (GPU 동기화)
    cudaEvent_t captureComplete;
    cudaEvent_t inferenceComplete;
    cudaEvent_t mouseComplete;
    
public:
    GpuEventPipeline() {
        // Windows 이벤트 생성
        captureReadyEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        targetDetectedEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        mouseMovedEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        
        // CUDA 스트림 생성 (우선순위 설정)
        int priority_high, priority_low;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
        
        cudaStreamCreateWithPriority(&captureStream, cudaStreamNonBlocking, priority_low);
        cudaStreamCreateWithPriority(&inferenceStream, cudaStreamNonBlocking, 0);
        cudaStreamCreateWithPriority(&mouseStream, cudaStreamNonBlocking, priority_high);
        
        // CUDA 이벤트 생성
        cudaEventCreate(&captureComplete);
        cudaEventCreate(&inferenceComplete);
        cudaEventCreate(&mouseComplete);
    }
    
    ~GpuEventPipeline() {
        CloseHandle(captureReadyEvent);
        CloseHandle(targetDetectedEvent);
        CloseHandle(mouseMovedEvent);
        
        cudaStreamDestroy(captureStream);
        cudaStreamDestroy(inferenceStream);
        cudaStreamDestroy(mouseStream);
        
        cudaEventDestroy(captureComplete);
        cudaEventDestroy(inferenceComplete);
        cudaEventDestroy(mouseComplete);
    }
    
    // 캡처 스레드: GPU에서 직접 캡처하고 이벤트 발생
    void gpuCaptureEventThread() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
        
        while (running) {
            // GPU 캡처 (비동기)
            void* gpuFrame = captureFrameGpu(captureStream);
            
            // 캡처 완료 이벤트 기록
            cudaEventRecord(captureComplete, captureStream);
            
            // 프레임 큐에 추가
            {
                std::lock_guard<std::mutex> lock(captureMutex);
                frameQueue.push({gpuFrame, 0, captureStream});
            }
            
            // 캡처 완료 알림 (CPU 이벤트)
            SetEvent(captureReadyEvent);
            
            // 조건 변수로도 알림
            captureCV.notify_one();
            
            // GPU가 너무 빨라서 CPU를 압도하지 않도록
            std::this_thread::yield();
        }
    }
    
    // 탐지 스레드: 이벤트 대기하고 GPU에서 추론
    void detectionThread() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
        
        while (running) {
            // 캡처 이벤트 대기 (CPU 0% 사용)
            WaitForSingleObject(captureReadyEvent, INFINITE);
            
            if (!running) break;
            
            // 프레임 가져오기
            FrameData frame;
            {
                std::lock_guard<std::mutex> lock(captureMutex);
                if (frameQueue.empty()) continue;
                frame = frameQueue.front();
                frameQueue.pop();
            }
            
            // 이전 스트림 대기 (캡처 완료 대기)
            cudaStreamWaitEvent(inferenceStream, captureComplete, 0);
            
            // GPU 추론 (비동기)
            TargetData target = runInferenceGpu(frame.gpuData, inferenceStream);
            
            // 추론 완료 이벤트 기록
            cudaEventRecord(inferenceComplete, inferenceStream);
            
            // 타겟이 있으면 이벤트 발생
            if (target.valid) {
                {
                    std::lock_guard<std::mutex> lock(detectionMutex);
                    targetQueue.push(target);
                    targetAvailable = true;
                }
                
                // 타겟 탐지 알림
                SetEvent(targetDetectedEvent);
                detectionCV.notify_one();
            }
        }
    }
    
    // 마우스 스레드: 이벤트 대기하고 GPU에서 PID 계산
    void mouseThread() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        
        while (running) {
            // 타겟 탐지 이벤트 대기 (CPU 0% 사용)
            WaitForSingleObject(targetDetectedEvent, INFINITE);
            
            if (!running) break;
            
            // 타겟 가져오기
            TargetData target;
            {
                std::lock_guard<std::mutex> lock(detectionMutex);
                if (targetQueue.empty()) continue;
                target = targetQueue.front();
                targetQueue.pop();
            }
            
            // 이전 스트림 대기 (추론 완료 대기)
            cudaStreamWaitEvent(mouseStream, inferenceComplete, 0);
            
            // GPU에서 PID 계산 (비동기)
            float2 movement = calculatePidGpu(target, mouseStream);
            
            // 마우스 이동 완료 이벤트 기록
            cudaEventRecord(mouseComplete, mouseStream);
            
            // 실제 마우스 이동 (Host 코드)
            cudaStreamSynchronize(mouseStream);  // 계산 완료 대기
            moveMouse(movement.x, movement.y);
            
            // 이동 완료 알림
            SetEvent(mouseMovedEvent);
        }
    }
    
    // 조건 변수 기반 대기 (대안)
    void waitForTarget(TargetData& target) {
        std::unique_lock<std::mutex> lock(detectionMutex);
        detectionCV.wait(lock, [this]() { return targetAvailable.load() || !running; });
        
        if (!running) return;
        
        if (!targetQueue.empty()) {
            target = targetQueue.front();
            targetQueue.pop();
            if (targetQueue.empty()) {
                targetAvailable = false;
            }
        }
    }
    
private:
    // GPU 함수들 (실제 구현 필요)
    void* captureFrameGpu(cudaStream_t stream) {
        // GPU 직접 캡처
        return nullptr;
    }
    
    TargetData runInferenceGpu(void* frame, cudaStream_t stream) {
        // GPU 추론
        return {0, 0, 0, 0, 0, false};
    }
    
    float2 calculatePidGpu(const TargetData& target, cudaStream_t stream) {
        // GPU PID 계산
        return {0, 0};
    }
    
    void moveMouse(float dx, float dy) {
        // 마우스 이동
    }
};

// 완전 비동기 파이프라인 매니저
class AsyncPipelineManager {
private:
    GpuEventPipeline pipeline;
    
    std::thread captureWorker;
    std::thread detectionWorker;
    std::thread mouseWorker;
    
    std::atomic<bool> running{true};
    
public:
    void start() {
        // 각 스레드를 다른 CPU 코어에 할당
        captureWorker = std::thread([this]() {
            SetThreadAffinityMask(GetCurrentThread(), 1 << 2);
            pipeline.captureThread();
        });
        
        detectionWorker = std::thread([this]() {
            SetThreadAffinityMask(GetCurrentThread(), 1 << 3);
            pipeline.detectionThread();
        });
        
        mouseWorker = std::thread([this]() {
            SetThreadAffinityMask(GetCurrentThread(), 1 << 4);
            pipeline.mouseThread();
        });
    }
    
    void stop() {
        running = false;
        
        // 모든 이벤트 트리거하여 스레드 깨우기
        SetEvent(pipeline.captureReadyEvent);
        SetEvent(pipeline.targetDetectedEvent);
        SetEvent(pipeline.mouseMovedEvent);
        
        if (captureWorker.joinable()) captureWorker.join();
        if (detectionWorker.joinable()) detectionWorker.join();
        if (mouseWorker.joinable()) mouseWorker.join();
    }
};