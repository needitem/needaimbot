#pragma once
#include <Windows.h>
#include <thread>
#include <chrono>
#include <atomic>

class CPUOptimizer {
private:
    static inline std::atomic<bool> highPerformanceMode{false};
    static inline std::atomic<int> targetFPS{60};
    
public:
    // CPU 친화성 설정 - 특정 코어에 스레드 고정
    static void SetThreadAffinity(int coreId) {
        DWORD_PTR mask = 1ULL << coreId;
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
    
    // 스레드 우선순위 최적화
    static void OptimizeThreadPriority(bool isMainThread = false) {
        if (isMainThread) {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
        } else {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
        }
    }
    
    // 적응형 슬립 - CPU 사용률과 FPS에 따라 동적 조정
    static void AdaptiveSleep(int targetFPSOverride = -1) {
        int fps = (targetFPSOverride > 0) ? targetFPSOverride : targetFPS.load();
        
        if (highPerformanceMode.load()) {
            // 고성능 모드: yield만 사용
            std::this_thread::yield();
        } else {
            // 일반 모드: 목표 FPS에 맞춰 슬립
            int sleepUs = 1000000 / fps;
            
            // 정밀한 타이밍을 위해 busy-wait와 sleep 조합
            auto start = std::chrono::high_resolution_clock::now();
            
            // 대부분의 시간은 sleep
            if (sleepUs > 1000) {
                std::this_thread::sleep_for(std::chrono::microseconds(sleepUs - 500));
            }
            
            // 나머지는 yield로 정밀 조정
            while (std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start).count() < sleepUs) {
                std::this_thread::yield();
            }
        }
    }
    
    // 조건부 yield - CPU 사용률 감소
    static void ConditionalYield(bool condition) {
        if (condition) {
            std::this_thread::yield();
        }
    }
    
    // 성능 모드 설정
    static void SetHighPerformanceMode(bool enable) {
        highPerformanceMode.store(enable);
        
        if (enable) {
            // Windows 타이머 정밀도 향상
            timeBeginPeriod(1);
        } else {
            // 기본 타이머 정밀도로 복원
            timeEndPeriod(1);
        }
    }
    
    // 목표 FPS 설정
    static void SetTargetFPS(int fps) {
        targetFPS.store(fps);
    }
    
    // 스마트 대기 - 이벤트 기반 대기
    template<typename Predicate>
    static bool WaitForCondition(Predicate pred, int timeoutMs = 100) {
        auto start = std::chrono::steady_clock::now();
        
        while (!pred()) {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count() > timeoutMs) {
                return false;
            }
            
            // CPU 사용률 감소를 위한 짧은 슬립
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        return true;
    }
    
    // 배치 처리를 위한 프레임 축적
    class FrameBatcher {
    private:
        int batchSize;
        int currentBatch;
        
    public:
        FrameBatcher(int size = 2) : batchSize(size), currentBatch(0) {}
        
        bool ShouldProcess() {
            currentBatch++;
            if (currentBatch >= batchSize) {
                currentBatch = 0;
                return true;
            }
            return false;
        }
        
        void Reset() {
            currentBatch = 0;
        }
    };
};

// RAII 패턴으로 스레드 최적화 자동 관리
class ThreadOptimizationScope {
private:
    HANDLE thread;
    DWORD_PTR oldMask;
    int oldPriority;
    
public:
    ThreadOptimizationScope(int coreId = -1, int priority = THREAD_PRIORITY_NORMAL) {
        thread = GetCurrentThread();
        
        // 이전 설정 저장
        oldMask = SetThreadAffinityMask(thread, (DWORD_PTR)-1);
        SetThreadAffinityMask(thread, oldMask);
        oldPriority = GetThreadPriority(thread);
        
        // 새 설정 적용
        if (coreId >= 0) {
            SetThreadAffinityMask(thread, 1ULL << coreId);
        }
        SetThreadPriority(thread, priority);
    }
    
    ~ThreadOptimizationScope() {
        // 원래 설정으로 복원
        SetThreadAffinityMask(thread, oldMask);
        SetThreadPriority(thread, oldPriority);
    }
};