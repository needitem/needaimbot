#pragma once
#include <Windows.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <atomic>
#include <functional>
#include <thread>
#include <future>

// 이벤트 기반 캡처 시스템 - CPU 사용률 최소화
class EventBasedCapture {
public:
    enum EventType {
        FRAME_READY,
        CAPTURE_REQUESTED,
        PROCESSING_COMPLETE,
        CONFIG_CHANGED,
        SHUTDOWN
    };

    struct Event {
        EventType type;
        std::chrono::steady_clock::time_point timestamp;
        void* data;
    };

private:
    // 이벤트 큐와 동기화
    std::queue<Event> eventQueue;
    std::mutex queueMutex;
    std::condition_variable eventCV;
    
    // 프레임 준비 알림을 위한 Windows 이벤트
    HANDLE frameReadyEvent;
    HANDLE captureRequestEvent;
    HANDLE shutdownEvent;
    
    // 콜백 함수들
    std::function<void()> onFrameReady;
    std::function<void()> onCaptureRequest;
    std::function<void()> onProcessingComplete;
    
    // 상태 플래그
    std::atomic<bool> running{false};
    std::atomic<bool> frameAvailable{false};
    
    // 성능 카운터
    std::atomic<uint64_t> eventsProcessed{0};
    std::atomic<uint64_t> idleTime{0};

public:
    EventBasedCapture() {
        // Windows 이벤트 생성 (자동 리셋, 초기 신호 없음)
        frameReadyEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        captureRequestEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        shutdownEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    }
    
    ~EventBasedCapture() {
        CloseHandle(frameReadyEvent);
        CloseHandle(captureRequestEvent);
        CloseHandle(shutdownEvent);
    }
    
    // 이벤트 대기 (CPU 사용률 0%에 가까움)
    Event waitForEvent(int timeoutMs = INFINITE) {
        HANDLE events[] = { frameReadyEvent, captureRequestEvent, shutdownEvent };
        
        // WaitForMultipleObjects는 CPU를 사용하지 않고 대기
        DWORD result = WaitForMultipleObjects(3, events, FALSE, timeoutMs);
        
        Event evt;
        evt.timestamp = std::chrono::steady_clock::now();
        
        switch (result) {
            case WAIT_OBJECT_0:  // frameReadyEvent
                evt.type = FRAME_READY;
                break;
            case WAIT_OBJECT_0 + 1:  // captureRequestEvent
                evt.type = CAPTURE_REQUESTED;
                break;
            case WAIT_OBJECT_0 + 2:  // shutdownEvent
                evt.type = SHUTDOWN;
                break;
            case WAIT_TIMEOUT:
                evt.type = CAPTURE_REQUESTED;  // 타임아웃시 캡처 요청으로 처리
                break;
        }
        
        eventsProcessed++;
        return evt;
    }
    
    // 이벤트 트리거
    void triggerFrameReady() {
        SetEvent(frameReadyEvent);
    }
    
    void triggerCaptureRequest() {
        SetEvent(captureRequestEvent);
    }
    
    void triggerShutdown() {
        SetEvent(shutdownEvent);
    }
    
    // 조건 변수를 사용한 대기 (크로스 플랫폼)
    template<typename Predicate>
    void waitForCondition(Predicate pred) {
        std::unique_lock<std::mutex> lock(queueMutex);
        eventCV.wait(lock, pred);
    }
    
    // 이벤트 푸시
    void pushEvent(const Event& evt) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            eventQueue.push(evt);
        }
        eventCV.notify_one();
    }
    
    // 이벤트 팝
    bool popEvent(Event& evt) {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (eventQueue.empty()) return false;
        
        evt = eventQueue.front();
        eventQueue.pop();
        return true;
    }
    
    // 콜백 설정
    void setFrameReadyCallback(std::function<void()> callback) {
        onFrameReady = callback;
    }
    
    void setCaptureRequestCallback(std::function<void()> callback) {
        onCaptureRequest = callback;
    }
    
    void setProcessingCompleteCallback(std::function<void()> callback) {
        onProcessingComplete = callback;
    }
};

// 비동기 파이프라인 - 각 단계가 독립적으로 실행
class AsyncPipeline {
private:
    struct PipelineStage {
        std::string name;
        std::function<void(void*)> process;
        std::condition_variable cv;
        std::mutex mutex;
        std::atomic<bool> hasWork{false};
        void* workData{nullptr};
    };
    
    std::vector<PipelineStage> stages;
    std::vector<std::thread> workers;
    std::atomic<bool> running{true};
    
public:
    // 파이프라인 스테이지 추가
    void addStage(const std::string& name, std::function<void(void*)> processFunc) {
        PipelineStage stage;
        stage.name = name;
        stage.process = processFunc;
        stages.push_back(std::move(stage));
    }
    
    // 워커 스레드 시작
    void start() {
        for (size_t i = 0; i < stages.size(); ++i) {
            workers.emplace_back([this, i]() {
                workerThread(i);
            });
        }
    }
    
    // 워커 스레드 함수
    void workerThread(size_t stageIndex) {
        auto& stage = stages[stageIndex];
        
        // CPU 친화성 설정 - 각 스테이지를 다른 코어에
        SetThreadAffinityMask(GetCurrentThread(), 1ULL << (stageIndex + 2));
        
        while (running) {
            // 작업 대기 (CPU 사용 없음)
            {
                std::unique_lock<std::mutex> lock(stage.mutex);
                stage.cv.wait(lock, [&stage, this]() {
                    return stage.hasWork.load() || !running.load();
                });
            }
            
            if (!running) break;
            
            // 작업 처리
            if (stage.hasWork) {
                stage.process(stage.workData);
                stage.hasWork = false;
                
                // 다음 스테이지 트리거
                if (stageIndex + 1 < stages.size()) {
                    submitWork(stageIndex + 1, stage.workData);
                }
            }
        }
    }
    
    // 작업 제출
    void submitWork(size_t stageIndex, void* data) {
        if (stageIndex >= stages.size()) return;
        
        auto& stage = stages[stageIndex];
        {
            std::lock_guard<std::mutex> lock(stage.mutex);
            stage.workData = data;
            stage.hasWork = true;
        }
        stage.cv.notify_one();
    }
    
    // 종료
    void stop() {
        running = false;
        for (auto& stage : stages) {
            stage.cv.notify_all();
        }
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
};

// 프레임 콜백 기반 캡처
class CallbackBasedCapture {
private:
    using FrameCallback = std::function<void(void*)>;
    
    // V-Sync 동기화를 위한 D3D 이벤트
    HANDLE vsyncEvent;
    
    // 콜백 체인
    std::vector<FrameCallback> callbacks;
    
    // 타이머 기반 캡처
    UINT_PTR timerId;
    
public:
    CallbackBasedCapture() {
        // V-Sync 이벤트 생성
        vsyncEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    }
    
    ~CallbackBasedCapture() {
        if (timerId) {
            KillTimer(NULL, timerId);
        }
        CloseHandle(vsyncEvent);
    }
    
    // 콜백 등록
    void registerCallback(FrameCallback callback) {
        callbacks.push_back(callback);
    }
    
    // V-Sync 기반 캡처 시작
    void startVSyncCapture() {
        // D3D Present 후크 또는 DWM 콜백 사용
        // 이것은 디스플레이 리프레시와 동기화되어 CPU 사용을 최소화
    }
    
    // 타이머 기반 캡처 (매우 낮은 CPU 사용)
    static void CALLBACK TimerCallback(HWND hwnd, UINT msg, UINT_PTR id, DWORD time) {
        // 타이머 콜백에서 캡처 실행
        // CPU는 타이머 이벤트가 발생할 때만 사용됨
    }
    
    void startTimerCapture(int fps) {
        int interval = 1000 / fps;
        timerId = SetTimer(NULL, 0, interval, TimerCallback);
    }
    
    // 프레임 처리
    void processFrame(void* frameData) {
        // 모든 콜백 실행
        for (auto& callback : callbacks) {
            callback(frameData);
        }
    }
};

// 완전한 이벤트 기반 캡처 스레드
class EventDrivenCaptureThread {
private:
    EventBasedCapture eventSystem;
    AsyncPipeline pipeline;
    CallbackBasedCapture callbackCapture;
    
    std::thread captureThread;
    std::atomic<bool> running{true};
    
public:
    void start() {
        // 파이프라인 스테이지 설정
        pipeline.addStage("Capture", [this](void* data) {
            // GPU 직접 캡처
        });
        
        pipeline.addStage("Process", [this](void* data) {
            // GPU 처리
        });
        
        pipeline.addStage("Detect", [this](void* data) {
            // AI 추론
        });
        
        pipeline.start();
        
        // 캡처 스레드 시작
        captureThread = std::thread([this]() {
            runEventLoop();
        });
    }
    
    void runEventLoop() {
        while (running) {
            // 이벤트 대기 (CPU 0% 사용)
            auto event = eventSystem.waitForEvent(16);  // 60 FPS = 16ms
            
            switch (event.type) {
                case EventBasedCapture::FRAME_READY:
                    // 프레임 준비됨 - 파이프라인에 제출
                    pipeline.submitWork(0, event.data);
                    break;
                    
                case EventBasedCapture::CAPTURE_REQUESTED:
                    // 캡처 요청 - 실제 캡처 수행
                    performCapture();
                    break;
                    
                case EventBasedCapture::SHUTDOWN:
                    running = false;
                    break;
            }
        }
    }
    
    void performCapture() {
        // 실제 캡처 로직
        // 완료 후 이벤트 트리거
        eventSystem.triggerFrameReady();
    }
    
    void stop() {
        running = false;
        eventSystem.triggerShutdown();
        pipeline.stop();
        if (captureThread.joinable()) {
            captureThread.join();
        }
    }
};