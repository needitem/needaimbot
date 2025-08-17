# 정밀 탐지인식 타격 시스템 - 구체적 구현 계획

## 📋 전체 일정: 4주 집중 리팩토링

### 목표
- **1-2주차**: 구조적 문제 해결 (God Object 제거, 싱글톤 제거)
- **3주차**: 아키텍처 패턴 적용 (파이프라인, 이벤트 버스)
- **4주차**: 테스트 및 최적화

---

## 🗓️ Week 1: AppContext 분해 (Day 1-5)

### Day 1-2: CaptureState 클래스 분리

#### 1. 새 파일 생성
```cpp
// needaimbot/core/states/CaptureState.h
#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "../cuda/simple_cuda_mat.h"

namespace Core {
    class CaptureState {
    private:
        // 캡처 버퍼
        std::vector<SimpleCudaMat> gpuBuffers_;
        std::atomic<int> gpuWriteIdx_{0};
        
        // 동기화
        mutable std::mutex frameMutex_;
        std::condition_variable frameCV_;
        std::atomic<bool> frameReady_{false};
        
        // 설정 변경 플래그
        std::atomic<bool> resolutionChanged_{false};
        std::atomic<bool> methodChanged_{false};
        
    public:
        CaptureState(size_t bufferCount = 4);
        
        // 버퍼 접근
        SimpleCudaMat& getWriteBuffer();
        const SimpleCudaMat& getReadBuffer() const;
        void swapBuffers();
        
        // 동기화
        void notifyFrameReady();
        bool waitForFrame(std::chrono::milliseconds timeout);
        
        // 설정 변경
        void markResolutionChanged() { resolutionChanged_ = true; }
        bool checkAndResetResolutionChange();
    };
}
```

#### 2. 구현 파일
```cpp
// needaimbot/core/states/CaptureState.cpp
#include "CaptureState.h"

namespace Core {
    CaptureState::CaptureState(size_t bufferCount) {
        gpuBuffers_.resize(bufferCount);
    }
    
    SimpleCudaMat& CaptureState::getWriteBuffer() {
        return gpuBuffers_[gpuWriteIdx_.load()];
    }
    
    void CaptureState::swapBuffers() {
        gpuWriteIdx_ = (gpuWriteIdx_ + 1) % gpuBuffers_.size();
        frameReady_ = true;
        frameCV_.notify_one();
    }
    
    bool CaptureState::waitForFrame(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(frameMutex_);
        return frameCV_.wait_for(lock, timeout, 
            [this] { return frameReady_.load(); });
    }
}
```

#### 3. AppContext 수정
```cpp
// AppContext.h 수정
class AppContext {
    // 제거할 멤버들:
    // - std::vector<SimpleCudaMat> captureGpuBuffer;
    // - std::atomic<int> captureGpuWriteIdx;
    // - std::mutex frame_mutex;
    // - std::condition_variable frame_cv;
    
    // 추가:
    std::unique_ptr<Core::CaptureState> captureState_;
    
public:
    Core::CaptureState& getCaptureState() { return *captureState_; }
};
```

### Day 2-3: DetectionState 클래스 분리

#### 1. 새 파일 생성
```cpp
// needaimbot/core/states/DetectionState.h
#pragma once
#include <atomic>
#include <mutex>
#include <vector>
#include "../cuda/detection/postProcess.h"

namespace Core {
    class DetectionState {
    private:
        // 타겟 정보
        mutable std::mutex targetMutex_;
        Target currentTarget_;
        std::vector<Target> allTargets_;
        std::atomic<bool> hasTarget_{false};
        
        // 상태 플래그
        std::atomic<bool> detectionPaused_{false};
        std::atomic<bool> modelChanged_{false};
        
        // 성능 메트릭
        std::atomic<float> inferenceTime_{0.0f};
        std::atomic<float> postProcessTime_{0.0f};
        
    public:
        // 타겟 관리
        void updateTargets(const std::vector<Target>& targets);
        Target getBestTarget() const;
        std::vector<Target> getAllTargets() const;
        
        // 상태 관리
        void pauseDetection() { detectionPaused_ = true; }
        void resumeDetection() { detectionPaused_ = false; }
        bool isPaused() const { return detectionPaused_.load(); }
        
        // 메트릭
        void setInferenceTime(float ms) { inferenceTime_ = ms; }
        float getInferenceTime() const { return inferenceTime_.load(); }
    };
}
```

### Day 3-4: PerformanceMetrics 클래스 분리

#### 1. 새 파일 생성
```cpp
// needaimbot/core/metrics/PerformanceMetrics.h
#pragma once
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace Core {
    class PerformanceMetrics {
    private:
        struct Metric {
            std::atomic<float> current{0.0f};
            std::vector<float> history;
            mutable std::mutex historyMutex;
            size_t maxHistorySize = 100;
            
            void update(float value);
            float getAverage() const;
            float getMin() const;
            float getMax() const;
        };
        
        std::unordered_map<std::string, std::unique_ptr<Metric>> metrics_;
        mutable std::mutex metricsMapMutex_;
        
    public:
        // 메트릭 등록 및 업데이트
        void registerMetric(const std::string& name);
        void updateMetric(const std::string& name, float value);
        
        // 조회
        float getCurrentValue(const std::string& name) const;
        std::vector<float> getHistory(const std::string& name) const;
        
        // 통계
        struct Stats {
            float current, average, min, max;
        };
        Stats getStats(const std::string& name) const;
        
        // 타이머 헬퍼
        class ScopedTimer {
            PerformanceMetrics& metrics_;
            std::string metricName_;
            std::chrono::high_resolution_clock::time_point start_;
            
        public:
            ScopedTimer(PerformanceMetrics& metrics, const std::string& name);
            ~ScopedTimer();
        };
    };
}
```

#### 2. 사용 예시
```cpp
// 기존 코드:
auto& ctx = AppContext::getInstance();
ctx.g_current_inference_time_ms = inference_time;
ctx.add_to_history(ctx.g_inference_time_history, inference_time, 
                   ctx.g_inference_history_mutex);

// 새 코드:
auto& metrics = performanceMetrics;  // 의존성 주입으로 받음
metrics.updateMetric("inference_time", inference_time);

// 또는 ScopedTimer 사용:
{
    Core::PerformanceMetrics::ScopedTimer timer(metrics, "inference_time");
    // 추론 코드
} // 자동으로 시간 측정 및 기록
```

### Day 4-5: MouseState & ConfigManager 분리

#### 1. MouseState 클래스
```cpp
// needaimbot/core/states/MouseState.h
#pragma once
#include <atomic>
#include <mutex>
#include <queue>

namespace Core {
    struct MouseMovement {
        int dx, dy;
        float confidence;
        bool hasTarget;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    class MouseState {
    private:
        // 이동 큐
        std::queue<MouseMovement> movementQueue_;
        mutable std::mutex queueMutex_;
        std::condition_variable queueCV_;
        
        // 상태
        std::atomic<bool> aiming_{false};
        std::atomic<bool> shooting_{false};
        std::atomic<bool> enabled_{true};
        
    public:
        // 큐 관리
        void pushMovement(const MouseMovement& movement);
        bool popMovement(MouseMovement& movement, 
                        std::chrono::milliseconds timeout);
        
        // 상태 관리
        void setAiming(bool aiming) { aiming_ = aiming; }
        bool isAiming() const { return aiming_.load(); }
        
        void setShooting(bool shooting) { shooting_ = shooting; }
        bool isShooting() const { return shooting_.load(); }
    };
}
```

#### 2. ConfigManager 클래스
```cpp
// needaimbot/core/config/ConfigManager.h
#pragma once
#include <atomic>
#include <mutex>
#include <functional>

namespace Core {
    class ConfigManager {
    private:
        Config config_;
        mutable std::mutex configMutex_;
        
        // 변경 콜백
        std::unordered_map<std::string, 
            std::vector<std::function<void()>>> changeCallbacks_;
        
    public:
        // 싱글톤 제거 - 일반 클래스로
        ConfigManager() = default;
        ConfigManager(const std::string& configPath);
        
        // 설정 접근
        template<typename T>
        T get(const std::string& key) const;
        
        template<typename T>
        void set(const std::string& key, const T& value);
        
        // 변경 알림
        void registerCallback(const std::string& key, 
                             std::function<void()> callback);
        
        // 파일 I/O
        bool loadFromFile(const std::string& path);
        bool saveToFile(const std::string& path) const;
    };
}
```

---

## 🗓️ Week 2: 의존성 주입 & 이벤트 시스템 (Day 6-10)

### Day 6-7: 의존성 주입 패턴 구현

#### 1. ServiceLocator 패턴
```cpp
// needaimbot/core/ServiceLocator.h
#pragma once
#include <memory>
#include <typeindex>
#include <unordered_map>

namespace Core {
    class ServiceLocator {
    private:
        std::unordered_map<std::type_index, 
                          std::shared_ptr<void>> services_;
        
    public:
        template<typename T>
        void registerService(std::shared_ptr<T> service) {
            services_[std::type_index(typeid(T))] = service;
        }
        
        template<typename T>
        std::shared_ptr<T> getService() {
            auto it = services_.find(std::type_index(typeid(T)));
            if (it != services_.end()) {
                return std::static_pointer_cast<T>(it->second);
            }
            return nullptr;
        }
    };
}
```

#### 2. Application 클래스 (main 함수 대체)
```cpp
// needaimbot/Application.h
#pragma once
#include "core/ServiceLocator.h"

class Application {
private:
    Core::ServiceLocator serviceLocator_;
    std::vector<std::unique_ptr<IModule>> modules_;
    std::atomic<bool> running_{true};
    
public:
    Application();
    
    // 초기화
    bool initialize(const std::string& configPath);
    
    // 모듈 등록
    void registerModule(std::unique_ptr<IModule> module);
    
    // 실행
    int run();
    
    // 종료
    void shutdown();
    
private:
    void initializeServices();
    void initializeModules();
    void startModules();
    void stopModules();
};
```

#### 3. 모듈 인터페이스
```cpp
// needaimbot/core/IModule.h
#pragma once

class IModule {
public:
    virtual ~IModule() = default;
    
    virtual bool initialize(Core::ServiceLocator& services) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void update(float deltaTime) {}
    
    virtual std::string getName() const = 0;
};
```

#### 4. Detector 리팩토링 예시
```cpp
// detector/Detector.h - 리팩토링 후
class Detector : public IModule {
private:
    // 의존성 주입으로 받은 서비스들
    std::shared_ptr<Core::DetectionState> detectionState_;
    std::shared_ptr<Core::ConfigManager> config_;
    std::shared_ptr<Core::PerformanceMetrics> metrics_;
    
public:
    // 싱글톤 제거, 생성자로 의존성 주입
    Detector() = default;
    
    bool initialize(Core::ServiceLocator& services) override {
        detectionState_ = services.getService<Core::DetectionState>();
        config_ = services.getService<Core::ConfigManager>();
        metrics_ = services.getService<Core::PerformanceMetrics>();
        
        if (!detectionState_ || !config_ || !metrics_) {
            return false;
        }
        
        // TensorRT 초기화 등
        return initializeTensorRT();
    }
    
    void start() override {
        // 추론 스레드 시작
        inferenceThread_ = std::thread(&Detector::inferenceLoop, this);
    }
    
    void stop() override {
        // 스레드 종료
        running_ = false;
        if (inferenceThread_.joinable()) {
            inferenceThread_.join();
        }
    }
    
    std::string getName() const override { return "Detector"; }
};
```

### Day 8-10: 이벤트 버스 시스템

#### 1. 이벤트 정의
```cpp
// needaimbot/core/events/Events.h
#pragma once
#include <variant>
#include <chrono>

namespace Core::Events {
    // 기본 이벤트
    struct FrameCapturedEvent {
        size_t frameId;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    struct TargetDetectedEvent {
        Target target;
        float confidence;
        size_t frameId;
    };
    
    struct MouseMoveRequestEvent {
        int dx, dy;
        float urgency;  // 0.0 ~ 1.0
    };
    
    struct ConfigChangedEvent {
        std::string key;
        std::string oldValue;
        std::string newValue;
    };
    
    // 이벤트 타입 정의
    using Event = std::variant<
        FrameCapturedEvent,
        TargetDetectedEvent,
        MouseMoveRequestEvent,
        ConfigChangedEvent
    >;
}
```

#### 2. EventBus 구현
```cpp
// needaimbot/core/events/EventBus.h
#pragma once
#include "Events.h"
#include <functional>
#include <queue>
#include <typeindex>

namespace Core {
    class EventBus {
    private:
        using Handler = std::function<void(const Events::Event&)>;
        
        // 타입별 핸들러 목록
        std::unordered_map<std::type_index, 
                          std::vector<Handler>> handlers_;
        mutable std::mutex handlersMutex_;
        
        // 비동기 이벤트 큐
        std::queue<Events::Event> eventQueue_;
        std::mutex queueMutex_;
        std::condition_variable queueCV_;
        
        // 처리 스레드
        std::thread processingThread_;
        std::atomic<bool> running_{true};
        
    public:
        EventBus();
        ~EventBus();
        
        // 구독
        template<typename EventType>
        void subscribe(std::function<void(const EventType&)> handler) {
            std::lock_guard<std::mutex> lock(handlersMutex_);
            
            auto wrapper = [handler](const Events::Event& event) {
                if (auto* e = std::get_if<EventType>(&event)) {
                    handler(*e);
                }
            };
            
            handlers_[std::type_index(typeid(EventType))].push_back(wrapper);
        }
        
        // 발행 (동기)
        void publish(const Events::Event& event);
        
        // 발행 (비동기)
        void publishAsync(const Events::Event& event);
        
    private:
        void processEvents();
    };
}
```

#### 3. 사용 예시
```cpp
// Capture 모듈에서
void CaptureModule::onFrameCaptured() {
    Core::Events::FrameCapturedEvent event{
        .frameId = currentFrameId_++,
        .timestamp = std::chrono::steady_clock::now()
    };
    
    eventBus_->publishAsync(event);
}

// Detector 모듈에서
void Detector::initialize(Core::ServiceLocator& services) {
    auto eventBus = services.getService<Core::EventBus>();
    
    // 프레임 캡처 이벤트 구독
    eventBus->subscribe<Core::Events::FrameCapturedEvent>(
        [this](const auto& event) {
            processFrame(event.frameId);
        });
}

// 타겟 탐지 시
void Detector::onTargetDetected(const Target& target) {
    Core::Events::TargetDetectedEvent event{
        .target = target,
        .confidence = target.confidence,
        .frameId = currentFrameId_
    };
    
    eventBus_->publish(event);  // 동기 발행
}
```

---

## 🗓️ Week 3: 파이프라인 패턴 & 최적화 (Day 11-15)

### Day 11-12: 파이프라인 패턴 구현

#### 1. 파이프라인 인터페이스
```cpp
// needaimbot/pipeline/IPipeline.h
#pragma once
#include <memory>
#include <vector>

template<typename TInput, typename TOutput>
class IPipelineStage {
public:
    virtual ~IPipelineStage() = default;
    virtual TOutput process(TInput input) = 0;
    virtual std::string getName() const = 0;
};

template<typename TData>
class Pipeline {
private:
    std::vector<std::unique_ptr<IPipelineStage<TData, TData>>> stages_;
    
public:
    void addStage(std::unique_ptr<IPipelineStage<TData, TData>> stage) {
        stages_.push_back(std::move(stage));
    }
    
    TData execute(TData input) {
        TData data = std::move(input);
        for (auto& stage : stages_) {
            data = stage->process(std::move(data));
        }
        return data;
    }
};
```

#### 2. 탐지 파이프라인 구현
```cpp
// needaimbot/pipeline/DetectionPipeline.h
#pragma once
#include "IPipeline.h"

struct FrameData {
    cv::cuda::GpuMat image;
    size_t frameId;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<Target> targets;
};

class PreprocessStage : public IPipelineStage<FrameData, FrameData> {
public:
    FrameData process(FrameData input) override {
        // BGR 변환, 리사이즈 등
        cv::cuda::cvtColor(input.image, input.image, cv::COLOR_BGRA2BGR);
        cv::cuda::resize(input.image, input.image, cv::Size(640, 640));
        return input;
    }
    
    std::string getName() const override { return "Preprocess"; }
};

class InferenceStage : public IPipelineStage<FrameData, FrameData> {
private:
    std::unique_ptr<TensorRTEngine> engine_;
    
public:
    FrameData process(FrameData input) override {
        // TensorRT 추론
        auto results = engine_->infer(input.image);
        input.targets = results;
        return input;
    }
    
    std::string getName() const override { return "Inference"; }
};

class PostProcessStage : public IPipelineStage<FrameData, FrameData> {
public:
    FrameData process(FrameData input) override {
        // NMS, 필터링
        input.targets = applyNMS(input.targets, 0.45f);
        input.targets = filterByConfidence(input.targets, 0.5f);
        return input;
    }
    
    std::string getName() const override { return "PostProcess"; }
};
```

### Day 13-15: Lock-Free 구조 적용

#### 1. Lock-Free 링 버퍼
```cpp
// needaimbot/core/concurrent/LockFreeRingBuffer.h
#pragma once
#include <atomic>
#include <array>

template<typename T, size_t Size>
class LockFreeRingBuffer {
private:
    struct alignas(64) CacheLine {  // 캐시 라인 정렬
        std::atomic<size_t> value{0};
    };
    
    CacheLine writePos_;
    CacheLine readPos_;
    std::array<T, Size> buffer_;
    
    static constexpr size_t MASK = Size - 1;  // Size는 2의 제곱수
    static_assert((Size & MASK) == 0, "Size must be power of 2");
    
public:
    bool tryPush(T&& item) {
        size_t write = writePos_.value.load(std::memory_order_relaxed);
        size_t next = (write + 1) & MASK;
        size_t read = readPos_.value.load(std::memory_order_acquire);
        
        if (next == read) {
            return false;  // 버퍼 풀
        }
        
        buffer_[write] = std::move(item);
        writePos_.value.store(next, std::memory_order_release);
        return true;
    }
    
    bool tryPop(T& item) {
        size_t read = readPos_.value.load(std::memory_order_relaxed);
        size_t write = writePos_.value.load(std::memory_order_acquire);
        
        if (read == write) {
            return false;  // 버퍼 비어있음
        }
        
        item = std::move(buffer_[read]);
        readPos_.value.store((read + 1) & MASK, std::memory_order_release);
        return true;
    }
    
    size_t size() const {
        size_t write = writePos_.value.load(std::memory_order_acquire);
        size_t read = readPos_.value.load(std::memory_order_acquire);
        return (write - read) & MASK;
    }
};
```

#### 2. 적용 예시
```cpp
// CaptureModule과 Detector 간 통신
class CaptureToDetectorQueue {
private:
    LockFreeRingBuffer<FrameData, 8> queue_;  // 8개 프레임 버퍼
    
public:
    bool pushFrame(FrameData&& frame) {
        return queue_.tryPush(std::move(frame));
    }
    
    bool popFrame(FrameData& frame) {
        return queue_.tryPop(frame);
    }
};
```

---

## 🗓️ Week 4: 테스트 및 마이그레이션 (Day 16-20)

### Day 16-17: 테스트 프레임워크 구축

#### 1. 테스트 설정
```cmake
# CMakeLists.txt 추가
enable_testing()
add_subdirectory(tests)

# Google Test 추가
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
```

#### 2. 단위 테스트 예시
```cpp
// tests/core/CaptureStateTest.cpp
#include <gtest/gtest.h>
#include "core/states/CaptureState.h"

class CaptureStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        captureState = std::make_unique<Core::CaptureState>(4);
    }
    
    std::unique_ptr<Core::CaptureState> captureState;
};

TEST_F(CaptureStateTest, BufferSwap) {
    auto& buffer1 = captureState->getWriteBuffer();
    captureState->swapBuffers();
    auto& buffer2 = captureState->getWriteBuffer();
    
    EXPECT_NE(&buffer1, &buffer2);
}

TEST_F(CaptureStateTest, FrameNotification) {
    std::atomic<bool> frameReceived{false};
    
    std::thread waiter([this, &frameReceived] {
        if (captureState->waitForFrame(std::chrono::milliseconds(100))) {
            frameReceived = true;
        }
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    captureState->notifyFrameReady();
    
    waiter.join();
    EXPECT_TRUE(frameReceived);
}
```

#### 3. 통합 테스트
```cpp
// tests/integration/PipelineTest.cpp
class PipelineIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 서비스 초기화
        serviceLocator = std::make_unique<Core::ServiceLocator>();
        
        auto captureState = std::make_shared<Core::CaptureState>();
        auto detectionState = std::make_shared<Core::DetectionState>();
        auto metrics = std::make_shared<Core::PerformanceMetrics>();
        
        serviceLocator->registerService(captureState);
        serviceLocator->registerService(detectionState);
        serviceLocator->registerService(metrics);
        
        // 파이프라인 구성
        pipeline = std::make_unique<Pipeline<FrameData>>();
        pipeline->addStage(std::make_unique<PreprocessStage>());
        pipeline->addStage(std::make_unique<MockInferenceStage>());  // Mock 사용
        pipeline->addStage(std::make_unique<PostProcessStage>());
    }
    
    std::unique_ptr<Core::ServiceLocator> serviceLocator;
    std::unique_ptr<Pipeline<FrameData>> pipeline;
};

TEST_F(PipelineIntegrationTest, EndToEndProcessing) {
    FrameData input;
    input.frameId = 1;
    input.image = createTestImage();  // 테스트 이미지 생성
    
    auto output = pipeline->execute(input);
    
    EXPECT_EQ(output.frameId, 1);
    EXPECT_FALSE(output.targets.empty());
}
```

### Day 18-20: 점진적 마이그레이션

#### 1. Feature Flag 시스템
```cpp
// needaimbot/core/FeatureFlags.h
class FeatureFlags {
private:
    std::unordered_map<std::string, bool> flags_;
    
public:
    void setFlag(const std::string& name, bool enabled) {
        flags_[name] = enabled;
    }
    
    bool isEnabled(const std::string& name) const {
        auto it = flags_.find(name);
        return it != flags_.end() && it->second;
    }
};

// 사용 예시
if (featureFlags.isEnabled("use_new_pipeline")) {
    // 새 파이프라인 사용
    newPipeline->execute(frame);
} else {
    // 기존 코드 사용
    legacyProcess(frame);
}
```

#### 2. 마이그레이션 스크립트
```python
# scripts/migrate_appcontext.py
import re
import os

def migrate_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # AppContext::getInstance() 호출 찾기
    pattern = r'AppContext::getInstance\(\)\.(\w+)'
    
    replacements = {
        'captureGpuBuffer': 'captureState_->getBuffer()',
        'aiming': 'mouseState_->isAiming()',
        'g_current_inference_time_ms': 'metrics_->getCurrentValue("inference_time")',
        # ... 더 많은 매핑
    }
    
    for old, new in replacements.items():
        content = re.sub(f'AppContext::getInstance\\(\\)\\.{old}', 
                        new, content)
    
    with open(filepath, 'w') as f:
        f.write(content)

# 모든 cpp 파일 마이그레이션
for root, dirs, files in os.walk('needaimbot'):
    for file in files:
        if file.endswith('.cpp') or file.endswith('.h'):
            migrate_file(os.path.join(root, file))
```

---

## 📊 구현 체크리스트

### Week 1 체크리스트
- [ ] CaptureState.h/cpp 생성
- [ ] DetectionState.h/cpp 생성
- [ ] PerformanceMetrics.h/cpp 생성
- [ ] MouseState.h/cpp 생성
- [ ] ConfigManager.h/cpp 생성
- [ ] AppContext에서 해당 멤버 제거
- [ ] 컴파일 에러 해결

### Week 2 체크리스트
- [ ] ServiceLocator 구현
- [ ] IModule 인터페이스 정의
- [ ] Application 클래스 구현
- [ ] EventBus 구현
- [ ] 이벤트 타입 정의
- [ ] 주요 모듈 IModule로 리팩토링

### Week 3 체크리스트
- [ ] Pipeline 인터페이스 구현
- [ ] DetectionPipeline 구현
- [ ] LockFreeRingBuffer 구현
- [ ] 파이프라인 스테이지 구현
- [ ] 성능 벤치마크

### Week 4 체크리스트
- [ ] Google Test 설정
- [ ] 단위 테스트 작성 (50개+)
- [ ] 통합 테스트 작성
- [ ] Feature Flag 시스템
- [ ] 마이그레이션 스크립트 실행
- [ ] 프로덕션 테스트

---

## 🎯 성공 지표

### 코드 품질
- [ ] 순환 복잡도 < 10
- [ ] 클래스당 책임 1개
- [ ] 테스트 커버리지 > 70%

### 성능
- [ ] CPU 사용률 < 25%
- [ ] 메모리 사용량 < 2GB
- [ ] 지연시간 < 10ms

### 유지보수성
- [ ] 새 기능 추가 시간 50% 단축
- [ ] 버그 수정 시간 70% 단축
- [ ] 빌드 시간 < 2분

---

## ⚠️ 리스크 관리

### 리스크 1: 기능 손실
**완화 전략**:
- 기존 코드 백업 (git branch)
- Feature Flag로 점진적 전환
- 각 단계마다 회귀 테스트

### 리스크 2: 성능 저하
**완화 전략**:
- 각 변경 후 벤치마크
- 프로파일링으로 병목 확인
- Lock-free 구조 우선 적용

### 리스크 3: 일정 지연
**완화 전략**:
- 일일 스탠드업 미팅
- 주간 진행률 체크
- 문제 발생 시 즉시 에스컬레이션

---

## 📝 일일 작업 로그 템플릿

```markdown
## Day X - [날짜]

### 완료한 작업
- [ ] 작업 1
- [ ] 작업 2

### 발견한 문제
- 문제 1: 설명
  - 해결책: 

### 내일 계획
- [ ] 작업 1
- [ ] 작업 2

### 메트릭
- LOC 변경: +X/-Y
- 컴파일 시간: X초
- 테스트 통과: X/Y
```

이 계획을 따라 단계적으로 구현하면 4주 내에 구조적 문제를 해결하고 유지보수 가능한 시스템으로 전환할 수 있습니다.