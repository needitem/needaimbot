# Legacy Code Cleanup Script

## 제거해야 할 레거시 코드 목록

이 문서는 v2 마이그레이션 후 `unified_graph_pipeline.cu/.h` (v1)에서 제거해야 할 레거시 코드를 정리합니다.

---

## 🗑️ 제거 대상

### 1. Enum 타입 (unified_graph_pipeline.h)

#### 위치: Line 40-56

```cpp
// ❌ 삭제
enum class CaptureState : uint8_t {
    IDLE,           // No capture scheduled
    CAPTURING,      // Async copy in flight
    READY,          // Frame ready for consumption
    CONSUMED        // Frame consumed by pipeline, needs new capture
};

enum class FrameFailureReason : uint8_t {
    NO_FRAME_READY,     // Waiting for capture completion - short yield
    INPUT_PENDING,      // Waiting for input to be reflected - event wait
    GRAPH_NOT_READY,    // Graph initialization - longer wait
    GPU_BUSY,           // GPU still processing previous frame - yield
    CAPTURE_FAILED,     // Capture error - backoff
    NONE                // Success
};
```

**대체:** Frame-ID 기반 처리 (v2의 `FrameMetadata`)

---

### 2. 멤버 변수 (unified_graph_pipeline.h)

#### 2.1 Capture 관련 레거시 변수

```cpp
// ❌ 삭제
std::atomic<CaptureState> m_captureState{CaptureState::IDLE};
bool m_qpcSupportChecked = false;  // ✅ m_qpcSupported만 유지
SimpleCudaMat m_captureBuffer;     // ✅ v2의 ring buffer로 대체
int m_stableCaptureRows = 0;
int m_stableCaptureCols = 0;
int m_stableCaptureChannels = 0;
bool m_captureBufferShapeDirty = true;
bool m_graphPrimed = false;
```

**이유:**
- `m_captureState` → v2는 Frame-ID로 상태 추적
- `m_captureBuffer` → v2는 `std::array<FrameSlot, 2>` 사용
- Shape tracking → Ring buffer가 자동 관리

#### 2.2 중복된 Atomic 플래그

```cpp
// ❌ 삭제 (v2에 통합됨)
std::atomic<bool> m_classFilterDirty{true};
std::atomic<int> m_cachedHeadClassId{-1};
std::atomic<size_t> m_cachedHeadClassNameHash{0};
std::atomic<size_t> m_cachedClassSettingsSize{0};
std::atomic<bool> m_pidConfigDirty{true};
```

**대체:** v2의 `CachedConfig::generation` (단일 카운터)

#### 2.3 Movement Filter Mutex

```cpp
// ❌ 삭제
mutable std::mutex m_movementFilterMutex;
```

**대체:** v2의 `MovementFilterState` (thread-local, no lock)

---

### 3. 메서드 (unified_graph_pipeline.cu)

#### 3.1 Frame Capture 레거시

```cpp
// ❌ 삭제 전체 구현
bool UnifiedGraphPipeline::waitForCaptureCompletion();
FrameFailureReason UnifiedGraphPipeline::scheduleNextFrameCapture(bool forceSync);
FrameFailureReason UnifiedGraphPipeline::ensureFrameReady();
bool UnifiedGraphPipeline::performFrameCapture();
bool UnifiedGraphPipeline::performFrameCaptureDirectToUnified();
```

**대체:**
- v2의 `scheduleCapture()` (non-blocking)
- v2의 `tryConsumeFrame()` (lock-free)

#### 3.2 Failure Handling

```cpp
// ❌ 삭제
void UnifiedGraphPipeline::handleFrameFailure(FrameFailureReason reason, int& consecutiveFails) {
    consecutiveFails++;

    switch (reason) {
    case FrameFailureReason::NO_FRAME_READY:
        m_perfMetrics.captureWaitCount++;
        // ... 50+ lines of complex wait logic
        break;
    // ... 5 more cases
    }
}
```

**대체:** v2는 failure를 자동으로 처리 (ring buffer overflow → drop)

#### 3.3 QPC Support Check

```cpp
// ❌ 삭제
bool UnifiedGraphPipeline::checkQPCSupport() {
    if (m_qpcSupportChecked) {
        return m_qpcSupported;
    }

    m_qpcSupportChecked = true;

    if (!m_capture) {
        m_qpcSupported = false;
        return false;
    }

    uint64_t testQpc = m_capture->GetLastPresentQpc();
    m_qpcSupported = (testQpc != 0);

    if (m_qpcSupported) {
        std::cout << "[Capture] QPC-based input latency reduction enabled" << std::endl;
    } else {
        std::cout << "[Capture] QPC not available, using timer-based input gating" << std::endl;
    }

    return m_qpcSupported;
}
```

**대체:** v2의 `initialize()` 내부에서 한 번만 체크

```cpp
// v2 - 초기화 시 한 번만
bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    // ...
    if (m_capture) {
        m_qpcSupported = (m_capture->GetLastPresentQpc() != 0);
    }
    // ...
}
```

#### 3.4 Normal Pipeline

```cpp
// ❌ 삭제 전체 (80+ lines)
bool UnifiedGraphPipeline::executeNormalPipeline(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    cudaStream_t activeStream = stream ? stream : (m_pipelineStream ? m_pipelineStream->get() : nullptr);
    if (!activeStream) {
        return false;
    }

    FrameFailureReason ensureReason = ensureFrameReady();
    if (ensureReason != FrameFailureReason::NONE) {
        return false;
    }

    // ... 60+ more lines
}
```

**대체:** v2의 `executeFrame()` (단일 메서드)

---

### 4. Config 접근 패턴 (unified_graph_pipeline.cu)

#### 4.1 Lock을 사용하는 Hot Path

**찾기:** 다음 패턴을 검색

```cpp
// ❌ 이 패턴을 모두 찾아서 제거
std::lock_guard<std::mutex> lock(ctx.configMutex);
float kp_x = ctx.config.pid_kp_x;
```

**대체:**

```cpp
// ✅ v2 패턴
const CachedConfig& cfg = m_cachedConfig;
float kp_x = cfg.pid.kp_x;
```

#### 4.2 제거할 함수들

```cpp
// ❌ performTargetSelection() 내부의 mutex lock
void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    // 🔴 이 부분 삭제
    if (!m_graphCaptured && m_pidConfigDirty.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(ctx.configMutex);  // ❌
        m_cachedPIDConfig.max_detections = ctx.config.max_detections;
        m_cachedPIDConfig.kp_x = ctx.config.pid_kp_x;
        // ... 10+ more lines
        m_pidConfigDirty.store(false, std::memory_order_release);
    }

    // 🔴 이 부분도 삭제
    m_deadbandEnterX = ctx.config.deadband_enter_x;  // ❌ Direct access
    m_deadbandExitX  = ctx.config.deadband_exit_x;
    // ...
}
```

**대체:**

```cpp
// ✅ v2 - 캐시된 config 사용 (NO LOCKS)
void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    const CachedConfig& cfg = m_cachedConfig;

    int maxDetections = cfg.detection.max_detections;
    float kp_x = cfg.pid.kp_x;
    float kp_y = cfg.pid.kp_y;
    // ... instant access
}
```

---

### 5. Movement Filter (unified_graph_pipeline.cu)

#### 5.1 Mutex 기반 필터

```cpp
// ❌ 삭제
MouseMovement UnifiedGraphPipeline::filterMouseMovement(const MouseMovement& rawMovement, bool movementEnabled) {
    std::lock_guard<std::mutex> lock(m_movementFilterMutex);  // 🔴 HOT PATH LOCK
    auto& ctx = AppContext::getInstance();

    if (!movementEnabled) {
        m_skipNextMovement = true;
        // ...
    }

    // ... 80+ lines with mutex held
}
```

**대체:**

```cpp
// ✅ v2 - Lock-free filter (thread-local state)
MouseMovement UnifiedGraphPipeline::filterMouseMovement(const MouseMovement& raw, bool enabled) {
    // No lock - m_filterState is thread-local
    if (!enabled) {
        m_filterState.skipNext = true;
        // ...
    }

    const auto& cfg = m_cachedConfig.filtering;  // Cache read, no lock
    // ... same logic, zero overhead
}
```

---

### 6. Preview 처리 (unified_graph_pipeline.cu)

#### 6.1 메인 스레드 Blocking

```cpp
// ❌ 문제: Preview가 메인 파이프라인 블록
void UnifiedGraphPipeline::updatePreviewBuffer(const SimpleCudaMat& currentBuffer) {
    auto& ctx = AppContext::getInstance();

    std::lock_guard<std::mutex> lock(m_previewMutex);  // 🔴 BLOCKS MAIN THREAD

    updatePreviewBufferAllocation();  // Long operation

    // ... 100+ lines while holding lock
}
```

**개선 (v2):**

```cpp
// ✅ v2 - Separate low-priority stream
void UnifiedGraphPipeline::updatePreviewBuffer(const SimpleCudaMat& frame) {
    // Quick lock - just enqueue async copy
    std::lock_guard<std::mutex> lock(m_previewMutex);

    if (!m_preview.enabled) return;

    // Async copy on separate stream (non-blocking)
    cudaMemcpy2DAsync(
        m_preview.previewBuffer.data(),
        m_preview.previewBuffer.step(),
        frame.data(),
        frame.step(),
        rowBytes,
        height,
        cudaMemcpyDeviceToDevice,
        m_previewStream->get()  // Low priority, independent
    );

    // Main pipeline doesn't wait
}
```

---

### 7. Graph Execution (unified_graph_pipeline.cu)

#### 7.1 복잡한 분기 로직

```cpp
// ❌ executeFrame() 내부 - 100+ lines of branching
bool UnifiedGraphPipeline::executeFrame(FrameFailureReason* outReason, cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    // 🔴 복잡한 상태 체크
    bool expected = false;
    if (!m_frameInFlight.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
        if (outReason) *outReason = FrameFailureReason::GPU_BUSY;
        return false;
    }

    // 🔴 Graph vs Normal 분기
    if (ctx.config.use_cuda_graph && m_state.graphReady && m_graphExec) {
        // Graph path - 60+ lines
        if (!m_graphPrimed) {
            FrameFailureReason primeReason = scheduleNextFrameCapture(false);
            // ... complex priming logic
        }

        (void)scheduleNextFrameCapture(false);

        // ... 40+ more lines
    } else {
        // Normal path
        if (!executeNormalPipeline(launchStream)) {
            // ... error handling
        }
    }

    // 🔴 Periodic CUDA memory trim (random latency spike!)
    static auto lastTrim = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::minutes>(now - lastTrim).count() >= 10) {
        cudaMemPoolTrimTo(nullptr, 0);  // 🔴 BLOCKING
        cudaDeviceGraphMemTrim(0);
        lastTrim = now;
    }

    return true;
}
```

**대체 (v2):**

```cpp
// ✅ v2 - Simple, predictable
bool UnifiedGraphPipeline::executeFrame(cudaStream_t stream) {
    // Single frame in flight
    bool expected = false;
    if (!m_frameInFlight.compare_exchange_strong(expected, true)) {
        return true;  // Not an error, just skip
    }

    // Try consume frame (lock-free)
    FrameMetadata metadata;
    SimpleCudaMat frameImage;
    if (!tryConsumeFrame(metadata, frameImage)) {
        m_frameInFlight.store(false);
        return true;  // No frame ready
    }

    // Process frame (unified path)
    performPreprocessing(frameImage, stream);
    performInference(stream);
    performPostProcessing(stream);
    performTargetSelection(stream);

    // Enqueue callback (releases m_frameInFlight)
    enqueueFrameCompletionCallback(stream, metadata);

    return true;
}
```

---

## 🔍 검색 & 교체 패턴

### Pattern 1: Config Mutex Lock

**찾기:**
```regex
std::lock_guard<std::mutex>\s+lock\(ctx\.configMutex\);
```

**확인 위치:**
- `performTargetSelection()`
- `updateClassFilterIfNeeded()`
- `findHeadClassId()`

**교체:**
```cpp
// Lock 제거 후 cached config 사용
const CachedConfig& cfg = m_cachedConfig;
```

---

### Pattern 2: Direct Config Access

**찾기:**
```regex
ctx\.config\.(pid_|deadband_|head_|body_|iou_)
```

**교체:**
```cpp
cfg.pid.kp_x       // ctx.config.pid_kp_x
cfg.filtering.deadband_enter_x  // ctx.config.deadband_enter_x
cfg.targeting.head_y_offset     // ctx.config.head_y_offset
```

---

### Pattern 3: CaptureState Usage

**찾기:**
```regex
m_captureState\.(load|store|compare_exchange)
```

**교체:** Frame-ID 기반 로직으로 재작성

---

## ✅ 정리 후 검증

### 1. 컴파일 체크

```bash
# 제거 후 빌드
cmake --build . --config Release 2>&1 | grep -E "(error|warning)"
```

**예상 결과:** 에러 없음

### 2. 사용되지 않는 코드 검색

```bash
# 제거 후 dead code 확인
grep -r "FrameFailureReason" --include="*.cpp" --include="*.h"
grep -r "CaptureState" --include="*.cpp" --include="*.h"
grep -r "m_captureBuffer" --include="*.cpp" --include="*.h"
```

**예상 결과:** 검색 결과 없음

### 3. Lock 검색

```bash
# Hot path에 남은 mutex 확인
grep -r "std::lock_guard.*configMutex" needaimbot/cuda/ --include="*.cu"
```

**예상 결과:** `refreshConfigCache()` 내부만 존재

---

## 📊 정리 전후 비교

### Before (v1)

```cpp
// unified_graph_pipeline.h - 655 lines
// unified_graph_pipeline.cu - 3102 lines
// Total: 3757 lines
```

### After (v2)

```cpp
// unified_graph_pipeline_v2.h - 485 lines (-170)
// unified_graph_pipeline_v2_core.cu - 580 lines (-2522!)
// Total: 1065 lines (-72% code reduction)
```

**코드 감소:**
- Header: 26% 감소
- Implementation: **81% 감소**
- 총: **72% 감소**

---

## 🚀 자동 정리 스크립트 (선택)

```bash
#!/bin/bash
# cleanup_legacy.sh

# Backup
cp unified_graph_pipeline.h unified_graph_pipeline.h.v1
cp unified_graph_pipeline.cu unified_graph_pipeline.cu.v1

# Remove legacy enums
sed -i '/enum class CaptureState/,/^};/d' unified_graph_pipeline.h
sed -i '/enum class FrameFailureReason/,/^};/d' unified_graph_pipeline.h

# Remove legacy methods
sed -i '/FrameFailureReason ensureFrameReady/d' unified_graph_pipeline.h
sed -i '/FrameFailureReason scheduleNextFrameCapture/d' unified_graph_pipeline.h
sed -i '/void handleFrameFailure/d' unified_graph_pipeline.h

echo "Legacy code removed. Review changes before committing."
```

---

**작성일:** 2025-01-14
**대상:** unified_graph_pipeline.cu/.h (v1)
**목표:** v2로 완전 마이그레이션
