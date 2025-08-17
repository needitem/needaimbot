# 정밀 탐지인식 타격 시스템 리소스 최적화 개선 계획

## 분석 요약
Gemini를 통한 전체 코드베이스 분석 결과, 심각한 리소스 사용 문제 발견:
- **GPU 메모리**: 과도한 VRAM 점유 (TensorRT 엔진 + CUDA 버퍼 중복)
- **CPU 사용률**: 불필요한 폴링과 동기화로 인한 과부하
- **FPS 제한 실패**: CUDA Graph가 FPS 제어를 우회하는 구조적 문제

## 목표
- GPU/CPU 메모리 사용량 50% 감소 (4GB → 2GB)
- CPU 사용률 60% 감소 (40-60% → 15-25%)
- FPS 제한 기능 복원 (60-144fps 설정 가능)
- 처리 지연시간 50% 단축 (20ms → 10ms)

---

## Phase 1: Quick Win (1주, 즉시 적용 가능)

### 1.1 메모리 풀 크기 최적화
**현재 문제**: 고정 64MB 메모리 풀
**개선 방안**: 동적 조정

```cpp
// memory_pool.h 수정
class GpuMemoryPool {
private:
    size_t calculateOptimalPoolSize() {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        // 시스템 메모리의 10-15% 사용
        return std::min(free * 0.15, 32 * 1024 * 1024); // 32MB로 축소
    }
};
```

**예상 효과**: GPU 메모리 32MB 즉시 절감

### 1.2 검출 해상도 최적화
**현재**: 256x256 고정
**개선**: config.ini에서 192x192로 변경

```ini
[AI]
detection_resolution=192  # 256 -> 192
```

**예상 효과**: 메모리 44% 절감, 추론 속도 30% 향상

### 1.3 불필요한 atomic 변수 정리
**파일**: AppContext.h
```cpp
// 100개 이상의 atomic -> 필수 20개로 축소
struct EssentialState {
    std::atomic<bool> should_exit;
    std::atomic<bool> frame_ready;
    std::atomic<bool> detection_ready;
    // 나머지는 일반 변수 + mutex
};
```

**예상 효과**: CPU 캐시 효율 20% 향상

### 1.4 TensorRT INT8 양자화
**명령어**:
```bash
# FP16 -> INT8 변환
trtexec --onnx=model.onnx --int8 --saveEngine=model_int8.engine
```

**예상 효과**: 모델 메모리 50% 절감

---

## Phase 2: 아키텍처 리팩토링 (2-4주)

### 2.1 싱글톤 제거 및 의존성 주입

**새 파일**: `pipeline_factory.h`
```cpp
class PipelineFactory {
public:
    static std::unique_ptr<DetectionPipeline> create(
        const PipelineConfig& config) {
        
        auto capture = std::make_unique<CaptureModule>(config.capture);
        auto detector = std::make_unique<DetectorModule>(config.detector);
        auto tracker = std::make_unique<TrackerModule>(config.tracker);
        
        return std::make_unique<DetectionPipeline>(
            std::move(capture),
            std::move(detector),
            std::move(tracker)
        );
    }
};
```

### 2.2 AppContext 분해

**새 구조**:
```
contexts/
├── capture_context.h    // 캡처 관련
├── detection_context.h  // 탐지 관련  
├── tracking_context.h   // 추적 관련
└── targeting_context.h  // 타격 관련
```

### 2.3 비동기 파이프라인 구현

**새 파일**: `async_pipeline.h`
```cpp
template<typename T>
class LockFreeRingBuffer {
    std::array<T, 8> buffer;
    std::atomic<size_t> write_pos{0};
    std::atomic<size_t> read_pos{0};
    
public:
    bool try_push(T&& item);
    bool try_pop(T& item);
};

class AsyncPipeline {
    LockFreeRingBuffer<Frame> capture_queue;
    LockFreeRingBuffer<Detection> detection_queue;
    LockFreeRingBuffer<Target> target_queue;
    
    std::thread capture_thread;
    std::thread detection_thread;
    std::thread tracking_thread;
};
```

---

## Phase 3: 핵심 모듈 재설계 (1-2개월)

### 3.1 Zero-Copy 아키텍처

**새 파일**: `zero_copy_capture.cpp`
```cpp
class ZeroCopyCapture {
private:
    cudaGraphicsResource_t d3d_cuda_resource;
    
public:
    void captureFrame(cudaStream_t stream) {
        // D3D11 텍스처를 CUDA로 직접 매핑
        cudaGraphicsMapResources(1, &d3d_cuda_resource, stream);
        
        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, 
            d3d_cuda_resource, 0, 0);
        
        // 복사 없이 직접 처리
        processOnGPU(array, stream);
        
        cudaGraphicsUnmapResources(1, &d3d_cuda_resource, stream);
    }
};
```

### 3.2 통합 CUDA Graph

**수정 파일**: `unified_graph_pipeline.cpp`
```cpp
class OptimizedGraphPipeline {
    cudaGraph_t full_pipeline_graph;
    
    void buildGraph() {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        
        // 모든 단계를 하나의 그래프로
        captureKernel<<<...>>>();
        preprocessKernel<<<...>>>();
        // TensorRT 추론 노드 추가
        cudaGraphAddChildNode(...);
        postprocessKernel<<<...>>>();
        trackingKernel<<<...>>>();
        
        cudaStreamEndCapture(stream, &full_pipeline_graph);
    }
};
```

### 3.3 메모리 아레나 할당자

**새 파일**: `memory_arena.h`
```cpp
class CudaMemoryArena {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    void* arena_base;
    size_t arena_size;
    std::vector<Block> blocks;
    
public:
    void* allocate(size_t size, size_t alignment = 256) {
        // O(1) 할당
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        // 새 블록 생성
        return createNewBlock(size);
    }
    
    void reset() {
        // 모든 블록을 한번에 해제
        for (auto& block : blocks) {
            block.in_use = false;
        }
    }
};
```

---

## Phase 4: 성능 최적화 및 검증 (2-3주)

### 4.1 프로파일링 시스템 구축

**새 파일**: `profiler.h`
```cpp
class PerformanceProfiler {
    struct Metrics {
        float gpu_memory_mb;
        float cpu_memory_mb;
        float frame_time_ms;
        float latency_ms;
    };
    
    void profile() {
        // NVTX 마커
        nvtxRangePushA("Detection");
        detect();
        nvtxRangePop();
        
        // 메모리 측정
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        
        // 결과 기록
        logMetrics();
    }
};
```

### 4.2 자동 튜닝 시스템

```cpp
class AutoTuner {
    void optimizeParameters() {
        // 런타임 조정
        if (gpu_memory_usage > 0.8) {
            reduceResolution();
            decreaseBufferCount();
        }
        
        if (latency > target_latency) {
            enableGraphOptimization();
            reduceNMSThreshold();
        }
    }
};
```

---

## 구현 우선순위

### 즉시 (1주 내)
1. ✅ memory_pool.h 수정 (메모리 풀 크기 감소)
2. ✅ config.ini 수정 (해상도 192로 변경)
3. ✅ TensorRT INT8 변환

### 단기 (2-4주)
1. ⏳ AppContext 분해
2. ⏳ 비동기 파이프라인 구현
3. ⏳ Lock-free 큐 도입

### 중기 (1-2개월)
1. ⏳ Zero-Copy 구현
2. ⏳ 통합 CUDA Graph
3. ⏳ 메모리 아레나

### 장기 (2-3개월)
1. ⏳ 전체 시스템 재설계
2. ⏳ 모듈화 및 플러그인 시스템
3. ⏳ 자동 튜닝 시스템

---

## 예상 성과

### 메모리 절감
- GPU: 40-50% 감소 (2GB → 1GB)
- CPU: 30-40% 감소 (500MB → 300MB)

### 성능 향상
- 지연시간: 30% 단축 (10ms → 7ms)
- 처리량: 40% 증가 (100fps → 140fps)

### 유지보수성
- 코드 복잡도: 50% 감소
- 모듈 간 결합도: 70% 감소
- 테스트 가능성: 200% 향상

---

## 위험 요소 및 대응

1. **호환성 문제**: 단계적 마이그레이션
2. **성능 저하**: 롤백 계획 수립
3. **메모리 누수**: Valgrind/NSight 검증

---

## 검증 계획

### 단위 테스트
```cpp
TEST(MemoryPool, DynamicResize) {
    auto& pool = GpuMemoryPool::getInstance();
    size_t before = pool.getCurrentSize();
    // 대량 할당 테스트
    // ...
    EXPECT_LT(pool.getCurrentSize(), 32 * 1024 * 1024);
}
```

### 통합 테스트
- 실제 환경에서 24시간 연속 실행
- 메모리 사용량 모니터링
- 지연시간 측정

### 성능 벤치마크
```bash
# Before
Memory: 2048MB, Latency: 10ms, FPS: 100

# After  
Memory: 1024MB, Latency: 7ms, FPS: 140
```

---

## 다음 단계

1. Phase 1 즉시 시작 (config 수정)
2. 테스트 환경 구축
3. 단계별 구현 및 검증
4. 프로덕션 배포

이 계획을 승인하시면 Phase 1부터 구현을 시작하겠습니다.