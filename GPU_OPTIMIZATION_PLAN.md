# GPU 사용량 최적화 구현 계획

## 문제 정의
- **현상**: RTX 3080에서 FPS를 120으로 제한해도 GPU 사용량이 100%
- **원인**: CPU만 대기하고 GPU는 계속 작업을 처리
- **목표**: FPS 제한 시 GPU도 실제로 유휴 상태가 되도록 수정

## 구현 단계

### Phase 1: Quick Fix (즉시 적용 가능) - 1일

#### 1.1 동기화 제거 및 Frame Pacing 구현
**파일**: `unified_graph_pipeline.cu`

**현재 코드 (라인 1786)**:
```cpp
cudaStreamSynchronize(stream);  // 강제 동기화
```

**수정 코드**:
```cpp
// Frame depth control을 위한 이벤트 풀
static std::array<cudaEvent_t, 3> frameEvents;
static bool eventsInitialized = false;
static int frameIndex = 0;

if (!eventsInitialized) {
    for (auto& event : frameEvents) {
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    }
    eventsInitialized = true;
}

// 현재 프레임의 GPU 작업 완료 기록
cudaEventRecord(frameEvents[frameIndex], stream);

// N-2 프레임이 완료될 때까지 대기 (파이프라인 깊이 제어)
int waitIndex = (frameIndex + 1) % 3;  // N-2 frame
cudaEventSynchronize(frameEvents[waitIndex]);

frameIndex = (frameIndex + 1) % 3;
```

**효과**: 
- GPU 작업 큐 깊이를 2-3 프레임으로 제한
- FPS 제한 시 GPU가 실제로 대기 상태가 됨
- 예상 GPU 사용량: 100% → 40-60%

#### 1.2 비동기 타겟 카운트 처리
**현재 코드 (라인 1779-1786)**:
```cpp
cudaMemcpyAsync(&finalCount, m_d_finalTargetsCount, sizeof(int), 
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);  // 즉시 동기화
```

**수정 코드**:
```cpp
// Pinned memory for async transfer
static int* h_finalCount_pinned = nullptr;
static cudaEvent_t countReadyEvent = nullptr;
static bool countInProgress = false;

if (!h_finalCount_pinned) {
    cudaHostAlloc(&h_finalCount_pinned, sizeof(int), cudaHostAllocDefault);
    cudaEventCreateWithFlags(&countReadyEvent, cudaEventDisableTiming);
}

// 이전 프레임 결과 확인
if (countInProgress && cudaEventQuery(countReadyEvent) == cudaSuccess) {
    finalCount = *h_finalCount_pinned;
    countInProgress = false;
}

// 현재 프레임 비동기 복사 시작
cudaMemcpyAsync(h_finalCount_pinned, m_d_finalTargetsCount, sizeof(int),
                cudaMemcpyDeviceToHost, stream);
cudaEventRecord(countReadyEvent, stream);
countInProgress = true;
```

### Phase 2: Stream Parallelism (2-3일)

#### 2.1 Multi-Stream Pipeline 활성화
**파일**: `unified_graph_pipeline.cu`

**현재 문제**:
- 5개 스트림 생성했지만 실제로는 `m_primaryStream`만 사용
- 모든 스트림이 동일한 우선순위(1)로 설정

**수정 방안**:
```cpp
// PipelineCoordinator 수정 (라인 116-118)
static constexpr int CAPTURE_PRIORITY = -1;     // 높음
static constexpr int INFERENCE_PRIORITY = 0;    // 보통  
static constexpr int POSTPROCESS_PRIORITY = 1;  // 낮음

// executeGraphNonBlocking 리팩토링
void executeGraphNonBlocking() {
    // Stage 1: Capture on captureStream
    captureDesktopFrame(m_coordinator->captureStream);
    cudaEventRecord(m_coordinator->captureComplete, m_coordinator->captureStream);
    
    // Stage 2: Preprocessing waits for capture
    cudaStreamWaitEvent(m_coordinator->preprocessStream, m_coordinator->captureComplete);
    cuda_unified_preprocessing(..., m_coordinator->preprocessStream);
    cudaEventRecord(m_coordinator->preprocessComplete, m_coordinator->preprocessStream);
    
    // Stage 3: Inference waits for preprocessing
    cudaStreamWaitEvent(m_coordinator->inferenceStream, m_coordinator->preprocessComplete);
    runInferenceAsync(m_coordinator->inferenceStream);
    cudaEventRecord(m_coordinator->inferenceComplete, m_coordinator->inferenceStream);
    
    // Stage 4: Post-processing waits for inference
    cudaStreamWaitEvent(m_coordinator->postprocessStream, m_coordinator->inferenceComplete);
    performIntegratedPostProcessing(m_coordinator->postprocessStream);
}
```

**효과**:
- 캡처, 전처리, 추론, 후처리가 병렬로 실행
- 파이프라인 처리량 30-50% 향상
- 지연시간 감소

### Phase 3: Memory Optimization (3-4일)

#### 3.1 Arena Allocator 구현
**현재 문제**: 20개 이상의 개별 `cudaMalloc` 호출

**수정 방안**:
```cpp
class CudaArenaAllocator {
private:
    void* arenaBase;
    size_t arenaSize;
    size_t currentOffset;
    
public:
    CudaArenaAllocator(size_t totalSize) {
        cudaMalloc(&arenaBase, totalSize);
        arenaSize = totalSize;
        currentOffset = 0;
    }
    
    template<typename T>
    T* allocate(size_t count) {
        size_t bytes = count * sizeof(T);
        void* ptr = (char*)arenaBase + currentOffset;
        currentOffset += bytes;
        return reinterpret_cast<T*>(ptr);
    }
    
    void reset() { currentOffset = 0; }
    ~CudaArenaAllocator() { cudaFree(arenaBase); }
};

// 사용 예
CudaArenaAllocator arena(100 * 1024 * 1024);  // 100MB
m_d_x1 = arena.allocate<float>(maxDetections);
m_d_y1 = arena.allocate<float>(maxDetections);
// ... 나머지 버퍼들
```

#### 3.2 불필요한 메모리 복사 제거
**현재**: Triple buffer → Capture buffer → YOLO input → TensorRT input

**최적화**:
```cpp
// TensorRT가 YOLO input buffer를 직접 사용하도록 설정
m_inputBindings[m_inputName] = m_d_yoloInput;  // 복사 제거

// Triple buffer를 capture buffer로 직접 사용
// 중간 복사 단계 제거
```

### Phase 4: Advanced Optimizations (선택적)

#### 4.1 CUDA Graph 완전 활용
```cpp
// 전체 파이프라인을 CUDA Graph로 캡처
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... 모든 GPU 작업 ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// 매 프레임마다 그래프 실행
cudaGraphLaunch(graphExec, stream);
```

#### 4.2 Dynamic Parallelism
- Kernel 내부에서 다른 kernel 호출
- NMS와 후처리를 단일 kernel로 통합

## 성능 목표

### 현재 상태
- GPU 사용량: 100% (FPS 제한과 무관)
- 실제 FPS: 110 (120 목표)
- 지연시간: 높음

### 목표 상태 (Phase 1 완료 후)
- GPU 사용량: 40-60% (120 FPS 제한 시)
- 실제 FPS: 120 (안정적)
- 지연시간: 현재 대비 30% 감소

### 최종 목표 (모든 Phase 완료)
- GPU 사용량: 30-40% (120 FPS 제한 시)
- 실제 FPS: 120 (안정적)
- 지연시간: 현재 대비 50% 감소
- 전력 소비: 현재 대비 40% 감소

## 테스트 계획

### 성능 측정
```bash
# GPU 사용량 모니터링
nvidia-smi dmon -s puc -i 0

# 프레임 타이밍 측정
nsys profile --stats=true ./needaimbot

# 전력 소비 측정
nvidia-smi -q -d POWER
```

### 검증 항목
- [ ] FPS 제한 작동 확인
- [ ] GPU 사용량 감소 확인
- [ ] 마우스 제어 정확도 유지
- [ ] 지연시간 개선 확인
- [ ] 메모리 사용량 확인

## 구현 우선순위

1. **즉시 (Day 1)**
   - Phase 1.1: 동기화 제거
   - Phase 1.2: 비동기 타겟 처리

2. **단기 (Week 1)**
   - Phase 2: Multi-Stream 활성화
   
3. **중기 (Week 2)**
   - Phase 3: Memory Optimization
   
4. **장기 (선택적)**
   - Phase 4: Advanced Optimizations

## 주의사항

- 각 수정 후 반드시 기능 테스트 수행
- 타겟 감지 정확도가 떨어지지 않는지 확인
- 게임 성능에 미치는 영향 측정
- 다른 GPU (RTX 4080, 4090)에서도 테스트

## 참고 자료

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Streams and Concurrency](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)
- [CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)