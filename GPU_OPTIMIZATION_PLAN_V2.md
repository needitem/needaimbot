# GPU 최적화 현실적 구현 계획 v2.0

## 현재 상태 분석
- **GPU**: RTX 3080, FPS 제한 120 설정
- **문제**: GPU 사용량 100% 지속 (FPS 제한이 GPU에 영향 없음)
- **핵심 병목**: 
  - 14개 이상의 동기화 지점 (`cudaStreamSynchronize`, `cudaDeviceSynchronize`)
  - 매 프레임 강제 동기화로 GPU가 지속적으로 작업 수행
  - CPU는 대기하지만 GPU는 계속 실행

## 현실적 목표 설정

### 단기 목표 (2주)
- GPU 사용량: 100% → 80-85%
- 실제 FPS: 110 → 120 (안정화)
- 지연시간: 10-15% 감소

### 중기 목표 (1개월)
- GPU 사용량: 80-85% → 70-75%
- 전력 소비: 15-20% 감소
- 지연시간: 20-25% 감소

### 장기 목표 (2개월)
- GPU 사용량: 70-75% → 60-65%
- 전력 소비: 25-30% 감소
- 지연시간: 30% 감소

## 구현 계획

### Phase 0: 프로파일링 및 기준선 설정 (3일)

#### 0.1 현재 성능 측정
```bash
# GPU 메트릭 수집 스크립트
#!/bin/bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format=csv,noheader -l 1 > gpu_baseline.csv &

# 애플리케이션 실행
./needaimbot --fps-limit 120

# 프로파일링
nsys profile --stats=true --output=baseline ./needaimbot
```

#### 0.2 병목 지점 정량화
- 각 동기화 지점의 대기 시간 측정
- kernel 실행 시간 분석
- 메모리 전송 패턴 분석

### Phase 1: 저위험 최적화 (1주)

#### 1.1 불필요한 동기화 제거
**파일**: `unified_graph_pipeline.cu:1786`

**현재 코드**:
```cpp
cudaMemcpyAsync(&finalCount, m_d_finalTargetsCount, sizeof(int), 
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);  // 매 프레임 강제 대기
```

**수정 방안**:
```cpp
// Double buffering for count
class TargetCountBuffer {
private:
    int* h_count[2];  // Pinned memory
    cudaEvent_t events[2];
    int current_idx = 0;
    bool first_frame = true;
    
public:
    TargetCountBuffer() {
        for (int i = 0; i < 2; i++) {
            cudaHostAlloc(&h_count[i], sizeof(int), cudaHostAllocDefault);
            cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        }
    }
    
    void startCopy(int* d_count, cudaStream_t stream) {
        cudaMemcpyAsync(h_count[current_idx], d_count, sizeof(int),
                       cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(events[current_idx], stream);
    }
    
    int getCount() {
        if (first_frame) {
            // 첫 프레임은 동기화 필요
            cudaEventSynchronize(events[current_idx]);
            first_frame = false;
        } else {
            // 이전 프레임 결과 사용 (1 프레임 지연 허용)
            int prev_idx = 1 - current_idx;
            if (cudaEventQuery(events[prev_idx]) == cudaSuccess) {
                current_idx = prev_idx;
            }
        }
        return *h_count[current_idx];
    }
};
```

**리스크**: 낮음 (1프레임 지연은 120FPS에서 8.3ms)
**예상 효과**: GPU 사용량 5-10% 감소

#### 1.2 디버그 동기화 조건부 컴파일
**파일**: `draw_debug.cpp`, `cuda_error_check.h`

```cpp
#ifdef DEBUG_MODE
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR();
#else
    // Production: 동기화 제거
    #ifdef PERIODIC_CHECK
        static int frame_count = 0;
        if (++frame_count % 100 == 0) {  // 100프레임마다 체크
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                log_error("CUDA error: %s", cudaGetErrorString(err));
            }
        }
    #endif
#endif
```

**리스크**: 낮음 (디버그 코드만 영향)
**예상 효과**: GPU 사용량 3-5% 감소

### Phase 2: 중위험 최적화 (2주)

#### 2.1 Adaptive Frame Limiting
**새 파일**: `adaptive_fps_controller.cu`

```cpp
class AdaptiveFPSController {
private:
    cudaEvent_t frame_start, frame_end;
    float target_frame_time_ms;
    float actual_frame_time_ms;
    float sleep_ratio = 0.0f;  // 0.0 ~ 1.0
    
public:
    AdaptiveFPSController(int target_fps) {
        target_frame_time_ms = 1000.0f / target_fps;
        cudaEventCreate(&frame_start);
        cudaEventCreate(&frame_end);
    }
    
    void frameStart(cudaStream_t stream) {
        cudaEventRecord(frame_start, stream);
    }
    
    void frameEnd(cudaStream_t stream) {
        cudaEventRecord(frame_end, stream);
        
        // GPU에 sleep 삽입
        if (sleep_ratio > 0.0f) {
            int sleep_cycles = (int)(sleep_ratio * 1000000);  // microseconds
            insertGPUSleep<<<1, 1, 0, stream>>>(sleep_cycles);
        }
        
        // 프레임 시간 측정 (비동기)
        static cudaEvent_t measure_event = nullptr;
        if (!measure_event) {
            cudaEventCreate(&measure_event);
        }
        
        cudaEventRecord(measure_event, stream);
        if (cudaEventQuery(measure_event) == cudaSuccess) {
            cudaEventElapsedTime(&actual_frame_time_ms, frame_start, frame_end);
            adjustSleepRatio();
        }
    }
    
private:
    void adjustSleepRatio() {
        float error = actual_frame_time_ms - target_frame_time_ms;
        
        if (error < -1.0f) {  // 너무 빠름
            sleep_ratio = min(1.0f, sleep_ratio + 0.05f);
        } else if (error > 1.0f) {  // 너무 느림
            sleep_ratio = max(0.0f, sleep_ratio - 0.05f);
        }
    }
};

__global__ void insertGPUSleep(int cycles) {
    clock_t start = clock();
    clock_t end = start + cycles;
    while (clock() < end);
}
```

**리스크**: 중간 (타이밍 민감)
**예상 효과**: GPU 사용량 10-15% 감소

#### 2.2 Selective Stream Usage
**파일**: `unified_graph_pipeline.cu`

```cpp
class StreamScheduler {
private:
    cudaStream_t high_priority;   // 추론용
    cudaStream_t low_priority;    // 후처리용
    cudaEvent_t inference_done;
    
public:
    StreamScheduler() {
        int high_prio, low_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        
        cudaStreamCreateWithPriority(&high_priority, cudaStreamNonBlocking, high_prio);
        cudaStreamCreateWithPriority(&low_priority, cudaStreamNonBlocking, low_prio);
        cudaEventCreateWithFlags(&inference_done, cudaEventDisableTiming);
    }
    
    void scheduleInference(/* params */) {
        // 중요 작업은 high priority stream
        runTensorRTInference(high_priority);
        cudaEventRecord(inference_done, high_priority);
    }
    
    void schedulePostProcess(/* params */) {
        // 덜 중요한 작업은 low priority stream
        cudaStreamWaitEvent(low_priority, inference_done);
        performNMS(low_priority);
    }
};
```

**리스크**: 중간 (동기화 복잡도 증가)
**예상 효과**: 지연시간 10-15% 감소

### Phase 3: 고위험 최적화 (3주)

#### 3.1 GPU Power Management API 활용
```cpp
class GPUPowerManager {
private:
    nvmlDevice_t device;
    unsigned int original_limit;
    
public:
    GPUPowerManager() {
        nvmlInit();
        nvmlDeviceGetHandleByIndex(0, &device);
        nvmlDeviceGetPowerManagementLimit(device, &original_limit);
    }
    
    void setFPSMode(int target_fps) {
        unsigned int new_limit;
        
        if (target_fps <= 60) {
            new_limit = original_limit * 0.6;  // 60% power
        } else if (target_fps <= 120) {
            new_limit = original_limit * 0.8;  // 80% power
        } else {
            new_limit = original_limit;        // 100% power
        }
        
        nvmlDeviceSetPowerManagementLimit(device, new_limit);
    }
    
    ~GPUPowerManager() {
        nvmlDeviceSetPowerManagementLimit(device, original_limit);
        nvmlShutdown();
    }
};
```

**리스크**: 높음 (시스템 전체 영향)
**예상 효과**: 전력 소비 20-30% 감소

#### 3.2 Memory Pool with Alignment
```cpp
class AlignedMemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
        size_t alignment;
    };
    
    std::vector<Block> blocks;
    void* pool_base;
    size_t pool_size;
    
public:
    AlignedMemoryPool(size_t total_size) : pool_size(total_size) {
        cudaMalloc(&pool_base, pool_size);
    }
    
    template<typename T>
    T* allocate(size_t count, size_t alignment = 256) {
        size_t bytes = count * sizeof(T);
        bytes = (bytes + alignment - 1) & ~(alignment - 1);  // Align size
        
        // Find free block or allocate new
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= bytes && 
                block.alignment == alignment) {
                block.in_use = true;
                return reinterpret_cast<T*>(block.ptr);
            }
        }
        
        // Allocate new block from pool
        // ... (implementation details)
    }
    
    void deallocate(void* ptr) {
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
};
```

**리스크**: 높음 (메모리 관리 복잡)
**예상 효과**: 메모리 대역폭 10-15% 개선

## 테스트 및 검증 전략

### 자동화된 성능 테스트
```python
import subprocess
import pandas as pd
import nvidia_ml_py as nvml

class GPUOptimizationTester:
    def __init__(self):
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
    def run_test(self, version, duration=60):
        results = {
            'version': version,
            'gpu_util': [],
            'power': [],
            'fps': [],
            'latency': []
        }
        
        proc = subprocess.Popen(['./needaimbot', f'--version={version}'])
        
        for _ in range(duration):
            util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
            power = nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            
            results['gpu_util'].append(util.gpu)
            results['power'].append(power)
            time.sleep(1)
            
        proc.terminate()
        return results
    
    def compare_versions(self, baseline, optimized):
        baseline_results = self.run_test(baseline)
        optimized_results = self.run_test(optimized)
        
        improvement = {
            'gpu_util': (mean(baseline_results['gpu_util']) - 
                        mean(optimized_results['gpu_util'])) / 
                        mean(baseline_results['gpu_util']) * 100,
            'power': (mean(baseline_results['power']) - 
                     mean(optimized_results['power'])) / 
                     mean(baseline_results['power']) * 100
        }
        
        return improvement
```

### 회귀 테스트
```cpp
class AccuracyValidator {
    float baseline_accuracy;
    float acceptable_degradation = 0.02f;  // 2% 허용
    
    bool validateOptimization(const OptimizedVersion& ver) {
        float new_accuracy = measureDetectionAccuracy(ver);
        
        if (new_accuracy < baseline_accuracy - acceptable_degradation) {
            log_error("Accuracy degradation: %.2f%% -> %.2f%%",
                     baseline_accuracy * 100, new_accuracy * 100);
            return false;
        }
        
        return true;
    }
};
```

### A/B 테스트 프레임워크
```cpp
enum class OptimizationLevel {
    NONE = 0,
    PHASE_1 = 1,
    PHASE_2 = 2,
    PHASE_3 = 3
};

class ABTestFramework {
    OptimizationLevel current_level;
    
    void switchOptimization(OptimizationLevel level) {
        current_level = level;
        
        switch(level) {
            case OptimizationLevel::PHASE_1:
                enableAsyncTargetCount();
                disableDebugSync();
                break;
            case OptimizationLevel::PHASE_2:
                enableAdaptiveFPS();
                enableStreamPriority();
                break;
            case OptimizationLevel::PHASE_3:
                enablePowerManagement();
                enableMemoryPool();
                break;
        }
    }
};
```

## 구현 일정

### Week 1-2: Phase 0 + Phase 1
- Day 1-3: 프로파일링 및 기준선 설정
- Day 4-7: 불필요한 동기화 제거
- Day 8-10: 디버그 코드 최적화
- Day 11-14: 테스트 및 검증

### Week 3-4: Phase 2
- Day 15-18: Adaptive FPS Controller 구현
- Day 19-22: Stream Priority 구현
- Day 23-28: 통합 테스트

### Week 5-7: Phase 3
- Day 29-35: Power Management API 통합
- Day 36-42: Memory Pool 구현
- Day 43-49: 최종 테스트 및 튜닝

### Week 8: 최종 검증
- 전체 시스템 테스트
- 다양한 GPU에서 검증
- 문서화 및 배포 준비

## 리스크 관리

### 롤백 계획
```cpp
class OptimizationRollback {
    std::stack<std::function<void()>> rollback_stack;
    
    void applyOptimization(const Optimization& opt) {
        try {
            opt.apply();
            rollback_stack.push(opt.getRevertFunction());
        } catch (...) {
            rollbackAll();
            throw;
        }
    }
    
    void rollbackAll() {
        while (!rollback_stack.empty()) {
            rollback_stack.top()();
            rollback_stack.pop();
        }
    }
};
```

### 모니터링 및 알림
```python
def monitor_optimization():
    if gpu_util > 90 and fps < target_fps * 0.9:
        send_alert("Optimization not effective")
        rollback_to_previous_version()
    
    if detection_accuracy < baseline * 0.98:
        send_alert("Accuracy degradation detected")
        disable_optimization()
```

## 성공 지표

### 필수 달성 목표
- [ ] GPU 사용량 20% 이상 감소
- [ ] FPS 안정성 99% 이상
- [ ] 검출 정확도 98% 이상 유지
- [ ] 지연시간 20% 이상 개선

### 선택적 목표
- [ ] 전력 소비 30% 감소
- [ ] 메모리 사용량 20% 감소
- [ ] 다중 GPU 지원

## 참고 자료
- [NVIDIA GPU Gems: GPU Occupancy](https://developer.nvidia.com/content/gpu-gems)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVML API Reference](https://developer.nvidia.com/nvidia-management-library-nvml)