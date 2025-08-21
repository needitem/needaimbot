# GPU 최적화 최종 실행 계획 (Gemini 협의 완료)

## 핵심 문제 진단
- **RTX 3080이 FPS 120 제한에도 GPU 100% 사용**
- **근본 원인**: 14개 이상의 동기화 지점과 불필요한 GPU↔CPU 데이터 왕복
- **가장 치명적인 병목**: `postProcessGpu.cu:1088`의 blocking `cudaMemcpy`

## 최우선 타겟 (발견된 주요 병목점)

### 1. Blocking cudaMemcpy (즉시 수정 필요)
```cpp
// postProcessGpu.cu:1088 - 현재 코드 (차단)
cudaMemcpy(&best_index_host, d_best_index, sizeof(int), cudaMemcpyDeviceToHost);

// 수정안 - 비동기 전환
cudaMemcpyAsync(&best_index_host, d_best_index, sizeof(int), 
                cudaMemcpyDeviceToHost, stream);
cudaEventRecord(bestIndexReady, stream);
```

### 2. GPU→CPU→GPU 왕복 제거
```cpp
// unified_graph_pipeline.cu:1779-1786 - 현재 코드
cudaMemcpyAsync(&finalCount, m_d_finalTargetsCount, sizeof(int), 
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);  // 강제 대기!
// ... finalCount를 다시 GPU 커널에 전달

// 수정안 - GPU에 유지
findClosestTargetGpu(
    m_d_finalTargets,
    m_d_finalTargetsCount,  // 디바이스 포인터 직접 전달
    crosshairX, crosshairY,
    m_d_bestTargetIndex, m_d_bestTarget,
    stream
);
```

## 구현 일정 (4주 계획)

### Week 1: 즉각적 성능 개선

#### Day 1-2: 프로파일링 및 측정
```bash
# 기준선 설정
nsys profile --stats=true --output=baseline.nsys-rep ./needaimbot
ncu --set full --output baseline.ncu-rep ./needaimbot

# 주요 측정 지표
# - 각 cudaStreamSynchronize의 대기 시간
# - blocking cudaMemcpy의 차단 시간
# - GPU 커널 실행 시간 및 occupancy
```

#### Day 3-4: 치명적 병목 제거 (우선순위 순)
1. **최우선**: 모든 blocking `cudaMemcpy`를 `cudaMemcpyAsync`로 전환
   ```cpp
   // postProcessGpu.cu:1088
   // gpu_tracker.cu의 모든 blocking 전송
   ```

2. **다음**: `finalCount`를 GPU에 유지
   - `findClosestTargetGpu` 커널 수정하여 device pointer 직접 읽기

3. **그 다음**: 필수 D2H 전송 최적화
   ```cpp
   // Best target 데이터 (마우스 이동에 필요)
   // Pinned memory 사용 + 비동기 전송
   cudaHostAlloc(&h_bestTarget_pinned, sizeof(Target), cudaHostAllocDefault);
   cudaMemcpyAsync(h_bestTarget_pinned, m_d_bestTarget, sizeof(Target),
                   cudaMemcpyDeviceToHost, stream);
   cudaEventRecord(targetReady, stream);
   ```

4. **마지막**: 디버그 전용 D2H 전송 제거
   ```cpp
   #ifdef PRODUCTION_BUILD
   // 디버그 목적 데이터 전송 생략
   #endif
   ```

#### Day 5-7: CPU측 프레임 제한 구현
```cpp
class CPUFrameLimiter {
private:
    std::chrono::steady_clock::time_point last_frame;
    std::chrono::microseconds target_frame_time;
    
public:
    CPUFrameLimiter(int target_fps) 
        : target_frame_time(1000000 / target_fps) {
        last_frame = std::chrono::steady_clock::now();
    }
    
    void limitFrame() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - last_frame;
        
        if (elapsed < target_frame_time) {
            std::this_thread::sleep_until(last_frame + target_frame_time);
        }
        
        last_frame = std::chrono::steady_clock::now();
    }
};
```

### Week 2-3: CUDA Graph 구현

#### CUDA Graph 전략
```cpp
class OptimizedPipeline {
private:
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // 동적 업데이트용 노드 핸들
    cudaGraphNode_t captureNode;
    cudaGraphNode_t inferenceNode;
    
public:
    void createStaticGraph() {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        
        // 1. 화면 캡처 (포인터만 업데이트)
        cudaMemcpyAsync(d_input, h_capture, capture_size, 
                       cudaMemcpyHostToDevice, stream);
        
        // 2. 전처리
        preprocessKernel<<<blocks, threads, 0, stream>>>(d_input, d_processed);
        
        // 3. TensorRT 추론
        context->enqueueV2(bindings, stream, nullptr);
        
        // 4. NMS 후처리
        nmsKernel<<<blocks, threads, 0, stream>>>(
            d_detections, d_finalTargets, d_finalTargetsCount);
        
        // 5. 타겟 선택 (device-side count 사용)
        findClosestTargetGpu<<<1, 256, 0, stream>>>(
            d_finalTargets, d_finalTargetsCount,  // GPU에 유지
            crosshairX, crosshairY,
            d_bestTargetIndex, d_bestTarget);
        
        // 6. 마우스 이동용 데이터만 CPU로 (비동기)
        cudaMemcpyAsync(h_bestTarget_pinned, d_bestTarget, sizeof(Target),
                       cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(targetReady, stream);
        
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    }
    
    void executeFrame(void* new_capture_ptr) {
        // 동적 요소만 업데이트
        cudaGraphExecKernelNodeSetParams(graphExec, captureNode, &new_capture_ptr);
        
        // 그래프 실행
        cudaGraphLaunch(graphExec, stream);
        
        // CPU는 마우스 이동을 위해 비동기로 대기
        processMouseMovementAsync();
    }
};
```

#### 동적 부분 처리
```cpp
void processMouseMovementAsync() {
    // 별도 스레드에서 실행
    static std::thread mouseThread;
    
    if (mouseThread.joinable()) {
        mouseThread.join();
    }
    
    mouseThread = std::thread([this]() {
        // Best target 데이터 준비될 때까지 대기
        cudaEventSynchronize(targetReady);
        
        // 마우스 이동 수행
        if (h_bestTarget_pinned->confidence > threshold) {
            moveMouse(h_bestTarget_pinned->x, h_bestTarget_pinned->y);
        }
    });
}
```

### Week 4: 검증 및 미세 조정

#### 정확도 검증 (0% 저하 요구사항)
```cpp
class AccuracyMonitor {
    struct Metrics {
        float hit_rate;
        float false_positive_rate;
        float avg_target_distance;
        float tracking_consistency;
    };
    
    Metrics baseline;
    Metrics current;
    
    bool validateNoRegression() {
        // 각 지표가 baseline 대비 저하되지 않았는지 확인
        return current.hit_rate >= baseline.hit_rate &&
               current.false_positive_rate <= baseline.false_positive_rate &&
               current.tracking_consistency >= baseline.tracking_consistency;
    }
};
```

#### 성능 측정 스크립트
```python
import subprocess
import nvidia_ml_py as nvml
import pandas as pd
import matplotlib.pyplot as plt

def measure_optimization_impact():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    configs = [
        ('baseline', './needaimbot --no-optimization'),
        ('phase1', './needaimbot --async-only'),
        ('phase2', './needaimbot --cuda-graphs'),
        ('final', './needaimbot --full-optimization')
    ]
    
    results = {}
    for name, command in configs:
        gpu_utils = []
        power_draws = []
        
        proc = subprocess.Popen(command.split())
        
        for _ in range(60):  # 60초 측정
            util = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            
            gpu_utils.append(util)
            power_draws.append(power)
            time.sleep(1)
            
        proc.terminate()
        
        results[name] = {
            'avg_gpu_util': np.mean(gpu_utils),
            'avg_power': np.mean(power_draws),
            'peak_gpu_util': np.max(gpu_utils)
        }
    
    return results
```

## 예상 성과

### 단계별 GPU 사용량 감소
1. **Week 1 완료**: 100% → 75-80% (blocking 제거 + 비동기화)
2. **Week 2-3 완료**: 75-80% → 60-65% (CUDA Graphs)
3. **Week 4 완료**: 60-65% → 55-60% (미세 조정)

### 지연시간 개선
- **현재**: ~15ms (동기화 대기 포함)
- **목표**: ~8-10ms (비동기 파이프라인)

### 전력 소비
- **현재**: 320W (RTX 3080 풀로드)
- **목표**: 200-240W (실제 작업량에 비례)

## 리스크 관리

### 롤백 전략
```cpp
class OptimizationManager {
    enum Level {
        BASELINE = 0,
        ASYNC_TRANSFERS = 1,
        CUDA_GRAPHS = 2,
        FULL_OPTIMIZATION = 3
    };
    
    Level current_level = BASELINE;
    
    void applyLevel(Level level) {
        switch(level) {
            case ASYNC_TRANSFERS:
                enableAsyncTransfers();
                break;
            case CUDA_GRAPHS:
                enableCudaGraphs();
                break;
            // ...
        }
    }
    
    void rollback() {
        if (current_level > BASELINE) {
            applyLevel(static_cast<Level>(current_level - 1));
        }
    }
};
```

### 실시간 모니터링
```cpp
void monitorPerformance() {
    static PerformanceMonitor monitor;
    
    // 매 100프레임마다 체크
    if (frame_count % 100 == 0) {
        auto metrics = monitor.getMetrics();
        
        if (metrics.gpu_util > 90 && metrics.fps < target_fps * 0.95) {
            log_warning("Performance degradation detected");
            optimization_manager.rollback();
        }
        
        if (metrics.accuracy < baseline_accuracy * 0.98) {
            log_critical("Accuracy drop detected");
            optimization_manager.disable();
        }
    }
}
```

## 핵심 원칙
1. **정확도 우선**: 어떤 최적화도 검출 정확도를 희생하지 않음
2. **점진적 적용**: 각 변경 후 완전한 테스트
3. **측정 기반**: 추측이 아닌 프로파일링 데이터로 결정
4. **안정성 보장**: 모든 최적화는 롤백 가능해야 함

## 다음 단계 체크리스트
- [ ] nsys/ncu 프로파일링 완료
- [ ] blocking cudaMemcpy 모두 제거
- [ ] finalCount GPU 유지 구현
- [ ] CUDA Graph 프로토타입 테스트
- [ ] 정확도 회귀 테스트 통과
- [ ] 다양한 GPU에서 검증 (RTX 3080, 4080, 4090)