# CPU 최적화 분석 보고서

## 개요
needaimbot 프로젝트의 CPU 사용률을 줄이기 위한 종합적인 분석과 최적화 방안을 정리한 문서입니다.

## 현재 스레드 구조 분석

### 스레드별 역할 및 CPU 사용 패턴

| 스레드 | 역할 | CPU 패턴 | 우선순위 | 주요 이슈 |
|--------|------|----------|----------|-----------|
| **Main Thread** | 애플리케이션 생명주기 관리 | 이벤트 기반 (낮음) | Normal | - |
| **CaptureThread** | 화면 캡처 및 GPU 전송 | **폴링 (매우 높음)** | TIME_CRITICAL | CPU 최다 사용 |
| **InferenceThread** | AI 추론 및 후처리 | 연산 집약적/이벤트 | HIGHEST | GPU 동기화 오버헤드 |
| **MouseThread** | 마우스 제어 | 이벤트 기반 | TIME_CRITICAL | 입력 지연 가능성 |
| **OverlayThread** | UI 렌더링 | 제한적 폴링 | BELOW_NORMAL | - |
| **KeyboardThread** | 키보드 입력 감지 | **폴링 (중간)** | Normal | 불필요한 CPU 사이클 |

### 스레드 간 통신 방식
- **AppContext 싱글턴**: 전역 설정 및 상태 공유
- **Condition Variables**: InferenceThread → MouseThread 이벤트 알림
- **Lockless Queue**: MouseThread 내부 명령 큐
- **Atomic 플래그**: 설정 변경 감지

## 주요 CPU 병목 지점

### 1. CaptureThread - 가장 심각한 CPU 사용
- **현재 방식**: `while` 루프에서 지속적인 화면 폴링
- **문제점**: 
  - FPS 제한이 없거나 높을 경우 CPU 100% 사용 가능
  - 새 프레임이 없어도 계속 확인하는 비효율
  - TIME_CRITICAL 우선순위로 다른 프로세스에도 영향

### 2. KeyboardThread - 불필요한 폴링
- **현재 방식**: 10ms마다 `GetAsyncKeyState()` 호출
- **문제점**: 키 입력이 없어도 지속적인 CPU 사용

### 3. InferenceThread - GPU 동기화 오버헤드
- **현재 방식**: 동기식 `cudaMemcpy` 사용
- **문제점**: CPU가 GPU 작업 완료를 대기하며 블로킹

## 최적화 방안

### 1. 폴링 → 이벤트 기반 아키텍처 전환

#### CaptureThread 최적화
```cpp
// Windows Graphics Capture API의 이벤트 기반 캡처
class WindowsGraphicsCapture {
    // FrameArrived 이벤트 사용
    void OnFrameArrived(Direct3D11CaptureFramePool const& sender) {
        auto frame = sender.TryGetNextFrame();
        // 새 프레임이 있을 때만 처리
        if (frame && m_frameCallback) {
            m_frameCallback(ConvertToMat(frame));
        }
    }
};
```
**예상 개선**: CPU 사용률 50-70% 감소

#### KeyboardThread 최적화
```cpp
// Windows Hook 사용
LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    // 키 이벤트 발생 시에만 실행
    if (nCode == HC_ACTION) {
        KBDLLHOOKSTRUCT* pkbhs = (KBDLLHOOKSTRUCT*)lParam;
        ProcessKeyEvent(pkbhs->vkCode, wParam == WM_KEYDOWN);
    }
    return CallNextHookEx(hook, nCode, wParam, lParam);
}
```
**예상 개선**: KeyboardThread CPU 사용률 거의 0%로 감소

### 2. GPU 동기화 최적화

#### 비동기 메모리 복사 및 CUDA Stream 활용
```cpp
void Detector::RunInferenceAsync(cudaStream_t stream) {
    // TensorRT 비동기 추론
    m_context->enqueueV2(&bindings[0], stream, nullptr);
    
    // 비동기 메모리 복사 (Pinned Memory 사용)
    cudaMemcpyAsync(pinned_output, gpu_output, size, 
                    cudaMemcpyDeviceToHost, stream);
    
    // 이벤트 기록
    cudaEventRecord(m_inference_done, stream);
}

// CPU는 다른 작업 수행 가능
PrepareNextFrame();

// 필요시에만 동기화
cudaEventSynchronize(m_inference_done);
```
**예상 개선**: GPU 대기 시간 20-30% 감소

### 3. 캐시 최적화

#### SoA (Structure of Arrays) 변환
```cpp
// 기존 AoS (Array of Structures)
struct Target {
    float x, y, width, height;
    int class_id;
    float confidence;
};
std::vector<Target> targets;

// 최적화된 SoA
struct Targets {
    std::vector<float> x, y, width, height;
    std::vector<int> class_ids;
    std::vector<float> confidences;
};
```
**예상 개선**: 캐시 미스 30-40% 감소, 추적 성능 15-20% 향상

### 4. SIMD 명령어 활용

#### 벡터 연산 최적화
```cpp
// AVX2를 사용한 거리 계산
__m256 CalculateDistances8(const float* x1, const float* y1, 
                           const float* x2, const float* y2) {
    __m256 dx = _mm256_sub_ps(_mm256_load_ps(x1), _mm256_load_ps(x2));
    __m256 dy = _mm256_sub_ps(_mm256_load_ps(y1), _mm256_load_ps(y2));
    __m256 dx2 = _mm256_mul_ps(dx, dx);
    __m256 dy2 = _mm256_mul_ps(dy, dy);
    return _mm256_sqrt_ps(_mm256_add_ps(dx2, dy2));
}
```
**예상 개선**: 수학 연산 4-8배 가속

### 5. GPU 오프로딩 확대

#### ByteTracker GPU 이전
```cpp
// 기존 CPU 칼만 필터를 GPU 버전으로 교체
class GPUByteTracker {
    GPUKalmanFilter gpu_kalman;  // cuda/gpu_kalman_filter.cu 활용
    
    void Update(const Detection* detections, int count) {
        // GPU에서 거리 계산 및 매칭
        gpu_kalman.BatchPredict(tracks_gpu, track_count);
        gpu_kalman.BatchUpdate(detections_gpu, detection_count);
    }
};
```
**예상 개선**: 추적 단계 CPU 부하 70-80% 감소

## 구현 우선순위

1. **즉시 구현 (높은 효과)**
   - CaptureThread 이벤트 기반 전환
   - KeyboardThread Windows Hook 적용

2. **단기 구현 (중간 효과)**
   - GPU 비동기 처리 최적화
   - Lockless Queue 전면 적용

3. **중장기 구현 (누적 효과)**
   - SoA 데이터 구조 변환
   - SIMD 명령어 적용
   - ByteTracker GPU 이전

## 예상 전체 성능 개선

| 지표 | 현재 | 최적화 후 | 개선율 |
|------|------|-----------|--------|
| **유휴 CPU 사용률** | 15-25% | 3-5% | 80% 감소 |
| **활성 CPU 사용률** | 40-60% | 20-30% | 50% 감소 |
| **프레임 처리 지연** | 8-12ms | 4-6ms | 40% 감소 |
| **전력 소비** | 높음 | 중간 | 30-40% 감소 |

## 추가 고려사항

### CPU 친화성 설정
```cpp
// 각 스레드를 특정 CPU 코어에 고정
SetThreadAffinityMask(capture_thread, 1 << 0);  // Core 0
SetThreadAffinityMask(inference_thread, 1 << 2); // Core 2
SetThreadAffinityMask(mouse_thread, 1 << 4);     // Core 4
```

### 동적 성능 조절
- 게임 상태에 따른 캡처 FPS 자동 조절
- 타겟 없을 시 추론 빈도 감소
- 배터리 모드에서 성능 프로파일 전환

## 결론

현재 needaimbot의 주요 CPU 병목은 **폴링 기반 아키텍처**에서 발생하고 있습니다. 이벤트 기반으로 전환하고 GPU 활용을 확대하면 CPU 사용률을 크게 줄이면서도 성능을 유지하거나 향상시킬 수 있습니다.

가장 효과적인 최적화는:
1. CaptureThread의 이벤트 기반 전환 (CPU 50-70% 감소)
2. GPU 오프로딩 확대 (CPU 부하 이전)
3. 캐시 친화적 데이터 구조 (성능 15-20% 향상)

이러한 최적화를 통해 더 많은 시스템 리소스를 게임에 할당할 수 있고, 발열과 전력 소비를 줄여 안정적인 성능을 제공할 수 있습니다.