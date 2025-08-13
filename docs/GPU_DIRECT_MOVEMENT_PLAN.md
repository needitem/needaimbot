# GPU Direct Movement Calculation Implementation Plan

## 현재 상황 분석

### 문제점
1. **과도한 데이터 전송**: GPU → CPU로 전체 Target 구조체 복사
2. **CPU에서 dx/dy 계산**: 불필요한 CPU 연산
3. **동기화 오버헤드**: 매 프레임마다 여러 번의 cudaEventSynchronize
4. **높은 CPU 사용량**: 동기화 대기로 인한 CPU 블로킹

### 기존 리소스
- ✅ `selectTargetAndCalculateMovementKernel` - GPU에서 dx/dy 직접 계산하는 커널 존재
- ✅ `MouseMovement` 구조체 정의됨
- ❌ 하지만 현재 사용하지 않음 (`findClosestTargetGpu` 사용 중)

## 구현 계획

### Phase 1: 기반 구조 준비 (1-2시간)

#### 1.1 Detector 클래스 수정
```cpp
// detector.h에 추가
private:
    // GPU-calculated mouse movement
    CudaBuffer<MouseMovement> m_mouseMovementGpu;
    MouseMovement m_mouseMovementHost;
    cudaEvent_t m_movementReadyEvent;
```

#### 1.2 초기화 코드 추가
- `Detector::initialize()`에서 MouseMovement 버퍼 할당
- CUDA 이벤트 생성

### Phase 2: GPU 커널 통합 (2-3시간)

#### 2.1 타겟 선택 커널 교체
```cpp
// 기존 코드 (detector.cpp:1420)
findClosestTargetGpu(...);

// 새 코드
selectTargetAndCalculateMovementKernel<<<...>>>(
    m_finalTargetsGpu.get(),
    m_finalTargetsCountHost,
    m_mouseMovementGpu.get(),
    m_bestTargetIndexGpu.get(),
    maxDistance
);
```

#### 2.2 외부 함수 선언 추가
```cpp
// detector.cpp 상단에 추가
extern "C" void updateTargetSelectionConstants(
    float centerX, float centerY, float scopeMultiplier,
    float headYOffset, float bodyYOffset, int headClassId,
    cudaStream_t stream);

extern "C" void selectTargetAndCalculateMovementKernel(
    const Target* detections,
    int numDetections,
    MouseMovement* movement,
    int* bestIdx,
    float maxDistance);
```

### Phase 3: 데이터 흐름 최적화 (2-3시간)

#### 3.1 비동기 복사 파이프라인
```cpp
// GPU에서 MouseMovement만 복사 (8 bytes vs Target 구조체 전체)
cudaMemcpyAsync(&m_mouseMovementHost, m_mouseMovementGpu.get(), 
                sizeof(MouseMovement), cudaMemcpyDeviceToHost, 
                postprocessStream);
cudaEventRecord(m_movementReadyEvent, postprocessStream);
```

#### 3.2 이벤트 전달 수정
```cpp
// 이벤트 생성 시 GPU 계산된 dx/dy 직접 사용
MouseEvent event;
event.dx = m_mouseMovementHost.dx;
event.dy = m_mouseMovementHost.dy;
event.has_target = m_mouseMovementHost.hasTarget;
```

### Phase 4: 동기화 최적화 (1-2시간)

#### 4.1 더블 버퍼링 구현
```cpp
// 이전 프레임 처리 중 다음 프레임 캡처
MouseMovement m_movementBuffer[2];
int m_currentBufferIdx = 0;
```

#### 4.2 비블로킹 체크
```cpp
// cudaEventSynchronize 대신 cudaEventQuery 사용
if (cudaEventQuery(m_movementReadyEvent) == cudaSuccess) {
    // Movement ready, process it
} else {
    // Still processing, use previous frame data
}
```

### Phase 5: 추가 최적화 (선택사항)

#### 5.1 CUDA Graph 통합
- 전체 추론 파이프라인을 CUDA Graph로 캡처
- CPU 개입 최소화

#### 5.2 Zero-Copy 메모리
- Pinned memory 사용으로 복사 오버헤드 감소

#### 5.3 스트림 병렬화
- 캡처, 추론, 후처리를 서로 다른 스트림에서 실행

## 예상 성능 개선

### 현재
- CPU → GPU → CPU → CPU 계산 → Event
- 데이터 전송: ~100 bytes (Target 구조체)
- 동기화: 3-4회/프레임

### 개선 후
- GPU → GPU 계산 → CPU(dx,dy만) → Event
- 데이터 전송: 8 bytes (dx, dy)
- 동기화: 1회/프레임

### 예상 결과
- **CPU 사용량**: 30-50% 감소
- **레이턴시**: 2-3ms 감소
- **FPS 안정성**: 크게 향상

## 구현 우선순위

1. **필수 (Phase 1-3)**: 기본 GPU 직접 계산 구현
2. **권장 (Phase 4)**: 동기화 최적화
3. **선택 (Phase 5)**: 추가 최적화

## 위험 요소 및 대응

### 위험 1: 호환성 문제
- **문제**: 기존 코드와의 호환성
- **해결**: 기존 로직을 백업하고 플래그로 전환 가능하게 구현

### 위험 2: 정밀도 차이
- **문제**: GPU/CPU 부동소수점 연산 차이
- **해결**: 필요시 정밀도 보정 파라미터 추가

### 위험 3: 디버깅 어려움
- **문제**: GPU 코드 디버깅 복잡성
- **해결**: 상세한 로깅 및 CPU 폴백 모드 구현

## 테스트 계획

1. **단위 테스트**
   - MouseMovement 계산 정확도
   - 동기화 타이밍
   
2. **통합 테스트**
   - 전체 파이프라인 동작
   - CPU 사용량 측정
   
3. **성능 테스트**
   - FPS 측정
   - 레이턴시 측정
   - CPU/GPU 사용률 모니터링

## 구현 체크리스트

- [ ] Detector 클래스에 MouseMovement 버퍼 추가
- [ ] CUDA 이벤트 초기화
- [ ] selectTargetAndCalculateMovementKernel 통합
- [ ] 상수 메모리 업데이트 함수 호출
- [ ] MouseMovement 복사 파이프라인 구현
- [ ] MouseEvent 생성 로직 수정
- [ ] 동기화 최적화 (cudaEventQuery)
- [ ] 테스트 및 검증
- [ ] 성능 측정 및 문서화

## 예상 소요 시간

- **총 예상 시간**: 6-10시간
- **최소 구현** (Phase 1-3): 5-8시간
- **전체 최적화** (Phase 1-5): 8-12시간

## 참고 자료

- `needaimbot/postprocess/targetSelectionGpu.cu` - GPU 커널 구현
- `needaimbot/detector/detector.h` - MouseMovement 구조체 정의
- `needaimbot/detector/detector.cpp:1376-1540` - 현재 타겟 선택 로직