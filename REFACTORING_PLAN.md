# NeedAimbot GPU Pipeline 리팩토링 계획

## 현재 문제점
1. **역할 혼재**: `gpu_only_capture.cpp`가 캡처, 검출, 마우스 제어를 모두 수행
2. **중복 구조**: `unified_graph_pipeline.cu`가 있지만 사용되지 않음
3. **비효율적 흐름**: GPU→CPU→GPU 불필요한 데이터 이동
4. **모듈 간 결합도 높음**: capture 모듈이 detector, mouse 등에 의존

## 목표 아키텍처

```
Main Thread
    ↓
UnifiedGraphPipeline (전체 조율)
    ├── Capture (화면 캡처만)
    ├── Detector (YOLO 추론)
    ├── Tracker (Kalman 필터)
    ├── Controller (PID 제어)
    └── Mouse (입력 실행)
```

## 구현 단계

### Phase 1: Capture 모듈 단순화
**목표**: capture 폴더는 오직 화면 캡처만 담당

#### 1.1 gpu_only_capture.cpp 수정
- [ ] Detector 관련 코드 제거
- [ ] Mouse 계산 코드 제거
- [ ] 단순히 GPU 버퍼에 프레임 캡처만 수행
- [ ] UnifiedGraphPipeline에 버퍼 전달 인터페이스 추가

#### 1.2 새로운 capture 인터페이스
```cpp
class GPUCapture {
public:
    // 캡처된 프레임을 GPU 버퍼로 반환
    SimpleCudaMat* getCapturedFrame();
    // 캡처 준비 상태 확인
    bool isFrameReady();
};
```

### Phase 2: UnifiedGraphPipeline 활성화
**목표**: 모든 GPU 작업을 UnifiedGraphPipeline에서 통합 관리

#### 2.1 Pipeline 초기화 (needaimbot.cpp)
- [ ] UnifiedGraphPipeline 인스턴스 생성
- [ ] 각 모듈 연결 설정
- [ ] CUDA Graph 최적화 활성화

#### 2.2 실행 루프 구현
```cpp
// Main loop in needaimbot.cpp
UnifiedGraphPipeline pipeline;
pipeline.initialize(config);

while (!ctx.should_exit) {
    // 1. 캡처 트리거
    if (capture.isFrameReady()) {
        // 2. 파이프라인 실행 (모든 GPU 작업)
        pipeline.execute(capture.getGPUBuffer());
    }
}
```

#### 2.3 Pipeline 내부 흐름
1. **Capture Stage**: D3D11 텍스처 → CUDA 버퍼
2. **Preprocess Stage**: BGRA → RGB, Resize, Normalize (Fused Kernel)
3. **Detection Stage**: YOLO 추론
4. **Postprocess Stage**: NMS, 타겟 선택
5. **Tracking Stage**: Kalman 필터 (옵션)
6. **Control Stage**: PID 계산
7. **Execution Stage**: 마우스 이동 실행

### Phase 3: 모듈 간 인터페이스 정리
**목표**: 각 모듈이 독립적으로 동작

#### 3.1 Detector 인터페이스
- [ ] processFrame() 제거 (UnifiedGraphPipeline이 직접 호출)
- [ ] GPU 버퍼 직접 접근 인터페이스 추가

#### 3.2 Mouse 인터페이스
- [ ] Legacy 메서드 완전 제거
- [ ] executeMovement(), executePress(), executeRelease()만 유지

#### 3.3 AppContext 정리
- [ ] 불필요한 이벤트 큐 제거
- [ ] mouseDataReady, mouseDataCV 등 제거
- [ ] 단순 상태 플래그만 유지

### Phase 4: CUDA Graph 최적화
**목표**: 전체 파이프라인을 CUDA Graph로 실행

#### 4.1 Graph Capture
- [ ] 첫 프레임에서 전체 파이프라인 캡처
- [ ] 동적 파라미터 업데이트 지원

#### 4.2 Two-Stage Pipeline
- [ ] Detection Graph (항상 실행)
- [ ] Tracking/Control Graph (타겟 있을 때만 실행)

### Phase 5: 성능 최적화
**목표**: 최소 레이턴시, 최대 처리량

#### 5.1 Multi-Stream 활용
- [ ] Capture Stream (최고 우선순위)
- [ ] Inference Stream
- [ ] Postprocess Stream
- [ ] Control Stream

#### 5.2 Triple Buffering
- [ ] 캡처, 추론, 디스플레이 버퍼 분리
- [ ] 비동기 파이프라인 구현

## 예상 파일 변경

### 수정 필요 파일
1. `needaimbot.cpp` - UnifiedGraphPipeline 통합
2. `gpu_only_capture.cpp` - 캡처만 담당하도록 단순화
3. `unified_graph_pipeline.cu` - 마우스 실행 추가
4. `mouse.h/cpp` - Legacy 코드 제거

### 삭제 예정 파일
1. `needaimbot_event.cpp` - 이미 삭제됨
2. `mouse_old.h/cpp` - 이미 삭제됨

### 새로 추가할 파일
없음 (기존 구조 활용)

## 구현 우선순위
1. **[높음]** gpu_only_capture.cpp 단순화
2. **[높음]** needaimbot.cpp에서 UnifiedGraphPipeline 초기화
3. **[중간]** 모듈 간 인터페이스 정리
4. **[낮음]** CUDA Graph 최적화
5. **[낮음]** Multi-Stream 최적화

## 예상 효과
- **레이턴시 감소**: GPU→CPU→GPU 왕복 제거
- **코드 명확성**: 각 모듈 역할 분명
- **유지보수성**: 모듈 간 낮은 결합도
- **성능 향상**: CUDA Graph로 커널 실행 오버헤드 최소화

## 테스트 계획
1. 단위 테스트: 각 모듈 독립 테스트
2. 통합 테스트: 전체 파이프라인 동작 확인
3. 성능 테스트: 레이턴시 및 FPS 측정
4. 안정성 테스트: 장시간 실행 테스트

## 롤백 계획
각 Phase별로 git 커밋을 분리하여 문제 발생 시 즉시 롤백 가능하도록 함

## 일정
- Phase 1-2: 1일 (핵심 리팩토링)
- Phase 3: 0.5일 (인터페이스 정리)
- Phase 4-5: 1일 (최적화)
- 테스트: 0.5일

총 예상 기간: 3일