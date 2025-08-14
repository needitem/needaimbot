# UnifiedGraphPipeline 중앙집중식 리팩토링 계획

## 현재 문제점
1. **책임 분산**: 캡처는 gpu_only_capture.cpp가, 처리는 UnifiedGraphPipeline이 담당
2. **불완전한 파이프라인**: UnifiedGraphPipeline이 실제 YOLO 추론을 호출하지 않음
3. **복잡한 의존성**: 여러 스레드와 모듈이 서로 의존
4. **비효율적 구조**: GPU 리소스가 여러 곳에서 관리됨

## 목표 아키텍처

```
Main Thread
    ↓
UnifiedGraphPipeline (모든 GPU 작업 통합 관리)
    ├── captureFrame() - D3D11 캡처 직접 수행
    ├── detectTargets() - YOLO 추론 실행
    ├── trackTargets() - Kalman 필터 적용
    ├── calculateControl() - PID 제어 계산
    └── executeMovement() - 마우스 이동 실행
```

## Phase 1: UnifiedGraphPipeline에 캡처 기능 통합

### 1.1 GPUCaptureManager 통합
```cpp
class UnifiedGraphPipeline {
private:
    // 캡처 관련 멤버 추가
    ComPtr<IDXGIOutputDuplication> m_duplication;
    ComPtr<ID3D11Device> m_d3dDevice;
    ComPtr<ID3D11DeviceContext> m_d3dContext;
    ComPtr<ID3D11Texture2D> m_stagingTexture;
    
public:
    // 새로운 메서드
    bool initializeCapture(int width, int height);
    bool captureNextFrame();  // 내부에서 직접 캡처
};
```

### 1.2 실행 흐름 변경
- gpu_only_capture.cpp의 gpuOnlyCaptureThread를 단순 루프로 변경
- 모든 실제 작업은 UnifiedGraphPipeline::execute()에서 수행

## Phase 2: Detection 통합

### 2.1 Detector 직접 호출
```cpp
bool UnifiedGraphPipeline::executeDirect(cudaStream_t stream) {
    // 1. 캡처
    if (!captureNextFrame()) return false;
    
    // 2. 전처리
    preprocessFrame();
    
    // 3. YOLO 추론 - 실제 호출 추가
    if (m_detector) {
        m_detector->processFrameAsync(m_d_yoloInput, m_d_inferenceOutput, stream);
    }
    
    // 4. 후처리 (NMS, 타겟 선택)
    postprocessDetections();
    
    // 5. 트래킹
    if (m_tracker) trackTargets();
    
    // 6. PID 제어
    if (m_pidController) calculateControl();
    
    // 7. 마우스 실행
    executeMouseMovement();
}
```

### 2.2 Detector 인터페이스 개선
```cpp
class Detector {
public:
    // 비동기 추론 메서드 추가
    bool processFrameAsync(float* input, float* output, cudaStream_t stream);
    cudaEvent_t getInferenceCompleteEvent();
};
```

## Phase 3: 스레드 구조 단순화

### 3.1 메인 실행 루프
```cpp
// needaimbot.cpp의 메인 루프
void mainPipelineLoop() {
    auto& pipeline = PipelineManager::getInstance().getPipeline();
    
    while (!ctx.should_exit) {
        // UnifiedGraphPipeline이 모든 것을 처리
        pipeline->execute();
        
        // FPS 제한 (옵션)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
```

### 3.2 gpu_only_capture.cpp 제거
- 더 이상 필요 없음
- 모든 기능을 UnifiedGraphPipeline으로 이동

## Phase 4: 리소스 관리 중앙화

### 4.1 단일 리소스 매니저
```cpp
class UnifiedGraphPipeline {
private:
    // 모든 GPU 리소스를 여기서 관리
    struct Resources {
        // D3D11 리소스
        cudaGraphicsResource_t cudaResource;
        
        // CUDA 버퍼
        SimpleCudaMat captureBuffer;
        float* yoloInput;
        float* inferenceOutput;
        Target* detections;
        
        // 동기화 객체
        cudaStream_t mainStream;
        cudaEvent_t captureEvent;
        cudaEvent_t inferenceEvent;
    } m_resources;
};
```

### 4.2 생명주기 관리
- initialize(): 모든 리소스 할당
- shutdown(): 모든 리소스 해제
- 중간에 리소스 재할당 없음

## Phase 5: CUDA Graph 최적화

### 5.1 전체 파이프라인 Graph 캡처
```cpp
bool UnifiedGraphPipeline::captureFullPipeline() {
    cudaStreamBeginCapture(m_mainStream);
    
    // 전체 파이프라인을 한 번에 캡처
    captureNextFrame();
    preprocessFrame();
    runInference();
    postprocessDetections();
    trackTargets();
    calculateControl();
    executeMouseMovement();
    
    cudaStreamEndCapture(m_mainStream, &m_fullGraph);
    cudaGraphInstantiate(&m_fullGraphExec, m_fullGraph);
}
```

### 5.2 동적 업데이트
- 파라미터만 변경 시 Graph 재캡처 불필요
- cudaGraphExecKernelNodeSetParams 사용

## 구현 순서

1. **[즉시]** executeDirect()에 detector->processFrame() 호출 추가
2. **[1일차]** UnifiedGraphPipeline에 캡처 기능 통합
3. **[2일차]** gpu_only_capture.cpp 제거 및 메인 루프 단순화
4. **[3일차]** 리소스 관리 중앙화
5. **[4일차]** CUDA Graph 최적화

## 예상 파일 변경

### 수정할 파일
1. `unified_graph_pipeline.cu` - 캡처 및 추론 통합
2. `unified_graph_pipeline.h` - 인터페이스 확장
3. `needaimbot.cpp` - 메인 루프 단순화
4. `detector.h/cpp` - 비동기 인터페이스 추가

### 삭제할 파일
1. `gpu_only_capture.cpp` - UnifiedGraphPipeline으로 통합
2. `gpu_capture_manager.h/cpp` - UnifiedGraphPipeline으로 통합

## 즉시 수정 사항 (Detection 문제 해결)

```cpp
// unified_graph_pipeline.cu의 executeDirect() 수정
// Line 607 이후에 추가:

// 2.5. YOLO 추론 실행
if (m_detector) {
    cudaStream_t inferenceStream = m_coordinator->inferenceStream;
    
    // 전처리 완료 대기
    cudaStreamWaitEvent(inferenceStream, m_coordinator->preprocessComplete, 0);
    
    // SimpleCudaMat으로 래핑
    SimpleCudaMat inputMat(640, 640, 3, m_d_yoloInput);
    
    // Detector의 processFrame 호출
    m_detector->processFrame(inputMat);
    
    // 추론 완료 시그널
    m_coordinator->synchronizeInference(m_coordinator->postprocessStream);
}

// 3. 후처리 추가
if (m_detector) {
    // Detection 결과 가져오기
    auto detections = m_detector->getLatestDetectionsGPU();
    if (detections.first && detections.second > 0) {
        // NMS 및 타겟 선택
        // ... 기존 후처리 로직
    }
}
```

## 성능 목표
- **레이턴시**: < 10ms (캡처부터 마우스 이동까지)
- **처리량**: > 144 FPS
- **GPU 사용률**: < 50% (게임 성능 보장)
- **메모리**: < 500MB GPU 메모리

## 테스트 계획
1. **단위 테스트**: 각 파이프라인 단계별 테스트
2. **통합 테스트**: 전체 파이프라인 동작 확인
3. **성능 테스트**: 레이턴시 및 FPS 측정
4. **안정성 테스트**: 24시간 연속 실행

## 위험 요소 및 대응
1. **D3D11-CUDA Interop 문제**: 기존 GPUCaptureManager 코드 참조
2. **Graph 캡처 실패**: Fallback으로 Direct 실행 유지
3. **메모리 부족**: 동적 해상도 조정 구현

## 결론
UnifiedGraphPipeline을 진정한 중앙 허브로 만들어 모든 GPU 작업을 통합 관리하면:
- 코드 복잡도 감소
- 디버깅 용이
- 성능 최적화 가능
- 유지보수성 향상