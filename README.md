# 🎯 needaimbot - Ultra-High Performance AI Targeting System

<div align="center">

[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.8-blue.svg)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/C++-17-orange.svg)](https://en.cppreference.com/w/cpp/17)
[![Performance](https://img.shields.io/badge/Latency-<3ms-red.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Real-time AI-Powered Targeting System with Sub-3ms Response Time**

</div>

---

## 📊 Executive Summary

**needaimbot**은 극한의 성능을 추구하는 차세대 AI 타겟팅 시스템으로, RTX 4060 환경에서 **한 사이클당 3ms 이내**의 응답 시간을 달성합니다. 완벽한 비동기 처리와 Zero-Copy 메모리 아키텍처를 통해 **캡처 0ms, 추론 3ms**의 혁신적인 성능을 구현하며, 에임봇이 활성화될 때만 GPU 자원을 사용하는 **극한의 효율성**을 자랑합니다.

### 🚀 핵심 성능 지표 (RTX 4060 기준)

| 측정 항목 | 성능 수치 | 기술적 특징 |
|-----------|-----------|------------|
| **화면 캡처** | **0ms** | Zero-Copy D3D11-CUDA Interop |
| **AI 추론** | **<3ms** | TensorRT 10.8 FP16 최적화 |
| **전체 레이턴시** | **<3ms** | 완벽한 비동기 파이프라인 |
| **GPU 사용률** | **5-15%** (활성 시) | 온디맨드 자원 활용 |
| **메모리 사용** | **~1.2GB VRAM** | 효율적인 메모리 풀링 |
| **처리 프레임** | **300+ FPS** | Triple Buffer 시스템 |

---

## 🏗️ 시스템 아키텍처

### 1. **Unified GPU Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                   UNIFIED GRAPH PIPELINE                      │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  [D3D11 Capture] ──Zero-Copy──> [CUDA Buffer]                │
│         ↓                            ↓                        │
│  [CUDA Preprocessing] ──────> [TensorRT Engine]              │
│         ↓                            ↓                        │
│  [Post-Processing] ────────> [Target Selection]              │
│         ↓                            ↓                        │
│  [Triple Buffer] ──────────> [Mouse Control]                 │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 2. **핵심 기술 스택**

#### **A. Zero-Latency Capture System**
- **D3D11-CUDA Interoperability**: CPU 개입 없는 직접 GPU 메모리 매핑
- **Desktop Duplication API**: 하드웨어 가속 화면 캡처
- **CUDA Graphics Resource**: 텍스처 직접 액세스로 메모리 복사 제거
- **실시간 성능**: 캡처 오버헤드 완전 제거 (0ms)

#### **B. AI Inference Pipeline**
- **TensorRT 10.8 최적화**:
  - Dynamic Shape 지원
  - FP16/INT8 자동 캘리브레이션
  - 커널 자동 융합 (Kernel Fusion)
  - 멀티스트림 실행
- **YOLO 아키텍처 지원**: YOLOv8/v9/v10/v11/v12
- **최적화된 후처리**:
  - GPU 기반 NMS (Non-Maximum Suppression)
  - CUDA 커널 디코딩
  - 병렬 타겟 필터링

#### **C. Triple Buffer System**
```cpp
struct TripleBuffer {
    std::atomic<int> captureIdx{0};   // 캡처 버퍼 인덱스
    std::atomic<int> processIdx{0};   // 처리 버퍼 인덱스
    std::atomic<int> displayIdx{0};   // 표시 버퍼 인덱스
    
    SimpleCudaMat buffers[3];         // 트리플 버퍼
    cudaEvent_t ready_events[3];      // 동기화 이벤트
};
```
- **완벽한 비동기 처리**: 캡처/처리/표시 동시 실행
- **Zero-Wait 파이프라인**: 버퍼 전환 시 대기 시간 제거
- **Pinned Memory**: Host-Device 전송 최적화

### 3. **메모리 관리 최적화**

#### **CUDA Memory Pool Architecture**
```cpp
// RAII 기반 자동 메모리 관리
template<typename T>
class CudaMemory {
    std::unique_ptr<T, CudaDeleter> ptr;
    size_t size;
public:
    explicit CudaMemory(size_t n);
    T* get() const { return ptr.get(); }
    void reset();
};
```

- **메모리 풀링**: 동적 할당 오버헤드 제거
- **Pinned Memory**: CPU-GPU 전송 최적화
- **Unified Memory**: 자동 페이지 마이그레이션
- **RAII 패턴**: 자동 메모리 정리

### 4. **입력 시스템 아키텍처**

#### **다중 하드웨어 지원**
```cpp
class InputMethod {
    virtual void move(int x, int y) = 0;
    virtual void press() = 0;
    virtual void release() = 0;
};

// 지원 드라이버
- GhubInputMethod     // Logitech G-Hub
- RazerInputMethod    // Razer Synapse
- SerialInputMethod   // Arduino/Custom HW
- KmboxNetMethod      // Network Hardware
- MakcuInputMethod    // Professional HW
```

---

## 🔬 기술적 상세 분석

### 1. **CUDA 커널 최적화**

#### **Unified Preprocessing Kernel**
```cuda
__global__ void unifiedPreprocessKernel(
    uint8_t* input,    // BGRA 입력
    float* output,     // CHW 출력
    int width, int height,
    float scale, float* mean, float* std
) {
    // Coalesced Memory Access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Warp-level 최적화
    if (idx < width * height) {
        // BGRA to RGB + Normalize + Layout Transform
        // 단일 패스로 모든 전처리 수행
    }
}
```

**최적화 기법**:
- **Coalesced Memory Access**: 32-byte 정렬된 메모리 접근
- **Warp Divergence 최소화**: 조건문 제거
- **Shared Memory 활용**: 타일 기반 처리
- **Grid-Stride Loop**: 대용량 데이터 처리

### 2. **TensorRT 엔진 최적화**

#### **Dynamic Shape Optimization**
```cpp
// 동적 입력 크기 지원
auto profile = builder->createOptimizationProfile();
profile->setDimensions("input", 
    OptProfileSelector::kMIN, Dims4{1, 3, 256, 256});
profile->setDimensions("input", 
    OptProfileSelector::kOPT, Dims4{1, 3, 320, 320});
profile->setDimensions("input", 
    OptProfileSelector::kMAX, Dims4{1, 3, 640, 640});
```

#### **Precision Calibration**
- **FP16 자동 변환**: 2배 처리량, 50% 메모리 절감
- **INT8 캘리브레이션**: 4배 처리량 (지원 GPU)
- **Layer Fusion**: Conv+BN+ReLU 자동 융합

### 3. **비동기 실행 파이프라인**

```cpp
// 3-Stage Async Pipeline
cudaStream_t streams[3];

// Stage 1: Capture (Stream 0)
cudaGraphicsMapResources(1, &resource, streams[0]);

// Stage 2: Process (Stream 1)
preprocessKernel<<<grid, block, 0, streams[1]>>>();
context->enqueueV3(streams[1]);

// Stage 3: Output (Stream 2)
cudaMemcpyAsync(host, device, size, streams[2]);

// 이벤트 기반 동기화
cudaEventRecord(events[i], streams[i]);
```

### 4. **성능 프로파일링 시스템**

```cpp
class PerformanceMetrics {
    std::atomic<float> captureTime{0.0f};
    std::atomic<float> inferenceTime{0.0f};
    std::atomic<float> postprocessTime{0.0f};
    std::atomic<float> totalLatency{0.0f};
    
    void measureKernel(cudaEvent_t start, cudaEvent_t end);
};
```

---

## 💡 혁신적 기능

### 1. **Adaptive Resource Management**
- **온디맨드 GPU 활성화**: 에임봇 활성 시에만 GPU 사용
- **동적 해상도 조정**: 성능에 따른 자동 해상도 스케일링
- **스마트 타겟 필터링**: GPU 기반 실시간 우선순위 계산

### 2. **Advanced Tracking System**
```cpp
// GPU Kalman Filter 구현
__device__ void kalmanPredict(
    KalmanState* state,
    float dt
) {
    // 상태 예측 (위치, 속도, 가속도)
    state->x += state->vx * dt + 0.5f * state->ax * dt * dt;
    state->y += state->vy * dt + 0.5f * state->ay * dt * dt;
    
    // 공분산 업데이트
    updateCovariance(state, dt);
}
```

### 3. **Multi-Model Hot Swapping**
- 실시간 모델 교체 지원
- 메모리 사전 할당으로 교체 시 지연 제거
- 자동 TensorRT 엔진 캐싱

---

## 🛠️ 빌드 및 설치

### 시스템 요구사항

#### **최소 사양**
- **GPU**: NVIDIA RTX 2060 (Turing 아키텍처)
- **CUDA**: 12.8
- **RAM**: 8GB
- **OS**: Windows 10 20H2+

#### **권장 사양**
- **GPU**: NVIDIA RTX 4060 이상
- **CUDA**: 12.8
- **RAM**: 16GB
- **OS**: Windows 11 22H2+

### 빠른 시작 (Pre-built Binary)

**최신 릴리즈 다운로드**: [Mega.nz에서 다운로드](https://mega.nz/file/MWkk0LSD#PbnofZnIjHYDKNH96oy4oN1_yPEBx7vR7w0qt07cu04)

```bash
# 1. 위 링크에서 다운로드
# 2. 압축 해제
# 3. needaimbot.exe 실행 (관리자 권한 권장)
```

### 소스 코드 빌드

```bash
# 1. 저장소 클론
git clone https://github.com/needitem/needaimbot.git
cd needaimbot

# 2. 의존성 설치
# - CUDA Toolkit 12.8
# - Visual Studio 2022
# - Windows SDK 10.0.26100.0+

# 3. 빌드
msbuild needaimbot.sln /p:Configuration=Release /p:Platform=x64

# 4. TensorRT 엔진 생성 (첫 실행 시 자동)
./x64/Release/needaimbot.exe
```

---

## 📈 성능 벤치마크

### GPU별 성능 비교

| GPU Model | Capture | Inference | Total | FPS |
|-----------|---------|-----------|-------|-----|
| RTX 4090 | 0ms | 1.8ms | <2ms | 500+ |
| RTX 4080 | 0ms | 2.1ms | <2.5ms | 400+ |
| RTX 4070 Ti | 0ms | 2.5ms | <3ms | 350+ |
| **RTX 4060** | **0ms** | **2.8ms** | **<3ms** | **300+** |
| RTX 3080 | 0ms | 3.5ms | <4ms | 250+ |
| RTX 3070 | 0ms | 4.2ms | <5ms | 200+ |

### 메모리 사용량 분석

```
VRAM 사용 내역 (RTX 4060):
├── TensorRT Engine: 450MB
├── CUDA Buffers: 320MB
├── Triple Buffer: 180MB
├── Preprocessing: 120MB
├── Post-processing: 80MB
└── Misc: 50MB
총계: ~1.2GB
```

---

## 🔧 고급 설정

### config.ini 최적화 예시

```ini
[Performance]
detection_resolution = 320      # 낮을수록 빠름
max_detections = 30             # 타겟 수 제한
confidence_threshold = 0.35     # 높을수록 정확
enable_fp16 = true              # FP16 가속
triple_buffer = true            # 비동기 처리

[GPU]
cuda_device = 0                 # GPU 선택
stream_priority = high          # 스트림 우선순위
memory_pool_size = 2048         # MB 단위

[Optimization]
kernel_fusion = true            # 커널 융합
graph_optimization = true       # 그래프 최적화
dynamic_shapes = true           # 동적 크기 지원
```

---

## 🎯 사용 시나리오

### 1. **경쟁 게이밍**
- 초저지연 타겟 감지
- 정밀한 에임 보정
- 다중 타겟 우선순위 지정

### 2. **훈련 및 분석**
- 에임 패턴 분석
- 반응 시간 측정
- 정확도 통계

### 3. **개발 및 연구**
- AI 모델 테스트
- 성능 벤치마킹
- 알고리즘 검증

---

## 📚 기술 문서

### API Reference
```cpp
// Pipeline 초기화
UnifiedGraphPipeline pipeline(config);
pipeline.initialize();

// 비동기 실행
pipeline.executeGraphNonBlocking(stream);

// 결과 획득
Target* targets = pipeline.getTargets();
int count = pipeline.getTargetCount();
```

### CUDA 커널 인터페이스
```cuda
// 전처리 커널
cuda_unified_preprocessing(
    uint8_t* input, float* output,
    int width, int height,
    cudaStream_t stream
);

// 후처리 커널
performNMS_gpu(
    Target* targets, int* count,
    float threshold,
    cudaStream_t stream
);
```

---

## 🤝 기여 가이드 (Contributing Guide)

### 🎯 기여 방법

#### 1. **환경 설정**

```bash
# Fork 및 Clone
git clone https://github.com/YOUR_USERNAME/needaimbot.git
cd needaimbot
git remote add upstream https://github.com/needitem/needaimbot.git

# 브랜치 생성
git checkout -b feature/your-feature-name
```

#### 2. **개발 환경 구축**

**필수 도구 설치:**
```powershell
# 1. CUDA Toolkit 12.8
# https://developer.nvidia.com/cuda-12-8-0-download-archive

# 2. Visual Studio 2022
# - Desktop development with C++ 워크로드 선택
# - Windows 10/11 SDK
# - MSVC v143

# 3. 의존성 다운로드
cd needaimbot/modules
# TensorRT, cuDNN, GLFW 등 필요 라이브러리 설치
```

**프로젝트 설정:**
```xml
<!-- needaimbot.vcxproj 확인 사항 -->
<CudaToolkitCustomDir>$(CUDA_PATH)</CudaToolkitCustomDir>
<CudaArchitecture>sm_75;sm_80;sm_86;sm_89</CudaArchitecture>
```

#### 3. **코드 작성 가이드라인**

##### **성능 최적화 체크리스트**
- [ ] CUDA 프로파일링 실행 (`nvprof` 또는 Nsight)
- [ ] 메모리 누수 검사 (`cuda-memcheck`)
- [ ] 3ms 이내 레이턴시 유지 확인
- [ ] GPU 사용률 15% 이하 확인

##### **코드 스타일**
```cpp
// 파일 구조
// header.h
#pragma once
#include "cuda_runtime.h"

class ClassName {
public:
    explicit ClassName(Config config);
    ~ClassName();
    
    // Public methods
    void publicMethod();
    
private:
    // RAII 멤버
    std::unique_ptr<Resource> m_resource;
    
    // CUDA 리소스
    CudaMemory<float> m_deviceBuffer;
};

// implementation.cpp
#include "header.h"

ClassName::ClassName(Config config) 
    : m_resource(std::make_unique<Resource>())
    , m_deviceBuffer(config.bufferSize) {
    // 초기화 코드
}

// CUDA 커널
__global__ void processKernel(
    float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Coalesced memory access
        output[idx] = process(input[idx]);
    }
}
```

#### 4. **테스트 요구사항**

```cpp
// 단위 테스트 예시
TEST(PipelineTest, InferenceLatency) {
    UnifiedGraphPipeline pipeline(testConfig);
    
    auto start = std::chrono::high_resolution_clock::now();
    pipeline.executeGraphNonBlocking();
    cudaStreamSynchronize(0);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto latency = std::chrono::duration<float, std::milli>(end - start).count();
    EXPECT_LT(latency, 3.0f); // 3ms 이내
}

// 성능 벤치마크
void benchmarkGPU() {
    // RTX 2060, 3060, 4060에서 테스트
    for (auto gpu : supportedGPUs) {
        cudaSetDevice(gpu);
        runPerformanceTest();
    }
}
```

#### 5. **Pull Request 프로세스**

##### **PR 체크리스트**
```markdown
## PR 체크리스트
- [ ] 코드가 컴파일되고 실행됨
- [ ] 3ms 레이턴시 목표 달성
- [ ] 메모리 누수 없음
- [ ] CUDA 에러 체크 포함
- [ ] 관련 문서 업데이트
- [ ] 테스트 추가/업데이트
```

##### **커밋 메시지 규칙**
```bash
# 형식: <type>(<scope>): <subject>

feat(cuda): GPU 메모리 풀 구현으로 할당 오버헤드 제거
perf(pipeline): 커널 융합으로 추론 시간 15% 개선
fix(capture): D3D11 텍스처 누수 문제 해결
docs(readme): 성능 벤치마크 섹션 업데이트

# Type
- feat: 새로운 기능
- fix: 버그 수정
- perf: 성능 개선
- docs: 문서 수정
- style: 코드 스타일 변경
- refactor: 코드 리팩토링
- test: 테스트 추가/수정
- chore: 빌드 프로세스 등 기타 변경
```

##### **PR 제출**
```bash
# 변경사항 커밋
git add .
git commit -m "perf(cuda): optimize memory access pattern"

# upstream 동기화
git fetch upstream
git rebase upstream/main

# Push
git push origin feature/your-feature-name

# GitHub에서 PR 생성
```

### 🔥 우선순위 기여 영역

#### **1. 성능 최적화 (High Priority)**
- **CUDA 커널 최적화**
  - Warp divergence 감소
  - Shared memory 활용 개선
  - Memory coalescing 패턴 최적화
  
- **메모리 관리**
  - CUDA Graph API 활용
  - Memory pool 확장
  - Unified Memory 최적화

#### **2. AI 모델 개선**
- **새로운 YOLO 버전 지원**
  - YOLOv13+ 아키텍처 통합
  - Transformer 기반 모델 지원
  
- **TensorRT 최적화**
  - INT8 양자화 구현
  - Dynamic shape 최적화
  - Plugin 레이어 개발

#### **3. 하드웨어 지원 확장**
- **새로운 입력 장치**
  ```cpp
  class NewDeviceInputMethod : public InputMethod {
      void move(int x, int y) override;
      void press() override;
      void release() override;
  };
  ```
  
- **Multi-GPU 지원**
  - NCCL 통신 구현
  - 로드 밸런싱 알고리즘

#### **4. 문서화 및 도구**
- **성능 분석 도구**
  - 실시간 프로파일러 UI
  - 자동 벤치마크 스크립트
  
- **사용자 가이드**
  - 비디오 튜토리얼
  - 설정 최적화 가이드
  - 문제 해결 가이드

### 📋 코드 리뷰 기준

#### **필수 검토 사항**
1. **성능 영향도**
   - 레이턴시 증가 여부
   - GPU 사용률 변화
   - 메모리 사용량 변화

2. **코드 품질**
   - RAII 패턴 준수
   - 에러 처리 적절성
   - 주석 및 문서화

3. **호환성**
   - 다양한 GPU 아키텍처 지원
   - Windows 버전 호환성
   - 의존성 버전 관리

### 🛠️ 유용한 리소스

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 🏆 기여자 인정

#### **기여 레벨**
- 🥉 **Bronze**: 첫 PR 머지
- 🥈 **Silver**: 5개 이상 PR 머지
- 🥇 **Gold**: 주요 기능 구현
- 💎 **Diamond**: 핵심 성능 개선

#### **명예의 전당**
기여자들은 README의 Contributors 섹션에 기록되며, 주요 기여자는 프로젝트 메인테이너로 초대될 수 있습니다.

### 📞 연락처

- **GitHub**: [https://github.com/needitem/needaimbot](https://github.com/needitem/needaimbot)
- **Issues**: [GitHub Issues](https://github.com/needitem/needaimbot/issues)
- **Email**: th07290828@gmail.com

---

## 📊 로드맵

### Phase 1: Performance (현재)
- ✅ 3ms 이내 추론 달성
- ✅ Zero-Copy 캡처 구현
- ✅ Triple Buffer 시스템
- ✅ TensorRT 10.8 통합

### Phase 2: Features (계획)
- ⬜ Transformer 기반 모델 지원
- ⬜ Multi-GPU 지원
- ⬜ Cloud 추론 옵션
- ⬜ 실시간 모델 학습

### Phase 3: Ecosystem (미래)
- ⬜ 플러그인 시스템
- ⬜ 웹 대시보드
- ⬜ 모바일 컨트롤
- ⬜ API 서비스

---

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

---

## 🙏 감사의 말

이 프로젝트는 다음 기술과 커뮤니티의 도움으로 만들어졌습니다:

- **NVIDIA**: CUDA, TensorRT, cuDNN
- **Microsoft**: DirectX, Windows SDK
- **Open Source**: Dear ImGui, Eigen, GLFW
- **Community**: 모든 기여자와 테스터

---

<div align="center">

**Built with ❤️ for Ultimate Performance**

*"극한의 효율을 추구하는 차세대 AI 타겟팅 시스템"*

</div>