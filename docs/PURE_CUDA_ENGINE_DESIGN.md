# 순수 CUDA 추론 엔진 설계 문서

## 목표
- TensorRT/cuDNN 의존성 제거
- `nvinfer_10.dll`, `cudnn64_9.dll` Import 없음
- 순수 CUDA Runtime (`nvcuda.dll`) + cuBLAS (`cublas64_*.dll`)만 사용
- YOLOv8n 640x640 기준 100+ FPS 목표

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    PureCudaEngine                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ WeightLoader│  │ LayerRegistry│  │ MemoryPool │     │
│  │ (.weights) │  │ (Conv,BN,..)│  │ (GPU Arena)│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              NetworkGraph                        │   │
│  │  Input → Conv → BN → SiLU → ... → Output        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Inference Engine                    │   │
│  │  - im2col + cuBLAS GEMM (Conv2D)                │   │
│  │  - Fused BN+Activation kernels                  │   │
│  │  - FP16 Tensor Core support                     │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 필요한 CUDA 커널

### 1. Convolution (im2col + GEMM 방식)
```cpp
// im2col: 이미지를 행렬로 변환
__global__ void im2col_kernel(
    const float* data_im,    // [C, H, W]
    float* data_col,         // [C*kH*kW, outH*outW]
    int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW
);

// GEMM: cuBLAS 사용 (cublasSgemm / cublasHgemm)
// Conv = weights * im2col(input)
```

### 2. BatchNorm + Activation (Fused)
```cpp
// BN + SiLU 퓨전 커널
__global__ void bn_silu_kernel(
    float* data,             // in-place
    const float* gamma,      // scale
    const float* beta,       // bias
    const float* mean,
    const float* var,
    float epsilon,
    int size
);
```

### 3. Pooling
```cpp
__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW
);
```

### 4. Upsample
```cpp
__global__ void upsample_nearest_kernel(
    const float* input,
    float* output,
    int C, int H, int W,
    int scale_factor
);
```

### 5. Concat
```cpp
__global__ void concat_kernel(
    const float* input1,
    const float* input2,
    float* output,
    int C1, int C2, int H, int W
);
```

## YOLOv8n 네트워크 구조 (간략화)

```
Input: [1, 3, 640, 640]
  │
  ├─ Conv 3x3/2 → [1, 16, 320, 320]
  ├─ Conv 3x3/2 → [1, 32, 160, 160]
  ├─ C2f block  → [1, 32, 160, 160]
  ├─ Conv 3x3/2 → [1, 64, 80, 80]
  ├─ C2f block  → [1, 64, 80, 80]
  ├─ Conv 3x3/2 → [1, 128, 40, 40]
  ├─ C2f block  → [1, 128, 40, 40]
  ├─ Conv 3x3/2 → [1, 256, 20, 20]
  ├─ C2f block  → [1, 256, 20, 20]
  ├─ SPPF       → [1, 256, 20, 20]
  │
  ├─ Upsample   → [1, 256, 40, 40]
  ├─ Concat     → [1, 384, 40, 40]
  ├─ C2f block  → [1, 128, 40, 40]
  │
  ├─ Upsample   → [1, 128, 80, 80]
  ├─ Concat     → [1, 192, 80, 80]
  ├─ C2f block  → [1, 64, 80, 80]
  │
  ├─ Detection heads (3 scales)
  │
Output: [1, 84, 8400]  (80 classes + 4 bbox)
```

## 구현 단계

### Phase 1: 기본 인프라 (1-2일)
- [ ] `PureCudaEngine` 클래스 구조
- [ ] 메모리 풀 (GPU Arena)
- [ ] 가중치 로더 (.weights 또는 .bin 포맷)

### Phase 2: 핵심 커널 (3-5일)
- [ ] im2col 커널
- [ ] cuBLAS GEMM 래퍼
- [ ] Conv2D 레이어 (im2col + GEMM)
- [ ] BatchNorm + SiLU 퓨전 커널
- [ ] MaxPool, Upsample, Concat 커널

### Phase 3: 네트워크 구성 (2-3일)
- [ ] YOLOv8n 네트워크 그래프 하드코딩
- [ ] 레이어 연결 및 버퍼 관리
- [ ] 추론 루프 구현

### Phase 4: 최적화 (1-2주)
- [ ] FP16 지원 (cublasHgemm)
- [ ] Tensor Core 활용 (WMMA)
- [ ] 커널 퓨전 추가
- [ ] 메모리 레이아웃 최적화

### Phase 5: 통합 (2-3일)
- [ ] 기존 파이프라인과 교체 가능한 인터페이스
- [ ] 빌드 옵션 (`-DUSE_PURE_CUDA=ON`)
- [ ] 테스트 및 벤치마크

## 파일 구조

```
needaimbot/cuda/
├── pure_cuda_engine/
│   ├── pure_cuda_engine.h       # 메인 엔진 클래스
│   ├── pure_cuda_engine.cu      # 엔진 구현
│   ├── layers/
│   │   ├── conv2d.cu            # Conv2D (im2col + GEMM)
│   │   ├── batchnorm.cu         # BatchNorm + Activation
│   │   ├── pooling.cu           # MaxPool, AvgPool
│   │   ├── upsample.cu          # Nearest/Bilinear upsample
│   │   └── concat.cu            # Tensor concatenation
│   ├── kernels/
│   │   ├── im2col.cu            # im2col 커널
│   │   ├── activations.cu       # SiLU, ReLU, etc.
│   │   └── gemm_wrapper.cu      # cuBLAS 래퍼
│   ├── utils/
│   │   ├── memory_pool.h        # GPU 메모리 풀
│   │   └── weight_loader.h      # 가중치 로더
│   └── yolov8n_network.cu       # YOLOv8n 네트워크 정의
```

## 의존성

### 제거되는 DLL
- `nvinfer_10.dll`
- `nvinfer_plugin_10.dll`
- `nvonnxparser_10.dll`
- `cudnn64_9.dll`

### 유지되는 DLL
- `nvcuda.dll` (CUDA Runtime) - 필수
- `cublas64_*.dll` (cuBLAS) - GEMM용

## 빌드 옵션

```cmake
option(USE_PURE_CUDA "Use pure CUDA engine instead of TensorRT" OFF)

if(USE_PURE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_PURE_CUDA)
    # TensorRT 링크 제거
    # cuBLAS만 링크
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas)
else()
    # 기존 TensorRT 사용
    target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer_10 nvonnxparser_10)
endif()
```

## 예상 성능

| 구현 단계 | YOLOv8n 640x640 | FPS |
|-----------|-----------------|-----|
| Phase 2 완료 (기본) | ~20-30ms | 30-50 |
| Phase 4 완료 (FP16) | ~8-12ms | 80-120 |
| 추가 최적화 | ~5-8ms | 120-200 |
| TensorRT (참고) | ~2-3ms | 300-500 |

## 참고 자료

- darknet: https://github.com/AlexeyAB/darknet
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
