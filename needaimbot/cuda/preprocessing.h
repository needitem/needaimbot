#pragma once
#include <cuda_runtime.h>
#include "simple_cuda_mat.h"

// 통합 전처리 함수 - SimpleCudaMat 버전
extern "C" cudaError_t unifiedPreprocessing(
    const SimpleCudaMat& src_bgra,      // BGRA 입력 (uchar4)
    float* dst_rgb_chw,                 // RGB CHW 출력 (float)
    int target_width,                   // 목표 너비
    int target_height,                  // 목표 높이
    cudaStream_t stream = 0
);

// 통합 전처리 함수 - 포인터 버전 (더 범용적)
extern "C" cudaError_t cuda_unified_preprocessing(
    const void* src_bgra_data,          // BGRA 입력 데이터 포인터
    float* dst_rgb_chw,                 // RGB CHW 출력 (float)
    int src_width, int src_height,      // 입력 크기
    int src_step,                       // 입력 스트라이드 (바이트)
    int target_width, int target_height, // 목표 크기
    cudaStream_t stream = 0
);

// BGR to RGBA 변환 함수
extern "C" cudaError_t cuda_bgr2rgba(const uint8_t* src, uint8_t* dst,
                                     int width, int height,
                                     int src_pitch, int dst_pitch,
                                     uint8_t alpha = 255, cudaStream_t stream = 0);

// BGRA to RGBA 변환 함수
extern "C" cudaError_t cuda_bgra2rgba(const uint8_t* src, uint8_t* dst,
                                      int width, int height,
                                      int src_pitch, int dst_pitch,
                                      cudaStream_t stream = 0);

/*
 * 통합 전처리 기능:
 * 1. BGRA → RGB 변환 (알파 채널 제거 + R/B 교환)
 * 2. Bilinear interpolation 크기 조정
 * 3. 정규화 (0-255 → 0.0-1.0)
 * 4. 채널 순서 변경 (HWC → CHW)
 * 
 * 입력: BGRA uchar4 형식 (NVFBC 출력)
 * 출력: RGB float CHW 형식 (YOLO 모델 입력)
 */