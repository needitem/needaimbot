#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include "simple_cuda_mat.h"

// ============================================================================
// 동일 해상도 전용 커널 (리사이즈 없음) - 최적화 버전
// ============================================================================

// 동일 해상도 전처리 커널 (FP16): BGRA → RGB + Normalize + HWC→CHW (리사이즈 없음)
__global__ void directPreprocessKernelFP16(
    const uchar4* __restrict__ src,     // BGRA 입력
    __half* __restrict__ dst,           // RGB CHW 출력 (정규화된 FP16)
    int width, int height,              // 입력/출력 크기 (동일)
    int src_step,                       // 입력 스트라이드
    float scale_factor                  // 정규화 인수 (1/255.0f)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 직접 픽셀 읽기 (bilinear interpolation 없음)
    const uchar4* src_row = (const uchar4*)((const char*)src + y * src_step);
    uchar4 pixel = src_row[x];

    // CHW 레이아웃으로 저장
    int hw_size = width * height;
    int dst_idx = y * width + x;

    // BGRA → RGB 변환 + 정규화 + FP16 변환
    dst[dst_idx] = __float2half(pixel.z * scale_factor);               // R (from BGRA.z)
    dst[dst_idx + hw_size] = __float2half(pixel.y * scale_factor);     // G (from BGRA.y)
    dst[dst_idx + 2 * hw_size] = __float2half(pixel.x * scale_factor); // B (from BGRA.x)
}

// 동일 해상도 전처리 커널 (FP32): BGRA → RGB + Normalize + HWC→CHW (리사이즈 없음)
__global__ void directPreprocessKernel(
    const uchar4* __restrict__ src,     // BGRA 입력
    float* __restrict__ dst,            // RGB CHW 출력 (정규화된 float)
    int width, int height,              // 입력/출력 크기 (동일)
    int src_step,                       // 입력 스트라이드
    float scale_factor                  // 정규화 인수 (1/255.0f)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 직접 픽셀 읽기 (bilinear interpolation 없음)
    const uchar4* src_row = (const uchar4*)((const char*)src + y * src_step);
    uchar4 pixel = src_row[x];

    // CHW 레이아웃으로 저장
    int hw_size = width * height;
    int dst_idx = y * width + x;

    // BGRA → RGB 변환 + 정규화
    dst[dst_idx] = pixel.z * scale_factor;               // R (from BGRA.z)
    dst[dst_idx + hw_size] = pixel.y * scale_factor;     // G (from BGRA.y)
    dst[dst_idx + 2 * hw_size] = pixel.x * scale_factor; // B (from BGRA.x)
}

// ============================================================================
// 리사이즈 포함 커널 (기존)
// ============================================================================

// 통합 전처리 커널 (FP16 출력): BGRA → RGB + Resize + Normalize + HWC→CHW
__global__ void integratedPreprocessKernelFP16(
    const uchar4* __restrict__ src,     // BGRA 입력
    __half* __restrict__ dst,           // RGB CHW 출력 (정규화된 FP16)
    int src_width, int src_height,      // 입력 크기
    int dst_width, int dst_height,      // 출력 크기
    int src_step,                       // 입력 스트라이드
    float scale_factor                  // 정규화 인수 (1/255.0f)
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height) return;

    // Bilinear interpolation을 위한 소스 좌표 계산
    float src_x_f = (dst_x + 0.5f) * src_width / dst_width - 0.5f;
    float src_y_f = (dst_y + 0.5f) * src_height / dst_height - 0.5f;

    // 정수 부분과 소수 부분 분리
    int src_x = __float2int_rd(src_x_f);
    int src_y = __float2int_rd(src_y_f);
    float alpha = src_x_f - src_x;
    float beta = src_y_f - src_y;

    // 경계 처리
    src_x = max(0, min(src_x, src_width - 2));
    src_y = max(0, min(src_y, src_height - 2));

    // 4개 픽셀 샘플링 (bilinear interpolation)
    const uchar4* src_row0 = (const uchar4*)((const char*)src + src_y * src_step);
    const uchar4* src_row1 = (const uchar4*)((const char*)src + (src_y + 1) * src_step);

    uchar4 p00 = src_row0[src_x];
    uchar4 p01 = src_row0[src_x + 1];
    uchar4 p10 = src_row1[src_x];
    uchar4 p11 = src_row1[src_x + 1];

    // Bilinear interpolation 계산
    float b_interp = (1 - alpha) * (1 - beta) * p00.x + alpha * (1 - beta) * p01.x +
                     (1 - alpha) * beta * p10.x + alpha * beta * p11.x;
    float g_interp = (1 - alpha) * (1 - beta) * p00.y + alpha * (1 - beta) * p01.y +
                     (1 - alpha) * beta * p10.y + alpha * beta * p11.y;
    float r_interp = (1 - alpha) * (1 - beta) * p00.z + alpha * (1 - beta) * p01.z +
                     (1 - alpha) * beta * p10.z + alpha * beta * p11.z;

    // CHW 레이아웃으로 저장: [R채널][G채널][B채널]
    int hw_size = dst_width * dst_height;
    int dst_idx = dst_y * dst_width + dst_x;

    // BGRA → RGB 변환 + 정규화 (0-255 → 0.0-1.0) + FP16 변환
    dst[dst_idx] = __float2half(r_interp * scale_factor);               // R 채널
    dst[dst_idx + hw_size] = __float2half(g_interp * scale_factor);     // G 채널
    dst[dst_idx + 2 * hw_size] = __float2half(b_interp * scale_factor); // B 채널
}

// 통합 전처리 커널 (FP32 출력): BGRA → RGB + Resize + Normalize + HWC→CHW
__global__ void integratedPreprocessKernel(
    const uchar4* __restrict__ src,     // BGRA 입력
    float* __restrict__ dst,            // RGB CHW 출력 (정규화된 float)
    int src_width, int src_height,      // 입력 크기
    int dst_width, int dst_height,      // 출력 크기
    int src_step,                       // 입력 스트라이드
    float scale_factor                  // 정규화 인수 (1/255.0f)
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    // Bilinear interpolation을 위한 소스 좌표 계산
    float src_x_f = (dst_x + 0.5f) * src_width / dst_width - 0.5f;
    float src_y_f = (dst_y + 0.5f) * src_height / dst_height - 0.5f;
    
    // 정수 부분과 소수 부분 분리
    int src_x = __float2int_rd(src_x_f);
    int src_y = __float2int_rd(src_y_f);
    float alpha = src_x_f - src_x;
    float beta = src_y_f - src_y;
    
    // 경계 처리
    src_x = max(0, min(src_x, src_width - 2));
    src_y = max(0, min(src_y, src_height - 2));
    
    // 4개 픽셀 샘플링 (bilinear interpolation)
    const uchar4* src_row0 = (const uchar4*)((const char*)src + src_y * src_step);
    const uchar4* src_row1 = (const uchar4*)((const char*)src + (src_y + 1) * src_step);
    
    uchar4 p00 = src_row0[src_x];
    uchar4 p01 = src_row0[src_x + 1];
    uchar4 p10 = src_row1[src_x];
    uchar4 p11 = src_row1[src_x + 1];
    
    // Bilinear interpolation 계산
    float b_interp = (1 - alpha) * (1 - beta) * p00.x + alpha * (1 - beta) * p01.x + 
                     (1 - alpha) * beta * p10.x + alpha * beta * p11.x;
    float g_interp = (1 - alpha) * (1 - beta) * p00.y + alpha * (1 - beta) * p01.y + 
                     (1 - alpha) * beta * p10.y + alpha * beta * p11.y;
    float r_interp = (1 - alpha) * (1 - beta) * p00.z + alpha * (1 - beta) * p01.z + 
                     (1 - alpha) * beta * p10.z + alpha * beta * p11.z;
    
    // CHW 레이아웃으로 저장: [R채널][G채널][B채널]
    int hw_size = dst_width * dst_height;
    int dst_idx = dst_y * dst_width + dst_x;
    
    // BGRA → RGB 변환 + 정규화 (0-255 → 0.0-1.0)
    dst[dst_idx] = r_interp * scale_factor;               // R 채널
    dst[dst_idx + hw_size] = g_interp * scale_factor;     // G 채널
    dst[dst_idx + 2 * hw_size] = b_interp * scale_factor; // B 채널
}

// 통합 전처리 함수 - 모든 작업을 하나의 커널로 수행
extern "C" cudaError_t unifiedPreprocessing(
    const SimpleCudaMat& src_bgra,      // BGRA 입력 (uchar4)
    float* dst_rgb_chw,                 // RGB CHW 출력 (float)
    int target_width,                   // 목표 너비
    int target_height,                  // 목표 높이
    cudaStream_t stream = 0
) {
    if (src_bgra.empty() || !dst_rgb_chw) {
        return cudaErrorInvalidValue;
    }
    
    // 블록과 그리드 크기 설정
    dim3 block(16, 16);  // 256 스레드 per block
    dim3 grid((target_width + block.x - 1) / block.x, 
              (target_height + block.y - 1) / block.y);
    
    // 정규화 인수 (0-255 → 0.0-1.0)
    const float scale_factor = 1.0f / 255.0f;
    
    // 통합 전처리 커널 실행
    integratedPreprocessKernel<<<grid, block, 0, stream>>>(
        (const uchar4*)src_bgra.data(),
        dst_rgb_chw,
        src_bgra.cols(),
        src_bgra.rows(),
        static_cast<int>(src_bgra.step()),
        target_width,
        target_height,
        scale_factor
    );
    
    return cudaGetLastError();
}

// 헤더에서 사용할 함수 선언을 위한 별도 버전
extern "C" cudaError_t cuda_unified_preprocessing(
    const void* src_bgra_data,          // BGRA 입력 데이터 포인터
    void* dst_rgb_chw,                  // RGB CHW 출력 (void* - FP32 or FP16)
    int src_width, int src_height,      // 입력 크기
    int src_step,                       // 입력 스트라이드
    int target_width, int target_height, // 목표 크기
    bool use_fp16,                      // true = FP16, false = FP32
    cudaStream_t stream = 0
) {
    if (!src_bgra_data || !dst_rgb_chw) {
        return cudaErrorInvalidValue;
    }

    const float scale_factor = 1.0f / 255.0f;

    // 동일 해상도인 경우 최적화된 커널 사용 (리사이즈 스킵)
    if (src_width == target_width && src_height == target_height) {
        dim3 block(32, 8);
        dim3 grid((target_width + block.x - 1) / block.x,
                  (target_height + block.y - 1) / block.y);

        if (use_fp16) {
            directPreprocessKernelFP16<<<grid, block, 0, stream>>>(
                (const uchar4*)src_bgra_data,
                (__half*)dst_rgb_chw,
                target_width, target_height,
                src_step,
                scale_factor
            );
        } else {
            directPreprocessKernel<<<grid, block, 0, stream>>>(
                (const uchar4*)src_bgra_data,
                (float*)dst_rgb_chw,
                target_width, target_height,
                src_step,
                scale_factor
            );
        }
    } else {
        // 다른 해상도인 경우 리사이즈 포함 커널 사용
        dim3 block(32, 8);
        dim3 grid((target_width + block.x - 1) / block.x,
                  (target_height + block.y - 1) / block.y);

        if (use_fp16) {
            integratedPreprocessKernelFP16<<<grid, block, 0, stream>>>(
                (const uchar4*)src_bgra_data,
                (__half*)dst_rgb_chw,
                src_width, src_height,
                target_width, target_height,
                src_step,
                scale_factor
            );
        } else {
            integratedPreprocessKernel<<<grid, block, 0, stream>>>(
                (const uchar4*)src_bgra_data,
                (float*)dst_rgb_chw,
                src_width, src_height,
                target_width, target_height,
                src_step,
                scale_factor
            );
        }
    }

    return cudaGetLastError();
}

// ============================================================================
// RGB 입력 전용 커널 (UDP 캡처용)
// ============================================================================

// RGB 입력 전처리 커널 (FP16): RGB → Normalize + HWC→CHW
__global__ void rgbPreprocessKernelFP16(
    const uint8_t* __restrict__ src,    // RGB 입력 (uint8_t * 3)
    __half* __restrict__ dst,           // RGB CHW 출력 (정규화된 FP16)
    int src_width, int src_height,      // 입력 크기
    int dst_width, int dst_height,      // 출력 크기
    int src_step,                       // 입력 스트라이드 (바이트)
    float scale_factor                  // 정규화 인수 (1/255.0f)
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height) return;

    // Bilinear interpolation을 위한 소스 좌표 계산
    float src_x_f = (dst_x + 0.5f) * src_width / dst_width - 0.5f;
    float src_y_f = (dst_y + 0.5f) * src_height / dst_height - 0.5f;

    int src_x = __float2int_rd(src_x_f);
    int src_y = __float2int_rd(src_y_f);
    float alpha = src_x_f - src_x;
    float beta = src_y_f - src_y;

    src_x = max(0, min(src_x, src_width - 2));
    src_y = max(0, min(src_y, src_height - 2));

    // 4개 픽셀 샘플링 (RGB * 3 bytes per pixel)
    const uint8_t* row0 = src + src_y * src_step;
    const uint8_t* row1 = src + (src_y + 1) * src_step;

    int idx00 = src_x * 3;
    int idx01 = (src_x + 1) * 3;

    // Bilinear interpolation for each channel
    float r_interp = (1 - alpha) * (1 - beta) * row0[idx00 + 0] + alpha * (1 - beta) * row0[idx01 + 0] +
                     (1 - alpha) * beta * row1[idx00 + 0] + alpha * beta * row1[idx01 + 0];
    float g_interp = (1 - alpha) * (1 - beta) * row0[idx00 + 1] + alpha * (1 - beta) * row0[idx01 + 1] +
                     (1 - alpha) * beta * row1[idx00 + 1] + alpha * beta * row1[idx01 + 1];
    float b_interp = (1 - alpha) * (1 - beta) * row0[idx00 + 2] + alpha * (1 - beta) * row0[idx01 + 2] +
                     (1 - alpha) * beta * row1[idx00 + 2] + alpha * beta * row1[idx01 + 2];

    // CHW 레이아웃으로 저장
    int hw_size = dst_width * dst_height;
    int dst_idx = dst_y * dst_width + dst_x;

    dst[dst_idx] = __float2half(r_interp * scale_factor);
    dst[dst_idx + hw_size] = __float2half(g_interp * scale_factor);
    dst[dst_idx + 2 * hw_size] = __float2half(b_interp * scale_factor);
}

// RGB 입력 전처리 커널 (FP32): RGB → Normalize + HWC→CHW
__global__ void rgbPreprocessKernel(
    const uint8_t* __restrict__ src,    // RGB 입력 (uint8_t * 3)
    float* __restrict__ dst,            // RGB CHW 출력 (정규화된 float)
    int src_width, int src_height,      // 입력 크기
    int dst_width, int dst_height,      // 출력 크기
    int src_step,                       // 입력 스트라이드 (바이트)
    float scale_factor                  // 정규화 인수 (1/255.0f)
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height) return;

    // Bilinear interpolation을 위한 소스 좌표 계산
    float src_x_f = (dst_x + 0.5f) * src_width / dst_width - 0.5f;
    float src_y_f = (dst_y + 0.5f) * src_height / dst_height - 0.5f;

    int src_x = __float2int_rd(src_x_f);
    int src_y = __float2int_rd(src_y_f);
    float alpha = src_x_f - src_x;
    float beta = src_y_f - src_y;

    src_x = max(0, min(src_x, src_width - 2));
    src_y = max(0, min(src_y, src_height - 2));

    // 4개 픽셀 샘플링 (RGB * 3 bytes per pixel)
    const uint8_t* row0 = src + src_y * src_step;
    const uint8_t* row1 = src + (src_y + 1) * src_step;

    int idx00 = src_x * 3;
    int idx01 = (src_x + 1) * 3;

    // Bilinear interpolation for each channel
    float r_interp = (1 - alpha) * (1 - beta) * row0[idx00 + 0] + alpha * (1 - beta) * row0[idx01 + 0] +
                     (1 - alpha) * beta * row1[idx00 + 0] + alpha * beta * row1[idx01 + 0];
    float g_interp = (1 - alpha) * (1 - beta) * row0[idx00 + 1] + alpha * (1 - beta) * row0[idx01 + 1] +
                     (1 - alpha) * beta * row1[idx00 + 1] + alpha * beta * row1[idx01 + 1];
    float b_interp = (1 - alpha) * (1 - beta) * row0[idx00 + 2] + alpha * (1 - beta) * row0[idx01 + 2] +
                     (1 - alpha) * beta * row1[idx00 + 2] + alpha * beta * row1[idx01 + 2];

    // CHW 레이아웃으로 저장
    int hw_size = dst_width * dst_height;
    int dst_idx = dst_y * dst_width + dst_x;

    dst[dst_idx] = r_interp * scale_factor;
    dst[dst_idx + hw_size] = g_interp * scale_factor;
    dst[dst_idx + 2 * hw_size] = b_interp * scale_factor;
}

// RGB 입력 전처리 함수 (UDP 캡처용)
extern "C" cudaError_t cuda_rgb_preprocessing(
    const void* src_rgb_data,           // RGB 입력 데이터 포인터
    void* dst_rgb_chw,                  // RGB CHW 출력 (void* - FP32 or FP16)
    int src_width, int src_height,      // 입력 크기
    int src_step,                       // 입력 스트라이드
    int target_width, int target_height, // 목표 크기
    bool use_fp16,                      // true = FP16, false = FP32
    cudaStream_t stream
) {
    if (!src_rgb_data || !dst_rgb_chw) {
        return cudaErrorInvalidValue;
    }

    const float scale_factor = 1.0f / 255.0f;

    dim3 block(32, 8);
    dim3 grid((target_width + block.x - 1) / block.x,
              (target_height + block.y - 1) / block.y);

    if (use_fp16) {
        rgbPreprocessKernelFP16<<<grid, block, 0, stream>>>(
            (const uint8_t*)src_rgb_data,
            (__half*)dst_rgb_chw,
            src_width, src_height,
            target_width, target_height,
            src_step,
            scale_factor
        );
    } else {
        rgbPreprocessKernel<<<grid, block, 0, stream>>>(
            (const uint8_t*)src_rgb_data,
            (float*)dst_rgb_chw,
            src_width, src_height,
            target_width, target_height,
            src_step,
            scale_factor
        );
    }

    return cudaGetLastError();
}

// BGR to RGBA 변환 커널
__global__ void bgr2rgba_kernel(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                int width, int height,
                                int src_pitch, int dst_pitch,
                                uint8_t alpha = 255) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint8_t* src_pixel = src + y * src_pitch + x * 3;  // BGR
    uint8_t* dst_pixel = dst + y * dst_pitch + x * 4;        // RGBA
    
    // BGR → RGBA 변환 (색상 순서 변경)
    dst_pixel[0] = src_pixel[2];  // R (from BGR[2])
    dst_pixel[1] = src_pixel[1];  // G (from BGR[1])
    dst_pixel[2] = src_pixel[0];  // B (from BGR[0])
    dst_pixel[3] = alpha;         // A
}

// BGRA to RGBA 변환 커널
__global__ void bgra2rgba_kernel(const uint8_t* __restrict__ src,
                                 uint8_t* __restrict__ dst,
                                 int width, int height,
                                 int src_pitch, int dst_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint8_t* src_pixel = src + y * src_pitch + x * 4;  // BGRA
    uint8_t* dst_pixel = dst + y * dst_pitch + x * 4;        // RGBA
    
    // BGRA → RGBA 변환 (B와 R 교환) - 알파는 항상 255로 설정
    dst_pixel[0] = src_pixel[2];  // R (from BGRA[2])
    dst_pixel[1] = src_pixel[1];  // G (from BGRA[1])
    dst_pixel[2] = src_pixel[0];  // B (from BGRA[0])
    dst_pixel[3] = 255;           // Opaque alpha for preview usage
}

// BGR to RGBA 변환 함수
extern "C" cudaError_t cuda_bgr2rgba(const uint8_t* src, uint8_t* dst,
                                     int width, int height,
                                     int src_pitch, int dst_pitch,
                                     uint8_t alpha = 255, cudaStream_t stream = 0) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    bgr2rgba_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, src_pitch, dst_pitch, alpha);
    
    return cudaGetLastError();
}

// BGRA to RGBA 변환 함수
extern "C" cudaError_t cuda_bgra2rgba(const uint8_t* src, uint8_t* dst,
                                      int width, int height,
                                      int src_pitch, int dst_pitch,
                                      cudaStream_t stream = 0) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    bgra2rgba_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, src_pitch, dst_pitch);
    
    return cudaGetLastError();
}