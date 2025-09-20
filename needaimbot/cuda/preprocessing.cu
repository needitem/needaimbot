#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simple_cuda_mat.h"

// 통합 전처리 커널: BGRA → RGB + Resize + Normalize + HWC→CHW
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
        src_bgra.step(),
        target_width,
        target_height,
        scale_factor
    );
    
    return cudaGetLastError();
}

// 헤더에서 사용할 함수 선언을 위한 별도 버전
extern "C" cudaError_t cuda_unified_preprocessing(
    const void* src_bgra_data,          // BGRA 입력 데이터 포인터
    float* dst_rgb_chw,                 // RGB CHW 출력 (float)
    int src_width, int src_height,      // 입력 크기
    int src_step,                       // 입력 스트라이드
    int target_width, int target_height, // 목표 크기
    cudaStream_t stream = 0
) {
    if (!src_bgra_data || !dst_rgb_chw) {
        return cudaErrorInvalidValue;
    }
    
    // 블록과 그리드 크기 설정
    dim3 block(16, 16);
    dim3 grid((target_width + block.x - 1) / block.x, 
              (target_height + block.y - 1) / block.y);
    
    const float scale_factor = 1.0f / 255.0f;
    
    integratedPreprocessKernel<<<grid, block, 0, stream>>>(
        (const uchar4*)src_bgra_data,
        dst_rgb_chw,
        src_width, src_height,
        target_width, target_height,
        src_step,
        scale_factor
    );
    
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