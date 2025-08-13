#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Target structure definition
struct Target {
    float x, y, w, h;
    float confidence;
    int class_id;
};

// GPU에서 마우스 움직임을 직접 계산하는 커널
__global__ void calculateMouseMovementKernel(
    const Target* best_target,
    const bool has_target,
    const int screen_width,
    const int screen_height,
    const float sensitivity_x,
    const float sensitivity_y,
    const float smooth_factor,
    float* dx_out,
    float* dy_out,
    int* should_move_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // 타겟이 없으면 움직이지 않음
    if (!has_target || !best_target) {
        *dx_out = 0.0f;
        *dy_out = 0.0f;
        *should_move_flag = 0;
        return;
    }
    
    // 화면 중앙 계산
    float center_x = screen_width * 0.5f;
    float center_y = screen_height * 0.5f;
    
    // 타겟까지의 거리 계산
    float target_center_x = best_target->x + best_target->w * 0.5f;
    float target_center_y = best_target->y + best_target->h * 0.5f;
    
    float delta_x = target_center_x - center_x;
    float delta_y = target_center_y - center_y;
    
    // 거리가 너무 작으면 움직이지 않음 (데드존)
    const float deadzone = 2.0f;
    if (fabsf(delta_x) < deadzone && fabsf(delta_y) < deadzone) {
        *dx_out = 0.0f;
        *dy_out = 0.0f;
        *should_move_flag = 0;
        return;
    }
    
    // 스무딩 적용
    delta_x *= smooth_factor;
    delta_y *= smooth_factor;
    
    // 감도 적용
    delta_x *= sensitivity_x;
    delta_y *= sensitivity_y;
    
    // 최대 이동 거리 제한
    const float max_movement = 100.0f;
    delta_x = fminf(fmaxf(delta_x, -max_movement), max_movement);
    delta_y = fminf(fmaxf(delta_y, -max_movement), max_movement);
    
    // 결과 저장
    *dx_out = delta_x;
    *dy_out = delta_y;
    *should_move_flag = 1;
}

// PID 제어기를 GPU에서 실행
__global__ void pidMouseControlKernel(
    const Target* best_target,
    const bool has_target,
    const int screen_width,
    const int screen_height,
    const float kp_x, const float ki_x, const float kd_x,
    const float kp_y, const float ki_y, const float kd_y,
    float* integral_x, float* integral_y,  // 적분 상태 (persistent)
    float* prev_error_x, float* prev_error_y,  // 이전 에러 (persistent)
    float* dx_out,
    float* dy_out,
    int* should_move_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    if (!has_target || !best_target) {
        // 타겟이 없으면 적분 리셋
        *integral_x = 0.0f;
        *integral_y = 0.0f;
        *prev_error_x = 0.0f;
        *prev_error_y = 0.0f;
        *dx_out = 0.0f;
        *dy_out = 0.0f;
        *should_move_flag = 0;
        return;
    }
    
    // 화면 중앙과 타겟 중심 계산
    float center_x = screen_width * 0.5f;
    float center_y = screen_height * 0.5f;
    float target_center_x = best_target->x + best_target->w * 0.5f;
    float target_center_y = best_target->y + best_target->h * 0.5f;
    
    // 에러 계산
    float error_x = target_center_x - center_x;
    float error_y = target_center_y - center_y;
    
    // 데드존 체크
    const float deadzone = 1.5f;
    if (fabsf(error_x) < deadzone && fabsf(error_y) < deadzone) {
        *dx_out = 0.0f;
        *dy_out = 0.0f;
        *should_move_flag = 0;
        return;
    }
    
    // PID 계산
    // P (비례)
    float p_x = kp_x * error_x;
    float p_y = kp_y * error_y;
    
    // I (적분) - 적분 windup 방지
    const float integral_limit = 50.0f;
    *integral_x = fminf(fmaxf(*integral_x + error_x, -integral_limit), integral_limit);
    *integral_y = fminf(fmaxf(*integral_y + error_y, -integral_limit), integral_limit);
    float i_x = ki_x * (*integral_x);
    float i_y = ki_y * (*integral_y);
    
    // D (미분)
    float d_x = kd_x * (error_x - *prev_error_x);
    float d_y = kd_y * (error_y - *prev_error_y);
    
    // 이전 에러 업데이트
    *prev_error_x = error_x;
    *prev_error_y = error_y;
    
    // 최종 출력
    float output_x = p_x + i_x + d_x;
    float output_y = p_y + i_y + d_y;
    
    // 출력 제한
    const float max_output = 100.0f;
    output_x = fminf(fmaxf(output_x, -max_output), max_output);
    output_y = fminf(fmaxf(output_y, -max_output), max_output);
    
    *dx_out = output_x;
    *dy_out = output_y;
    *should_move_flag = 1;
}

// Host 인터페이스 클래스
class GpuMouseController {
private:
    // Pinned memory for zero-copy access
    float *h_dx, *h_dy;
    int *h_should_move;
    
    // Device memory
    float *d_dx, *d_dy;
    int *d_should_move;
    
    // PID 상태 (GPU 메모리에 유지)
    float *d_integral_x, *d_integral_y;
    float *d_prev_error_x, *d_prev_error_y;
    
    // Stream과 Event
    cudaStream_t mouseStream;
    cudaEvent_t mouseCalcComplete;
    
    int screen_width, screen_height;
    
public:
    GpuMouseController(int width, int height) 
        : screen_width(width), screen_height(height) {
        
        // Pinned memory 할당 (Zero-copy를 위해)
        cudaHostAlloc(&h_dx, sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc(&h_dy, sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc(&h_should_move, sizeof(int), cudaHostAllocMapped);
        
        // Device memory 할당
        cudaMalloc(&d_dx, sizeof(float));
        cudaMalloc(&d_dy, sizeof(float));
        cudaMalloc(&d_should_move, sizeof(int));
        
        // PID 상태 메모리
        cudaMalloc(&d_integral_x, sizeof(float));
        cudaMalloc(&d_integral_y, sizeof(float));
        cudaMalloc(&d_prev_error_x, sizeof(float));
        cudaMalloc(&d_prev_error_y, sizeof(float));
        
        // 초기화
        cudaMemset(d_integral_x, 0, sizeof(float));
        cudaMemset(d_integral_y, 0, sizeof(float));
        cudaMemset(d_prev_error_x, 0, sizeof(float));
        cudaMemset(d_prev_error_y, 0, sizeof(float));
        
        // Stream과 Event 생성
        cudaStreamCreateWithPriority(&mouseStream, cudaStreamNonBlocking, -1);
        cudaEventCreate(&mouseCalcComplete);
    }
    
    ~GpuMouseController() {
        cudaFreeHost(h_dx);
        cudaFreeHost(h_dy);
        cudaFreeHost(h_should_move);
        
        cudaFree(d_dx);
        cudaFree(d_dy);
        cudaFree(d_should_move);
        cudaFree(d_integral_x);
        cudaFree(d_integral_y);
        cudaFree(d_prev_error_x);
        cudaFree(d_prev_error_y);
        
        cudaStreamDestroy(mouseStream);
        cudaEventDestroy(mouseCalcComplete);
    }
    
    // 비동기 마우스 움직임 계산 시작
    void calculateMouseAsync(
        const Target* d_best_target,
        bool has_target,
        float kp_x, float ki_x, float kd_x,
        float kp_y, float ki_y, float kd_y
    ) {
        // GPU에서 PID 계산
        pidMouseControlKernel<<<1, 1, 0, mouseStream>>>(
            d_best_target,
            has_target,
            screen_width,
            screen_height,
            kp_x, ki_x, kd_x,
            kp_y, ki_y, kd_y,
            d_integral_x, d_integral_y,
            d_prev_error_x, d_prev_error_y,
            d_dx, d_dy,
            d_should_move
        );
        
        // Device에서 Pinned memory로 복사
        cudaMemcpyAsync(h_dx, d_dx, sizeof(float), cudaMemcpyDeviceToHost, mouseStream);
        cudaMemcpyAsync(h_dy, d_dy, sizeof(float), cudaMemcpyDeviceToHost, mouseStream);
        cudaMemcpyAsync(h_should_move, d_should_move, sizeof(int), cudaMemcpyDeviceToHost, mouseStream);
        
        // 완료 이벤트 기록
        cudaEventRecord(mouseCalcComplete, mouseStream);
    }
    
    // CPU에서 결과 대기 (블로킹)
    bool waitForMouseMovement(float& dx, float& dy) {
        // GPU 계산 완료 대기
        cudaEventSynchronize(mouseCalcComplete);
        
        if (*h_should_move) {
            dx = *h_dx;
            dy = *h_dy;
            return true;
        }
        return false;
    }
    
    // 논블로킹 체크
    bool checkMouseMovement(float& dx, float& dy) {
        cudaError_t result = cudaEventQuery(mouseCalcComplete);
        if (result == cudaSuccess) {
            if (*h_should_move) {
                dx = *h_dx;
                dy = *h_dy;
                return true;
            }
        }
        return false;
    }
    
    // PID 상태 리셋
    void resetPID() {
        cudaMemsetAsync(d_integral_x, 0, sizeof(float), mouseStream);
        cudaMemsetAsync(d_integral_y, 0, sizeof(float), mouseStream);
        cudaMemsetAsync(d_prev_error_x, 0, sizeof(float), mouseStream);
        cudaMemsetAsync(d_prev_error_y, 0, sizeof(float), mouseStream);
    }
};

// C++ 인터페이스를 위한 extern 함수들
extern "C" {
    GpuMouseController* createGpuMouseController(int width, int height) {
        return new GpuMouseController(width, height);
    }
    
    void destroyGpuMouseController(GpuMouseController* controller) {
        delete controller;
    }
    
    void calculateMouseAsync(
        GpuMouseController* controller,
        const Target* d_best_target,
        bool has_target,
        float kp_x, float ki_x, float kd_x,
        float kp_y, float ki_y, float kd_y
    ) {
        controller->calculateMouseAsync(d_best_target, has_target, 
                                       kp_x, ki_x, kd_x, 
                                       kp_y, ki_y, kd_y);
    }
    
    bool waitForMouseMovement(GpuMouseController* controller, float* dx, float* dy) {
        return controller->waitForMouseMovement(*dx, *dy);
    }
}