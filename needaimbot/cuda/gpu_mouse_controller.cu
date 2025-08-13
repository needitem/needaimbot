#include "gpu_mouse_controller.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

namespace needaimbot {
namespace cuda {

// GPU에서 최적 타겟 선택 커널
__global__ void selectBestTargetKernel(
    const Detection* detections,
    int numDetections,
    float screenCenterX,
    float screenCenterY,
    float confidenceThreshold,
    int* bestTargetIdx,
    float* bestDistance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections) return;
    
    const Detection& det = detections[idx];
    
    // 신뢰도 체크
    if (det.confidence < confidenceThreshold) return;
    
    // 타겟 중심 계산
    float targetCenterX = det.x + det.width * 0.5f;
    float targetCenterY = det.y + det.height * 0.5f;
    
    // 화면 중심으로부터의 거리
    float dx = targetCenterX - screenCenterX;
    float dy = targetCenterY - screenCenterY;
    float distance = sqrtf(dx * dx + dy * dy);
    
    // 원자적으로 최소 거리 업데이트
    float oldDistance = atomicExch(bestDistance, distance);
    if (distance < oldDistance) {
        atomicExch(bestTargetIdx, idx);
    } else {
        atomicExch(bestDistance, oldDistance);
    }
}

// GPU에서 마우스 이동량 계산 (PID 제어 포함)
__global__ void calculateMovementKernel(
    const Detection* detections,
    int bestTargetIdx,
    float screenCenterX,
    float screenCenterY,
    const GPUMouseConfig config,
    float* pidState,  // [integral_x, integral_y, prev_error_x, prev_error_y, prev_dx, prev_dy]
    MouseMovement* movement
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // 타겟이 없는 경우
    if (bestTargetIdx < 0) {
        // PID 상태 리셋
        pidState[0] = pidState[1] = 0.0f;  // integral
        pidState[2] = pidState[3] = 0.0f;  // prev_error
        pidState[4] = pidState[5] = 0.0f;  // prev_dx/dy
        
        movement->dx = 0.0f;
        movement->dy = 0.0f;
        movement->confidence = 0.0f;
        movement->shouldMove = false;
        movement->targetId = -1;
        return;
    }
    
    const Detection& target = detections[bestTargetIdx];
    
    // 타겟 중심 계산
    float targetCenterX = target.x + target.width * 0.5f;
    float targetCenterY = target.y + target.height * 0.5f;
    
    // 에러 계산 (타겟 - 현재 위치)
    float errorX = targetCenterX - screenCenterX;
    float errorY = targetCenterY - screenCenterY;
    
    // 데드존 체크
    const float deadzone = 2.0f;
    if (fabsf(errorX) < deadzone && fabsf(errorY) < deadzone) {
        movement->dx = 0.0f;
        movement->dy = 0.0f;
        movement->confidence = target.confidence;
        movement->shouldMove = false;
        movement->targetId = bestTargetIdx;
        return;
    }
    
    // PID 제어 계산
    // P (비례)
    float pX = config.kp_x * errorX;
    float pY = config.kp_y * errorY;
    
    // I (적분) - windup 방지
    const float integralLimit = 50.0f;
    pidState[0] = fminf(fmaxf(pidState[0] + errorX * 0.016f, -integralLimit), integralLimit);  // dt = 16ms
    pidState[1] = fminf(fmaxf(pidState[1] + errorY * 0.016f, -integralLimit), integralLimit);
    float iX = config.ki_x * pidState[0];
    float iY = config.ki_y * pidState[1];
    
    // D (미분)
    float dX = config.kd_x * (errorX - pidState[2]) / 0.016f;
    float dY = config.kd_y * (errorY - pidState[3]) / 0.016f;
    
    // 이전 에러 저장
    pidState[2] = errorX;
    pidState[3] = errorY;
    
    // PID 출력
    float outputX = pX + iX + dX;
    float outputY = pY + iY + dY;
    
    // 스무딩 적용 (이전 출력과 혼합)
    outputX = config.smoothing * pidState[4] + (1.0f - config.smoothing) * outputX;
    outputY = config.smoothing * pidState[5] + (1.0f - config.smoothing) * outputY;
    
    // 이전 출력 저장
    pidState[4] = outputX;
    pidState[5] = outputY;
    
    // 감도 적용
    outputX *= config.sensitivity;
    outputY *= config.sensitivity;
    
    // 최대 이동량 제한
    const float maxMovement = 100.0f;
    outputX = fminf(fmaxf(outputX, -maxMovement), maxMovement);
    outputY = fminf(fmaxf(outputY, -maxMovement), maxMovement);
    
    // 결과 저장
    movement->dx = outputX;
    movement->dy = outputY;
    movement->confidence = target.confidence;
    movement->shouldMove = true;
    movement->targetId = bestTargetIdx;
}

// GPUMouseController 구현
GPUMouseController::GPUMouseController() {}

GPUMouseController::~GPUMouseController() {
    Cleanup();
}

bool GPUMouseController::Initialize(const GPUMouseConfig& config) {
    if (initialized_) return true;
    
    config_ = config;
    
    // CUDA 리소스 할당
    cudaError_t err;
    
    // 마우스 이동 결과 메모리
    err = cudaMalloc(&d_movementResult_, sizeof(MouseMovement));
    if (err != cudaSuccess) {
        printf("Failed to allocate movement result: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&d_previousMovement_, sizeof(MouseMovement));
    if (err != cudaSuccess) {
        printf("Failed to allocate previous movement: %s\n", cudaGetErrorString(err));
        Cleanup();
        return false;
    }
    
    // PID 상태 (6개 float)
    err = cudaMalloc(&d_pidState_, 6 * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate PID state: %s\n", cudaGetErrorString(err));
        Cleanup();
        return false;
    }
    cudaMemset(d_pidState_, 0, 6 * sizeof(float));
    
    // 스트림 생성 (고우선순위)
    err = cudaStreamCreateWithPriority(&computeStream_, cudaStreamNonBlocking, -1);
    if (err != cudaSuccess) {
        printf("Failed to create compute stream: %s\n", cudaGetErrorString(err));
        Cleanup();
        return false;
    }
    
    // 이벤트 생성
    err = cudaEventCreate(&startEvent_);
    if (err != cudaSuccess) {
        printf("Failed to create start event: %s\n", cudaGetErrorString(err));
        Cleanup();
        return false;
    }
    
    err = cudaEventCreate(&endEvent_);
    if (err != cudaSuccess) {
        printf("Failed to create end event: %s\n", cudaGetErrorString(err));
        Cleanup();
        return false;
    }
    
    initialized_ = true;
    return true;
}

void GPUMouseController::Cleanup() {
    if (d_movementResult_) {
        cudaFree(d_movementResult_);
        d_movementResult_ = nullptr;
    }
    
    if (d_previousMovement_) {
        cudaFree(d_previousMovement_);
        d_previousMovement_ = nullptr;
    }
    
    if (d_pidState_) {
        cudaFree(d_pidState_);
        d_pidState_ = nullptr;
    }
    
    if (computeStream_) {
        cudaStreamDestroy(computeStream_);
        computeStream_ = nullptr;
    }
    
    if (startEvent_) {
        cudaEventDestroy(startEvent_);
        startEvent_ = nullptr;
    }
    
    if (endEvent_) {
        cudaEventDestroy(endEvent_);
        endEvent_ = nullptr;
    }
    
    initialized_ = false;
}

bool GPUMouseController::CalculateMovement(Detection* d_detections, int numDetections, MouseMovement& movement) {
    if (!initialized_ || !d_detections || numDetections <= 0) {
        movement.dx = 0;
        movement.dy = 0;
        movement.shouldMove = false;
        return false;
    }
    
    // 비동기 계산 시작
    if (!CalculateMovementAsync(d_detections, numDetections, computeStream_)) {
        return false;
    }
    
    // 결과 대기
    return GetMovementResult(movement);
}

bool GPUMouseController::CalculateMovementAsync(Detection* d_detections, int numDetections, cudaStream_t stream) {
    if (!initialized_ || !d_detections || numDetections <= 0) {
        return false;
    }
    
    // 타이밍 시작
    cudaEventRecord(startEvent_, stream);
    
    // 최적 타겟 선택을 위한 임시 메모리
    int* d_bestTargetIdx;
    float* d_bestDistance;
    cudaMalloc(&d_bestTargetIdx, sizeof(int));
    cudaMalloc(&d_bestDistance, sizeof(float));
    
    // 초기값 설정
    int initIdx = -1;
    float initDist = FLT_MAX;
    cudaMemcpyAsync(d_bestTargetIdx, &initIdx, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bestDistance, &initDist, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // 화면 중심
    float screenCenterX = config_.screenWidth * 0.5f;
    float screenCenterY = config_.screenHeight * 0.5f;
    
    // 최적 타겟 선택 커널 실행
    int threadsPerBlock = 256;
    int blocksPerGrid = (numDetections + threadsPerBlock - 1) / threadsPerBlock;
    
    selectBestTargetKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_detections,
        numDetections,
        screenCenterX,
        screenCenterY,
        config_.confidenceThreshold,
        d_bestTargetIdx,
        d_bestDistance
    );
    
    // 최적 타겟 인덱스 가져오기
    int bestIdx;
    cudaMemcpyAsync(&bestIdx, d_bestTargetIdx, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // 여기서 동기화 필요
    
    // 마우스 이동량 계산 커널 실행
    calculateMovementKernel<<<1, 1, 0, stream>>>(
        d_detections,
        bestIdx,
        screenCenterX,
        screenCenterY,
        config_,
        d_pidState_,
        d_movementResult_
    );
    
    // 임시 메모리 해제
    cudaFree(d_bestTargetIdx);
    cudaFree(d_bestDistance);
    
    // 타이밍 종료
    cudaEventRecord(endEvent_, stream);
    
    processedFrames_++;
    
    return true;
}

bool GPUMouseController::GetMovementResult(MouseMovement& movement) {
    if (!initialized_) return false;
    
    // GPU 계산 완료 대기
    cudaEventSynchronize(endEvent_);
    
    // 결과 복사
    cudaMemcpy(&movement, d_movementResult_, sizeof(MouseMovement), cudaMemcpyDeviceToHost);
    
    // 계산 시간 측정
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent_, endEvent_);
    lastCalculationTime_ = milliseconds;
    
    return true;
}

void GPUMouseController::UpdateConfig(const GPUMouseConfig& config) {
    config_ = config;
}

void GPUMouseController::SetSensitivity(float sensitivity) {
    config_.sensitivity = sensitivity;
}

void GPUMouseController::SetSmoothing(float smoothing) {
    config_.smoothing = smoothing;
}

// C 스타일 인터페이스 구현
extern "C" {
    GPUMouseController* createGpuMouseController(int width, int height) {
        GPUMouseController* controller = new GPUMouseController();
        GPUMouseConfig config;
        config.screenWidth = width;
        config.screenHeight = height;
        
        if (!controller->Initialize(config)) {
            delete controller;
            return nullptr;
        }
        
        return controller;
    }
    
    void destroyGpuMouseController(GPUMouseController* controller) {
        if (controller) {
            delete controller;
        }
    }
    
    bool calculateMouseMovement(
        GPUMouseController* controller,
        Detection* d_detections,
        int numDetections,
        float* dx, float* dy
    ) {
        if (!controller) return false;
        
        MouseMovement movement;
        if (controller->CalculateMovement(d_detections, numDetections, movement)) {
            *dx = movement.dx;
            *dy = movement.dy;
            return movement.shouldMove;
        }
        
        return false;
    }
}

} // namespace cuda
} // namespace needaimbot