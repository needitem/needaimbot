#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <atomic>

namespace needaimbot {
namespace cuda {

// YOLO 감지 결과 구조체 (Target과 호환되도록)
struct Detection {
    int classId;           // 클래스 ID
    int x, y;              // 바운딩 박스 좌상단
    int width, height;     // 바운딩 박스 크기
    int id;                // 트래킹 ID
    float confidence;      // 신뢰도
    float center_x, center_y;  // 중심점
    float velocity_x, velocity_y;  // 속도
};

// GPU 마우스 컨트롤러 설정
struct GPUMouseConfig {
    float sensitivity = 1.0f;
    float smoothing = 0.5f;
    int screenWidth = 1920;
    int screenHeight = 1080;
    int targetSize = 50;
    float confidenceThreshold = 0.7f;
    
    // PID 제어 파라미터
    float kp_x = 0.5f, ki_x = 0.1f, kd_x = 0.05f;
    float kp_y = 0.5f, ki_y = 0.1f, kd_y = 0.05f;
};

// GPU에서 계산된 마우스 이동량
struct MouseMovement {
    float dx;
    float dy;
    float confidence;
    bool shouldMove;
    int targetId;  // 선택된 타겟 ID
};

class GPUMouseController {
public:
    GPUMouseController();
    ~GPUMouseController();

    // 초기화 및 정리
    bool Initialize(const GPUMouseConfig& config);
    void Cleanup();

    // GPU에서 직접 마우스 이동량 계산 (동기)
    bool CalculateMovement(Detection* d_detections, int numDetections, MouseMovement& movement);

    // 비동기 계산 (스트림 사용)
    bool CalculateMovementAsync(Detection* d_detections, int numDetections, cudaStream_t stream);
    bool GetMovementResult(MouseMovement& movement);

    // 설정 업데이트
    void UpdateConfig(const GPUMouseConfig& config);
    void SetSensitivity(float sensitivity);
    void SetSmoothing(float smoothing);

    // 성능 통계
    float GetLastCalculationTime() const { return lastCalculationTime_; }
    int GetProcessedFrames() const { return processedFrames_.load(); }

private:
    // CUDA 리소스
    MouseMovement* d_movementResult_ = nullptr;
    MouseMovement* d_previousMovement_ = nullptr;
    float* d_pidState_ = nullptr;  // PID 제어 상태 (6개 값: integral_x, integral_y, prev_error_x, prev_error_y, prev_dx, prev_dy)
    
    cudaStream_t computeStream_ = nullptr;
    cudaEvent_t startEvent_ = nullptr;
    cudaEvent_t endEvent_ = nullptr;
    
    GPUMouseConfig config_;
    bool initialized_ = false;
    
    float lastCalculationTime_ = 0.0f;
    std::atomic<int> processedFrames_{0};
};

// C 스타일 인터페이스 (기존 호환성)
extern "C" {
    GPUMouseController* createGpuMouseController(int width, int height);
    void destroyGpuMouseController(GPUMouseController* controller);
    bool calculateMouseMovement(
        GPUMouseController* controller,
        Detection* d_detections,
        int numDetections,
        float* dx, float* dy
    );
}

} // namespace cuda
} // namespace needaimbot