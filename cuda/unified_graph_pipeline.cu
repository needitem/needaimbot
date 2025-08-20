#include "unified_graph_pipeline.h"
#include <cuda_runtime.h>
#include <cudaGraph.h>
#include <cstdio>
#include <cmath>

// PD Controller structure for GPU
struct PDController {
    float kp;           // Proportional gain
    float kd;           // Derivative gain
    float prevError;    // Previous error for derivative calculation
    float deltaTime;    // Time step
    float maxOutput;    // Maximum output clamp
    float minOutput;    // Minimum output clamp
    float deadzone;     // Deadzone threshold
    float smoothing;    // Smoothing factor for output
    float prevOutput;   // Previous output for smoothing
    bool isInitialized; // Initialization flag
};

// GPU memory for PD controllers (one per potential target)
__device__ PDController* d_pdControllers;
__constant__ int MAX_TARGETS = 10;

// CUDA kernel for PD controller update
__global__ void updatePDController(
    PDController* controllers,
    float* targetPositionsX,
    float* targetPositionsY,
    float* currentPositionX,
    float* currentPositionY,
    float* outputX,
    float* outputY,
    int numTargets,
    float screenCenterX,
    float screenCenterY
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTargets) return;
    
    PDController* pd = &controllers[tid];
    
    // Calculate error (distance from screen center to target)
    float errorX = targetPositionsX[tid] - screenCenterX;
    float errorY = targetPositionsY[tid] - screenCenterY;
    
    // Apply deadzone
    float errorMagnitude = sqrtf(errorX * errorX + errorY * errorY);
    if (errorMagnitude < pd->deadzone) {
        outputX[tid] = 0.0f;
        outputY[tid] = 0.0f;
        pd->prevError = 0.0f;
        return;
    }
    
    // Calculate derivative (rate of change of error)
    float derivativeX = 0.0f;
    float derivativeY = 0.0f;
    
    if (pd->isInitialized) {
        // Use previous error for derivative calculation
        float prevErrorX = pd->prevError * (errorX / errorMagnitude);
        float prevErrorY = pd->prevError * (errorY / errorMagnitude);
        derivativeX = (errorX - prevErrorX) / pd->deltaTime;
        derivativeY = (errorY - prevErrorY) / pd->deltaTime;
    }
    
    // PD control law: output = Kp * error + Kd * derivative
    float controlX = pd->kp * errorX + pd->kd * derivativeX;
    float controlY = pd->kp * errorY + pd->kd * derivativeY;
    
    // Clamp output
    controlX = fminf(fmaxf(controlX, pd->minOutput), pd->maxOutput);
    controlY = fminf(fmaxf(controlY, pd->minOutput), pd->maxOutput);
    
    // Apply smoothing filter
    if (pd->isInitialized && pd->smoothing > 0.0f) {
        float prevOutX = pd->prevOutput * (controlX / (fabsf(controlX) + fabsf(controlY) + 0.001f));
        float prevOutY = pd->prevOutput * (controlY / (fabsf(controlX) + fabsf(controlY) + 0.001f));
        controlX = pd->smoothing * prevOutX + (1.0f - pd->smoothing) * controlX;
        controlY = pd->smoothing * prevOutY + (1.0f - pd->smoothing) * controlY;
    }
    
    // Store output
    outputX[tid] = controlX;
    outputY[tid] = controlY;
    
    // Update state for next iteration
    pd->prevError = errorMagnitude;
    pd->prevOutput = sqrtf(controlX * controlX + controlY * controlY);
    pd->isInitialized = true;
}

// CUDA kernel for adaptive gain adjustment based on target velocity
__global__ void adjustPDGains(
    PDController* controllers,
    float* targetVelocitiesX,
    float* targetVelocitiesY,
    int numTargets,
    float baseKp,
    float baseKd,
    float velocityScaling
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTargets) return;
    
    PDController* pd = &controllers[tid];
    
    // Calculate target velocity magnitude
    float velocityMagnitude = sqrtf(
        targetVelocitiesX[tid] * targetVelocitiesX[tid] + 
        targetVelocitiesY[tid] * targetVelocitiesY[tid]
    );
    
    // Adaptive gain adjustment based on target velocity
    // Higher velocity targets need higher derivative gain for prediction
    float velocityFactor = 1.0f + velocityScaling * velocityMagnitude;
    
    pd->kp = baseKp;
    pd->kd = baseKd * velocityFactor;
}

// Host-side PD controller management
class UnifiedGraphPipeline {
private:
    // PD Controller parameters
    PDController* h_pdControllers;
    PDController* d_pdControllersPtr;
    
    // Target and output buffers
    float* d_targetPosX;
    float* d_targetPosY;
    float* d_targetVelX;
    float* d_targetVelY;
    float* d_outputX;
    float* d_outputY;
    
    // CUDA Graph handles
    cudaGraph_t pdGraph;
    cudaGraphExec_t pdGraphExec;
    
    // Pipeline parameters
    int maxTargets;
    float screenCenterX;
    float screenCenterY;
    
public:
    UnifiedGraphPipeline(int maxTargets = 10) : maxTargets(maxTargets) {
        // Allocate host memory
        h_pdControllers = new PDController[maxTargets];
        
        // Initialize PD controllers with default values
        for (int i = 0; i < maxTargets; i++) {
            h_pdControllers[i].kp = 0.4f;          // Proportional gain
            h_pdControllers[i].kd = 0.15f;         // Derivative gain
            h_pdControllers[i].prevError = 0.0f;
            h_pdControllers[i].deltaTime = 0.016f;  // 60 FPS
            h_pdControllers[i].maxOutput = 10.0f;   // Max mouse movement
            h_pdControllers[i].minOutput = -10.0f;  // Min mouse movement
            h_pdControllers[i].deadzone = 5.0f;     // 5 pixel deadzone
            h_pdControllers[i].smoothing = 0.3f;    // Smoothing factor
            h_pdControllers[i].prevOutput = 0.0f;
            h_pdControllers[i].isInitialized = false;
        }
        
        // Allocate device memory
        cudaMalloc(&d_pdControllersPtr, maxTargets * sizeof(PDController));
        cudaMalloc(&d_targetPosX, maxTargets * sizeof(float));
        cudaMalloc(&d_targetPosY, maxTargets * sizeof(float));
        cudaMalloc(&d_targetVelX, maxTargets * sizeof(float));
        cudaMalloc(&d_targetVelY, maxTargets * sizeof(float));
        cudaMalloc(&d_outputX, maxTargets * sizeof(float));
        cudaMalloc(&d_outputY, maxTargets * sizeof(float));
        
        // Copy controllers to device
        cudaMemcpy(d_pdControllersPtr, h_pdControllers, 
                   maxTargets * sizeof(PDController), cudaMemcpyHostToDevice);
        
        // Set screen center (can be updated dynamically)
        screenCenterX = 960.0f;  // Default 1920/2
        screenCenterY = 540.0f;  // Default 1080/2
        
        // Create CUDA Graph for PD control pipeline
        createPDControlGraph();
    }
    
    ~UnifiedGraphPipeline() {
        // Clean up host memory
        delete[] h_pdControllers;
        
        // Clean up device memory
        cudaFree(d_pdControllersPtr);
        cudaFree(d_targetPosX);
        cudaFree(d_targetPosY);
        cudaFree(d_targetVelX);
        cudaFree(d_targetVelY);
        cudaFree(d_outputX);
        cudaFree(d_outputY);
        
        // Clean up CUDA Graph
        if (pdGraphExec) cudaGraphExecDestroy(pdGraphExec);
        if (pdGraph) cudaGraphDestroy(pdGraph);
    }
    
    void createPDControlGraph() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Start graph capture
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        
        // Define grid and block dimensions
        dim3 blockSize(256);
        dim3 gridSize((maxTargets + blockSize.x - 1) / blockSize.x);
        
        // Add kernels to graph
        // 1. Adaptive gain adjustment
        adjustPDGains<<<gridSize, blockSize, 0, stream>>>(
            d_pdControllersPtr,
            d_targetVelX,
            d_targetVelY,
            maxTargets,
            0.4f,   // base Kp
            0.15f,  // base Kd
            0.001f  // velocity scaling factor
        );
        
        // 2. PD controller update
        updatePDController<<<gridSize, blockSize, 0, stream>>>(
            d_pdControllersPtr,
            d_targetPosX,
            d_targetPosY,
            nullptr,  // current position (can be added if needed)
            nullptr,
            d_outputX,
            d_outputY,
            maxTargets,
            screenCenterX,
            screenCenterY
        );
        
        // End graph capture
        cudaStreamEndCapture(stream, &pdGraph);
        
        // Create executable graph
        cudaGraphInstantiate(&pdGraphExec, pdGraph, nullptr, nullptr, 0);
        
        cudaStreamDestroy(stream);
    }
    
    // Update PD controller parameters
    void updatePDParameters(float kp, float kd, float deadzone, float smoothing) {
        for (int i = 0; i < maxTargets; i++) {
            h_pdControllers[i].kp = kp;
            h_pdControllers[i].kd = kd;
            h_pdControllers[i].deadzone = deadzone;
            h_pdControllers[i].smoothing = smoothing;
        }
        
        // Update device memory
        cudaMemcpy(d_pdControllersPtr, h_pdControllers, 
                   maxTargets * sizeof(PDController), cudaMemcpyHostToDevice);
    }
    
    // Execute PD control graph
    void executePDControl(cudaStream_t stream) {
        cudaGraphLaunch(pdGraphExec, stream);
    }
    
    // Set target positions (call before execution)
    void setTargetPositions(float* hostPosX, float* hostPosY, int numTargets) {
        cudaMemcpy(d_targetPosX, hostPosX, 
                   numTargets * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_targetPosY, hostPosY, 
                   numTargets * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Set target velocities for adaptive control
    void setTargetVelocities(float* hostVelX, float* hostVelY, int numTargets) {
        cudaMemcpy(d_targetVelX, hostVelX, 
                   numTargets * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_targetVelY, hostVelY, 
                   numTargets * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Get control outputs (call after execution)
    void getControlOutputs(float* hostOutputX, float* hostOutputY, int numTargets) {
        cudaMemcpy(hostOutputX, d_outputX, 
                   numTargets * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostOutputY, d_outputY, 
                   numTargets * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Update screen center for different resolutions
    void updateScreenCenter(float centerX, float centerY) {
        screenCenterX = centerX;
        screenCenterY = centerY;
        
        // Recreate graph with new parameters
        if (pdGraphExec) cudaGraphExecDestroy(pdGraphExec);
        if (pdGraph) cudaGraphDestroy(pdGraph);
        createPDControlGraph();
    }
    
    // Reset PD controller states
    void resetControllers() {
        for (int i = 0; i < maxTargets; i++) {
            h_pdControllers[i].prevError = 0.0f;
            h_pdControllers[i].prevOutput = 0.0f;
            h_pdControllers[i].isInitialized = false;
        }
        
        cudaMemcpy(d_pdControllersPtr, h_pdControllers, 
                   maxTargets * sizeof(PDController), cudaMemcpyHostToDevice);
    }
    
    // Get device pointers for integration with other CUDA kernels
    float* getTargetPosXDevice() { return d_targetPosX; }
    float* getTargetPosYDevice() { return d_targetPosY; }
    float* getOutputXDevice() { return d_outputX; }
    float* getOutputYDevice() { return d_outputY; }
};