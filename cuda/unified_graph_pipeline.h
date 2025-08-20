#pragma once

#include <cuda_runtime.h>

class UnifiedGraphPipeline {
private:
    struct PDController* h_pdControllers;
    struct PDController* d_pdControllersPtr;
    
    float* d_targetPosX;
    float* d_targetPosY;
    float* d_targetVelX;
    float* d_targetVelY;
    float* d_outputX;
    float* d_outputY;
    
    cudaGraph_t pdGraph;
    cudaGraphExec_t pdGraphExec;
    
    int maxTargets;
    float screenCenterX;
    float screenCenterY;
    
    void createPDControlGraph();
    
public:
    UnifiedGraphPipeline(int maxTargets = 10);
    ~UnifiedGraphPipeline();
    
    void updatePDParameters(float kp, float kd, float deadzone, float smoothing);
    void executePDControl(cudaStream_t stream);
    void setTargetPositions(float* hostPosX, float* hostPosY, int numTargets);
    void setTargetVelocities(float* hostVelX, float* hostVelY, int numTargets);
    void getControlOutputs(float* hostOutputX, float* hostOutputY, int numTargets);
    void updateScreenCenter(float centerX, float centerY);
    void resetControllers();
    
    float* getTargetPosXDevice();
    float* getTargetPosYDevice();
    float* getOutputXDevice();
    float* getOutputYDevice();
};