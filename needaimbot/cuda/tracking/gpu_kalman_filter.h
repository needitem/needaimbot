#pragma once

#include <cuda_runtime.h>
#include "../detection/postProcess.h"

// Forward declaration
class GPUKalmanTracker;

// Export functions for GPU Kalman filter
extern "C" {
    // Create and destroy tracker
    GPUKalmanTracker* createGPUKalmanTracker(int maxStates = 100, int maxMeasurements = 100);
    void destroyGPUKalmanTracker(GPUKalmanTracker* tracker);
    
    // Initialize CUDA graph for optimized execution
    void initializeKalmanGraph(GPUKalmanTracker* tracker, cudaStream_t stream);
    
    // Process tracking with Kalman filter
    void processKalmanFilter(GPUKalmanTracker* tracker,
                            const Target* d_measurements, int numMeasurements,
                            Target* d_output, int* d_outputCount,
                            cudaStream_t stream, bool useGraph, float lookaheadFrames = 1.0f);
    
    // Update filter settings
    void updateKalmanFilterSettings(float dt, float processNoise, 
                                   float measurementNoise, cudaStream_t stream);
}

// Configuration structure for Kalman filter
struct KalmanConfig {
    bool enabled = false;           // Enable/disable Kalman filter
    bool useGraph = true;           // Use CUDA graph for optimization
    float dt = 0.033f;              // Time delta (30 FPS default)
    float processNoise = 1.0f;      // Process noise scale
    float measurementNoise = 1.0f;  // Measurement noise scale
    int minHits = 3;                // Minimum hits before track is confirmed
    int maxAge = 5;                 // Maximum frames without detection
};