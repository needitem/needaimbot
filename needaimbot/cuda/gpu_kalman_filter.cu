#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include "../postprocess/postProcess.h"

namespace cg = cooperative_groups;

// Kalman filter state for each tracked object
struct KalmanState {
    // State vector: [x, y, vx, vy, w, h]
    float state[6];
    
    // Covariance matrix (6x6)
    float P[36];
    
    // Track ID and metadata
    int trackId;
    int age;
    int hits;
    int timeSinceUpdate;
    bool isActive;
};

// Kalman filter constants in constant memory
__constant__ float c_processNoise[36];      // Q matrix
__constant__ float c_measurementNoise[16];  // R matrix  
__constant__ float c_dt;                    // Time delta

// Initialize Kalman filter constants
__host__ void initKalmanConstants(float dt, float processNoiseScale, float measurementNoiseScale, cudaStream_t stream) {
    // Process noise covariance matrix Q (6x6)
    float Q[36] = {0};
    Q[0] = Q[7] = processNoiseScale * 0.01f;  // Position noise
    Q[14] = Q[21] = processNoiseScale * 1.0f;  // Velocity noise
    Q[28] = Q[35] = processNoiseScale * 0.01f; // Size noise
    
    // Measurement noise covariance matrix R (4x4)
    float R[16] = {0};
    R[0] = R[5] = R[10] = R[15] = measurementNoiseScale;
    
    cudaMemcpyToSymbolAsync(c_processNoise, Q, sizeof(Q), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_measurementNoise, R, sizeof(R), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_dt, &dt, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
}

// Kernel: Predict step for all Kalman filters
__global__ void kalmanPredictKernel(
    KalmanState* states,
    int numStates,
    float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStates || !states[idx].isActive) return;
    
    KalmanState& kf = states[idx];
    
    // State prediction: x' = F * x
    // F is the state transition matrix
    kf.state[0] += kf.state[2] * dt;  // x = x + vx * dt
    kf.state[1] += kf.state[3] * dt;  // y = y + vy * dt
    // w, h, vx, vy remain the same
    
    // Covariance prediction: P' = F * P * F^T + Q
    // Simplified for our constant velocity model
    float temp[36];
    
    // F * P
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            temp[i * 6 + j] = kf.P[i * 6 + j];
            if (i < 2 && j >= 2 && j < 4) {
                temp[i * 6 + j] += kf.P[(j - 2) * 6 + j] * dt;
            }
        }
    }
    
    // F * P * F^T
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            kf.P[i * 6 + j] = temp[i * 6 + j];
            if (j < 2 && i >= 2 && i < 4) {
                kf.P[i * 6 + j] += temp[i * 6 + (i - 2)] * dt;
            }
        }
    }
    
    // Add process noise
    for (int i = 0; i < 36; i++) {
        kf.P[i] += c_processNoise[i];
    }
    
    kf.timeSinceUpdate++;
}

// Kernel: Update step with measurements
__global__ void kalmanUpdateKernel(
    KalmanState* states,
    const Target* measurements,
    const int* associations,  // measurement idx -> state idx mapping
    int numMeasurements)
{
    int measIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (measIdx >= numMeasurements) return;
    
    int stateIdx = associations[measIdx];
    if (stateIdx < 0) return;  // Unassociated measurement
    
    KalmanState& kf = states[stateIdx];
    const Target& meas = measurements[measIdx];
    
    // Measurement vector z = [x, y, w, h]
    float z[4] = {
        static_cast<float>(meas.x) + static_cast<float>(meas.width) * 0.5f,
        static_cast<float>(meas.y) + static_cast<float>(meas.height) * 0.5f,
        static_cast<float>(meas.width),
        static_cast<float>(meas.height)
    };
    
    // Innovation: y = z - H * x
    // H is the measurement matrix (4x6)
    float y[4];
    y[0] = z[0] - kf.state[0];
    y[1] = z[1] - kf.state[1];
    y[2] = z[2] - kf.state[4];
    y[3] = z[3] - kf.state[5];
    
    // Innovation covariance: S = H * P * H^T + R
    float S[16] = {0};
    S[0] = kf.P[0] + c_measurementNoise[0];
    S[5] = kf.P[7] + c_measurementNoise[5];
    S[10] = kf.P[28] + c_measurementNoise[10];
    S[15] = kf.P[35] + c_measurementNoise[15];
    
    // Kalman gain: K = P * H^T * S^-1
    // Simplified for diagonal S
    float K[24] = {0};  // 6x4 matrix
    K[0] = kf.P[0] / S[0];
    K[7] = kf.P[7] / S[5];
    K[16] = kf.P[28] / S[10];
    K[23] = kf.P[35] / S[15];
    
    // State update: x = x + K * y
    kf.state[0] += K[0] * y[0];
    kf.state[1] += K[7] * y[1];
    kf.state[4] += K[16] * y[2];
    kf.state[5] += K[23] * y[3];
    
    // Estimate velocities from position change
    if (kf.hits > 0) {
        kf.state[2] = (kf.state[0] - (static_cast<float>(meas.x) + static_cast<float>(meas.width) * 0.5f)) / c_dt;
        kf.state[3] = (kf.state[1] - (static_cast<float>(meas.y) + static_cast<float>(meas.height) * 0.5f)) / c_dt;
    }
    
    // Covariance update: P = (I - K * H) * P
    float factor = 1.0f - K[0];
    kf.P[0] *= factor;
    kf.P[7] *= (1.0f - K[7]);
    kf.P[28] *= (1.0f - K[16]);
    kf.P[35] *= (1.0f - K[23]);
    
    // Update metadata
    kf.hits++;
    kf.timeSinceUpdate = 0;
}

// Kernel: Extract predictions as Target structs
__global__ void extractPredictionsKernel(
    const KalmanState* states,
    Target* predictions,
    int* numPredictions,
    int numStates,
    int minHits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStates) return;
    
    const KalmanState& kf = states[idx];
    
    // Only output confirmed tracks
    if (kf.isActive && kf.hits >= minHits && kf.timeSinceUpdate < 3) {
        int outIdx = atomicAdd(numPredictions, 1);
        
        Target& pred = predictions[outIdx];
        pred.x = static_cast<int>(kf.state[0] - kf.state[4] * 0.5f);
        pred.y = static_cast<int>(kf.state[1] - kf.state[5] * 0.5f);
        pred.width = static_cast<int>(kf.state[4]);
        pred.height = static_cast<int>(kf.state[5]);
        pred.id = kf.trackId;  // Use 'id' instead of 'trackId'
        pred.confidence = min(1.0f, kf.hits / 10.0f);  // Confidence based on hits
        pred.classId = 0;  // Will be updated from measurement
    }
}

// Kernel: Create new tracks for unassociated measurements
__global__ void createNewTracksKernel(
    KalmanState* states,
    const Target* measurements,
    const bool* isAssociated,
    int* nextTrackId,
    int numMeasurements,
    int maxStates)
{
    int measIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (measIdx >= numMeasurements || isAssociated[measIdx]) return;
    
    // Find free slot
    for (int i = 0; i < maxStates; i++) {
        if (!states[i].isActive) {
            // Try to claim this slot
            bool expected = false;
            if (atomicCAS((int*)&states[i].isActive, expected, true) == expected) {
                // Successfully claimed slot
                KalmanState& kf = states[i];
                const Target& meas = measurements[measIdx];
                
                // Initialize state
                kf.state[0] = static_cast<float>(meas.x) + static_cast<float>(meas.width) * 0.5f;
                kf.state[1] = static_cast<float>(meas.y) + static_cast<float>(meas.height) * 0.5f;
                kf.state[2] = 0.0f;  // vx
                kf.state[3] = 0.0f;  // vy
                kf.state[4] = static_cast<float>(meas.width);
                kf.state[5] = static_cast<float>(meas.height);
                
                // Initialize covariance
                for (int j = 0; j < 36; j++) kf.P[j] = 0.0f;
                kf.P[0] = kf.P[7] = 100.0f;  // Position uncertainty
                kf.P[14] = kf.P[21] = 1000.0f;  // Velocity uncertainty
                kf.P[28] = kf.P[35] = 100.0f;  // Size uncertainty
                
                // Initialize metadata
                kf.trackId = atomicAdd(nextTrackId, 1);
                kf.age = 0;
                kf.hits = 1;
                kf.timeSinceUpdate = 0;
                
                break;
            }
        }
    }
}

// Kernel: Clean up dead tracks
__global__ void cleanupTracksKernel(
    KalmanState* states,
    int numStates,
    int maxAge)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStates) return;
    
    KalmanState& kf = states[idx];
    if (kf.isActive) {
        kf.age++;
        
        // Deactivate old tracks
        if (kf.timeSinceUpdate > maxAge) {
            kf.isActive = false;
        }
    }
}

// GPU Kalman Filter Manager Class
class GPUKalmanTracker {
private:
    KalmanState* d_states;
    Target* d_predictions;
    int* d_numPredictions;
    int* d_associations;
    bool* d_isAssociated;
    int* d_nextTrackId;
    
    cudaGraph_t kalmanGraph;
    cudaGraphExec_t kalmanGraphExec;
    
    int maxStates;
    int maxMeasurements;
    bool graphCreated;
    
    // Graph nodes
    cudaGraphNode_t predictNode;
    cudaGraphNode_t updateNode;
    cudaGraphNode_t extractNode;
    cudaGraphNode_t newTracksNode;
    cudaGraphNode_t cleanupNode;
    
public:
    GPUKalmanTracker(int maxStates = 100, int maxMeasurements = 100) 
        : maxStates(maxStates), maxMeasurements(maxMeasurements), graphCreated(false) {
        
        // Allocate GPU memory
        cudaMalloc(&d_states, maxStates * sizeof(KalmanState));
        cudaMalloc(&d_predictions, maxMeasurements * sizeof(Target));
        cudaMalloc(&d_numPredictions, sizeof(int));
        cudaMalloc(&d_associations, maxMeasurements * sizeof(int));
        cudaMalloc(&d_isAssociated, maxMeasurements * sizeof(bool));
        cudaMalloc(&d_nextTrackId, sizeof(int));
        
        // Initialize
        cudaMemset(d_states, 0, maxStates * sizeof(KalmanState));
        cudaMemset(d_numPredictions, 0, sizeof(int));
        int initialId = 1;
        cudaMemcpy(d_nextTrackId, &initialId, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    ~GPUKalmanTracker() {
        if (graphCreated) {
            cudaGraphExecDestroy(kalmanGraphExec);
            cudaGraphDestroy(kalmanGraph);
        }
        
        cudaFree(d_states);
        cudaFree(d_predictions);
        cudaFree(d_numPredictions);
        cudaFree(d_associations);
        cudaFree(d_isAssociated);
        cudaFree(d_nextTrackId);
    }
    
    void createGraph(cudaStream_t stream) {
        if (graphCreated) return;
        
        // Start graph capture
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        
        // Add kernels to graph
        dim3 blockSize(256);
        dim3 gridSize((maxStates + blockSize.x - 1) / blockSize.x);
        
        // Predict step
        kalmanPredictKernel<<<gridSize, blockSize, 0, stream>>>(
            d_states, maxStates, 0.033f
        );
        
        // Update step (placeholder - will be updated with actual measurements)
        kalmanUpdateKernel<<<gridSize, blockSize, 0, stream>>>(
            d_states, nullptr, d_associations, 0
        );
        
        // Extract predictions
        extractPredictionsKernel<<<gridSize, blockSize, 0, stream>>>(
            d_states, d_predictions, d_numPredictions, maxStates, 3
        );
        
        // Create new tracks
        createNewTracksKernel<<<gridSize, blockSize, 0, stream>>>(
            d_states, nullptr, d_isAssociated, d_nextTrackId, 0, maxStates
        );
        
        // Cleanup old tracks
        cleanupTracksKernel<<<gridSize, blockSize, 0, stream>>>(
            d_states, maxStates, 5
        );
        
        // End capture and create graph
        cudaStreamEndCapture(stream, &kalmanGraph);
        cudaGraphInstantiate(&kalmanGraphExec, kalmanGraph, nullptr, nullptr, 0);
        
        graphCreated = true;
    }
    
    void process(const Target* d_measurements, int numMeasurements, 
                 Target* d_output, int* d_outputCount,
                 cudaStream_t stream, bool useGraph) {
        
        if (useGraph && graphCreated) {
            // Update graph with new parameters
            updateGraphParams(d_measurements, numMeasurements);
            
            // Launch graph
            cudaGraphLaunch(kalmanGraphExec, stream);
        } else {
            // Direct kernel launches
            dim3 blockSize(256);
            dim3 gridSize((maxStates + blockSize.x - 1) / blockSize.x);
            
            // Predict
            kalmanPredictKernel<<<gridSize, blockSize, 0, stream>>>(
                d_states, maxStates, 0.033f
            );
            
            // Association (simplified - would use Hungarian algorithm)
            // For now, just using nearest neighbor
            associateMeasurements(d_measurements, numMeasurements, stream);
            
            // Update
            if (numMeasurements > 0) {
                dim3 measGrid((numMeasurements + blockSize.x - 1) / blockSize.x);
                kalmanUpdateKernel<<<measGrid, blockSize, 0, stream>>>(
                    d_states, d_measurements, d_associations, numMeasurements
                );
            }
            
            // Extract predictions
            cudaMemset(d_numPredictions, 0, sizeof(int));
            extractPredictionsKernel<<<gridSize, blockSize, 0, stream>>>(
                d_states, d_predictions, d_numPredictions, maxStates, 3
            );
            
            // Create new tracks for unassociated measurements
            if (numMeasurements > 0) {
                dim3 measGrid((numMeasurements + blockSize.x - 1) / blockSize.x);
                createNewTracksKernel<<<measGrid, blockSize, 0, stream>>>(
                    d_states, d_measurements, d_isAssociated, 
                    d_nextTrackId, numMeasurements, maxStates
                );
            }
            
            // Cleanup
            cleanupTracksKernel<<<gridSize, blockSize, 0, stream>>>(
                d_states, maxStates, 5
            );
        }
        
        // Copy results
        cudaMemcpyAsync(d_output, d_predictions, 
                       maxMeasurements * sizeof(Target), 
                       cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_outputCount, d_numPredictions, 
                       sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }
    
private:
    void updateGraphParams(const Target* d_measurements, int numMeasurements) {
        // Update kernel parameters in the graph
        // This would involve cudaGraphKernelNodeSetParams calls
        // For simplicity, showing the concept
    }
    
    void associateMeasurements(const Target* d_measurements, int numMeasurements, 
                               cudaStream_t stream) {
        // Simple nearest neighbor association
        // In production, use Hungarian algorithm or GNN
        cudaMemset(d_associations, -1, numMeasurements * sizeof(int));
        cudaMemset(d_isAssociated, 0, numMeasurements * sizeof(bool));
        
        // Association kernel would go here
    }
};

// Export functions for integration
extern "C" {
    GPUKalmanTracker* createGPUKalmanTracker(int maxStates, int maxMeasurements) {
        return new GPUKalmanTracker(maxStates, maxMeasurements);
    }
    
    void destroyGPUKalmanTracker(GPUKalmanTracker* tracker) {
        delete tracker;
    }
    
    void initializeKalmanGraph(GPUKalmanTracker* tracker, cudaStream_t stream) {
        tracker->createGraph(stream);
    }
    
    void processKalmanFilter(GPUKalmanTracker* tracker,
                            const Target* d_measurements, int numMeasurements,
                            Target* d_output, int* d_outputCount,
                            cudaStream_t stream, bool useGraph) {
        tracker->process(d_measurements, numMeasurements, 
                        d_output, d_outputCount, stream, useGraph);
    }
    
    void updateKalmanFilterSettings(float dt, float processNoise, 
                                   float measurementNoise, cudaStream_t stream) {
        initKalmanConstants(dt, processNoise, measurementNoise, stream);
    }
}