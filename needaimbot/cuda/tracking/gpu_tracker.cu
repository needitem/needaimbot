#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "../core/Target.h"

namespace cg = cooperative_groups;

// ============================================================================
// GPU-ACCELERATED OBJECT TRACKING WITH CUDA GRAPH SUPPORT
// ============================================================================

// Maximum number of tracked objects and detections
#define MAX_TRACKS 128
#define MAX_DETECTIONS 128
#define KALMAN_STATE_DIM 8
#define KALMAN_MEASURE_DIM 4

// Tracked object structure for GPU - aligned with unified Target
struct GPUTrackedObject {
    float x, y, width, height;  // Changed w,h to width,height for consistency
    float center_x, center_y;
    float velocity_x, velocity_y;
    float kalman_state[KALMAN_STATE_DIM];  // [cx, cy, w, h, vx, vy, vw, vh]
    float confidence;
    int classId;  // Changed class_id to classId for consistency
    int track_id;
    int age;
    int hits;
    int time_since_update;
    bool active;
};

// GPU tracking context - persistent memory for CUDA Graph
struct GPUTrackingContext {
    GPUTrackedObject* d_tracks;
    Target* d_detections;
    float* d_iou_matrix;
    int* d_assignment;
    float* d_cost_matrix;
    int* d_num_tracks;
    int* d_num_detections;
    int* d_next_id;
    
    // Kalman filter matrices
    float* d_kalman_F;  // State transition matrix
    float* d_kalman_H;  // Measurement matrix
    float* d_kalman_Q;  // Process noise
    float* d_kalman_R;  // Measurement noise
    float* d_kalman_P;  // Covariance matrices
    
    cublasHandle_t cublas_handle;
    
    int max_age;
    int min_hits;
    float iou_threshold;
};

// ============================================================================
// CUDA KERNELS FOR TRACKING
// ============================================================================

// 1. Batch IOU calculation kernel - fully parallelized
__global__ void __launch_bounds__(256, 4)
batchIOUKernel(
    const Target* __restrict__ detections,
    const GPUTrackedObject* __restrict__ tracks,
    float* __restrict__ iou_matrix,
    int num_detections,
    int num_tracks)
{
    const int det_idx = blockIdx.x;
    const int track_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (det_idx >= num_detections || track_idx >= num_tracks) return;
    
    // Load detection and track data
    const Target& det = detections[det_idx];
    const GPUTrackedObject& track = tracks[track_idx];
    
    if (!track.active) {
        iou_matrix[det_idx * num_tracks + track_idx] = 0.0f;
        return;
    }
    
    // Calculate IOU using fast min/max operations (convert int to float)
    float det_x = static_cast<float>(det.x);
    float det_y = static_cast<float>(det.y);
    float det_x2 = det_x + static_cast<float>(det.width);
    float det_y2 = det_y + static_cast<float>(det.height);
    
    float x1 = fmaxf(det_x, track.x);
    float y1 = fmaxf(det_y, track.y);
    float x2 = fminf(det_x2, track.x + track.width);
    float y2 = fminf(det_y2, track.y + track.height);
    
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area1 = static_cast<float>(det.width * det.height);
    float area2 = track.width * track.height;
    float union_area = area1 + area2 - intersection;
    
    float iou = (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
    iou_matrix[det_idx * num_tracks + track_idx] = iou;
}

// 2. Kalman predict kernel - vectorized state prediction
__global__ void __launch_bounds__(128, 8)
kalmanPredictKernel(
    GPUTrackedObject* __restrict__ tracks,
    int num_tracks,
    float dt)
{
    const int track_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (track_idx >= num_tracks) return;
    
    GPUTrackedObject& track = tracks[track_idx];
    if (!track.active) return;
    
    // Simple constant velocity model prediction
    float* state = track.kalman_state;
    
    // Predict: x = x + v * dt
    state[0] += state[4] * dt;  // cx += vx * dt
    state[1] += state[5] * dt;  // cy += vy * dt
    state[2] += state[6] * dt;  // w += vw * dt
    state[3] += state[7] * dt;  // h += vh * dt
    
    // Update track bbox from predicted state
    track.center_x = state[0];
    track.center_y = state[1];
    track.width = fmaxf(1.0f, state[2]);
    track.height = fmaxf(1.0f, state[3]);
    track.x = state[0] - track.width * 0.5f;
    track.y = state[1] - track.height * 0.5f;
    
    // Update tracking metadata
    track.age++;
    track.time_since_update++;
}

// 3. Kalman update kernel - measurement update
__global__ void __launch_bounds__(128, 8)
kalmanUpdateKernel(
    GPUTrackedObject* __restrict__ tracks,
    const Target* __restrict__ detections,
    const int* __restrict__ assignment,
    int num_detections,
    int num_tracks,
    float dt,
    float alpha)  // Measurement weight
{
    const int det_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (det_idx >= num_detections) return;
    
    int track_idx = assignment[det_idx];
    if (track_idx < 0 || track_idx >= num_tracks) return;
    
    GPUTrackedObject& track = tracks[track_idx];
    const Target& det = detections[det_idx];
    
    // Measurement: convert detection to state space (int to float)
    float meas_cx = static_cast<float>(det.x) + static_cast<float>(det.width) * 0.5f;
    float meas_cy = static_cast<float>(det.y) + static_cast<float>(det.height) * 0.5f;
    float meas_w = static_cast<float>(det.width);
    float meas_h = static_cast<float>(det.height);
    
    float* state = track.kalman_state;
    
    // Calculate velocity from position change
    if (track.hits > 0 && dt > 0.0f) {
        state[4] = (meas_cx - state[0]) / dt * 0.5f;  // Smooth velocity
        state[5] = (meas_cy - state[1]) / dt * 0.5f;
        state[6] = (meas_w - state[2]) / dt * 0.5f;
        state[7] = (meas_h - state[3]) / dt * 0.5f;
    }
    
    // Update state with measurement (exponential smoothing)
    state[0] = alpha * meas_cx + (1.0f - alpha) * state[0];
    state[1] = alpha * meas_cy + (1.0f - alpha) * state[1];
    state[2] = alpha * meas_w + (1.0f - alpha) * state[2];
    state[3] = alpha * meas_h + (1.0f - alpha) * state[3];
    
    // Update track properties
    track.center_x = state[0];
    track.center_y = state[1];
    track.width = state[2];
    track.height = state[3];
    track.x = state[0] - track.width * 0.5f;
    track.y = state[1] - track.height * 0.5f;
    track.velocity_x = state[4];
    track.velocity_y = state[5];
    track.confidence = det.confidence;
    track.classId = det.classId;
    
    // Update tracking statistics
    track.hits++;
    track.time_since_update = 0;
}

// Count active tracks kernel
__global__ void countActiveTracks(
    const GPUTrackedObject* __restrict__ tracks,
    const int* __restrict__ num_tracks,
    int* __restrict__ active_count)
{
    __shared__ int s_count;
    
    if (threadIdx.x == 0) {
        s_count = 0;
    }
    __syncthreads();
    
    const int track_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_tracks = *num_tracks;
    
    if (track_idx < total_tracks) {
        // Count only valid, active tracks (more lenient for stable tracking)
        if (tracks[track_idx].active && 
            tracks[track_idx].hits >= 1 &&  // Show tracks immediately
            tracks[track_idx].time_since_update <= 5 &&  // Allow 5 frames without update
            (tracks[track_idx].x != 0.0f || tracks[track_idx].y != 0.0f)) {
            atomicAdd(&s_count, 1);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(active_count, s_count);
    }
}

// Convert GPUTrackedObject to Target format (compact output - only active tracks)
__global__ void convertTracksToTargets(
    const GPUTrackedObject* __restrict__ tracks,
    Target* __restrict__ targets,
    const int* __restrict__ num_tracks,
    int* __restrict__ num_targets)
{
    const int track_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_tracks = *num_tracks;
    
    // Exit early if beyond total tracks or max buffer
    if (track_idx >= total_tracks || track_idx >= MAX_TRACKS) return;
    
    // Check if track is active and valid (more lenient for stable tracking)
    if (tracks[track_idx].active && 
        tracks[track_idx].hits >= 1 &&  // Show immediately on first hit
        tracks[track_idx].time_since_update <= 5 &&  // Allow 5 frames without update
        (tracks[track_idx].x != 0.0f || tracks[track_idx].y != 0.0f)) {
        
        // Atomically get output index for this active track
        int output_idx = atomicAdd(num_targets, 1);
        
        // Compact active tracks to beginning of output array
        if (output_idx < MAX_DETECTIONS) {
            Target& target = targets[output_idx];
            const GPUTrackedObject& track = tracks[track_idx];
            
            target.x = static_cast<int>(track.x);
            target.y = static_cast<int>(track.y);
            target.width = static_cast<int>(track.width);
            target.height = static_cast<int>(track.height);
            target.confidence = track.confidence;
            target.classId = track.classId;
            target.id = track.track_id;
            target.center_x = track.center_x;
            target.center_y = track.center_y;
            target.velocity_x = track.velocity_x;
            target.velocity_y = track.velocity_y;
        }
    }
}

// 4. Hungarian assignment kernel - simplified greedy assignment
__global__ void __launch_bounds__(32, 1)
greedyAssignmentKernel(
    const float* __restrict__ iou_matrix,
    int* __restrict__ assignment,
    int num_detections,
    int num_tracks,
    float iou_threshold)
{
    // Simple greedy assignment - one thread handles entire assignment
    if (threadIdx.x != 0) return;
    
    // Initialize assignments
    for (int i = 0; i < num_detections; i++) {
        assignment[i] = -1;
    }
    
    // Track which tracks have been assigned
    bool track_assigned[MAX_TRACKS];
    for (int i = 0; i < num_tracks; i++) {
        track_assigned[i] = false;
    }
    
    // Greedy assignment: for each detection, find best unassigned track
    for (int det_idx = 0; det_idx < num_detections; det_idx++) {
        float best_iou = iou_threshold;
        int best_track = -1;
        
        for (int track_idx = 0; track_idx < num_tracks; track_idx++) {
            if (!track_assigned[track_idx]) {
                float iou = iou_matrix[det_idx * num_tracks + track_idx];
                if (iou > best_iou) {
                    best_iou = iou;
                    best_track = track_idx;
                }
            }
        }
        
        if (best_track >= 0) {
            assignment[det_idx] = best_track;
            track_assigned[best_track] = true;
        }
    }
}

// 5. Track management kernel - create new tracks and remove old ones
__global__ void __launch_bounds__(128, 8)
trackManagementKernel(
    GPUTrackedObject* __restrict__ tracks,
    const Target* __restrict__ detections,
    const int* __restrict__ assignment,
    int* __restrict__ next_id,
    int num_detections,
    int* __restrict__ num_tracks,
    int max_tracks,
    int max_age,
    int min_hits)
{
    const int tid = threadIdx.x;
    
    // Ensure num_tracks doesn't exceed MAX_TRACKS
    if (tid == 0 && *num_tracks > MAX_TRACKS) {
        *num_tracks = MAX_TRACKS;
    }
    __syncthreads();
    
    // Phase 1: Mark dead tracks for removal (parallel)
    if (tid < *num_tracks && tid < MAX_TRACKS) {
        GPUTrackedObject& track = tracks[tid];
        if (track.active && track.time_since_update > max_age) {
            track.active = false;
        }
    }
    __syncthreads();
    
    // Phase 2: Compact tracks - remove inactive ones (sequential on thread 0)
    if (tid == 0) {
        int write_idx = 0;
        for (int read_idx = 0; read_idx < *num_tracks && read_idx < MAX_TRACKS; read_idx++) {
            if (tracks[read_idx].active) {
                if (write_idx != read_idx) {
                    tracks[write_idx] = tracks[read_idx];
                }
                write_idx++;
            }
        }
        *num_tracks = write_idx;  // Update count to only active tracks
    }
    __syncthreads();
    
    // Phase 3: Create new tracks for unassigned detections (sequential)
    if (tid == 0) {
        for (int det_idx = 0; det_idx < num_detections; det_idx++) {
            if (assignment[det_idx] < 0 && *num_tracks < max_tracks && *num_tracks < MAX_TRACKS) {
                // Add new track at the end
                int slot = (*num_tracks)++;
                
                if (slot >= 0 && slot < MAX_TRACKS) {
                    const Target& det = detections[det_idx];
                    GPUTrackedObject& track = tracks[slot];
                    
                    // Initialize new track (convert int to float)
                    track.active = true;
                    int new_id = atomicAdd(next_id, 1);
                    // Reset ID counter if it gets too large
                    if (new_id > 10000) {
                        atomicExch(next_id, 0);
                        new_id = 0;
                    }
                    track.track_id = new_id;
                    track.x = static_cast<float>(det.x);
                    track.y = static_cast<float>(det.y);
                    track.width = static_cast<float>(det.width);
                    track.height = static_cast<float>(det.height);
                    track.center_x = static_cast<float>(det.x) + static_cast<float>(det.width) * 0.5f;
                    track.center_y = static_cast<float>(det.y) + static_cast<float>(det.height) * 0.5f;
                    track.velocity_x = 0.0f;
                    track.velocity_y = 0.0f;
                    track.confidence = det.confidence;
                    track.classId = det.classId;
                    track.age = 0;
                    track.hits = 1;
                    track.time_since_update = 0;
                    
                    // Initialize Kalman state
                    track.kalman_state[0] = track.center_x;
                    track.kalman_state[1] = track.center_y;
                    track.kalman_state[2] = track.width;
                    track.kalman_state[3] = track.height;
                    track.kalman_state[4] = 0.0f;  // vx
                    track.kalman_state[5] = 0.0f;  // vy
                    track.kalman_state[6] = 0.0f;  // vw
                    track.kalman_state[7] = 0.0f;  // vh
                }
            }
        }
    }
}

// ============================================================================
// HOST FUNCTIONS FOR GPU TRACKER
// ============================================================================

extern "C" {

// Initialize GPU tracking context
GPUTrackingContext* initGPUTracker(int max_age, int min_hits, float iou_threshold) {
    GPUTrackingContext* ctx = new GPUTrackingContext();
    
    ctx->max_age = max_age;
    ctx->min_hits = min_hits;
    ctx->iou_threshold = iou_threshold;
    
    // Allocate GPU memory
    cudaMalloc(&ctx->d_tracks, MAX_TRACKS * sizeof(GPUTrackedObject));
    cudaMalloc(&ctx->d_detections, MAX_DETECTIONS * sizeof(Target));
    cudaMalloc(&ctx->d_iou_matrix, MAX_DETECTIONS * MAX_TRACKS * sizeof(float));
    cudaMalloc(&ctx->d_assignment, MAX_DETECTIONS * sizeof(int));
    cudaMalloc(&ctx->d_cost_matrix, MAX_DETECTIONS * MAX_TRACKS * sizeof(float));
    cudaMalloc(&ctx->d_num_tracks, sizeof(int));
    cudaMalloc(&ctx->d_num_detections, sizeof(int));
    cudaMalloc(&ctx->d_next_id, sizeof(int));
    
    // Initialize counters asynchronously
    int zero = 0;
    cudaMemcpyAsync(ctx->d_num_tracks, &zero, sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(ctx->d_num_detections, &zero, sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(ctx->d_next_id, &zero, sizeof(int), cudaMemcpyHostToDevice, 0);
    
    // Clear tracks
    cudaMemset(ctx->d_tracks, 0, MAX_TRACKS * sizeof(GPUTrackedObject));
    
    // Create cuBLAS handle for matrix operations
    cublasCreate(&ctx->cublas_handle);
    
    return ctx;
}

// Update tracking with CUDA Graph support
void updateGPUTracker(
    GPUTrackingContext* ctx,
    const Target* h_detections,
    int num_detections,
    GPUTrackedObject* h_output_tracks,
    int* h_num_output_tracks,
    cudaStream_t stream,
    float dt = 0.033f)  // ~30 FPS
{
    if (num_detections > MAX_DETECTIONS) {
        num_detections = MAX_DETECTIONS;
    }
    
    // Copy detections to GPU
    cudaMemcpyAsync(ctx->d_detections, h_detections, 
                    num_detections * sizeof(Target), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx->d_num_detections, &num_detections, 
                    sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // Use device-side counter directly in kernels to avoid synchronization
    // We'll pass d_num_tracks directly to kernels instead of copying to host
    int h_num_tracks = MAX_TRACKS;  // Use max for grid size, kernel will check actual count
    
    // 1. Kalman Predict
    if (h_num_tracks > 0) {
        dim3 predictBlock(128);
        dim3 predictGrid((h_num_tracks + 127) / 128);
        kalmanPredictKernel<<<predictGrid, predictBlock, 0, stream>>>(
            ctx->d_tracks, h_num_tracks, dt);
    }
    
    // 2. Calculate IOU matrix
    if (num_detections > 0 && h_num_tracks > 0) {
        dim3 iouBlock(32);
        dim3 iouGrid(num_detections, (h_num_tracks + 31) / 32);
        batchIOUKernel<<<iouGrid, iouBlock, 0, stream>>>(
            ctx->d_detections, ctx->d_tracks, ctx->d_iou_matrix,
            num_detections, h_num_tracks);
    }
    
    // 3. Hungarian assignment (simplified greedy)
    if (num_detections > 0 && h_num_tracks > 0) {
        greedyAssignmentKernel<<<1, 32, 0, stream>>>(
            ctx->d_iou_matrix, ctx->d_assignment,
            num_detections, h_num_tracks, ctx->iou_threshold);
    } else if (num_detections > 0) {
        // No tracks - all detections are unassigned
        cudaMemsetAsync(ctx->d_assignment, -1, 
                       num_detections * sizeof(int), stream);
    }
    
    // 4. Kalman Update for matched tracks
    if (num_detections > 0) {
        dim3 updateBlock(128);
        dim3 updateGrid((num_detections + 127) / 128);
        kalmanUpdateKernel<<<updateGrid, updateBlock, 0, stream>>>(
            ctx->d_tracks, ctx->d_detections, ctx->d_assignment,
            num_detections, h_num_tracks, dt, 0.7f);
    }
    
    // 5. Track management (create new, remove old)
    trackManagementKernel<<<1, 128, 0, stream>>>(
        ctx->d_tracks, ctx->d_detections, ctx->d_assignment,
        ctx->d_next_id, num_detections, ctx->d_num_tracks,
        MAX_TRACKS, ctx->max_age, ctx->min_hits);
    
    // 6. Copy active tracks back to host
    cudaMemcpyAsync(h_output_tracks, ctx->d_tracks,
                    MAX_TRACKS * sizeof(GPUTrackedObject),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_num_output_tracks, ctx->d_num_tracks,
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
}

// Direct GPU-to-GPU tracking update (no host copies)
void updateGPUTrackerDirect(
    GPUTrackingContext* ctx,
    const Target* d_detections,  // Already on GPU
    int num_detections,
    Target* d_output_tracks,      // Output directly to GPU
    int* d_num_output_tracks,
    cudaStream_t stream,
    float dt = 0.033f)
{
    if (num_detections > MAX_DETECTIONS) {
        num_detections = MAX_DETECTIONS;
    }
    
    // Copy GPU detections to internal buffer (device to device)
    cudaMemcpyAsync(ctx->d_detections, d_detections, 
                    num_detections * sizeof(Target), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(ctx->d_num_detections, &num_detections, 
                    sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // Use device-side counter directly in kernels to avoid synchronization
    // We'll pass d_num_tracks directly to kernels instead of copying to host
    int h_num_tracks = MAX_TRACKS;  // Use max for grid size, kernel will check actual count
    
    // 1. Kalman Predict
    if (h_num_tracks > 0) {
        dim3 predictBlock(128);
        dim3 predictGrid((h_num_tracks + 127) / 128);
        kalmanPredictKernel<<<predictGrid, predictBlock, 0, stream>>>(
            ctx->d_tracks, h_num_tracks, dt);
    }
    
    // 2. Calculate IOU matrix
    if (num_detections > 0 && h_num_tracks > 0) {
        dim3 iouBlock(32);
        dim3 iouGrid(num_detections, (h_num_tracks + 31) / 32);
        batchIOUKernel<<<iouGrid, iouBlock, 0, stream>>>(
            ctx->d_detections, ctx->d_tracks, ctx->d_iou_matrix,
            num_detections, h_num_tracks);
    }
    
    // 3. Hungarian assignment
    if (num_detections > 0 && h_num_tracks > 0) {
        greedyAssignmentKernel<<<1, 32, 0, stream>>>(
            ctx->d_iou_matrix, ctx->d_assignment,
            num_detections, h_num_tracks, ctx->iou_threshold);
    } else if (num_detections > 0) {
        cudaMemsetAsync(ctx->d_assignment, -1, 
                       num_detections * sizeof(int), stream);
    }
    
    // 4. Kalman Update
    if (num_detections > 0) {
        dim3 updateBlock(128);
        dim3 updateGrid((num_detections + 127) / 128);
        kalmanUpdateKernel<<<updateGrid, updateBlock, 0, stream>>>(
            ctx->d_tracks, ctx->d_detections, ctx->d_assignment,
            num_detections, h_num_tracks, dt, 0.7f);
    }
    
    // 5. Track management
    trackManagementKernel<<<1, 128, 0, stream>>>(
        ctx->d_tracks, ctx->d_detections, ctx->d_assignment,
        ctx->d_next_id, num_detections, ctx->d_num_tracks,
        MAX_TRACKS, ctx->max_age, ctx->min_hits);
    
    // 6. Clear output count first, then convert active tracks
    cudaMemsetAsync(d_num_output_tracks, 0, sizeof(int), stream);
    
    // Ensure num_tracks doesn't exceed MAX_TRACKS before conversion
    int h_num_tracks_clamped = h_num_tracks;
    if (h_num_tracks_clamped > MAX_TRACKS) {
        h_num_tracks_clamped = MAX_TRACKS;
        cudaMemcpyAsync(ctx->d_num_tracks, &h_num_tracks_clamped, sizeof(int), cudaMemcpyHostToDevice, stream);
    }
    
    // Convert GPUTrackedObject to Target format and output (only active tracks)
    convertTracksToTargets<<<(MAX_TRACKS + 127) / 128, 128, 0, stream>>>(
        ctx->d_tracks, d_output_tracks, ctx->d_num_tracks, d_num_output_tracks);
}

// Cleanup GPU tracker
void destroyGPUTracker(GPUTrackingContext* ctx) {
    if (ctx) {
        cudaFree(ctx->d_tracks);
        cudaFree(ctx->d_detections);
        cudaFree(ctx->d_iou_matrix);
        cudaFree(ctx->d_assignment);
        cudaFree(ctx->d_cost_matrix);
        cudaFree(ctx->d_num_tracks);
        cudaFree(ctx->d_num_detections);
        cudaFree(ctx->d_next_id);
        
        cublasDestroy(ctx->cublas_handle);
        delete ctx;
    }
}

} // extern "C"