#include "unified_graph_pipeline.h"
#include "detection/cuda_float_processing.h"
#include "simple_cuda_mat.h"
#include "../AppContext.h"
#include "../capture/capture_interface.h"
#include "../core/logger.h"
#include "cuda_error_check.h"
#include "preprocessing.h"
#include "../include/other_tools.h"
#include "../core/constants.h"
#include "detection/postProcess.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <mutex>
#include <atomic>
#include <functional>
#include <random>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Forward declare the mouse control function
extern "C" {
    void executeMouseMovement(int dx, int dy);
}

namespace gpa {

// Use blocking CUDA events to avoid CPU yield storms while waiting for GPU work
static constexpr unsigned int kBlockingEventFlags = cudaEventDisableTiming | cudaEventBlockingSync;

// Gaussian noise generator for humanization (thread-local for callback safety)
static thread_local std::mt19937 g_noiseGen{std::random_device{}()};
static thread_local std::normal_distribution<float> g_noiseDistX(0.0f, 1.0f);
static thread_local std::normal_distribution<float> g_noiseDistY(0.0f, 1.0f);

// ============================================================================
// LOCK-FREE CONFIG UPDATE (v2)
// NOTE: This translation unit must be rebuilt whenever AppContext/Config layout changes.
// ============================================================================

void UnifiedGraphPipeline::refreshConfigCache(const AppContext& ctx) {
    // Called periodically from main loop or when config changes.
    // NOT called in the hot path (executeFrame).
    // NOTE: Does NOT increment generation counter - that is only done by
    // markPidConfigDirty() to signal external config changes.

    // Atomic read of config with single mutex lock
    AppContext& mutableCtx = const_cast<AppContext&>(ctx);
    {
        std::lock_guard<std::mutex> lock(mutableCtx.configMutex);

        // PID
        m_cachedConfig.pid.kp_x = ctx.config.profile().pid_kp_x;
        m_cachedConfig.pid.kp_y = ctx.config.profile().pid_kp_y;
        m_cachedConfig.pid.ki_x = ctx.config.profile().pid_ki_x;
        m_cachedConfig.pid.ki_y = ctx.config.profile().pid_ki_y;
        m_cachedConfig.pid.kd_x = ctx.config.profile().pid_kd_x;
        m_cachedConfig.pid.kd_y = ctx.config.profile().pid_kd_y;
        m_cachedConfig.pid.integral_max = ctx.config.profile().pid_integral_max;
        m_cachedConfig.pid.derivative_max = ctx.config.profile().pid_derivative_max;

        // Targeting
        m_cachedConfig.targeting.head_y_offset = ctx.config.profile().head_y_offset;
        m_cachedConfig.targeting.body_y_offset = ctx.config.profile().body_y_offset;
        m_cachedConfig.targeting.iou_stickiness_threshold = ctx.config.profile().iou_stickiness_threshold;

        // Resolve head class ID
        m_cachedConfig.targeting.head_class_id = -1;
        if (!ctx.config.profile().class_settings.empty()) {
            for (const auto& cs : ctx.config.profile().class_settings) {
                if (cs.name == ctx.config.profile().head_class_name) {
                    m_cachedConfig.targeting.head_class_id = cs.id;
                    break;
                }
            }
        }

        // Detection
        m_cachedConfig.detection.max_detections = ctx.config.profile().max_detections;
        m_cachedConfig.detection.detection_resolution = ctx.config.profile().detection_resolution;
        m_cachedConfig.detection.confidence_threshold = ctx.config.profile().confidence_threshold;
        m_cachedConfig.detection.enable_nms = ctx.config.profile().enable_nms;
        m_cachedConfig.detection.nms_iou_threshold = ctx.config.profile().nms_iou_threshold;

        // Class filter - fixed size array (cache-friendly)
        m_cachedConfig.detection.class_filter.fill(0);
        if (!ctx.config.profile().class_settings.empty()) {
            for (const auto& cs : ctx.config.profile().class_settings) {
                if (cs.id >= 0 && cs.id < 80) {
                    m_cachedConfig.detection.class_filter[cs.id] = cs.allow ? 1 : 0;
                }
            }
        }

        // Movement filtering
        m_cachedConfig.filtering.deadband_enter_x = ctx.config.profile().deadband_enter_x;
        m_cachedConfig.filtering.deadband_exit_x  = ctx.config.profile().deadband_exit_x;
        m_cachedConfig.filtering.deadband_enter_y = ctx.config.profile().deadband_enter_y;
        m_cachedConfig.filtering.deadband_exit_y  = ctx.config.profile().deadband_exit_y;
        m_cachedConfig.filtering.disable_upward_aim = ctx.disable_upward_aim.load(std::memory_order_relaxed);

        // Color filter for target selection
        m_cachedConfig.color_filter.enabled = ctx.config.profile().color_filter_target_enabled && ctx.config.profile().color_filter_enabled;
        m_cachedConfig.color_filter.color_mode = ctx.config.profile().color_filter_mode;
        m_cachedConfig.color_filter.target_mode = ctx.config.profile().color_filter_target_mode;
        m_cachedConfig.color_filter.comparison = ctx.config.profile().color_filter_comparison;
        m_cachedConfig.color_filter.r_min = ctx.config.profile().color_filter_r_min;
        m_cachedConfig.color_filter.r_max = ctx.config.profile().color_filter_r_max;
        m_cachedConfig.color_filter.g_min = ctx.config.profile().color_filter_g_min;
        m_cachedConfig.color_filter.g_max = ctx.config.profile().color_filter_g_max;
        m_cachedConfig.color_filter.b_min = ctx.config.profile().color_filter_b_min;
        m_cachedConfig.color_filter.b_max = ctx.config.profile().color_filter_b_max;
        m_cachedConfig.color_filter.h_min = ctx.config.profile().color_filter_h_min;
        m_cachedConfig.color_filter.h_max = ctx.config.profile().color_filter_h_max;
        m_cachedConfig.color_filter.s_min = ctx.config.profile().color_filter_s_min;
        m_cachedConfig.color_filter.s_max = ctx.config.profile().color_filter_s_max;
        m_cachedConfig.color_filter.v_min = ctx.config.profile().color_filter_v_min;
        m_cachedConfig.color_filter.v_max = ctx.config.profile().color_filter_v_max;
        m_cachedConfig.color_filter.min_ratio = ctx.config.profile().color_filter_min_ratio;
        m_cachedConfig.color_filter.max_ratio = ctx.config.profile().color_filter_max_ratio;
        m_cachedConfig.color_filter.min_count = ctx.config.profile().color_filter_min_count;
        m_cachedConfig.color_filter.max_count = ctx.config.profile().color_filter_max_count;
    }
}

void UnifiedGraphPipeline::updateConfig(const AppContext& ctx) {
    refreshConfigCache(ctx);

    // Upload class filter to GPU only if changed
    if (m_smallBufferArena.allowFlags && m_pipelineStream && m_pipelineStream->get()) {
        if (m_cachedConfig.detection.class_filter != m_cachedConfig.detection.prev_class_filter) {
            cudaMemcpyAsync(
                m_smallBufferArena.allowFlags,
                m_cachedConfig.detection.class_filter.data(),
                80 * sizeof(unsigned char),
                cudaMemcpyHostToDevice,
                m_pipelineStream->get());
            m_cachedConfig.detection.prev_class_filter = m_cachedConfig.detection.class_filter;
        }
    }
}

void UnifiedGraphPipeline::markPidConfigDirty() {
    // Signal that external config has changed.
    // The main loop's change-detection will pick this up and call updateConfig.
    m_cachedConfig.generation.fetch_add(1, std::memory_order_release);
}

void UnifiedGraphPipeline::fillHostDeviceConfig() {
    const CachedConfig& cfg = m_cachedConfig;
    int detRes = cfg.detection.detection_resolution;

    DeviceConfig& h = m_hostDeviceConfig;
    // PID
    h.kp_x = cfg.pid.kp_x;   h.kp_y = cfg.pid.kp_y;
    h.ki_x = cfg.pid.ki_x;   h.ki_y = cfg.pid.ki_y;
    h.kd_x = cfg.pid.kd_x;   h.kd_y = cfg.pid.kd_y;
    h.integral_max = cfg.pid.integral_max;
    h.derivative_max = cfg.pid.derivative_max;
    // Targeting
    h.head_y_offset = cfg.targeting.head_y_offset;
    h.body_y_offset = cfg.targeting.body_y_offset;
    h.iou_stickiness_threshold = cfg.targeting.iou_stickiness_threshold;
    h.head_class_id = cfg.targeting.head_class_id;
    // Screen geometry
    h.max_detections = cfg.detection.max_detections;
    h.detection_resolution = detRes;
    h.screen_center_x = detRes / 2.0f;
    h.screen_center_y = detRes / 2.0f;
    // Color filter
    h.color_mode = cfg.color_filter.color_mode;
    h.target_mode = cfg.color_filter.target_mode;
    h.comparison = cfg.color_filter.comparison;
    h.r_min = cfg.color_filter.r_min; h.r_max = cfg.color_filter.r_max;
    h.g_min = cfg.color_filter.g_min; h.g_max = cfg.color_filter.g_max;
    h.b_min = cfg.color_filter.b_min; h.b_max = cfg.color_filter.b_max;
    h.h_min = cfg.color_filter.h_min; h.h_max = cfg.color_filter.h_max;
    h.s_min = cfg.color_filter.s_min; h.s_max = cfg.color_filter.s_max;
    h.v_min = cfg.color_filter.v_min; h.v_max = cfg.color_filter.v_max;
    h.cf_min_ratio = cfg.color_filter.min_ratio;
    h.cf_max_ratio = cfg.color_filter.max_ratio;
    h.min_count = cfg.color_filter.min_count;
    h.max_count = cfg.color_filter.max_count;
    // Image geometry
    if (!m_captureBuffer.empty()) {
        h.imageWidth = m_captureBuffer.cols();
        h.imageHeight = m_captureBuffer.rows();
        h.imageStep = static_cast<int>(m_captureBuffer.step());
    }
    // NMS
    h.nms_iou_threshold = cfg.detection.nms_iou_threshold;
}

void UnifiedGraphPipeline::uploadDeviceConfig(cudaStream_t stream) {
    if (!m_smallBufferArena.deviceConfig) return;
    fillHostDeviceConfig();
    // Copy from persistent m_hostDeviceConfig (class member at fixed address).
    // In CUDA Graph replay, the captured memcpy node reads from this same address.
    cudaMemcpyAsync(m_smallBufferArena.deviceConfig, &m_hostDeviceConfig,
                    sizeof(DeviceConfig), cudaMemcpyHostToDevice, stream);
}

bool UnifiedGraphPipeline::graphTopologyChanged() const {
    const CachedConfig& cfg = m_cachedConfig;
    return m_graphTopology.nmsEnabled != cfg.detection.enable_nms ||
           m_graphTopology.colorFilterEnabled != cfg.color_filter.enabled ||
           m_graphTopology.detectionResolution != cfg.detection.detection_resolution ||
           m_graphTopology.maxDetections != cfg.detection.max_detections ||
           m_graphTopology.confidenceThreshold != cfg.detection.confidence_threshold ||
           m_graphTopology.nmsIouThreshold != cfg.detection.nms_iou_threshold;
}

void UnifiedGraphPipeline::snapshotGraphTopology() {
    const CachedConfig& cfg = m_cachedConfig;
    m_graphTopology.nmsEnabled = cfg.detection.enable_nms;
    m_graphTopology.colorFilterEnabled = cfg.color_filter.enabled;
    m_graphTopology.detectionResolution = cfg.detection.detection_resolution;
    m_graphTopology.maxDetections = cfg.detection.max_detections;
    m_graphTopology.confidenceThreshold = cfg.detection.confidence_threshold;
    m_graphTopology.nmsIouThreshold = cfg.detection.nms_iou_threshold;
}

void UnifiedGPUArena::initializePointers(uint8_t* basePtr, int maxDetections, int yoloSize, PipelinePrecision prec) {
    precision = prec;
    size_t elementSize = getElementSize(prec);
    size_t offset = 0;
    
    // Align to 16 bytes for optimal memory access (works for FP32, FP16, FP8)
    offset = (offset + 15) & ~15;
    yoloInput = reinterpret_cast<void*>(basePtr + offset);
    offset += yoloSize * yoloSize * 3 * elementSize;
    
    
    offset = (offset + alignof(Target) - 1) & ~(alignof(Target) - 1);
    decodedTargets = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);

    finalTargets = decodedTargets;  // Alias final targets to decoded buffer
    
    // NMS temporary buffer for compaction
    offset = (offset + alignof(Target) - 1) & ~(alignof(Target) - 1);
    nmsTemp = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);
}

size_t UnifiedGPUArena::calculateArenaSize(int maxDetections, int yoloSize, PipelinePrecision prec) {
    size_t elementSize = getElementSize(prec);
    size_t size = 0;
    
    // Align to 16 bytes for optimal memory access
    size = (size + 15) & ~15;
    size += yoloSize * yoloSize * 3 * elementSize;
    
    size = (size + alignof(Target) - 1) & ~(alignof(Target) - 1);
    size += maxDetections * sizeof(Target);
    
    // NMS temporary buffer
    size = (size + alignof(Target) - 1) & ~(alignof(Target) - 1);
    size += maxDetections * sizeof(Target);
    
    return size;
}


namespace {

// Block size is now fixed at 32 (single warp) for warp shuffle optimization

}  // namespace

// Batched memset kernel - clears multiple int buffers in a single kernel launch
__global__ void batchedMemsetKernel(
    int* __restrict__ buf0,
    int* __restrict__ buf1,
    int* __restrict__ buf2,
    int* __restrict__ buf3,
    int val0, int val1, int val2, int val3)
{
    if (threadIdx.x == 0) {
        if (buf0) *buf0 = val0;
        if (buf1) *buf1 = val1;
        if (buf2) *buf2 = val2;
        if (buf3) *buf3 = val3;
    }
}

// Extended batched memset for detection buffer clearing
__global__ void batchedDetectionClearKernel(
    int* __restrict__ decodedCount,
    int* __restrict__ finalTargetsCount,
    int* __restrict__ classFilteredCount,
    int* __restrict__ bestTargetIndex,
    Target* __restrict__ bestTarget)
{
    if (threadIdx.x == 0) {
        if (decodedCount) *decodedCount = 0;
        if (finalTargetsCount) *finalTargetsCount = 0;
        if (classFilteredCount) *classFilteredCount = 0;
        if (bestTargetIndex) *bestTargetIndex = -1;
        if (bestTarget) {
            bestTarget->classId = -1;
            bestTarget->confidence = 0.0f;
            bestTarget->x = 0;
            bestTarget->y = 0;
            bestTarget->width = 0;
            bestTarget->height = 0;
        }
    }
}

// RGB to HSV conversion (device function)
__device__ void rgbToHsvDevice(int r, int g, int b, int& h, int& s, int& v) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;

    float maxVal = fmaxf(fmaxf(rf, gf), bf);
    float minVal = fminf(fminf(rf, gf), bf);
    float delta = maxVal - minVal;

    v = static_cast<int>(maxVal * 255);

    if (maxVal > 0.0f) {
        s = static_cast<int>((delta / maxVal) * 255);
    } else {
        s = 0;
    }

    if (delta < 0.00001f) {
        h = 0;
    } else if (maxVal == rf) {
        h = static_cast<int>(60.0f * fmodf((gf - bf) / delta, 6.0f));
    } else if (maxVal == gf) {
        h = static_cast<int>(60.0f * ((bf - rf) / delta + 2.0f));
    } else {
        h = static_cast<int>(60.0f * ((rf - gf) / delta + 4.0f));
    }

    if (h < 0) h += 360;
    h = h / 2; // Convert to 0-179 range (OpenCV style)
}

// Color match kernel - computes color match ratio for each target
// One block per target, threads cooperatively sample pixels within bbox
__global__ void computeColorMatchRatioKernel(
    Target* __restrict__ targets,
    const int* __restrict__ targetCount,
    const unsigned char* __restrict__ imageData,
    const gpa::DeviceConfig* __restrict__ config
) {
    // Read runtime values from device config
    const int imageWidth = config->imageWidth;
    const int imageHeight = config->imageHeight;
    const int imageStep = config->imageStep;
    const int colorMode = config->color_mode;
    const int r_min = config->r_min, r_max = config->r_max;
    const int g_min = config->g_min, g_max = config->g_max;
    const int b_min = config->b_min, b_max = config->b_max;
    const int h_min = config->h_min, h_max = config->h_max;
    const int s_min = config->s_min, s_max = config->s_max;
    const int v_min = config->v_min, v_max = config->v_max;
    int targetIdx = blockIdx.x;
    int count = *targetCount;

    if (targetIdx >= count) return;

    Target& t = targets[targetIdx];
    if (t.width <= 0 || t.height <= 0 || t.classId < 0) {
        t.colorMatchRatio = -1.0f;
        t.colorMatchCount = -1;
        return;
    }

    // Clamp bbox to image bounds
    int x1 = max(0, t.x);
    int y1 = max(0, t.y);
    int x2 = min(imageWidth, t.x + t.width);
    int y2 = min(imageHeight, t.y + t.height);

    int bboxWidth = x2 - x1;
    int bboxHeight = y2 - y1;

    if (bboxWidth <= 0 || bboxHeight <= 0) {
        t.colorMatchRatio = -1.0f;
        t.colorMatchCount = -1;
        return;
    }

    // Count ALL pixels in bbox for accurate pixel count
    int totalPixels = bboxWidth * bboxHeight;

    // Use shared memory for reduction
    __shared__ int matchCount;

    if (threadIdx.x == 0) {
        matchCount = 0;
    }
    __syncthreads();

    int localMatches = 0;

    // Each thread processes multiple pixels - iterate through ALL pixels
    for (int i = threadIdx.x; i < totalPixels; i += blockDim.x) {
        int localX = i % bboxWidth;
        int localY = i / bboxWidth;

        int px = x1 + localX;
        int py = y1 + localY;

        // Read BGRA pixel
        const unsigned char* pixel = imageData + py * imageStep + px * 4;
        int b = pixel[0];
        int g = pixel[1];
        int r = pixel[2];

        bool matches = false;

        if (colorMode == 0) {
            // RGB mode
            matches = (r >= r_min && r <= r_max &&
                      g >= g_min && g <= g_max &&
                      b >= b_min && b <= b_max);
        } else {
            // HSV mode
            int h, s, v;
            rgbToHsvDevice(r, g, b, h, s, v);

            // Handle hue wraparound
            bool hueMatch;
            if (h_min <= h_max) {
                hueMatch = (h >= h_min && h <= h_max);
            } else {
                hueMatch = (h >= h_min || h <= h_max);
            }

            matches = hueMatch &&
                     (s >= s_min && s <= s_max) &&
                     (v >= v_min && v <= v_max);
        }

        if (matches) localMatches++;
    }

    // Warp-level reduction (256 atomics → 8 atomics)
    const unsigned int FULL_MASK = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        localMatches += __shfl_down_sync(FULL_MASK, localMatches, offset);
    }
    // Only lane 0 of each warp does atomicAdd (8 warps × 1 = 8 ops)
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&matchCount, localMatches);
    }
    __syncthreads();

    // Thread 0 writes the final ratio and count
    if (threadIdx.x == 0) {
        t.colorMatchCount = matchCount;  // Actual pixel count
        t.colorMatchRatio = static_cast<float>(matchCount) / static_cast<float>(totalPixels);
    }
}

// Separate kernel to apply color filter - marks filtered targets by setting confidence to 0
// This runs AFTER computeColorMatchRatioKernel and BEFORE target selection
__global__ void applyColorFilterKernel(
    Target* __restrict__ targets,
    const int* __restrict__ targetCount,
    const gpa::DeviceConfig* __restrict__ config
) {
    // Read runtime values from device config
    const int target_mode = config->target_mode;
    const int comparison = config->comparison;
    const float min_ratio = config->cf_min_ratio;
    const float max_ratio = config->cf_max_ratio;
    const int min_count = config->min_count;
    const int max_count = config->max_count;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = *targetCount;

    if (idx >= count) return;

    Target& t = targets[idx];
    if (t.colorMatchRatio < 0.0f) return;  // Not computed

    bool passFilter = false;

    if (target_mode == 0) {
        // Ratio mode
        float value = t.colorMatchRatio;
        if (comparison == 0) {
            passFilter = (value >= min_ratio);
        } else if (comparison == 1) {
            passFilter = (value <= max_ratio);
        } else {
            passFilter = (value >= min_ratio && value <= max_ratio);
        }
    } else {
        // Absolute count mode
        int value = t.colorMatchCount;
        if (comparison == 0) {
            passFilter = (value >= min_count);
        } else if (comparison == 1) {
            passFilter = (value <= max_count);
        } else {
            passFilter = (value >= min_count && value <= max_count);
        }
    }

    // Mark filtered targets by zeroing confidence
    if (!passFilter) {
        t.confidence = 0.0f;
    }
}

__device__ float computeBoundingBoxIoU(const Target& a, const Target& b) {
    if (a.classId < 0 || b.classId < 0 || a.width <= 0 || a.height <= 0 ||
        b.width <= 0 || b.height <= 0) {
        return 0.0f;
    }

    int a_x2 = a.x + a.width;
    int a_y2 = a.y + a.height;
    int b_x2 = b.x + b.width;
    int b_y2 = b.y + b.height;

    int inter_x1 = (a.x > b.x) ? a.x : b.x;
    int inter_y1 = (a.y > b.y) ? a.y : b.y;
    int inter_x2 = (a_x2 < b_x2) ? a_x2 : b_x2;
    int inter_y2 = (a_y2 < b_y2) ? a_y2 : b_y2;

    int inter_w = inter_x2 - inter_x1;
    int inter_h = inter_y2 - inter_y1;
    if (inter_w <= 0 || inter_h <= 0) {
        return 0.0f;
    }

    int inter_area = inter_w * inter_h;
    int area_a = a.width * a.height;
    int area_b = b.width * b.height;
    int union_area = area_a + area_b - inter_area;

    if (union_area <= 0) {
        return 0.0f;
    }

    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

__global__ void fusedTargetSelectionAndMovementKernel(
    Target* __restrict__ finalTargets,
    int* __restrict__ finalTargetsCount,
    const gpa::DeviceConfig* __restrict__ config,
    Target* __restrict__ selectedTarget,
    int* __restrict__ bestTargetIndex,
    Target* __restrict__ bestTarget,
    gpa::MouseMovement* __restrict__ output_movement,
    gpa::PIDState* __restrict__ pidState
) {
    // Read runtime values from device config (updated each frame via cudaMemcpyAsync)
    const int maxDetections = config->max_detections;
    const float screen_center_x = config->screen_center_x;
    const float screen_center_y = config->screen_center_y;
    const int head_class_id = config->head_class_id;
    const float kp_x = config->kp_x;
    const float kp_y = config->kp_y;
    const float ki_x = config->ki_x;
    const float ki_y = config->ki_y;
    const float kd_x = config->kd_x;
    const float kd_y = config->kd_y;
    const float integral_max = config->integral_max;
    const float derivative_max = config->derivative_max;
    const float iou_stickiness_threshold = config->iou_stickiness_threshold;
    const float head_y_offset = config->head_y_offset;
    const float body_y_offset = config->body_y_offset;
    const int detection_resolution = config->detection_resolution;
    // Using warp shuffle for reduction - no shared memory arrays needed
    __shared__ Target s_prevTarget;
    __shared__ bool s_prevValid;

    if (threadIdx.x == 0) {
        Target emptyTarget = {};
        s_prevTarget = emptyTarget;
        s_prevValid = false;

        if (selectedTarget) {
            Target cached = *selectedTarget;
            s_prevTarget = cached;
            s_prevValid = (cached.classId >= 0) && (cached.confidence > 0.0f) &&
                          (cached.width > 0) && (cached.height > 0);
        }

        *bestTargetIndex = -1;
        *bestTarget = emptyTarget;
        output_movement->dx = 0;
        output_movement->dy = 0;
    }
    __syncthreads();

    int count = *finalTargetsCount;
    if (count <= 0 || count > maxDetections) {
        if (threadIdx.x == 0 && selectedTarget) {
            Target emptyTarget = {};
            *selectedTarget = emptyTarget;
        }
        return;
    }

    Target prevTarget = s_prevTarget;
    bool prevValid = s_prevValid;

    int localBestIdx = -1;
    float localBestDistX = 1e9f;
    float localBestIoU = -1.0f;
    int localBestIoUIdx = -1;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        // Read target data (compiler will optimize read-only access)
        Target t = finalTargets[i];

        // Skip invalid targets (includes those filtered by color filter kernel)
        if (t.x < -1000.0f || t.x > 10000.0f ||
            t.y < -1000.0f || t.y > 10000.0f ||
            t.width <= 0 || t.width > detection_resolution ||
            t.height <= 0 || t.height > detection_resolution ||
            t.confidence <= 0.0f || t.confidence > 1.0f) {
            continue;
        }

        float centerX = t.x + t.width / 2.0f;
        float dx = fabsf(centerX - screen_center_x);

        if (dx < localBestDistX) {
            localBestDistX = dx;
            localBestIdx = i;
        }

        if (prevValid) {
            float iou = computeBoundingBoxIoU(t, prevTarget);
            if (iou > localBestIoU) {
                localBestIoU = iou;
                localBestIoUIdx = i;
            }
        }
    }

    // Warp-level reduction using shuffle (no shared memory needed for single warp)
    // This is faster than shared memory reduction for blockSize <= 32
    const unsigned int FULL_MASK = 0xffffffff;

    float bestDistX = localBestDistX;
    int bestIdx = localBestIdx;
    float bestIoU = localBestIoU;
    int bestIoUIdx = localBestIoUIdx;

    // Warp shuffle reduction - find min distance and max IoU
    for (int offset = 16; offset > 0; offset >>= 1) {
        float otherDistX = __shfl_down_sync(FULL_MASK, bestDistX, offset);
        int otherIdx = __shfl_down_sync(FULL_MASK, bestIdx, offset);
        float otherIoU = __shfl_down_sync(FULL_MASK, bestIoU, offset);
        int otherIoUIdx = __shfl_down_sync(FULL_MASK, bestIoUIdx, offset);

        // Min distance reduction
        if (otherDistX < bestDistX) {
            bestDistX = otherDistX;
            bestIdx = otherIdx;
        }
        // Max IoU reduction
        if (otherIoU > bestIoU) {
            bestIoU = otherIoU;
            bestIoUIdx = otherIoUIdx;
        }
    }

    // Only thread 0 has the final results
    if (threadIdx.x == 0) {
        int candidateIndex = bestIdx;
        bool candidateValid = candidateIndex >= 0;
        Target candidateTarget = candidateValid ? finalTargets[candidateIndex] : Target{};

        // Hysteresis: prefer previous target if IoU stays above threshold
        const float iouStickinessThreshold = iou_stickiness_threshold;

        int chosenIndex = candidateIndex;
        Target chosenTarget = candidateTarget;
        bool haveTarget = candidateValid;
        // bestIoU and bestIoUIdx are from warp reduction above
        if (prevValid && bestIoUIdx >= 0 && bestIoU > iouStickinessThreshold) {
            chosenIndex = bestIoUIdx;
            chosenTarget = finalTargets[bestIoUIdx];
            haveTarget = true;
        }

        if (haveTarget) {
            *bestTargetIndex = chosenIndex;
            *bestTarget = chosenTarget;
            if (selectedTarget) {
                *selectedTarget = chosenTarget;
            }

            float target_center_x = chosenTarget.x + chosenTarget.width / 2.0f;
            float target_center_y;

            if (chosenTarget.classId == head_class_id) {
                target_center_y = chosenTarget.y + chosenTarget.height * head_y_offset;
            } else {
                target_center_y = chosenTarget.y + chosenTarget.height * body_y_offset;
            }

            // Current error
            float error_x = target_center_x - screen_center_x;
            float error_y = target_center_y - screen_center_y;

            // Load previous PID state
            float prev_error_x = pidState->prev_error_x;
            float prev_error_y = pidState->prev_error_y;
            float integral_x = pidState->integral_x;
            float integral_y = pidState->integral_y;

            // Reset integral when very close to target (deadzone)
            // Decouple axes: apply deadzone per-axis, not by combined magnitude
            const float deadzone_threshold = 5.0f;  // pixels
            if (fabsf(error_x) < deadzone_threshold) {
                integral_x = 0.0f;
            }
            if (fabsf(error_y) < deadzone_threshold) {
                integral_y = 0.0f;
            }

            // Update integral (with anti-windup clamping)
            integral_x += error_x;
            integral_y += error_y;

            // Clamp integral to prevent windup
            if (integral_x > integral_max) integral_x = integral_max;
            if (integral_x < -integral_max) integral_x = -integral_max;
            if (integral_y > integral_max) integral_y = integral_max;
            if (integral_y < -integral_max) integral_y = -integral_max;

            // Calculate derivative (error change)
            float derivative_x = error_x - prev_error_x;
            float derivative_y = error_y - prev_error_y;

            // Clamp derivative to prevent excessive oscillation from large movements
            if (derivative_x > derivative_max) derivative_x = derivative_max;
            if (derivative_x < -derivative_max) derivative_x = -derivative_max;
            if (derivative_y > derivative_max) derivative_y = derivative_max;
            if (derivative_y < -derivative_max) derivative_y = -derivative_max;

            // PID controller: P + I + D
            float movement_x = kp_x * error_x + ki_x * integral_x + kd_x * derivative_x;
            float movement_y = kp_y * error_y + ki_y * integral_y + kd_y * derivative_y;

            // Save current state for next iteration
            pidState->prev_error_x = error_x;
            pidState->prev_error_y = error_y;
            pidState->integral_x = integral_x;
            pidState->integral_y = integral_y;

            // Round to nearest int
            int emit_dx = static_cast<int>(lroundf(movement_x));
            int emit_dy = static_cast<int>(lroundf(movement_y));

            output_movement->dx = emit_dx;
            output_movement->dy = emit_dy;
        } else {
            Target emptyTarget = {};
            *bestTargetIndex = -1;
            *bestTarget = emptyTarget;
            if (selectedTarget) {
                *selectedTarget = emptyTarget;
            }
            output_movement->dx = 0;
            output_movement->dy = 0;

            // Reset PID state when no target
            pidState->prev_error_x = 0.0f;
            pidState->prev_error_y = 0.0f;
            pidState->integral_x = 0.0f;
            pidState->integral_y = 0.0f;
        }
    }
}


UnifiedGraphPipeline::UnifiedGraphPipeline() {
    m_state.startEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    m_state.endEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    resetMovementFilter();
    m_perfMetrics.reset();
}


bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    m_config = config;
    
    int leastPriority, greatestPriority;
    cudaError_t err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    if (err == cudaSuccess) {
        cudaStream_t priorityStream;
        err = cudaStreamCreateWithPriority(&priorityStream, cudaStreamNonBlocking, greatestPriority);
        if (err == cudaSuccess) {
            m_pipelineStream = std::make_unique<CudaStream>(priorityStream);
        } else {
            m_pipelineStream = std::make_unique<CudaStream>();
        }
    } else {
        m_pipelineStream = std::make_unique<CudaStream>();
    }

    // Create a low-priority stream for preview conversions/copies
    cudaStream_t previewStreamHandle = nullptr;
    cudaError_t previewErr = cudaSuccess;
    if (err == cudaSuccess) {
        previewErr = cudaStreamCreateWithPriority(&previewStreamHandle, cudaStreamNonBlocking, leastPriority);
    }
    if (previewErr == cudaSuccess && previewStreamHandle) {
        m_previewStream = std::make_unique<CudaStream>(previewStreamHandle);
    } else {
        try {
            m_previewStream = std::make_unique<CudaStream>();
        } catch (...) {
            m_previewStream.reset();
        }
    }

    m_previewReadyEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    
    if (m_config.modelPath.empty()) {
        std::cerr << "[UnifiedGraph] ERROR: Model path is required for TensorRT integration" << std::endl;
        return false;
    }
    
    if (!initializeTensorRT(m_config.modelPath)) {
        std::cerr << "[UnifiedGraph] CRITICAL: TensorRT initialization failed" << std::endl;
        return false;
    }

    // Allocate GPU arenas and small buffers first to avoid duplicate
    // allocations with TensorRT input bindings.
    if (!allocateBuffers()) {
        std::cerr << "[UnifiedGraph] Failed to allocate buffers" << std::endl;
        return false;
    }

    // Allocate TensorRT bindings after arena allocation so we can alias
    // the primary input directly to the unified arena instead of
    // temporarily allocating a duplicate input buffer.
    try {
        getBindings();
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] Failed to allocate TensorRT bindings: " << e.what() << std::endl;
        return false;
    }
    
    if (m_config.useGraphOptimization) {
        // Auto warmup iterations to stabilize TensorRT/cuBLAS autotuning
        // Use a small fixed number to avoid user misconfiguration
        int warmupIters = 3;

        // Ensure pipeline stream is initialized before warmup
        if (!m_pipelineStream || !m_pipelineStream->get()) {
            std::cerr << "[Pipeline] Warning: Pipeline stream not available for warmup" << std::endl;
        } else {
            std::cout << "[Pipeline] Performing warmup inference..." << std::endl;
            for (int i = 0; i < warmupIters; i++) {
                auto bindingIt = m_inputBindings.find(m_inputName);
                if (bindingIt != m_inputBindings.end() && bindingIt->second) {
                    cudaMemsetAsync(bindingIt->second->get(), 0,
                                    bindingIt->second->size(),
                                    m_pipelineStream->get());
                }

                if (!bindStaticTensorAddresses()) {
                    break;
                }

                if (m_context) {
                    m_context->enqueueV3(m_pipelineStream->get());
                }
            }
            cudaStreamSynchronize(m_pipelineStream->get());
            std::cout << "[Pipeline] Warmup complete (3 iterations)" << std::endl;
        }

        m_state.needsRebuild = true;
    }

    // Initialize cached config and upload class filter to GPU once at startup.
    {
        auto& ctx = AppContext::getInstance();
        updateConfig(ctx);
    }
    
    return true;
}


bool UnifiedGraphPipeline::captureGraph(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    auto& ctx = AppContext::getInstance();
    
    if (!stream) {
        if (!m_pipelineStream || !m_pipelineStream->get()) {
            std::cerr << "[UnifiedGraph] Pipeline stream not initialized for graph capture" << std::endl;
            return false;
        }
        stream = m_pipelineStream->get();
    }
    
    
    cleanupGraph();
    
    m_captureNodes.clear();
    m_inferenceNodes.clear();
    m_postprocessNodes.clear();
    
    if (!bindStaticTensorAddresses()) {
        return false;
    }

    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to begin capture: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Frame capture is NOT part of the graph - acquireFrameSync() runs
    // outside graph capture, providing m_captureBuffer for preprocessing.
    
    // Preprocessing captured into graph
    if (m_unifiedArena.yoloInput && !m_captureBuffer.empty()) {
        int modelRes = getModelInputResolution();
        bool use_fp16 = (m_inputDataType == nvinfer1::DataType::kHALF);
        cuda_unified_preprocessing(
            m_captureBuffer.data(),
            m_unifiedArena.yoloInput,
            m_captureBuffer.cols(),
            m_captureBuffer.rows(),
            static_cast<int>(m_captureBuffer.step()),
            modelRes,
            modelRes,
            use_fp16,
            stream
        );
        
        ensurePrimaryInputBindingAliased();

        void* inputBinding = (m_primaryInputIndex >= 0 &&
                              m_primaryInputIndex < static_cast<int>(m_inputAddressCache.size()))
                                 ? m_inputAddressCache[m_primaryInputIndex]
                                 : nullptr;
        if (inputBinding && inputBinding != m_unifiedArena.yoloInput) {
            size_t inputSize = modelRes * modelRes * 3 * sizeof(float);
            cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize,
                           cudaMemcpyDeviceToDevice, stream);
        }
    }

    // TensorRT inference captured into graph (uses graph-compatible mode)
    if (m_context && m_config.enableDetection) {
        if (!m_context->enqueueV3(stream)) {
            std::cerr << "Warning: TensorRT enqueue failed during graph capture" << std::endl;
        }
    }
    
    if (m_config.enableDetection) {
        performIntegratedPostProcessing(stream);
        performTargetSelection(stream);

        // Mouse movement D2H copy (captured in graph)
        if (!m_mouseMovementUsesMappedMemory) {
            cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement,
                           sizeof(MouseMovement), cudaMemcpyDeviceToHost, stream);
        }

        // Debug overlay D2H copies - always captured in graph (negligible overhead).
        // This avoids needing show_window as a topology parameter.
        if (m_h_targetCount && m_h_targetCount->get() && m_smallBufferArena.decodedCount) {
            cudaMemcpyAsync(m_h_targetCount->get(), m_smallBufferArena.decodedCount,
                           sizeof(int), cudaMemcpyDeviceToHost, stream);
        }
        if (m_h_targets && m_h_targets->get() && m_unifiedArena.decodedTargets) {
            cudaMemcpyAsync(m_h_targets->get(), m_unifiedArena.decodedTargets,
                           sizeof(Target) * MAX_HOST_TARGETS, cudaMemcpyDeviceToHost, stream);
        }

        // Completion callback - executes mouse movement on host after GPU work
        FrameMetadata graphMeta{};
        if (!enqueueFrameCompletionCallback(stream, graphMeta)) {
            std::cerr << "[UnifiedGraph] Failed to attach completion callback during graph capture" << std::endl;
        }
    }

    err = cudaStreamEndCapture(stream, &m_graph);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to end capture: " 
                  << cudaGetErrorString(err) << std::endl;
        if (m_graph) {
            cudaGraphDestroy(m_graph);
            m_graph = nullptr;
        }
        return false;
    }
    
    if (!validateGraph()) {
        std::cerr << "[UnifiedGraph] Graph validation failed" << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }
    
    err = cudaGraphInstantiateWithFlags(&m_graphExec, m_graph, m_config.graphInstantiateFlags);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to instantiate graph: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }
    
    m_state.graphReady = true;
    m_state.needsRebuild = false;
    m_graphCaptured = true;  // Graph 罹≪쿂 ?꾨즺 ?뚮옒洹??ㅼ젙
    
    return true;
}

bool UnifiedGraphPipeline::updateGraphExec() {
    if (!m_graphExec || !m_graph) {
        std::cerr << "[UnifiedGraph] Cannot update: graph not instantiated" << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(m_graphMutex);
    auto& ctx = AppContext::getInstance();

    // Create a new graph with updated parameters
    cudaGraph_t newGraph = nullptr;
    if (!m_pipelineStream || !m_pipelineStream->get()) {
        std::cerr << "[UnifiedGraph] Pipeline stream not initialized for graph update" << std::endl;
        return false;
    }
    cudaStream_t stream = m_pipelineStream->get();

    if (!bindStaticTensorAddresses()) {
        return false;
    }

    // Begin capture to create updated graph topology
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to begin update capture: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Capture same operations as original graph
    if (m_unifiedArena.yoloInput && !m_captureBuffer.empty()) {
        int modelRes = getModelInputResolution();
        bool use_fp16 = (m_inputDataType == nvinfer1::DataType::kHALF);
        cuda_unified_preprocessing(
            m_captureBuffer.data(),
            m_unifiedArena.yoloInput,
            m_captureBuffer.cols(),
            m_captureBuffer.rows(),
            static_cast<int>(m_captureBuffer.step()),
            modelRes,
            modelRes,
            use_fp16,
            stream
        );

        ensurePrimaryInputBindingAliased();

        void* inputBinding = (m_primaryInputIndex >= 0 &&
                              m_primaryInputIndex < static_cast<int>(m_inputAddressCache.size()))
                                 ? m_inputAddressCache[m_primaryInputIndex]
                                 : nullptr;
        if (inputBinding && inputBinding != m_unifiedArena.yoloInput) {
            size_t inputSize = modelRes * modelRes * 3 * sizeof(float);
            cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize,
                           cudaMemcpyDeviceToDevice, stream);
        }
    }

    if (m_context && m_config.enableDetection) {
        if (!m_context->enqueueV3(stream)) {
            cudaStreamEndCapture(stream, &newGraph);
            if (newGraph) cudaGraphDestroy(newGraph);
            return false;
        }
    }

    if (m_config.enableDetection) {
        performIntegratedPostProcessing(stream);
        performTargetSelection(stream);

        if (!m_mouseMovementUsesMappedMemory) {
            cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement,
                           sizeof(MouseMovement), cudaMemcpyDeviceToHost, stream);
        }

        // Debug overlay D2H copies (same as captureGraph)
        if (m_h_targetCount && m_h_targetCount->get() && m_smallBufferArena.decodedCount) {
            cudaMemcpyAsync(m_h_targetCount->get(), m_smallBufferArena.decodedCount,
                           sizeof(int), cudaMemcpyDeviceToHost, stream);
        }
        if (m_h_targets && m_h_targets->get() && m_unifiedArena.decodedTargets) {
            cudaMemcpyAsync(m_h_targets->get(), m_unifiedArena.decodedTargets,
                           sizeof(Target) * MAX_HOST_TARGETS, cudaMemcpyDeviceToHost, stream);
        }

        FrameMetadata graphMeta{};  // Empty metadata for graph update
        if (!enqueueFrameCompletionCallback(stream, graphMeta)) {
            cudaStreamEndCapture(stream, &newGraph);
            if (newGraph) cudaGraphDestroy(newGraph);
            return false;
        }
    }

    err = cudaStreamEndCapture(stream, &newGraph);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to end update capture: "
                  << cudaGetErrorString(err) << std::endl;
        if (newGraph) cudaGraphDestroy(newGraph);
        return false;
    }

    // Try to update the existing executable
    cudaGraphExecUpdateResultInfo updateResultInfo{};
    err = cudaGraphExecUpdate(m_graphExec, newGraph, &updateResultInfo);

    if (err == cudaSuccess && updateResultInfo.result == cudaGraphExecUpdateSuccess) {
        // Update succeeded - just clean up the temporary graph
        cudaGraphDestroy(newGraph);
        std::cout << "[UnifiedGraph] Graph exec updated successfully (fast path)" << std::endl;
        m_state.needsRebuild = false;
        return true;
    }

    // Update failed or topology changed - need full reinstantiation
    if (updateResultInfo.result == cudaGraphExecUpdateErrorTopologyChanged) {
        std::cout << "[UnifiedGraph] Topology changed, performing full graph reinstantiation" << std::endl;
    } else if (updateResultInfo.result == cudaGraphExecUpdateErrorNodeTypeChanged) {
        std::cout << "[UnifiedGraph] Node type changed, performing full graph reinstantiation" << std::endl;
    } else if (updateResultInfo.result == cudaGraphExecUpdateErrorUnsupportedFunctionChange) {
        std::cout << "[UnifiedGraph] Function signature changed, performing full graph reinstantiation" << std::endl;
    } else {
        std::cerr << "[UnifiedGraph] Graph update failed: " << cudaGetErrorString(err)
                  << ", result=" << updateResultInfo.result << std::endl;
    }

    // Destroy old exec and reinstantiate with new graph
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }

    if (m_graph) {
        cudaGraphDestroy(m_graph);
    }

    m_graph = newGraph;

    err = cudaGraphInstantiateWithFlags(&m_graphExec, m_graph, m_config.graphInstantiateFlags);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to reinstantiate graph: "
                  << cudaGetErrorString(err) << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }

    std::cout << "[UnifiedGraph] Graph reinstantiated successfully (slow path)" << std::endl;
    m_state.needsRebuild = false;
    return true;
}





bool UnifiedGraphPipeline::validateGraph() {
    if (!m_graph) return false;
    
    size_t numNodes = 0;
    cudaError_t err = cudaGraphGetNodes(m_graph, nullptr, &numNodes);
    if (err != cudaSuccess || numNodes == 0) {
        std::cerr << "[UnifiedGraph] Graph has no nodes" << std::endl;
        return false;
    }
    
    return true;
}

void UnifiedGraphPipeline::cleanupGraph() {
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }

    if (m_graph) {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
    }

    if (m_updateGraph) {
        cudaGraphDestroy(m_updateGraph);
        m_updateGraph = nullptr;
    }

    m_state.graphReady = false;
}

bool UnifiedGraphPipeline::allocateBuffers() {
    auto& ctx = AppContext::getInstance();
    const int width = ctx.config.profile().detection_resolution;
    const int height = ctx.config.profile().detection_resolution;
    const int yoloSize = getModelInputResolution();
    const int maxDetections = ctx.config.profile().max_detections;
    
    // Determine pipeline precision based on TensorRT model input type
    PipelinePrecision precision = PipelinePrecision::FP32;
    if (m_inputDataType == nvinfer1::DataType::kHALF) {
        precision = PipelinePrecision::FP16;
        std::cout << "[UnifiedGraph] Using FP16 precision for entire pipeline (2x memory bandwidth savings)" << std::endl;
    } else if (m_inputDataType == nvinfer1::DataType::kFP8) {
        precision = PipelinePrecision::FP8;
        std::cout << "[UnifiedGraph] Using FP8 precision for entire pipeline (4x memory bandwidth savings)" << std::endl;
    } else if (m_inputDataType == nvinfer1::DataType::kINT8) {
        precision = PipelinePrecision::INT8;
        std::cout << "[UnifiedGraph] Using INT8 precision for entire pipeline" << std::endl;
    }
    
    try {
        m_captureBuffer.create(height, width, 4);

        size_t arenaSize = SmallBufferArena::calculateArenaSize();
        m_smallBufferArena.arenaBuffer = std::make_unique<CudaMemory<uint8_t>>(arenaSize);
        m_smallBufferArena.initializePointers(m_smallBufferArena.arenaBuffer->get());
        invalidateSelectedTarget(nullptr);

        size_t unifiedArenaSize = UnifiedGPUArena::calculateArenaSize(maxDetections, yoloSize, precision);
        m_unifiedArena.megaArena = std::make_unique<CudaMemory<uint8_t>>(unifiedArenaSize);
        m_unifiedArena.initializePointers(m_unifiedArena.megaArena->get(), maxDetections, yoloSize, precision);
        
        
        
        m_h_movement = std::make_unique<CudaPinnedMemory<MouseMovement>>(
            1, cudaHostAllocMapped | cudaHostAllocPortable);
        if (m_h_movement && m_h_movement->get()) {
            m_h_movement->get()->dx = 0;
            m_h_movement->get()->dy = 0;
        }
        m_mouseMovementUsesMappedMemory = configureMouseMovementBuffer();
        m_h_allowFlags = std::make_unique<CudaPinnedMemory<unsigned char>>(
            Constants::MAX_CLASSES_FOR_FILTERING);

        if (m_h_allowFlags && m_h_allowFlags->get()) {
            std::fill_n(m_h_allowFlags->get(),
                        Constants::MAX_CLASSES_FOR_FILTERING,
                        static_cast<unsigned char>(0));
        }
        
        // Allocate host pinned buffers for target data (debug overlay)
        m_h_targets = std::make_unique<CudaPinnedMemory<Target>>(MAX_HOST_TARGETS);
        m_h_targetCount = std::make_unique<CudaPinnedMemory<int>>(1);
        if (m_h_targetCount && m_h_targetCount->get()) {
            *m_h_targetCount->get() = 0;
        }

        // Defer aliasing until TensorRT bindings are created to avoid spurious warnings
        if (m_unifiedArena.yoloInput && !m_inputBindings.empty()) {
            (void)ensurePrimaryInputBindingAliased();
        }

        ensureFinalTargetAliases();

        // Initialize prev_class_filter to force first upload
        m_cachedConfig.detection.prev_class_filter.fill(0xFF);

        // Dynamic preview buffer allocation based on current state
        updatePreviewBufferAllocation();
        
        if (!m_captureBuffer.data()) {
            throw std::runtime_error("Capture buffer allocation failed");
        }
        
        size_t gpuMemory = (width * height * 4 + yoloSize * yoloSize * 3) * sizeof(float);
        gpuMemory += unifiedArenaSize;
        if (m_preview.enabled) {
            gpuMemory += width * height * 4 * sizeof(unsigned char);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedGraph] Buffer allocation failed: " << e.what() << std::endl;
        deallocateBuffers();
        return false;
    }
}

void UnifiedGraphPipeline::deallocateBuffers() {
    m_captureBuffer.release();

    // Always release preview buffer if allocated
    if (m_preview.hostPreviewPinned && m_preview.hostPreview.data()) {
        cudaHostUnregister(m_preview.hostPreview.data());
        m_preview.hostPreviewPinned = false;
        m_preview.hostPreviewPinnedSize = 0;
    }
    // Unpin and release both double-buffer host buffers
    for (int i = 0; i < 2; i++) {
        if (m_preview.hostPreviewPinnedFlags[i] && m_preview.hostPreviewBuffers[i].data()) {
            cudaHostUnregister(m_preview.hostPreviewBuffers[i].data());
            m_preview.hostPreviewPinnedFlags[i] = false;
            m_preview.hostPreviewPinnedSizes[i] = 0;
        }
        m_preview.hostPreviewBuffers[i].release();
    }
    m_preview.previewBuffer.release();
    m_preview.hostPreview.release();
    m_preview.finalTargets.clear();
    m_preview.enabled = false;
    m_preview.hasValidHostPreview = false;
    
    m_smallBufferArena.arenaBuffer.reset();
    m_smallBufferArena = SmallBufferArena{};

    m_unifiedArena.megaArena.reset();
    m_unifiedArena = UnifiedGPUArena{};

    m_d_inferenceOutput.reset();
    m_d_outputBuffer.reset();

    m_h_movement.reset();
    m_h_allowFlags.reset();
    m_mouseMovementUsesMappedMemory = false;

    m_inputBindings.clear();
    m_outputBindings.clear();
    m_inputAddressCache.clear();
    m_outputAddressCache.clear();
    m_bindingsNeedUpdate = true;
    m_primaryInputIndex = -1;

}

bool UnifiedGraphPipeline::ensurePrimaryInputBindingAliased() {
    if (m_primaryInputIndex < 0 ||
        m_primaryInputIndex >= static_cast<int>(m_inputAddressCache.size()) ||
        !m_unifiedArena.yoloInput) {
        return false;
    }

    void* currentBinding = m_inputAddressCache[m_primaryInputIndex];
    if (currentBinding == m_unifiedArena.yoloInput) {
        return true;
    }

    auto bindingIt = m_inputBindings.find(m_inputName);
    auto sizeIt = m_inputSizes.find(m_inputName);
    if (bindingIt == m_inputBindings.end() ||
        sizeIt == m_inputSizes.end() ||
        sizeIt->second == 0) {
        return false;
    }

    try {
        bindingIt->second = std::make_unique<CudaMemory<uint8_t>>(
            reinterpret_cast<uint8_t*>(m_unifiedArena.yoloInput),
            sizeIt->second,
            false);
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedGraph] Failed to alias TensorRT input binding: "
                  << e.what() << std::endl;
        return false;
    }

    refreshCachedBindings();

    if (m_primaryInputIndex < static_cast<int>(m_inputAddressCache.size()) &&
        m_inputAddressCache[m_primaryInputIndex] == m_unifiedArena.yoloInput) {
        m_bindingsNeedUpdate = true;
        return true;
    }

    return false;
}

void UnifiedGraphPipeline::ensureFinalTargetAliases() {
    bool updated = false;
    if (m_unifiedArena.decodedTargets &&
        m_unifiedArena.finalTargets != m_unifiedArena.decodedTargets) {
        m_unifiedArena.finalTargets = m_unifiedArena.decodedTargets;
        updated = true;
    }

    if (m_smallBufferArena.decodedCount &&
        m_smallBufferArena.finalTargetsCount != m_smallBufferArena.decodedCount) {
        m_smallBufferArena.finalTargetsCount = m_smallBufferArena.decodedCount;
        updated = true;
    }

    if (updated) {
        std::cerr << "[UnifiedGraph] Realigned post-processing buffers to decoded targets" << std::endl;
    }

}

bool UnifiedGraphPipeline::configureMouseMovementBuffer() {
    if (!m_smallBufferArena.mouseMovement) {
        return false;
    }

    if (!m_h_movement || !m_h_movement->get()) {
        return false;
    }

    MouseMovement* mappedPtr = nullptr;
    cudaError_t mapErr = cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&mappedPtr),
        m_h_movement->get(),
        0);

    if (mapErr == cudaSuccess && mappedPtr) {
        m_smallBufferArena.mouseMovement = mappedPtr;
        return true;
    }

    if (mapErr != cudaErrorInvalidValue && mapErr != cudaErrorNotSupported) {
        std::cerr << "[UnifiedGraph] Failed to map mouse movement buffer: "
                  << cudaGetErrorString(mapErr) << std::endl;
    }

    return false;
}


void UnifiedGraphPipeline::handleAimbotDeactivation() {
    auto& ctx = AppContext::getInstance();

    clearCountBuffers();
    clearMovementData();
    clearHostPreviewData(ctx);
    m_allowMovement.store(false, std::memory_order_release);

    // Reset capture region cache
    m_captureRegionCache = {};

    // Reset frame tracking
    m_lastProcessedPresentQpc.store(0, std::memory_order_release);

    // Reset filter state (lock-free)
    m_filterState.skipNext = true;
    m_filterState.inSettleX = false;
    m_filterState.inSettleY = false;
    m_filterState.lastEmitX = 0;
    m_filterState.lastEmitY = 0;
}

void UnifiedGraphPipeline::clearCountBuffers() {
    cudaStream_t stream = m_pipelineStream ? m_pipelineStream->get() : nullptr;

    // Use batched kernel instead of multiple cudaMemsetAsync calls
    batchedMemsetKernel<<<1, 1, 0, stream>>>(
        m_smallBufferArena.finalTargetsCount,
        (m_smallBufferArena.decodedCount != m_smallBufferArena.finalTargetsCount)
            ? m_smallBufferArena.decodedCount : nullptr,
        m_smallBufferArena.classFilteredCount,
        m_smallBufferArena.bestTargetIndex,
        0, 0, 0, -1);

    invalidateSelectedTarget(stream);
}

void UnifiedGraphPipeline::clearMovementData() {
    if (m_h_movement && m_h_movement->get()) {
        m_h_movement->get()->dx = 0;
        m_h_movement->get()->dy = 0;
    }

    if (m_smallBufferArena.mouseMovement && !m_mouseMovementUsesMappedMemory) {
        cudaError_t resetErr = cudaMemset(m_smallBufferArena.mouseMovement, 0, sizeof(MouseMovement));
        if (resetErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Failed to reset device movement buffer: "
                      << cudaGetErrorString(resetErr) << std::endl;
        }
    }

    // Reset PID state when aimbot is deactivated
    if (m_smallBufferArena.pidState) {
        cudaError_t pidResetErr = cudaMemset(m_smallBufferArena.pidState, 0, sizeof(PIDState));
        if (pidResetErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Failed to reset PID state buffer: "
                      << cudaGetErrorString(pidResetErr) << std::endl;
        }
    }

    invalidateSelectedTarget(m_pipelineStream ? m_pipelineStream->get() : nullptr);
    resetMovementFilter();
}

void UnifiedGraphPipeline::resetMovementFilter() {
    // Lock-free: direct member access
    m_filterState.skipNext = true;
    m_lastFrameTime = {};
    m_filterState.inSettleX = false;
    m_filterState.inSettleY = false;
    m_filterState.lastEmitX = 0;
    m_filterState.lastEmitY = 0;
}

void UnifiedGraphPipeline::invalidateSelectedTarget(cudaStream_t stream) {
    if (!m_smallBufferArena.selectedTarget) {
        return;
    }

    Target invalidTarget{};
    invalidTarget.classId = -1;
    invalidTarget.x = -1;
    invalidTarget.y = -1;
    invalidTarget.width = -1;
    invalidTarget.height = -1;
    invalidTarget.confidence = 0.0f;

    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(
            m_smallBufferArena.selectedTarget,
            &invalidTarget,
            sizeof(Target),
            cudaMemcpyHostToDevice,
            stream);
    } else {
        err = cudaMemcpy(
            m_smallBufferArena.selectedTarget,
            &invalidTarget,
            sizeof(Target),
            cudaMemcpyHostToDevice);
    }

    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to invalidate selected target: "
                  << cudaGetErrorString(err) << std::endl;
    }
}

MouseMovement UnifiedGraphPipeline::filterMouseMovement(const MouseMovement& raw, bool enabled) {
    if (!enabled) {
        m_filterState.skipNext = true;
        m_filterState.inSettleX = false;
        m_filterState.inSettleY = false;
        m_filterState.lastEmitX = 0;
        m_filterState.lastEmitY = 0;
        return {0, 0};
    }

    if (m_filterState.skipNext) {
        if (raw.dx != 0 || raw.dy != 0) {
            m_filterState.skipNext = false;
            return {0, 0};
        }
        return raw;
    }

    // Read cached filter config (NO LOCKS, no atomic loads)
    const auto& cfg = m_cachedConfig.filtering;

    int dx = raw.dx;
    int dy = raw.dy;

    // Block upward movement if disable_upward_aim is active (cached value)
    if (cfg.disable_upward_aim && dy < 0) {
        dy = 0;
    }

    // X-axis: sign-flip suppression
    int emitDx = dx;
    if (abs(dx) <= 1 && dx == -m_filterState.lastEmitX) {
        emitDx = 0;
    }

    // Y-axis: sign-flip suppression
    int emitDy = dy;
    if (abs(dy) <= 1 && dy == -m_filterState.lastEmitY) {
        emitDy = 0;
    }

    // X-axis hysteresis
    {
        const int enter2 = cfg.deadband_enter_x * cfg.deadband_enter_x;
        const int exit2  = cfg.deadband_exit_x * cfg.deadband_exit_x;
        const int mag2   = emitDx * emitDx;

        if (m_filterState.inSettleX) {
            if (mag2 < exit2) {
                emitDx = 0;
            } else {
                m_filterState.inSettleX = false;
            }
        } else {
            if (mag2 <= enter2) {
                m_filterState.inSettleX = true;
                emitDx = 0;
            }
        }
    }

    // Y-axis hysteresis
    {
        const int enter2 = cfg.deadband_enter_y * cfg.deadband_enter_y;
        const int exit2  = cfg.deadband_exit_y * cfg.deadband_exit_y;
        const int mag2   = emitDy * emitDy;

        if (m_filterState.inSettleY) {
            if (mag2 < exit2) {
                emitDy = 0;
            } else {
                m_filterState.inSettleY = false;
            }
        } else {
            if (mag2 <= enter2) {
                m_filterState.inSettleY = true;
                emitDy = 0;
            }
        }
    }

    // Update last emit (cache-local, no atomics)
    m_filterState.lastEmitX = (emitDx != 0) ? emitDx : 0;
    m_filterState.lastEmitY = (emitDy != 0) ? emitDy : 0;

    return {emitDx, emitDy};
}

void UnifiedGraphPipeline::clearHostPreviewData(AppContext& ctx) {
    if (m_preview.enabled) {
        m_preview.finalTargets.clear();
        m_preview.finalCount = 0;
        m_preview.copyInProgress = false;
        m_preview.hostPreview.release();
        m_preview.hasValidHostPreview = false;
    }

    ctx.clearTargets();
}

void UnifiedGraphPipeline::handleAimbotActivation() {
    m_allowMovement.store(false, std::memory_order_release);
    clearMovementData();

    // Reset frame tracking (synchronous model)
    m_nextFrameId.store(0, std::memory_order_release);
    m_lastProcessedPresentQpc.store(0, std::memory_order_release);
    m_pendingInputQpc.store(0, std::memory_order_release);

    // Reset capture region cache
    m_captureRegionCache = {};

    // Reset movement filter state (lock-free)
    m_filterState.skipNext = true;
    m_filterState.inSettleX = false;
    m_filterState.inSettleY = false;
    m_filterState.lastEmitX = 0;
    m_filterState.lastEmitY = 0;

    // CUDA Graph: check if existing graph is still valid.
    // If topology changed since last session, invalidate for re-warmup.
    // Otherwise keep the graph to avoid 5-frame warmup on quick re-activations.
    if (m_graphCaptured && graphTopologyChanged()) {
        cleanupGraph();
        m_graphCaptured = false;
    }
    // Reset frame count: if graph is valid it will be used immediately;
    // if invalidated, auto-capture triggers after 5 warmup frames.
    m_state.frameCount = 0;
}

// Frame-ID aware completion callback that updates lastProcessed metadata
// and enforces exactly-once semantics per PresentQpc.
bool UnifiedGraphPipeline::enqueueFrameCompletionCallback(cudaStream_t stream, const FrameMetadata& metadata) {
    if (!stream) {
        return false;
    }

    // Use pre-allocated member instead of heap alloc (safe: m_frameInFlight guards single-flight)
    m_callbackData = {this, metadata};

    cudaError_t err = cudaLaunchHostFunc(stream,
        [](void* userData) {
            auto* cbData = static_cast<CallbackData*>(userData);
            auto* pipeline = cbData->pipeline;
            const FrameMetadata& meta = cbData->metadata;

            auto& ctx = AppContext::getInstance();

            bool allowMovement = pipeline->m_allowMovement.load(std::memory_order_acquire);
            pipeline->m_allowMovement.store(false, std::memory_order_release);

            if (pipeline->m_h_movement && pipeline->m_h_movement->get()) {
                bool isSingleShot = ctx.single_shot_mode.load();
                bool shouldMove = allowMovement && (ctx.aiming || isSingleShot);

                if (shouldMove) {
                    MouseMovement rawMovement = *pipeline->m_h_movement->get();
                    MouseMovement filtered = pipeline->filterMouseMovement(rawMovement, true);

                    // === OPTIMIZATION 3: Update deadband state ===
                    // Track if target is in deadband (raw movement exists but filtered to zero)
                    bool wasMovementRequested = (rawMovement.dx != 0 || rawMovement.dy != 0);
                    bool wasFilteredToZero = (filtered.dx == 0 && filtered.dy == 0);
                    pipeline->m_adaptiveSkip.lastWasInDeadband = wasMovementRequested && wasFilteredToZero;

                    if (filtered.dx != 0 || filtered.dy != 0) {
                        // Apply Gaussian noise for humanization
                        float moveX = static_cast<float>(filtered.dx);
                        float moveY = static_cast<float>(filtered.dy);

                        if (ctx.config.profile().noise_enabled) {
                            float stddevX = ctx.config.profile().noise_stddev_x;
                            float stddevY = ctx.config.profile().noise_stddev_y;
                            moveX += g_noiseDistX(g_noiseGen) * stddevX;
                            moveY += g_noiseDistY(g_noiseGen) * stddevY;
                        }

                        int finalX = static_cast<int>(std::round(moveX));
                        int finalY = static_cast<int>(std::round(moveY));

                        if (finalX != 0 || finalY != 0) {
                            executeMouseMovement(finalX, finalY);
                        }

                        // Record input timestamp for latency tracking
                        LARGE_INTEGER qpc{};
                        if (QueryPerformanceCounter(&qpc)) {
                            pipeline->m_pendingInputQpc.store(
                                static_cast<uint64_t>(qpc.QuadPart),
                                std::memory_order_release);
                        }
                    }

                    // Mark this frame as processed (exactly-once guarantee)
                    pipeline->m_lastProcessedPresentQpc.store(meta.presentQpc, std::memory_order_release);

                    // Clear pending input flag if this frame includes our input
                    uint64_t pendingQpc = pipeline->m_pendingInputQpc.load(std::memory_order_acquire);
                    if (pendingQpc != 0 && meta.presentQpc >= pendingQpc) {
                        pipeline->m_pendingInputQpc.store(0, std::memory_order_release);
                    }
                } else {
                    pipeline->filterMouseMovement({0, 0}, false);
                    pipeline->m_adaptiveSkip.lastWasInDeadband = false;  // Reset when not aiming
                }

                if (isSingleShot) {
                    ctx.single_shot_mode = false;
                }
            }

            // Update AppContext with detected targets for debug overlay
            if (pipeline->m_h_targetCount && pipeline->m_h_targetCount->get() &&
                pipeline->m_h_targets && pipeline->m_h_targets->get()) {
                int targetCount = *pipeline->m_h_targetCount->get();
                if (targetCount > 0) {
                    targetCount = (targetCount > MAX_HOST_TARGETS) ? MAX_HOST_TARGETS : targetCount;
                    std::vector<Target> targets(pipeline->m_h_targets->get(),
                                                 pipeline->m_h_targets->get() + targetCount);
                    ctx.updateTargets(targets);
                } else {
                    ctx.clearTargets();
                }
            }

            // Release frame-in-flight lock
            pipeline->m_frameInFlight.store(false, std::memory_order_release);
        },
        &m_callbackData);

    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to enqueue completion callback: "
                  << cudaGetErrorString(err) << std::endl;
        m_allowMovement.store(false, std::memory_order_release);
        return false;
    }

    return true;
}

// ============================================================================
// MAIN LOOP - Synchronous Feedback Loop
//
// This pipeline maintains a strict serial feedback loop:
//   Capture → Preprocess → Infer → PostProcess → TargetSelect → MouseMove → (repeat)
//
// Each iteration of the inner while-loop calls executeFrame() which BLOCKS
// until the entire pipeline (including the mouse movement callback) completes.
// m_frameInFlight guarantees exactly one frame is in-flight at any time.
// The next capture only begins AFTER the previous frame's feedback (mouse move)
// has been fully applied, ensuring closed-loop control.
// ============================================================================
void UnifiedGraphPipeline::runMainLoop() {
    auto& ctx = AppContext::getInstance();

    // Raise priority to TIME_CRITICAL for minimal wake-up jitter
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    // Pin to last core to avoid Core 0 (handles OS interrupts/DPCs)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    DWORD lastCore = sysInfo.dwNumberOfProcessors - 1;
    DWORD_PTR affinityMask = 1ULL << lastCore;
    SetThreadAffinityMask(GetCurrentThread(), affinityMask);

    bool wasAiming = false;

    while (!m_shouldStop.load(std::memory_order_acquire) && !ctx.should_exit.load()) {
        // Wait for activation
        {
            std::unique_lock<std::mutex> lock(ctx.pipeline_activation_mutex);
            ctx.pipeline_activation_cv.wait(lock, [&ctx, this]() {
                return ctx.aiming.load() || ctx.single_shot_requested.load() ||
                       m_shouldStop.load(std::memory_order_acquire) ||
                       ctx.should_exit.load();
            });
        }

        if (m_shouldStop.load(std::memory_order_acquire) || ctx.should_exit.load()) {
            break;
        }

        // Single shot mode
        if (ctx.single_shot_requested.load()) {
            ctx.single_shot_requested = false;
            ctx.single_shot_mode = true;

            // Synchronous: execute one frame (acquires frame internally)
            executeFrame(nullptr);

            continue;
        }

        if (!ctx.aiming.load()) {
            if (wasAiming) {
                handleAimbotDeactivation();
                wasAiming = false;
            }
            continue;
        }

        if (!wasAiming) {
            handleAimbotActivation();
            wasAiming = true;
        }

        // Main loop: synchronous capture + processing
        while (ctx.aiming.load() && !m_shouldStop.load(std::memory_order_acquire) &&
               !ctx.should_exit.load()) {

            // Synchronous: execute frame (acquires, preprocesses, infers, outputs)
            if (!executeFrame(nullptr)) {
                // Critical error - exit loop
                break;
            }

            // Change-detection based config update (more efficient than frame-count based)
            // Only refresh when config actually changes, detected via generation counter
            // generation is only incremented by markPidConfigDirty(), not by refresh itself
            uint32_t currentGen = m_cachedConfig.generation.load(std::memory_order_acquire);
            if (currentGen != m_lastConfigGeneration) {
                updateConfig(ctx);
                m_lastConfigGeneration = currentGen;
            }

            // ================================================================
            // CUDA Graph lifecycle management
            // After config update, check topology for graph invalidation/capture.
            // ================================================================
            if (m_config.useGraphOptimization) {
                // Topology change → invalidate graph and reset for re-warmup
                if (m_graphCaptured && graphTopologyChanged()) {
                    if (m_pipelineStream) cudaStreamSynchronize(m_pipelineStream->get());
                    cleanupGraph();
                    m_graphCaptured = false;
                    m_state.frameCount = 0;
                }

                // Auto-capture after warmup frames (5 frames to stabilize TensorRT)
                if (!m_graphCaptured && m_state.frameCount >= 5) {
                    if (m_pipelineStream) cudaStreamSynchronize(m_pipelineStream->get());
                    if (captureGraph()) {
                        snapshotGraphTopology();
                    }
                }
            }

            // Smart yield: only sleep if delay is configured, otherwise rely on
            // frame-in-flight blocking which is more efficient
            int delay = ctx.config.profile().pipeline_loop_delay_ms;
            if (delay > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            }
        }

        if (wasAiming && !ctx.aiming.load()) {
            handleAimbotDeactivation();
            wasAiming = false;
        }
    }

    if (wasAiming) {
        handleAimbotDeactivation();
    }
}


void UnifiedGraphPipeline::stopMainLoop() {
    m_shouldStop.store(true, std::memory_order_release);

    auto& ctx = AppContext::getInstance();
    ctx.pipeline_activation_cv.notify_all();
}


bool UnifiedGraphPipeline::initializeTensorRT(const std::string& modelFile) {
    auto& ctx = AppContext::getInstance();

    if (!loadEngine(modelFile)) {
        std::cerr << "[Pipeline] Failed to load engine" << std::endl;
        return false;
    }
    
    
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "[Pipeline] Failed to create execution context" << std::endl;
        return false;
    }
    
    if (m_engine->getNbOptimizationProfiles() > 0 && m_pipelineStream && m_pipelineStream->get()) {
        m_context->setOptimizationProfileAsync(0, m_pipelineStream->get());
    }
    
    
    getInputNames();
    getOutputNames();
    
    if (!m_inputNames.empty()) {
        m_inputName = m_inputNames[0];
        m_primaryInputIndex = 0;
        m_inputDims = m_engine->getTensorShape(m_inputName.c_str());
        m_inputDataType = m_engine->getTensorDataType(m_inputName.c_str());

        if (m_inputDims.nbDims == 4) {
            m_modelInputResolution = m_inputDims.d[2];
        } else if (m_inputDims.nbDims == 3) {
            m_modelInputResolution = m_inputDims.d[1];
        }

        size_t inputSize = 1;
        for (int i = 0; i < m_inputDims.nbDims; ++i) {
            inputSize *= m_inputDims.d[i];
        }

        // Use correct element size based on actual data type
        size_t elementSize = sizeof(float); // default FP32
        if (m_inputDataType == nvinfer1::DataType::kHALF) {
            elementSize = sizeof(__half); // FP16 = 2 bytes
        } else if (m_inputDataType == nvinfer1::DataType::kINT8) {
            elementSize = 1;
        }

        inputSize *= elementSize;
        m_inputSizes[m_inputName] = inputSize;
    }
    
    for (const auto& outputName : m_outputNames) {
        nvinfer1::Dims outputDims = m_engine->getTensorShape(outputName.c_str());
        nvinfer1::DataType outputDataType = m_engine->getTensorDataType(outputName.c_str());

        size_t outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; ++i) {
            outputSize *= outputDims.d[i];
        }

        // Use correct element size based on actual data type
        size_t elementSize = sizeof(float); // default FP32
        if (outputDataType == nvinfer1::DataType::kHALF) {
            elementSize = sizeof(__half); // FP16 = 2 bytes
        } else if (outputDataType == nvinfer1::DataType::kINT8) {
            elementSize = 1;
        }

        outputSize *= elementSize;
        m_outputSizes[outputName] = outputSize;
    }
    
    // Defer getBindings() until after arena allocation so we can alias
    // the primary input to the unified arena and avoid duplicate memory.
    
    m_imgScale = static_cast<float>(ctx.config.profile().detection_resolution) / getModelInputResolution();
    
    
    const auto& outputShape = m_outputShapes[m_outputNames[0]];
    if (outputShape.size() >= 2) {
        m_numClasses = static_cast<int>(outputShape[1]) - 4;
    } else {
        m_numClasses = 80;
    }

    // NOTE: TensorRT warmup is performed in initialize() AFTER allocateBuffers() and getBindings()
    // to ensure all tensor addresses are properly bound before inference.

    return true;
}

void UnifiedGraphPipeline::getInputNames() {
    auto& ctx = AppContext::getInstance();
    m_inputNames.clear();
    m_inputSizes.clear();

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            m_inputNames.emplace_back(name);
        }
    }
}

void UnifiedGraphPipeline::getOutputNames() {
    auto& ctx = AppContext::getInstance();
    m_outputNames.clear();
    m_outputSizes.clear();
    m_outputShapes.clear();
    m_outputTypes.clear();

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            m_outputNames.emplace_back(name);

            auto dims = m_engine->getTensorShape(name);
            std::vector<int64_t> shape;
            for (int j = 0; j < dims.nbDims; ++j) {
                shape.push_back(dims.d[j]);
            }
            m_outputShapes[name] = shape;

            auto dataType = m_engine->getTensorDataType(name);
            m_outputTypes[name] = dataType;
        }
    }
}

void UnifiedGraphPipeline::getBindings() {
    m_inputBindings.clear();
    m_outputBindings.clear();

    for (const auto& name : m_inputNames) {
        size_t size = m_inputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for input '" << name << "'" << std::endl;
            continue;
        }
        
        try {
            // ?쇰떒 ?먮옒?濡?蹂듦뎄 - 吏곸젒 ?ъ씤???ъ슜? TensorRT 諛붿씤?⑷낵 ?명솚??臾몄젣 ?덉쓬
            m_inputBindings[name] = (name == m_inputName && m_unifiedArena.yoloInput)
                ? std::unique_ptr<CudaMemory<uint8_t>>(new CudaMemory<uint8_t>(
                      reinterpret_cast<uint8_t*>(m_unifiedArena.yoloInput),
                      size,
                      false))
                : std::unique_ptr<CudaMemory<uint8_t>>(new CudaMemory<uint8_t>(size));
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Failed to allocate input memory for '" << name << "': " << e.what() << std::endl;
            throw;
        }
    }

    for (const auto& name : m_outputNames) {
        size_t size = m_outputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for output '" << name << "'" << std::endl;
            continue;
        }
        
        try {
            // Simplified - same allocation for all outputs
            m_outputBindings[name] = std::make_unique<CudaMemory<uint8_t>>(size);
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Failed to allocate output memory for '" << name << "': " << e.what() << std::endl;
            throw;
        }
    }
    
    
    if (m_inputBindings.size() != m_inputNames.size()) {
        std::cerr << "[Pipeline] Warning: Input binding count mismatch" << std::endl;
    }
    if (m_outputBindings.size() != m_outputNames.size()) {
        std::cerr << "[Pipeline] Warning: Output binding count mismatch" << std::endl;
    }

    for (size_t i = 0; i < m_inputNames.size(); ++i) {
        if (m_inputNames[i] == m_inputName) {
            m_primaryInputIndex = static_cast<int>(i);
            break;
        }
    }

    refreshCachedBindings();

}

void UnifiedGraphPipeline::refreshCachedBindings() {
    m_inputAddressCache.assign(m_inputNames.size(), nullptr);
    for (size_t i = 0; i < m_inputNames.size(); ++i) {
        const auto& name = m_inputNames[i];
        auto bindingIt = m_inputBindings.find(name);
        if (bindingIt != m_inputBindings.end() && bindingIt->second) {
            m_inputAddressCache[i] = bindingIt->second->get();
        }
    }

    m_outputAddressCache.assign(m_outputNames.size(), nullptr);
    for (size_t i = 0; i < m_outputNames.size(); ++i) {
        const auto& name = m_outputNames[i];
        auto bindingIt = m_outputBindings.find(name);
        if (bindingIt != m_outputBindings.end() && bindingIt->second) {
            m_outputAddressCache[i] = bindingIt->second->get();
        }
    }

    m_bindingsNeedUpdate = true;
}

bool UnifiedGraphPipeline::bindStaticTensorAddresses() {
    if (!m_bindingsNeedUpdate) {
        return true;
    }

    if (!m_context) {
        return false;
    }

    for (size_t i = 0; i < m_inputNames.size(); ++i) {
        void* address = m_inputAddressCache[i];
        if (!address) {
            std::cerr << "[Pipeline] Input binding address missing for: " << m_inputNames[i] << std::endl;
            return false;
        }

        if (!m_context->setTensorAddress(m_inputNames[i].c_str(), address)) {
            std::cerr << "[Pipeline] Failed to bind input tensor: " << m_inputNames[i] << std::endl;
            return false;
        }
    }

    for (size_t i = 0; i < m_outputNames.size(); ++i) {
        void* address = m_outputAddressCache[i];
        if (!address) {
            std::cerr << "[Pipeline] Output binding address missing for: " << m_outputNames[i] << std::endl;
            return false;
        }

        if (!m_context->setTensorAddress(m_outputNames[i].c_str(), address)) {
            std::cerr << "[Pipeline] Failed to bind output tensor: " << m_outputNames[i] << std::endl;
            return false;
        }
    }

    m_bindingsNeedUpdate = false;
    return true;
}


bool UnifiedGraphPipeline::loadEngine(const std::string& modelFile) {
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();
    

    if (extension != ".engine") {
        std::cerr << "[Pipeline] Error: Only .engine files are supported. Please use EngineExport tool to convert ONNX to engine format." << std::endl;
        return false;
    }

    if (!fileExists(modelFile)) {
        std::cerr << "[Pipeline] Engine file does not exist: " << modelFile << std::endl;
        return false;
    }

    std::string engineFilePath = modelFile;

    class SimpleLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // Always log errors and warnings for debugging engine load issues
            if (severity <= Severity::kWARNING && 
                (strstr(msg, "defaultAllocator.cpp") == nullptr) &&
                (strstr(msg, "enqueueV3") == nullptr)) {
                std::cerr << "[TensorRT] " << msg << std::endl;
            }
        }
    };
    static SimpleLogger logger;

    std::ifstream file(engineFilePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Pipeline] Failed to open engine file: " << engineFilePath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    if (!m_runtime) {
        m_runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!m_runtime) {
            std::cerr << "[Pipeline] Failed to create runtime" << std::endl;
            return false;
        }
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(buffer.data(), size));
    
    if (m_engine) {
        return true;
    } else {
        std::cerr << "[Pipeline] Failed to load engine from: " << engineFilePath << std::endl;
        return false;
    }
}

int UnifiedGraphPipeline::getModelInputResolution() const {
    return m_modelInputResolution;
}

bool UnifiedGraphPipeline::runInferenceAsync(cudaStream_t stream) {
    if (!m_context || !m_engine) {
        std::cerr << "[Pipeline] TensorRT context or engine not initialized" << std::endl;
        return false;
    }
    
    if (!stream) {
        if (!m_pipelineStream) {
            std::cerr << "[Pipeline] Pipeline stream not initialized" << std::endl;
            return false;
        }
        stream = m_pipelineStream->get();
    }
    
    if (!bindStaticTensorAddresses()) {
        return false;
    }

    bool success = m_context->enqueueV3(stream);

    if (!success) {
        cudaError_t cudaErr = cudaGetLastError();
        std::cerr << "[Pipeline] TensorRT inference failed";
        if (cudaErr != cudaSuccess) {
            std::cerr << " - CUDA error: " << cudaGetErrorString(cudaErr);
        }
        std::cerr << std::endl;


        return false;
    }

    return true;
}

void gpa::PostProcessingConfig::updateFromContext(const AppContext& ctx, bool graphCaptured) {
    // Use lock-free reads - these are safe for reading primitive types
    // No mutex needed for hot path performance
    if (!graphCaptured) {
        max_detections = ctx.config.profile().max_detections;
        confidence_threshold = ctx.config.profile().confidence_threshold;
        postprocess = ctx.config.profile().postprocess;
    }
}

void UnifiedGraphPipeline::clearDetectionBuffers(const PostProcessingConfig& config, cudaStream_t stream) {
    if (!m_smallBufferArena.decodedCount || config.max_detections <= 0) {
        return;
    }

    // Use batched kernel instead of multiple cudaMemsetAsync calls
    batchedDetectionClearKernel<<<1, 1, 0, stream>>>(
        m_smallBufferArena.decodedCount,
        (m_smallBufferArena.finalTargetsCount != m_smallBufferArena.decodedCount)
            ? m_smallBufferArena.finalTargetsCount : nullptr,
        m_smallBufferArena.classFilteredCount,
        m_smallBufferArena.bestTargetIndex,
        m_smallBufferArena.bestTarget);

    invalidateSelectedTarget(stream);
}

cudaError_t UnifiedGraphPipeline::decodeYoloOutput(void* d_rawOutputPtr, nvinfer1::DataType outputType, 
                                                   const std::vector<int64_t>& shape, 
                                                   const PostProcessingConfig& config, cudaStream_t stream) {
    int maxDecodedTargets = config.max_detections;
    
    if (config.postprocess == "yolo_nms") {
        // Output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
        int num_detections = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        int output_features = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        if (output_features != 6) {
            std::cerr << "[Pipeline] Invalid output format. Expected 6 features [x1,y1,x2,y2,conf,class], got " << output_features << std::endl;
            return cudaErrorInvalidValue;
        }
        
        // Use existing decodeYolo10Gpu which handles pre-processed format
        return decodeYolo10Gpu(
            d_rawOutputPtr, outputType, shape, m_numClasses,
            config.confidence_threshold, m_imgScale,
            m_unifiedArena.decodedTargets, m_smallBufferArena.decodedCount,
            maxDecodedTargets, num_detections,
            m_smallBufferArena.allowFlags, m_numClasses, stream);
    } else if (config.postprocess == "yolo10") {
        int max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        
        return decodeYolo10Gpu(
            d_rawOutputPtr, outputType, shape, m_numClasses,
            config.confidence_threshold, m_imgScale,
            m_unifiedArena.decodedTargets, m_smallBufferArena.decodedCount,
            maxDecodedTargets, max_candidates,
            m_smallBufferArena.allowFlags, m_numClasses, stream);
    } else if (config.postprocess == "yolo8" || config.postprocess == "yolo9" || 
               config.postprocess == "yolo11" || config.postprocess == "yolo12") {
        int max_candidates = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        if (!validateYoloDecodeBuffers(maxDecodedTargets, max_candidates)) {
            return cudaErrorInvalidValue;
        }
        
        return decodeYolo11Gpu(
            d_rawOutputPtr, outputType, shape, m_numClasses,
            config.confidence_threshold, m_imgScale,
            m_unifiedArena.decodedTargets, m_smallBufferArena.decodedCount,
            maxDecodedTargets, max_candidates,
            m_smallBufferArena.allowFlags,
            Constants::MAX_CLASSES_FOR_FILTERING, stream);
    }
    
    std::cerr << "[Pipeline] Unsupported post-processing type: " << config.postprocess << std::endl;
    return cudaErrorNotSupported;
}

bool UnifiedGraphPipeline::validateYoloDecodeBuffers(int maxDecodedTargets, int max_candidates) {
    if (!m_unifiedArena.decodedTargets || !m_smallBufferArena.decodedCount) {
        std::cerr << "[Pipeline] Target buffers not allocated!" << std::endl;
        return false;
    }
    
    if (max_candidates <= 0 || maxDecodedTargets <= 0) {
        std::cerr << "[Pipeline] Invalid buffer sizes: max_candidates=" << max_candidates 
                  << ", maxDecodedTargets=" << maxDecodedTargets << std::endl;
        return false;
    }
    
    return true;
}

void UnifiedGraphPipeline::performIntegratedPostProcessing(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    if (m_outputNames.empty()) {
        std::cerr << "[Pipeline] No output names found for post-processing." << std::endl;
        return;
    }

    const std::string& primaryOutputName = m_outputNames[0];
    void* d_rawOutputPtr = m_outputAddressCache.empty() ? nullptr : m_outputAddressCache[0];
    nvinfer1::DataType outputType = m_outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = m_outputShapes[primaryOutputName];

    if (!d_rawOutputPtr) {
        std::cerr << "[Pipeline] Raw output GPU pointer is null for " << primaryOutputName << std::endl;
        return;
    }

    // Use member variable instead of static for thread safety
    m_postProcessConfig.updateFromContext(ctx, m_graphCaptured);
    
    clearDetectionBuffers(m_postProcessConfig, stream);
    
    cudaError_t decodeErr = decodeYoloOutput(d_rawOutputPtr, outputType, shape, m_postProcessConfig, stream);
    if (decodeErr != cudaSuccess) {
        std::cerr << "[Pipeline] GPU decoding failed: " << cudaGetErrorString(decodeErr) << std::endl;
        return;
    }
    
    // Ensure aliases point to decoded buffer before NMS (reset from previous frame)
    ensureFinalTargetAliases();

    // Perform NMS if enabled — compact to nmsTemp, then swap pointer
    if (m_cachedConfig.detection.enable_nms &&
        m_unifiedArena.decodedTargets &&
        m_unifiedArena.nmsTemp &&
        m_smallBufferArena.decodedCount &&
        m_smallBufferArena.keepFlags) {

        // Optimized: NMS on original → compact to nmsTemp (3 ops, no initial D2D copy)
        cudaError_t nmsErr = performNmsCompactGpu(
            m_unifiedArena.decodedTargets,
            m_smallBufferArena.decodedCount,
            m_smallBufferArena.keepFlags,
            m_unifiedArena.nmsTemp,
            m_smallBufferArena.decodedCount,  // reuse count — reset + atomicAdd inside
            m_cachedConfig.detection.nms_iou_threshold,
            m_cachedConfig.detection.max_detections,
            stream);

        if (nmsErr == cudaSuccess) {
            // Downstream reads from nmsTemp; next frame ensureFinalTargetAliases() restores
            m_unifiedArena.finalTargets = m_unifiedArena.nmsTemp;
        } else {
            std::cerr << "[Pipeline] NMS failed: " << cudaGetErrorString(nmsErr) << std::endl;
        }
    }
}

void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    if (!m_unifiedArena.finalTargets || !m_smallBufferArena.finalTargetsCount) {
        std::cerr << "[Pipeline] No final targets available for selection" << std::endl;
        return;
    }

    if (!m_smallBufferArena.bestTargetIndex || !m_smallBufferArena.bestTarget || !m_smallBufferArena.mouseMovement || !m_smallBufferArena.pidState) {
        std::cerr << "[Pipeline] Target selection buffers not allocated!" << std::endl;
        return;
    }

    // Use cached values (lock-free) - NO AppContext access in hot path
    const CachedConfig& cfg = m_cachedConfig;

    int cached_max_detections = cfg.detection.max_detections;

#ifdef _DEBUG
    cudaError_t staleError = cudaGetLastError();
    if (staleError != cudaSuccess) {
        std::cerr << "[Pipeline] Clearing stale CUDA error before target selection: "
                  << cudaGetErrorString(staleError) << std::endl;
    }
#endif

    // Upload DeviceConfig to GPU (small 200-byte H2D copy)
    uploadDeviceConfig(stream);

    // Run color filter kernels ONLY if color filter is enabled
    // When disabled, this entire block is skipped - zero overhead
    if (cfg.color_filter.enabled && !m_captureBuffer.empty() && m_captureBuffer.data()) {
        // Step 1: Compute color match ratio for each target
        computeColorMatchRatioKernel<<<cached_max_detections, 256, 0, stream>>>(
            m_unifiedArena.finalTargets,
            m_smallBufferArena.finalTargetsCount,
            m_captureBuffer.data(),
            m_smallBufferArena.deviceConfig
        );

        // Step 2: Apply filter - marks filtered targets by setting confidence to 0
        int filterBlockSize = 128;
        int filterGridSize = (cached_max_detections + filterBlockSize - 1) / filterBlockSize;
        applyColorFilterKernel<<<filterGridSize, filterBlockSize, 0, stream>>>(
            m_unifiedArena.finalTargets,
            m_smallBufferArena.finalTargetsCount,
            m_smallBufferArena.deviceConfig
        );

        // No synchronization needed - kernels on same stream execute sequentially
    }

    // For max_detections <= 32, use single warp (32 threads) with warp shuffle
    const int blockSize = 32;
    const int gridSize = 1;

    fusedTargetSelectionAndMovementKernel<<<gridSize, blockSize, 0, stream>>>(
        m_unifiedArena.finalTargets,
        m_smallBufferArena.finalTargetsCount,
        m_smallBufferArena.deviceConfig,
        m_smallBufferArena.selectedTarget,
        m_smallBufferArena.bestTargetIndex,
        m_smallBufferArena.bestTarget,
        m_smallBufferArena.mouseMovement,
        m_smallBufferArena.pidState
    );

#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Fused kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
    }
#endif
}

bool UnifiedGraphPipeline::updateDDACaptureRegion(const AppContext& ctx) {
    if (!m_capture) {
        return false;
    }

    int detectionRes = ctx.config.profile().detection_resolution;
    if (detectionRes <= 0) {
        return false;
    }

    bool useAimShootOffset = ctx.config.profile().enable_aim_shoot_offset && ctx.aiming && ctx.shooting;
    float offsetX = useAimShootOffset ? ctx.config.profile().aim_shoot_offset_x : ctx.config.profile().crosshair_offset_x;
    float offsetY = useAimShootOffset ? ctx.config.profile().aim_shoot_offset_y : ctx.config.profile().crosshair_offset_y;

    if (m_captureRegionCache.detectionRes == detectionRes &&
        m_captureRegionCache.offsetX == offsetX &&
        m_captureRegionCache.offsetY == offsetY &&
        m_captureRegionCache.usingAimShootOffset == useAimShootOffset) {
        return true;
    }

    int screenW = m_capture->GetScreenWidth();
    int screenH = m_capture->GetScreenHeight();
    if (screenW <= 0 || screenH <= 0) {
        return false;
    }

    int captureSize = std::min(detectionRes, std::min(screenW, screenH));
    if (captureSize <= 0) {
        return false;
    }

    int centerX = screenW / 2 + static_cast<int>(offsetX);
    int centerY = screenH / 2 + static_cast<int>(offsetY);

    int maxLeft = std::max(0, screenW - captureSize);
    int maxTop = std::max(0, screenH - captureSize);

    int left = std::clamp(centerX - captureSize / 2, 0, maxLeft);
    int top = std::clamp(centerY - captureSize / 2, 0, maxTop);

    if (!m_capture->SetCaptureRegion(left, top, captureSize, captureSize)) {
        return false;
    }

    m_captureRegionCache.detectionRes = detectionRes;
    m_captureRegionCache.offsetX = offsetX;
    m_captureRegionCache.offsetY = offsetY;
    m_captureRegionCache.usingAimShootOffset = useAimShootOffset;
    m_captureRegionCache.left = left;
    m_captureRegionCache.top = top;
    m_captureRegionCache.size = captureSize;
    return true;
}


// ============================================================================
// SYNCHRONOUS CAPTURE - Direct frame acquisition from pipeline thread
// ============================================================================

bool UnifiedGraphPipeline::acquireFrameSync(FrameMetadata& outMetadata) {
    if (!m_config.enableCapture || !m_capture) {
        return false;
    }

    auto& ctx = AppContext::getInstance();

    // Update capture region if needed (cached, minimal overhead)
    if (!updateDDACaptureRegion(ctx)) {
        return false;
    }

    // Synchronous frame acquisition - blocks until frame available or timeout
    cudaArray_t cudaArray = nullptr;
    unsigned int width = 0, height = 0;
    uint64_t presentQpc = 0;

    // Use the new synchronous API with frame prediction
    // DDACapture internally uses spin-wait + minimal DXGI timeout for lowest latency
    // The 16ms here is just a fallback maximum (should rarely be used)
    if (!m_capture->AcquireFrameSync(&cudaArray, &width, &height, &presentQpc, 16)) {
        // Timeout is normal, don't spam logs
        return false;
    }

    if (!cudaArray) {
        return false;
    }

    // Duplicate frame detection
    uint64_t lastPresentQpc = m_lastProcessedPresentQpc.load(std::memory_order_acquire);
    if (presentQpc != 0 && presentQpc <= lastPresentQpc) {
        m_perfMetrics.duplicateFrames++;
        return false;
    }

    // === OPTIMIZATION 2: Frame Age Skip ===
    // Skip frames that are too old (stale) to prevent processing outdated data
    // This occurs when GPU inference is slower than capture rate
    if (presentQpc != 0) {
        LARGE_INTEGER currentQpc{}, qpcFreq{};
        if (!QueryPerformanceCounter(&currentQpc) || !QueryPerformanceFrequency(&qpcFreq) || qpcFreq.QuadPart == 0) {
            // QPC not available or invalid frequency - skip age check, continue processing
            // This is a soft failure - we just skip the optimization
        } else {
            double frameAgeMs = static_cast<double>(currentQpc.QuadPart - presentQpc) 
                               * 1000.0 / qpcFreq.QuadPart;
        
            // Skip if frame is older than ~2 frames at 240Hz (approximately 8ms)
            // This ensures we always process recent frames under GPU backpressure
            constexpr double kMaxFrameAgeMs = 8.0;
            if (frameAgeMs > kMaxFrameAgeMs) {
                m_perfMetrics.frameSkipCount++;
                return false;  // Try again for a fresher frame
            }
        }
    }

    // Allocate or reuse capture buffer
    int heightInt = static_cast<int>(height);
    int widthInt = static_cast<int>(width);

    if (m_captureBuffer.empty() ||
        m_captureBuffer.rows() != heightInt ||
        m_captureBuffer.cols() != widthInt ||
        m_captureBuffer.channels() != 4) {
        try {
            m_captureBuffer.create(heightInt, widthInt, 4);
        } catch (const std::exception& e) {
            std::cerr << "[Capture] Failed to allocate capture buffer: " << e.what() << std::endl;
            return false;
        }
        // Buffer address changed - graph baked stale pointers, must recapture
        if (m_graphCaptured) {
            cleanupGraph();
            m_graphCaptured = false;
            m_state.frameCount = 0;
        }
    }

    // Copy from CUDA array to capture buffer (synchronous - D3D11 already flushed)
    cudaStream_t stream = m_pipelineStream ? m_pipelineStream->get() : nullptr;
    cudaError_t err = cudaMemcpy2DFromArrayAsync(
        m_captureBuffer.data(),
        m_captureBuffer.step(),
        cudaArray,
        0, 0,
        width * 4,  // BGRA8
        height,
        cudaMemcpyDeviceToDevice,
        stream);

    if (err != cudaSuccess) {
        std::cerr << "[Capture] cudaMemcpy2DFromArrayAsync failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Populate metadata
    outMetadata.frameId = m_nextFrameId.fetch_add(1, std::memory_order_relaxed);
    outMetadata.presentQpc = presentQpc;
    outMetadata.width = width;
    outMetadata.height = height;

    // Update preview if enabled
    if ((m_preview.enabled || ctx.config.global().show_window) && !m_captureBuffer.empty()) {
        updatePreviewBuffer(m_captureBuffer);
    }

    return true;
}

// ============================================================================
// PREPROCESSING - Uses m_captureBuffer directly
// ============================================================================

bool UnifiedGraphPipeline::performPreprocessing(cudaStream_t stream) {
    if (!m_unifiedArena.yoloInput || m_captureBuffer.empty() || !m_captureBuffer.data()) {
        return false;
    }

    int modelRes = getModelInputResolution();
    
    // Convert PipelinePrecision to PreprocessPrecision
    PreprocessPrecision prepPrecision;
    switch (m_unifiedArena.precision) {
        case PipelinePrecision::FP16:
            prepPrecision = PreprocessPrecision::FP16;
            break;
        case PipelinePrecision::FP8:
            prepPrecision = PreprocessPrecision::FP8;
            break;
        case PipelinePrecision::INT8:
            prepPrecision = PreprocessPrecision::INT8;
            break;
        case PipelinePrecision::FP32:
        default:
            prepPrecision = PreprocessPrecision::FP32;
            break;
    }
    
    cudaError_t err = cuda_unified_preprocessing_ex(
        m_captureBuffer.data(),
        m_unifiedArena.yoloInput,
        m_captureBuffer.cols(),
        m_captureBuffer.rows(),
        static_cast<int>(m_captureBuffer.step()),
        modelRes,
        modelRes,
        prepPrecision,
        stream);

    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Unified preprocessing failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

// ============================================================================
// MAIN EXECUTION - Frame-ID based exactly-once processing (v2)
// ============================================================================

// ============================================================================
// EXECUTE FRAME - Single-frame synchronous pipeline
//
// Pipeline stages (all on the same CUDA stream, serialized by stream ordering):
//   1. acquireFrameSync()           — DDA capture (CPU-blocking until new frame)
//   2. performPreprocessing()       — BGRA→RGB + resize to model input
//   3. performInference()           — TensorRT enqueueV3
//   4. performIntegratedPostProcessing() — YOLO decode + NMS
//   5. performTargetSelection()     — target pick + PID → mouse delta
//   6. enqueueFrameCompletionCallback() — stream callback: executeMouseMovement()
//
// The callback at the end sets m_frameInFlight=false, which unblocks the
// next iteration of runMainLoop(). This guarantees:
//   "capture N+1 starts only AFTER mouse movement from frame N is applied"
// ============================================================================
bool UnifiedGraphPipeline::executeFrame(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    // Single frame in flight - prevents duplicate processing
    // Use fast spin-wait pattern: spin briefly, then return to avoid blocking
    bool expected = false;
    if (!m_frameInFlight.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
        // Another frame is still processing - spin briefly before giving up
        constexpr int kMaxSpinIterations = 64;
        for (int i = 0; i < kMaxSpinIterations; ++i) {
            YieldProcessor();  // CPU hint for spin-wait (Windows intrinsic)
            expected = false;
            if (m_frameInFlight.compare_exchange_weak(expected, true, std::memory_order_acquire)) {
                goto acquired;
            }
        }
        // Still busy after spinning - return and let caller retry
        return true;
    }
acquired:

#ifdef _DEBUG
    auto frameStart = std::chrono::steady_clock::now();
#endif

    // Synchronous frame acquisition - blocks until frame available or timeout
    FrameMetadata metadata;
    if (!acquireFrameSync(metadata)) {
        // No frame available or timeout - release lock and return
        m_frameInFlight.store(false, std::memory_order_release);
        return true;
    }

    // We have a valid, fresh frame in m_captureBuffer - process it
    cudaStream_t execStream = stream ? stream : (m_pipelineStream ? m_pipelineStream->get() : nullptr);
    if (!execStream) {
        m_frameInFlight.store(false, std::memory_order_release);
        return false;
    }

    bool shouldRunDetection = m_config.enableDetection && !ctx.detection_paused.load();

    // === OPTIMIZATION 3: Adaptive Inference Skip ===
    // When target stays in deadband (no movement needed) for consecutive frames,
    // reduce inference rate to save GPU resources while maintaining responsiveness.
    // This feature is configurable: enable/disable and interval can be set in UI.
    bool shouldSkipInference = false;
    const bool adaptiveEnabled = ctx.config.profile().adaptive_skip_enabled;
    const int skipInterval = ctx.config.profile().adaptive_skip_interval;
    
    if (adaptiveEnabled && shouldRunDetection && m_adaptiveSkip.lastWasInDeadband) {
        m_adaptiveSkip.consecutiveDeadbandFrames++;
        
        // Only start skipping after establishing we're stable in deadband (>2 frames)
        if (m_adaptiveSkip.consecutiveDeadbandFrames > 2) {
            m_adaptiveSkip.skipCounter++;
            // Skip (interval-1) out of interval frames (e.g., 4 = 75% reduction)
            if (m_adaptiveSkip.skipCounter < static_cast<uint32_t>(skipInterval)) {
                shouldSkipInference = true;
            } else {
                m_adaptiveSkip.skipCounter = 0;  // Reset counter, run inference this frame
            }
        }
    } else {
        // Not in deadband or feature disabled - reset adaptive skip state
        m_adaptiveSkip.consecutiveDeadbandFrames = 0;
        m_adaptiveSkip.skipCounter = 0;
    }

    // Skip inference if adaptive skip is active
    if (shouldSkipInference) {
        m_frameInFlight.store(false, std::memory_order_release);
        return true;  // Successfully skipped, not an error
    }

    if (shouldRunDetection) {
        // ================================================================
        // CUDA Graph path: replay captured graph for minimal launch overhead.
        // Graph contains: DeviceConfig H2D → preprocess → infer → postprocess
        //   → target select → mouse D2H → debug overlay D2H → callback.
        // We update the persistent host buffer before launch; the baked
        // memcpy node reads from this fixed address on every replay.
        // ================================================================
        if (m_graphCaptured && m_graphExec) {
            // Update host-side config (CPU-only write, no CUDA call)
            fillHostDeviceConfig();

            // Update callback metadata (read by the baked host callback node)
            m_callbackData = {this, metadata};
            m_allowMovement.store(true, std::memory_order_release);

            cudaError_t graphErr = cudaGraphLaunch(m_graphExec, execStream);
            if (graphErr != cudaSuccess) {
                std::cerr << "[Pipeline] CUDA Graph launch failed: "
                          << cudaGetErrorString(graphErr) << std::endl;
                cleanupGraph();
                m_graphCaptured = false;
                m_allowMovement.store(false, std::memory_order_release);
                m_frameInFlight.store(false, std::memory_order_release);
                return false;
            }

            m_state.frameCount++;
        } else {
            // ============================================================
            // Individual-calls path: used during warmup and after topology
            // changes until graph is (re)captured.
            // ============================================================

            // Preprocessing - uses m_captureBuffer directly
            if (!performPreprocessing(execStream)) {
                m_frameInFlight.store(false, std::memory_order_release);
                return false;
            }

            // Inference
            if (!performInference(execStream)) {
                m_frameInFlight.store(false, std::memory_order_release);
                return false;
            }

            // Post-processing (uses cached config via device buffers)
            performIntegratedPostProcessing(execStream);

            // Target selection (uploads DeviceConfig + runs kernels)
            performTargetSelection(execStream);

            // Copy mouse movement to host (if not using mapped memory)
            if (!m_mouseMovementUsesMappedMemory && m_h_movement && m_smallBufferArena.mouseMovement) {
                cudaMemcpyAsync(
                    m_h_movement->get(),
                    m_smallBufferArena.mouseMovement,
                    sizeof(MouseMovement),
                    cudaMemcpyDeviceToHost,
                    execStream);
            }

            // Copy target data to host for debug overlay
            if (ctx.config.global().show_window) {
                if (m_h_targetCount && m_h_targetCount->get() && m_smallBufferArena.decodedCount) {
                    cudaMemcpyAsync(m_h_targetCount->get(), m_smallBufferArena.decodedCount,
                                   sizeof(int), cudaMemcpyDeviceToHost, execStream);
                }
                if (m_h_targets && m_h_targets->get() && m_unifiedArena.decodedTargets) {
                    int maxToCopy = std::min(MAX_HOST_TARGETS, ctx.config.profile().max_detections);
                    cudaMemcpyAsync(m_h_targets->get(), m_unifiedArena.decodedTargets,
                                   sizeof(Target) * maxToCopy, cudaMemcpyDeviceToHost, execStream);
                }
            }

            // Enqueue callback to execute mouse movement and mark frame complete
            m_allowMovement.store(true, std::memory_order_release);
            if (!enqueueFrameCompletionCallback(execStream, metadata)) {
                m_allowMovement.store(false, std::memory_order_release);
                m_frameInFlight.store(false, std::memory_order_release);
                return false;
            }

            m_state.frameCount++;
        }
    } else {
        // Detection paused - just release the frame
        m_frameInFlight.store(false, std::memory_order_release);
    }

#ifdef _DEBUG
    // Update performance metrics (debug only - avoid overhead in release)
    auto frameEnd = std::chrono::steady_clock::now();
    auto latencyMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
    m_perfMetrics.totalFrames++;
    m_perfMetrics.totalLatencyMs += latencyMs;
    m_perfMetrics.logIfNeeded("[Pipeline]");
#endif

    return true;
}

void UnifiedGraphPipeline::updatePreviewBufferAllocation() {
    auto& ctx = AppContext::getInstance();

    // Lazy allocation/deallocation - keep buffers allocated for fast re-enable
    if (ctx.config.global().show_window && !m_preview.enabled) {
        int width = ctx.config.profile().detection_resolution;
        int height = ctx.config.profile().detection_resolution;

        // Reuse existing buffers if they match dimensions
        bool needRealloc = m_preview.previewBuffer.empty() ||
                          m_preview.previewBuffer.rows() != height ||
                          m_preview.previewBuffer.cols() != width;

        if (needRealloc) {
            // Release old buffers first if they exist
            if (m_preview.hostPreviewPinned && m_preview.hostPreview.data()) {
                cudaHostUnregister(m_preview.hostPreview.data());
                m_preview.hostPreviewPinned = false;
            }
            // Unpin and release both double-buffer host buffers
            for (int i = 0; i < 2; i++) {
                if (m_preview.hostPreviewPinnedFlags[i] && m_preview.hostPreviewBuffers[i].data()) {
                    cudaHostUnregister(m_preview.hostPreviewBuffers[i].data());
                    m_preview.hostPreviewPinnedFlags[i] = false;
                    m_preview.hostPreviewPinnedSizes[i] = 0;
                }
                m_preview.hostPreviewBuffers[i].release();
            }
            m_preview.previewBuffer.release();
            m_preview.hostPreview.release();

            // Allocate new GPU preview buffer
            m_preview.previewBuffer.create(height, width, 4);

            // Allocate legacy single buffer for compatibility
            m_preview.hostPreview.create(height, width, 4);

            // Allocate both double-buffer host buffers
            m_preview.hostPreviewBuffers[0].create(height, width, 4);
            m_preview.hostPreviewBuffers[1].create(height, width, 4);

            // Pin legacy host preview buffer for faster async copies
            size_t bytes = static_cast<size_t>(m_preview.hostPreview.step()) * m_preview.hostPreview.rows();
            if (bytes > 0) {
                if (cudaHostRegister(m_preview.hostPreview.data(), bytes, cudaHostRegisterPortable) == cudaSuccess) {
                    m_preview.hostPreviewPinned = true;
                    m_preview.hostPreviewPinnedSize = bytes;
                } else {
                    m_preview.hostPreviewPinned = false;
                    m_preview.hostPreviewPinnedSize = 0;
                }
            }

            // Pin both double-buffer host buffers
            for (int i = 0; i < 2; i++) {
                size_t bufBytes = static_cast<size_t>(m_preview.hostPreviewBuffers[i].step()) * m_preview.hostPreviewBuffers[i].rows();
                if (bufBytes > 0) {
                    if (cudaHostRegister(m_preview.hostPreviewBuffers[i].data(), bufBytes, cudaHostRegisterPortable) == cudaSuccess) {
                        m_preview.hostPreviewPinnedFlags[i] = true;
                        m_preview.hostPreviewPinnedSizes[i] = bufBytes;
                    } else {
                        m_preview.hostPreviewPinnedFlags[i] = false;
                        m_preview.hostPreviewPinnedSizes[i] = 0;
                    }
                }
            }
        }

        m_preview.hasValidHostPreview = false;
        m_preview.finalTargets.reserve(ctx.config.profile().max_detections);
        m_preview.enabled = true;
        m_preview.lastCopyTime = {};
        m_preview.writeIndex.store(0, std::memory_order_release);
        m_preview.newFrameReady.store(false, std::memory_order_release);
    } else if (!ctx.config.global().show_window && m_preview.enabled) {
        // Lazy deallocation - just disable without releasing memory
        // Buffers stay allocated for instant re-enable
        m_preview.enabled = false;
        m_preview.hasValidHostPreview = false;
        // NOTE: Actual deallocation happens only in shutdown() to avoid
        // allocation churn when user rapidly toggles preview window
    }
}

void UnifiedGraphPipeline::updatePreviewBuffer(const SimpleCudaMat& currentBuffer) {
    auto& ctx = AppContext::getInstance();

    // First ensure preview buffer allocation is correct
    updatePreviewBufferAllocation();

    // m_preview.enabled reflects show_window state; avoid redundant check
    if (!m_preview.enabled || m_preview.previewBuffer.empty()) {
        return;
    }

    if (currentBuffer.channels() != 4) {
        std::cerr << "[Preview] Unexpected channel count for capture buffer: "
                  << currentBuffer.channels() << std::endl;
        return;
    }

    if (currentBuffer.empty() || !currentBuffer.data()) {
        return;
    }

    if (m_preview.previewBuffer.rows() != currentBuffer.rows() ||
        m_preview.previewBuffer.cols() != currentBuffer.cols() ||
        m_preview.previewBuffer.channels() != currentBuffer.channels()) {
        // If host preview was pinned, unregister before reallocating
        if (m_preview.hostPreviewPinned && m_preview.hostPreview.data()) {
            cudaHostUnregister(m_preview.hostPreview.data());
            m_preview.hostPreviewPinned = false;
            m_preview.hostPreviewPinnedSize = 0;
        }
        m_preview.previewBuffer.create(currentBuffer.rows(), currentBuffer.cols(), currentBuffer.channels());
        m_preview.hostPreview.create(currentBuffer.rows(), currentBuffer.cols(), currentBuffer.channels());
        // Re-pin host preview buffer after reallocation
        size_t bytes = static_cast<size_t>(m_preview.hostPreview.step()) * m_preview.hostPreview.rows();
        if (bytes > 0 && m_preview.enabled) {
            if (cudaHostRegister(m_preview.hostPreview.data(), bytes, cudaHostRegisterPortable) == cudaSuccess) {
                m_preview.hostPreviewPinned = true;
                m_preview.hostPreviewPinnedSize = bytes;
            }
        }
    }

    if (m_preview.previewBuffer.empty() || !m_preview.previewBuffer.data()) {
        return;
    }

    if (!m_pipelineStream || !m_pipelineStream->get()) {
        return;
    }

    size_t srcStep = currentBuffer.step();
    size_t dstStep = m_preview.previewBuffer.step();

    if (srcStep > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        dstStep > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "[Preview] Capture pitch exceeds supported range" << std::endl;
        return;
    }

    // Validate parameters before CUDA call
    if (!currentBuffer.data() || !m_preview.previewBuffer.data()) {
        std::cerr << "[Preview] Null buffer pointer - src:" << (void*)currentBuffer.data() 
                  << " dst:" << (void*)m_preview.previewBuffer.data() << std::endl;
        return;
    }
    
    int width = currentBuffer.cols();
    int height = currentBuffer.rows();
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        std::cerr << "[Preview] Invalid dimensions: " << width << "x" << height << std::endl;
        return;
    }

    cudaStream_t previewStream = (m_previewStream && m_previewStream->get()) 
                                  ? m_previewStream->get() 
                                  : m_pipelineStream->get();
    
    // Clear any previous CUDA error before our call
    cudaError_t prevErr = cudaGetLastError();
    if (prevErr != cudaSuccess) {
        std::cerr << "[Preview] Clearing previous CUDA error: " << cudaGetErrorString(prevErr) << std::endl;
    }
    
    cudaError_t convertErr = cuda_bgra2rgba(
        currentBuffer.data(),
        m_preview.previewBuffer.data(),
        width,
        height,
        static_cast<int>(srcStep),
        static_cast<int>(dstStep),
        previewStream);

    if (convertErr != cudaSuccess) {
        std::cerr << "[Preview] Failed to convert capture buffer for preview: "
                  << cudaGetErrorString(convertErr) 
                  << " (src:" << (void*)currentBuffer.data() 
                  << " dst:" << (void*)m_preview.previewBuffer.data()
                  << " " << width << "x" << height 
                  << " srcStep:" << srcStep << " dstStep:" << dstStep << ")" << std::endl;
        return;
    }
}

bool UnifiedGraphPipeline::getPreviewSnapshot(SimpleMat& outFrame) {
    auto& ctx = AppContext::getInstance();

    if (!m_preview.enabled) {
        return false;
    }

    if (!m_pipelineStream || !m_pipelineStream->get()) {
        return false;
    }

    if (m_preview.previewBuffer.empty() || m_preview.previewBuffer.data() == nullptr) {
        return false;
    }

    // Determine write buffer index (producer buffer)
    int writeIdx = m_preview.writeIndex.load(std::memory_order_acquire);
    int readIdx = 1 - writeIdx;  // Consumer reads from the other buffer

    // Ensure host preview buffers are allocated and match GPU buffer dimensions
    SimpleMat& writeBuffer = m_preview.hostPreviewBuffers[writeIdx];
    SimpleMat& readBuffer = m_preview.hostPreviewBuffers[readIdx];

    if (writeBuffer.empty() ||
        writeBuffer.rows() != m_preview.previewBuffer.rows() ||
        writeBuffer.cols() != m_preview.previewBuffer.cols() ||
        writeBuffer.channels() != m_preview.previewBuffer.channels()) {
        writeBuffer.create(m_preview.previewBuffer.rows(),
                          m_preview.previewBuffer.cols(),
                          m_preview.previewBuffer.channels());
        m_preview.hasValidHostPreview = false;
    }

    if (readBuffer.empty() ||
        readBuffer.rows() != m_preview.previewBuffer.rows() ||
        readBuffer.cols() != m_preview.previewBuffer.cols() ||
        readBuffer.channels() != m_preview.previewBuffer.channels()) {
        readBuffer.create(m_preview.previewBuffer.rows(),
                         m_preview.previewBuffer.cols(),
                         m_preview.previewBuffer.channels());
    }

    // Check if previous copy completed
    if (m_preview.copyInProgress) {
        if (m_previewReadyEvent && m_previewReadyEvent->get()) {
            cudaError_t queryStatus = cudaEventQuery(m_previewReadyEvent->get());
            if (queryStatus == cudaSuccess) {
                // Copy completed - flip buffers
                m_preview.writeIndex.store(1 - writeIdx, std::memory_order_release);
                m_preview.newFrameReady.store(true, std::memory_order_release);
                m_preview.copyInProgress = false;
                m_preview.hasValidHostPreview = true;
            } else if (queryStatus != cudaErrorNotReady) {
                std::cerr << "[Preview] Event query failed: " << cudaGetErrorString(queryStatus) << std::endl;
                m_preview.copyInProgress = false;
                m_preview.hasValidHostPreview = false;
            }
        } else if (m_pipelineStream && m_pipelineStream->get()) {
            cudaError_t queryStatus = cudaStreamQuery(m_pipelineStream->get());
            if (queryStatus == cudaSuccess) {
                // Copy completed - flip buffers
                m_preview.writeIndex.store(1 - writeIdx, std::memory_order_release);
                m_preview.newFrameReady.store(true, std::memory_order_release);
                m_preview.copyInProgress = false;
                m_preview.hasValidHostPreview = true;
            } else if (queryStatus != cudaErrorNotReady) {
                std::cerr << "[Preview] Stream query failed: " << cudaGetErrorString(queryStatus) << std::endl;
                m_preview.copyInProgress = false;
                m_preview.hasValidHostPreview = false;
            }
        }
        if (m_preview.copyInProgress) {
            // Copy still in progress, return old frame if available
            if (m_preview.hasValidHostPreview) {
                outFrame = readBuffer;
                return true;
            }
            return false;
        }
    }

    // Return the read buffer if we have valid data
    bool hasFrameToReturn = false;
    if (m_preview.hasValidHostPreview && readBuffer.data()) {
        outFrame = readBuffer;
        hasFrameToReturn = true;
    }

    // Throttle preview copies to ~15 FPS to reduce overhead
    auto now = std::chrono::steady_clock::now();
    if (m_preview.lastCopyTime.time_since_epoch().count() != 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_preview.lastCopyTime).count();
        if (elapsed < 66) {
            return hasFrameToReturn;
        }
    }

    // Start new D2H copy to the write buffer
    size_t rowBytes = static_cast<size_t>(m_preview.previewBuffer.cols()) * m_preview.previewBuffer.channels();
    cudaStream_t pstream = (m_previewStream && m_previewStream->get()) ? m_previewStream->get() : (m_pipelineStream ? m_pipelineStream->get() : nullptr);
    if (!pstream) {
        return hasFrameToReturn;
    }
    cudaError_t copyErr = cudaMemcpy2DAsync(
        writeBuffer.data(),
        writeBuffer.step(),
        m_preview.previewBuffer.data(),
        m_preview.previewBuffer.step(),
        rowBytes,
        m_preview.previewBuffer.rows(),
        cudaMemcpyDeviceToHost,
        pstream);
    if (copyErr != cudaSuccess) {
        std::cerr << "[Preview] Failed to copy preview to host: " << cudaGetErrorString(copyErr) << std::endl;
        m_preview.hasValidHostPreview = false;
        return hasFrameToReturn;
    }

    if (m_previewReadyEvent && m_previewReadyEvent->get()) {
        m_previewReadyEvent->record(pstream);
    }

    m_preview.copyInProgress = true;
    m_preview.lastCopyTime = now;
    return hasFrameToReturn;
}

bool UnifiedGraphPipeline::performInference(cudaStream_t stream) {
    if (!stream) {
        return false;
    }

    if (m_primaryInputIndex < 0 ||
        m_primaryInputIndex >= static_cast<int>(m_inputAddressCache.size()) ||
        !m_unifiedArena.yoloInput) {
        return false;
    }

    void* inputBinding = m_inputAddressCache[m_primaryInputIndex];
    if (!inputBinding) {
        return false;
    }
    
    // Aliasing should be set during initialization - no runtime retry.
    // If not aliased, copy is required (logged once during init).
    if (inputBinding != m_unifiedArena.yoloInput) {
        size_t elementSize = UnifiedGPUArena::getElementSize(m_unifiedArena.precision);
        size_t inputSize = getModelInputResolution() * getModelInputResolution() * 3 * elementSize;
        cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize,
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        // Successfully aliased - skip memcpy
        m_perfMetrics.memcpySkipCount++;
    }
    
    if (!runInferenceAsync(stream)) {
        std::cerr << "[UnifiedGraph] TensorRT inference failed" << std::endl;
        return false;
    }

    return true;
}

}
void gpa::UnifiedGraphPipeline::getCaptureStats(gpa::UnifiedGraphPipeline::CaptureStats& out) const {
    auto& ctx = AppContext::getInstance();
    out.lastWidth = m_lastCaptureW;
    out.lastHeight = m_lastCaptureH;
    out.roiLeft = m_captureRegionCache.left;
    out.roiTop = m_captureRegionCache.top;
    out.roiSize = m_captureRegionCache.size;
    out.gpuDirect = m_lastGpuDirect;

    // In synchronous model, hasFrame indicates capture buffer is valid
    out.hasFrame = !m_captureBuffer.empty() && m_captureBuffer.data() != nullptr;

    out.previewEnabled = m_preview.enabled;
    out.previewHasHost = m_preview.hasValidHostPreview;
    out.backend = ctx.config.profile().capture_method.c_str();
}
