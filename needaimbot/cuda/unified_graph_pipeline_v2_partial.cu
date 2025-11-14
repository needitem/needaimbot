// Core implementation of frame-ID based lock-free pipeline
// This file contains the critical hot-path code with all optimizations

#include "unified_graph_pipeline.h"
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

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

extern "C" {
    void executeMouseMovement(int dx, int dy);
}

namespace needaimbot {

// ============================================================================
// LOCK-FREE CONFIG UPDATE
// ============================================================================

void UnifiedGraphPipeline::refreshConfigCache(const AppContext& ctx) {
    // Called periodically from background thread or when config changes
    // NOT called in hot path (executeFrame)

    uint32_t currentGen = m_cachedConfig.generation.load(std::memory_order_acquire);

    // Atomic read of config with single mutex lock
    AppContext& mutableCtx = const_cast<AppContext&>(ctx);
    {
        std::lock_guard<std::mutex> lock(mutableCtx.configMutex);

        // PID
        m_cachedConfig.pid.kp_x = ctx.config.pid_kp_x;
        m_cachedConfig.pid.kp_y = ctx.config.pid_kp_y;
        m_cachedConfig.pid.ki_x = ctx.config.pid_ki_x;
        m_cachedConfig.pid.ki_y = ctx.config.pid_ki_y;
        m_cachedConfig.pid.kd_x = ctx.config.pid_kd_x;
        m_cachedConfig.pid.kd_y = ctx.config.pid_kd_y;
        m_cachedConfig.pid.integral_max = ctx.config.pid_integral_max;
        m_cachedConfig.pid.derivative_max = ctx.config.pid_derivative_max;

        // Targeting
        m_cachedConfig.targeting.head_y_offset = ctx.config.head_y_offset;
        m_cachedConfig.targeting.body_y_offset = ctx.config.body_y_offset;
        m_cachedConfig.targeting.iou_stickiness_threshold = ctx.config.iou_stickiness_threshold;

        // Find head class ID
        m_cachedConfig.targeting.head_class_id = -1;
        for (const auto& cs : ctx.config.class_settings) {
            if (cs.name == ctx.config.head_class_name) {
                m_cachedConfig.targeting.head_class_id = cs.id;
                break;
            }
        }

        // Detection
        m_cachedConfig.detection.max_detections = ctx.config.max_detections;
        m_cachedConfig.detection.confidence_threshold = ctx.config.confidence_threshold;

        // Class filter - fixed size array (cache-friendly)
        m_cachedConfig.detection.class_filter.fill(0);
        for (const auto& cs : ctx.config.class_settings) {
            if (cs.id >= 0 && cs.id < 80) {
                m_cachedConfig.detection.class_filter[cs.id] = cs.allow ? 1 : 0;
            }
        }

        // Movement filtering
        m_cachedConfig.filtering.deadband_enter_x = ctx.config.deadband_enter_x;
        m_cachedConfig.filtering.deadband_exit_x = ctx.config.deadband_exit_x;
        m_cachedConfig.filtering.deadband_enter_y = ctx.config.deadband_enter_y;
        m_cachedConfig.filtering.deadband_exit_y = ctx.config.deadband_exit_y;

        // Increment generation to signal update
        m_cachedConfig.generation.store(currentGen + 1, std::memory_order_release);
    }
}

void UnifiedGraphPipeline::updateConfig(const AppContext& ctx) {
    refreshConfigCache(ctx);

    // Upload class filter to GPU (async)
    if (m_smallBufferArena.allowFlags && m_pipelineStream) {
        cudaMemcpyAsync(
            m_smallBufferArena.allowFlags,
            m_cachedConfig.detection.class_filter.data(),
            80 * sizeof(unsigned char),
            cudaMemcpyHostToDevice,
            m_pipelineStream->get());
    }
}

// ============================================================================
// FRAME RING BUFFER - Lock-free producer/consumer
// ============================================================================

bool UnifiedGraphPipeline::scheduleCapture() {
    if (!m_capture) return false;

    auto& ctx = AppContext::getInstance();

    // Update capture region if needed (cached, minimal overhead)
    if (!updateCaptureRegion(ctx)) {
        return false;
    }

    // Get next slot to write to (lock-free)
    uint64_t captureSlot = m_nextCaptureSlot.load(std::memory_order_acquire);
    size_t slotIdx = captureSlot % FRAME_RING_SIZE;
    FrameSlot& slot = m_frameRing[slotIdx];

    // Check if slot is available (consumer must have marked it consumed)
    bool expectedConsumed = true;
    if (slotIdx > 0 && !slot.consumed.compare_exchange_strong(expectedConsumed, false, std::memory_order_acquire)) {
        // Slot still in use - consumer is slow, drop this capture
        m_perfMetrics.droppedFrames++;
        return false;
    }

    // Acquire frame from DDA
    cudaArray_t cudaArray = nullptr;
    unsigned int width = 0, height = 0;

    if (!m_capture->GetLatestFrameGPU(&cudaArray, &width, &height) || !cudaArray) {
        return false;
    }

    // Get frame metadata
    FrameMetadata metadata;
    metadata.frameId = m_nextFrameId.fetch_add(1, std::memory_order_relaxed);

    // Get PresentQpc (0 if not supported)
    metadata.presentQpc = m_qpcSupported ? m_capture->GetLastPresentQpc() : 0;

    metadata.width = width;
    metadata.height = height;

    LARGE_INTEGER qpc;
    metadata.captureTimeQpc = QueryPerformanceCounter(&qpc)
        ? static_cast<uint64_t>(qpc.QuadPart)
        : 0;

    // Allocate or reuse image buffer
    SimpleCudaMat& image = slot.image;
    if (image.empty() || image.rows() != (int)height || image.cols() != (int)width || image.channels() != 4) {
        try {
            image.create((int)height, (int)width, 4);
        } catch (const std::exception& e) {
            std::cerr << "[Capture] Failed to allocate frame buffer: " << e.what() << std::endl;
            return false;
        }
    }

    // GPU-direct copy (async)
    cudaStream_t stream = m_captureStream ? m_captureStream->get() : m_pipelineStream->get();
    cudaError_t err = cudaMemcpy2DFromArrayAsync(
        image.data(),
        image.step(),
        cudaArray,
        0, 0,
        width * 4,
        height,
        cudaMemcpyDeviceToDevice,
        stream
    );

    if (err != cudaSuccess) {
        std::cerr << "[Capture] cudaMemcpy2DFromArrayAsync failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Record event to track completion
    if (m_captureReadyEvent) {
        m_captureReadyEvent->record(stream);
    }

    // Write metadata and mark ready (AFTER copy is enqueued)
    slot.metadata = metadata;
    slot.ready.store(true, std::memory_order_release);

    // Advance producer index
    m_nextCaptureSlot.store(captureSlot + 1, std::memory_order_release);

    return true;
}

bool UnifiedGraphPipeline::tryConsumeFrame(FrameMetadata& outMetadata, SimpleCudaMat& outImage) {
    // Get oldest unconsumed frame (lock-free)
    uint64_t processSlot = m_nextProcessSlot.load(std::memory_order_acquire);
    size_t slotIdx = processSlot % FRAME_RING_SIZE;
    FrameSlot& slot = m_frameRing[slotIdx];

    // Check if frame is ready
    if (!slot.ready.load(std::memory_order_acquire)) {
        return false;  // No frame available yet
    }

    // Wait for capture to complete (if still in flight)
    if (m_captureReadyEvent) {
        cudaError_t q = cudaEventQuery(m_captureReadyEvent->get());
        if (q == cudaErrorNotReady) {
            return false;  // Capture still in progress
        }
        if (q != cudaSuccess) {
            std::cerr << "[Capture] Event query failed: " << cudaGetErrorString(q) << std::endl;
            return false;
        }
    }

    // Check frame freshness - skip stale frames
    const FrameMetadata& metadata = slot.metadata;

    // Duplicate frame detection (same PresentQpc)
    uint64_t lastPresentQpc = m_lastProcessedPresentQpc.load(std::memory_order_acquire);
    if (metadata.presentQpc != 0 && metadata.presentQpc <= lastPresentQpc) {
        // Same game frame presented multiple times - skip
        m_perfMetrics.duplicateFrames++;
        slot.ready.store(false, std::memory_order_release);
        slot.consumed.store(true, std::memory_order_release);
        m_nextProcessSlot.store(processSlot + 1, std::memory_order_release);
        return false;
    }

    // Already processed frame detection (same frameId)
    uint64_t lastFrameId = m_lastProcessedFrameId.load(std::memory_order_acquire);
    if (metadata.frameId <= lastFrameId) {
        // Already processed this frame - skip
        m_perfMetrics.duplicateFrames++;
        slot.ready.store(false, std::memory_order_release);
        slot.consumed.store(true, std::memory_order_release);
        m_nextProcessSlot.store(processSlot + 1, std::memory_order_release);
        return false;
    }

    // Input latency check - ensure frame includes our last input
    uint64_t pendingQpc = m_pendingInputQpc.load(std::memory_order_acquire);
    if (pendingQpc != 0 && metadata.presentQpc != 0 && metadata.presentQpc < pendingQpc) {
        // Frame was presented BEFORE our input - skip stale frame
        m_perfMetrics.frameSkipCount++;
        slot.ready.store(false, std::memory_order_release);
        slot.consumed.store(true, std::memory_order_release);
        m_nextProcessSlot.store(processSlot + 1, std::memory_order_release);
        return false;
    }

    // Frame is valid and fresh - consume it
    outMetadata = metadata;
    outImage = std::move(slot.image);  // Move ownership (ring buffer will recreate)

    // Mark slot as consumed (producer can reuse it)
    slot.ready.store(false, std::memory_order_release);
    slot.consumed.store(true, std::memory_order_release);

    // Advance consumer index
    m_nextProcessSlot.store(processSlot + 1, std::memory_order_release);

    return true;
}

// ============================================================================
// MAIN EXECUTION - Frame-ID based exactly-once processing
// ============================================================================

bool UnifiedGraphPipeline::executeFrame(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    // Single frame in flight - prevents duplicate processing
    bool expected = false;
    if (!m_frameInFlight.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
        // Another frame is still processing - skip (not an error)
        return true;
    }

    auto frameStart = std::chrono::steady_clock::now();

    // Try to consume next frame (lock-free)
    FrameMetadata metadata;
    SimpleCudaMat frameImage;

    if (!tryConsumeFrame(metadata, frameImage)) {
        // No frame available or frame was stale - release lock and return
        m_frameInFlight.store(false, std::memory_order_release);
        m_inflightCv.notify_all();
        return true;
    }

    // We have a valid, fresh frame - process it
    cudaStream_t execStream = stream ? stream : m_pipelineStream->get();
    if (!execStream) {
        m_frameInFlight.store(false, std::memory_order_release);
        m_inflightCv.notify_all();
        return false;
    }

    bool shouldRunDetection = m_config.enableDetection && !ctx.detection_paused.load();

    if (shouldRunDetection) {
        // Read cached config (NO LOCKS)
        const CachedConfig& cfg = m_cachedConfig;

        // Preprocessing
        if (!performPreprocessing(frameImage, execStream)) {
            m_frameInFlight.store(false, std::memory_order_release);
            m_inflightCv.notify_all();
            return false;
        }

        // Inference
        if (!performInference(execStream)) {
            m_frameInFlight.store(false, std::memory_order_release);
            m_inflightCv.notify_all();
            return false;
        }

        // Post-processing (uses cached config)
        performPostProcessing(execStream);

        // Target selection (uses cached config)
        performTargetSelection(execStream);

        // Copy mouse movement to host (if not using mapped memory)
        if (!m_mouseMovementUsesMappedMemory && m_h_movement && m_smallBufferArena.mouseMovement) {
            cudaMemcpyAsync(
                m_h_movement->get(),
                m_smallBufferArena.mouseMovement,
                sizeof(MouseMovement),
                cudaMemcpyDeviceToHost,
                execStream
            );
        }

        // Enqueue callback to execute mouse movement and mark frame complete
        m_allowMovement.store(true, std::memory_order_release);
        if (!enqueueFrameCompletionCallback(execStream, metadata)) {
            m_allowMovement.store(false, std::memory_order_release);
            m_frameInFlight.store(false, std::memory_order_release);
            m_inflightCv.notify_all();
            return false;
        }
    } else {
        // Detection paused - just release the frame
        m_frameInFlight.store(false, std::memory_order_release);
        m_inflightCv.notify_all();
    }

    // Update performance metrics
    auto frameEnd = std::chrono::steady_clock::now();
    auto latencyMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

    m_perfMetrics.totalFrames++;
    m_perfMetrics.totalLatencyMs += latencyMs;
    m_perfMetrics.logIfNeeded("[Pipeline]");

    return true;
}

// ============================================================================
// FRAME COMPLETION CALLBACK - Executes mouse movement
// ============================================================================

bool UnifiedGraphPipeline::enqueueFrameCompletionCallback(cudaStream_t stream, const FrameMetadata& metadata) {
    if (!stream) return false;

    // Capture metadata by value for callback
    struct CallbackData {
        UnifiedGraphPipeline* pipeline;
        FrameMetadata metadata;
    };

    auto* data = new CallbackData{this, metadata};

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

                    if (filtered.dx != 0 || filtered.dy != 0) {
                        executeMouseMovement(filtered.dx, filtered.dy);

                        // Record input timestamp for latency tracking
                        LARGE_INTEGER qpc;
                        if (QueryPerformanceCounter(&qpc)) {
                            pipeline->m_pendingInputQpc.store(
                                static_cast<uint64_t>(qpc.QuadPart),
                                std::memory_order_release
                            );
                        }
                    }

                    // Mark this frame as processed (exactly-once guarantee)
                    pipeline->m_lastProcessedFrameId.store(meta.frameId, std::memory_order_release);
                    pipeline->m_lastProcessedPresentQpc.store(meta.presentQpc, std::memory_order_release);

                    // Clear pending input flag if this frame includes our input
                    uint64_t pendingQpc = pipeline->m_pendingInputQpc.load(std::memory_order_acquire);
                    if (pendingQpc != 0 && meta.presentQpc >= pendingQpc) {
                        pipeline->m_pendingInputQpc.store(0, std::memory_order_release);
                    }
                } else {
                    pipeline->filterMouseMovement({0, 0}, false);
                }

                if (isSingleShot) {
                    ctx.single_shot_mode = false;
                }
            }

            // Release frame-in-flight lock
            pipeline->m_frameInFlight.store(false, std::memory_order_release);
            pipeline->m_inflightCv.notify_all();

            delete cbData;
        },
        data
    );

    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to enqueue completion callback: "
                  << cudaGetErrorString(err) << std::endl;
        delete data;
        m_allowMovement.store(false, std::memory_order_release);
        return false;
    }

    return true;
}

// ============================================================================
// MOVEMENT FILTERING - Per-axis hysteresis without locks
// ============================================================================

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

    // Read cached filter config (NO LOCKS)
    const auto& cfg = m_cachedConfig.filtering;

    int dx = raw.dx;
    int dy = raw.dy;

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
        const int exit2 = cfg.deadband_exit_x * cfg.deadband_exit_x;
        const int mag2 = emitDx * emitDx;

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
        const int exit2 = cfg.deadband_exit_y * cfg.deadband_exit_y;
        const int mag2 = emitDy * emitDy;

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

// ============================================================================
// MAIN LOOP - Producer/Consumer orchestration
// ============================================================================

void UnifiedGraphPipeline::runMainLoop() {
    auto& ctx = AppContext::getInstance();

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

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

            // Schedule capture (async, non-blocking)
            scheduleCapture();

            // Execute one frame
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

        // Main loop: parallel capture + serial processing
        while (ctx.aiming.load() && !m_shouldStop.load(std::memory_order_acquire) &&
               !ctx.should_exit.load()) {

            // Producer: schedule next capture (async, non-blocking)
            scheduleCapture();

            // Consumer: process oldest available frame (blocks if needed)
            if (!executeFrame(nullptr)) {
                // Critical error - exit loop
                break;
            }

            // Config update (periodic, low frequency)
            static uint64_t framesSinceConfigUpdate = 0;
            if (++framesSinceConfigUpdate >= 60) {  // Update every ~60 frames
                refreshConfigCache(ctx);
                framesSinceConfigUpdate = 0;
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

// ============================================================================
// LIFECYCLE
// ============================================================================

void UnifiedGraphPipeline::handleAimbotActivation() {
    m_state.frameCount = 0;
    m_allowMovement.store(false, std::memory_order_release);
    clearMovementData();

    // Reset frame tracking
    m_nextCaptureSlot.store(0, std::memory_order_release);
    m_nextProcessSlot.store(0, std::memory_order_release);
    m_lastProcessedFrameId.store(0, std::memory_order_release);
    m_lastProcessedPresentQpc.store(0, std::memory_order_release);
    m_pendingInputQpc.store(0, std::memory_order_release);

    // Reset movement filter
    m_filterState.skipNext = true;
    m_filterState.inSettleX = false;
    m_filterState.inSettleY = false;
    m_filterState.lastEmitX = 0;
    m_filterState.lastEmitY = 0;
}

void UnifiedGraphPipeline::handleAimbotDeactivation() {
    clearMovementData();
    m_allowMovement.store(false, std::memory_order_release);

    // Clear frame ring
    for (auto& slot : m_frameRing) {
        slot.ready.store(false, std::memory_order_release);
        slot.consumed.store(true, std::memory_order_release);
    }

    m_captureRegionCache = {};
}

void UnifiedGraphPipeline::clearMovementData() {
    if (m_h_movement && m_h_movement->get()) {
        m_h_movement->get()->dx = 0;
        m_h_movement->get()->dy = 0;
    }

    if (m_smallBufferArena.mouseMovement && !m_mouseMovementUsesMappedMemory) {
        cudaMemset(m_smallBufferArena.mouseMovement, 0, sizeof(MouseMovement));
    }

    if (m_smallBufferArena.pidState) {
        cudaMemset(m_smallBufferArena.pidState, 0, sizeof(PIDState));
    }

    invalidateSelectedTarget(m_pipelineStream ? m_pipelineStream->get() : nullptr);

    m_filterState.skipNext = true;
}

void UnifiedGraphPipeline::invalidateSelectedTarget(cudaStream_t stream) {
    if (!m_smallBufferArena.selectedTarget) return;

    Target invalidTarget{};
    invalidTarget.classId = -1;
    invalidTarget.x = -1;
    invalidTarget.y = -1;
    invalidTarget.width = -1;
    invalidTarget.height = -1;
    invalidTarget.confidence = 0.0f;

    if (stream) {
        cudaMemcpyAsync(
            m_smallBufferArena.selectedTarget,
            &invalidTarget,
            sizeof(Target),
            cudaMemcpyHostToDevice,
            stream
        );
    } else {
        cudaMemcpy(
            m_smallBufferArena.selectedTarget,
            &invalidTarget,
            sizeof(Target),
            cudaMemcpyHostToDevice
        );
    }
}

}  // namespace needaimbot
