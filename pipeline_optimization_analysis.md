# Unified Pipeline Optimization Analysis

## Current Pipeline Idle Points (병목 지점)

### 1. Capture Wait Synchronization (가장 큰 병목)
**Location**: `executeFrame()` → `waitForCaptureCompletion()`
- Line 2774-2782: Synchronous wait for first frame capture
- Line 2791-2793: `cudaStreamWaitEvent` blocks GPU stream
- **Impact**: Pipeline completely stalls, GPU idle time 5-10ms per frame

### 2. Single Buffer Dependency
**Location**: `m_captureBuffer` (unified_graph_pipeline.h:426)
- Triple-buffer ring exists but Graph path only uses single `m_captureBuffer`
- Cannot start next capture while graph executes
- **Impact**: Sequential capture-process-capture pattern, no overlap

### 3. Frame-in-flight Serialization
**Location**: `executeFrame()` line 2716-2720
- Atomic `m_frameInFlight` enforces single frame processing
- New capture cannot start if previous frame still processing
- **Impact**: CPU-side throttling even when GPU has capacity

### 4. Input Pending Wait
**Location**: `scheduleNextFrameCapture()` line 2201-2217
- QPC-based frame wait after mouse input (up to 6ms)
- Synchronous wait blocks entire pipeline
- **Impact**: 3-6ms added latency per input event

---

## Optimization Ideas

### Idea 1: Double-Buffered Capture for Graph Path ⭐⭐⭐
**Current Flow**:
```
Capture → Wait → Graph Execute → Schedule Next
```

**Improved Flow**:
```
Capture A → Graph Execute (A) ⚡ Capture B (parallel)
                              ↓
Next: Graph Execute (B) ⚡ Capture A (parallel)
```

**Implementation**:
- Use 2 capture buffers in ping-pong pattern for graph path
- Start next capture asynchronously during graph execution
- Eliminate synchronous `waitForCaptureCompletion()` call

**Expected Gain**: 5-10ms latency reduction, 30-50% throughput increase

**Risk**: Medium - requires careful buffer lifecycle management

---

### Idea 2: Async Capture Kickoff (Most Promising) ⭐⭐⭐⭐⭐
**Current Sequence**:
```
1. Check if capture ready
2. If not ready, wait
3. Execute graph
4. Schedule next capture (after graph finishes)
```

**Improved Sequence**:
```
1. Kickoff capture for frame N+1 (async, non-blocking)
2. Execute graph for frame N (if ready)
3. Process completes while N+1 captures in background
```

**Implementation Changes**:
```cpp
// In executeFrame(), move capture kickoff earlier:
// BEFORE: Wait for capture → Execute graph → Schedule next
// AFTER:  Schedule next (async) → Execute graph (if ready)

// Pseudo-code:
if (m_graphPrimed) {
    scheduleNextFrameCapture(false);  // Move before graph execute
}

// Remove blocking wait:
// if (!waitForCaptureCompletion()) { return false; }  // DELETE THIS

// Use event-based sync only:
if (m_captureReadyEvent) {
    cudaStreamWaitEvent(launchStream, m_captureReadyEvent->get(), 0);
}
```

**Expected Gain**:
- Pipeline bubble eliminated
- Continuous GPU utilization
- 40-60% throughput increase
- Latency reduction: 5-8ms

**Risk**: Low - minimal code changes, leverages existing async infrastructure

---

### Idea 3: Speculative Capture (투기적 캡처) ⭐⭐⭐⭐
**Concept**: Always keep 1-2 frames pre-captured and ready

**Implementation**:
```cpp
// On aimbot activation:
void handleAimbotActivation() {
    // Pre-capture 2-3 frames immediately
    for (int i = 0; i < 2; i++) {
        scheduleNextFrameCapture(false);
    }
    // Now pipeline has frames ready instantly
}

// In main loop:
// Maintain invariant: always 1-2 frames in capture queue
if (captureQueueDepth() < 1) {
    scheduleNextFrameCapture(false);
}
```

**Expected Gain**:
- First frame latency: -5ms (no cold start)
- Stable FPS, no pipeline stalls
- Better responsiveness on aim key press

**Risk**: Low - uses existing ring buffer, just fills it proactively

---

### Idea 4: Input Pending Optimization ⭐⭐
**Current**: Synchronous 6ms wait for QPC-based frame validation

**Improved**:
- Skip exactly 1 frame (fixed), no variable wait
- Use async callback when frame ready
- Continue other work during wait (e.g., preprocessing previous frame)

**Expected Gain**: 3-4ms average latency reduction

**Risk**: Medium - may need careful tuning to avoid using stale frames

---

## Recommended Implementation Priority

1. **Idea 2 (Async Capture Kickoff)** - Highest impact, lowest risk
2. **Idea 3 (Speculative Capture)** - Complements #1, easy to add
3. **Idea 1 (Double-Buffering)** - Larger refactor, implement after #2 if needed
4. **Idea 4 (Input Optimization)** - Nice-to-have, lower priority

---

## Code Locations for Changes

### executeFrame() (unified_graph_pipeline.cu:2711)
- Move `scheduleNextFrameCapture(false)` before graph launch
- Remove blocking `waitForCaptureCompletion()` in priming phase

### scheduleNextFrameCapture() (unified_graph_pipeline.cu:2165)
- Already supports async mode (forceSync=false)
- Ensure it's called early and often

### handleAimbotActivation() (location TBD)
- Add speculative pre-capture loop

---

## Validation Plan (using Codex)

1. Analyze current capture-execute timing with markers
2. Simulate double-buffered capture pattern
3. Measure theoretical vs actual overlap
4. Identify remaining synchronization points
5. Generate optimized code with proper event handling
