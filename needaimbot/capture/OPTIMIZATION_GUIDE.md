# Capture Pipeline Optimization Guide

## Current Performance Bottlenecks

1. **CPU Color Conversion (BGRA→BGR)**
   - Currently using simple loop pixel-by-pixel
   - Can be optimized with SIMD or parallel processing

2. **GPU Memory Allocation**
   - Currently allocating GPU memory each frame
   - Consider using memory pool or pre-allocation strategy

3. **CPU-GPU Transfer**
   - Using regular memory for transfers
   - Pinned memory can improve transfer speed by 2-3x

## Safe Optimizations to Implement

### 1. Parallel Color Conversion (CPU)
```cpp
// In simple_capture.cpp, replace the color conversion loop with:
#include <ppl.h>

// Parallel conversion
concurrency::parallel_for(0, m_height, [&](int y) {
    const uint8_t* srcRow = srcData + y * (m_width * 4);
    uint8_t* dstRow = bgrFrame.data() + y * bgrFrame.step();
    for (int x = 0; x < m_width; ++x) {
        dstRow[x * 3 + 0] = srcRow[x * 4 + 0]; // B
        dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
        dstRow[x * 3 + 2] = srcRow[x * 4 + 2]; // R
    }
});
```

### 2. Memory Pool for GPU Allocations
Use the already created `GpuMemoryPool` class to avoid repeated allocations.

### 3. Async Pipeline
- Use multiple CUDA streams
- Overlap capture, color conversion, and GPU upload
- Pipeline stages: Capture → Convert → Upload → Process

### 4. Direct GPU Processing
Instead of CPU color conversion, capture BGRA and convert on GPU:
- Modify detector to accept BGRA input
- Add GPU kernel for BGRA→BGR conversion
- Eliminates CPU processing entirely

## Performance Metrics to Monitor

1. Frame capture time
2. Color conversion time
3. GPU upload time
4. Total pipeline latency
5. GPU memory usage

## Next Steps

1. Implement parallel color conversion (easiest, immediate benefit)
2. Add performance timing to identify actual bottlenecks
3. Consider GPU-based color conversion for maximum performance
4. Implement memory pooling to reduce allocation overhead