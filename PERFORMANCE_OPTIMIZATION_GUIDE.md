# Performance Optimization Implementation Guide

## 1. GPU Memory Pool Implementation

### Current Issue
The detector allocates GPU memory in multiple places with individual `cudaMalloc` calls, causing fragmentation and overhead.

### Proposed Solution

```cpp
class CudaMemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks;
    void* pool_base;
    size_t pool_size;
    size_t alignment = 256; // GPU alignment requirement
    
public:
    CudaMemoryPool(size_t total_size) {
        cudaMalloc(&pool_base, total_size);
        pool_size = total_size;
    }
    
    void* allocate(size_t size) {
        size = align_up(size, alignment);
        // Find free block or allocate from pool
        // Return aligned pointer
    }
    
    void deallocate(void* ptr) {
        // Mark block as free for reuse
    }
};
```

### Integration Points
- Replace allocations in `detector.cpp:getBindings()`
- Replace allocations in `detector.cpp:initializeBuffers()`
- Pre-allocate pool size based on model requirements

### Expected Impact
- 20-30% reduction in allocation overhead
- Eliminated fragmentation
- Faster allocation/deallocation

## 2. Lock-Free Detection Queue

### Current Issue
Heavy mutex contention between detector and mouse threads when sharing detection results.

### Proposed Solution

```cpp
template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<T*> data;
        std::atomic<Node*> next;
    };
    
    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    
public:
    void push(T item) {
        Node* new_node = new Node;
        new_node->data.store(new T(std::move(item)));
        new_node->next.store(nullptr);
        
        Node* prev_tail = tail.exchange(new_node);
        prev_tail->next.store(new_node);
    }
    
    bool try_pop(T& result) {
        Node* head_node = head.load();
        Node* next = head_node->next.load();
        
        if (next == nullptr) return false;
        
        result = std::move(*next->data.load());
        head.store(next);
        delete head_node;
        return true;
    }
};
```

### Integration
- Replace mutex-protected detection sharing in `detector.cpp`
- Use for mouse thread communication

### Expected Impact
- 15-20% reduction in latency
- Eliminated thread blocking
- Better CPU utilization

## 3. Optimized NMS with Spatial Hashing

### Current Issue
O(n²) IoU calculations in NMS kernel, even for spatially distant boxes.

### Proposed Solution

```cuda
// Spatial hash grid for NMS
__constant__ int HASH_GRID_SIZE = 16;
__constant__ int CELLS_PER_DIM = 16;

struct SpatialHash {
    int* cell_starts;  // Start index for each cell
    int* cell_ends;    // End index for each cell
    int* sorted_indices; // Box indices sorted by cell
};

__global__ void buildSpatialHashKernel(
    const Detection* detections, int n,
    SpatialHash hash, int grid_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Calculate cell for this detection
    float cx = detections[idx].x + detections[idx].width * 0.5f;
    float cy = detections[idx].y + detections[idx].height * 0.5f;
    
    int cell_x = min(grid_size-1, int(cx / cell_width));
    int cell_y = min(grid_size-1, int(cy / cell_height));
    int cell_id = cell_y * grid_size + cell_x;
    
    // Atomic increment for cell assignment
    int pos = atomicAdd(&hash.cell_ends[cell_id], 1);
    hash.sorted_indices[pos] = idx;
}

__global__ void spatialNMSKernel(
    const Detection* detections,
    SpatialHash hash,
    bool* keep,
    float threshold) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !keep[idx]) return;
    
    // Get cell of current box
    int my_cell = getCell(detections[idx]);
    
    // Only check nearby cells (3x3 grid)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int check_cell = my_cell + dy * grid_size + dx;
            if (check_cell < 0 || check_cell >= total_cells) continue;
            
            // Check boxes in this cell
            for (int i = hash.cell_starts[check_cell]; 
                 i < hash.cell_ends[check_cell]; i++) {
                int other_idx = hash.sorted_indices[i];
                if (other_idx == idx) continue;
                
                // Perform IoU check only for nearby boxes
                float iou = calculateIoU(detections[idx], detections[other_idx]);
                if (iou > threshold && 
                    detections[other_idx].confidence > detections[idx].confidence) {
                    keep[idx] = false;
                    return;
                }
            }
        }
    }
}
```

### Expected Impact
- 40-50% reduction in NMS time
- Scales better with number of detections
- Reduced memory bandwidth

## 4. Config Value Caching System

### Current Issue
Frequent mutex locks to read config values in hot paths.

### Proposed Solution

```cpp
class CachedConfig {
private:
    struct ConfigCache {
        // Frequently accessed values
        float detection_resolution;
        float confidence_threshold;
        float nms_threshold;
        bool enable_aimbot;
        // ... other hot values
        
        std::atomic<uint64_t> version;
    };
    
    ConfigCache cache;
    std::atomic<uint64_t> current_version{0};
    
public:
    void updateCache(const Config& config) {
        ConfigCache new_cache;
        // Copy values
        new_cache.detection_resolution = config.detection_resolution;
        // ... copy other values
        
        new_cache.version = ++current_version;
        cache = new_cache; // Atomic on x64
    }
    
    ConfigCache getCache() const {
        return cache; // Copy is atomic on x64
    }
    
    bool needsUpdate(uint64_t last_version) const {
        return last_version != current_version.load();
    }
};
```

### Integration
- Update cache on config changes
- Hot paths read from cache without locks
- Periodic version checks for updates

### Expected Impact
- 5-10% CPU usage reduction
- Eliminated mutex contention for reads
- Better cache locality

## 5. CPU Thread Optimization

### Current Issue
Mouse thread uses busy-wait polling, consuming CPU even when idle.

### Proposed Solution

```cpp
class EventDrivenMouseThread {
private:
    std::condition_variable work_cv;
    std::mutex work_mutex;
    std::queue<DetectionEvent> work_queue;
    std::atomic<bool> should_exit{false};
    
public:
    void processLoop() {
        while (!should_exit) {
            DetectionEvent event;
            
            {
                std::unique_lock<std::mutex> lock(work_mutex);
                work_cv.wait(lock, [this] { 
                    return !work_queue.empty() || should_exit; 
                });
                
                if (should_exit) break;
                
                event = work_queue.front();
                work_queue.pop();
            }
            
            // Process event without holding lock
            processDetection(event);
        }
    }
    
    void submitWork(DetectionEvent event) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            work_queue.push(event);
        }
        work_cv.notify_one();
    }
};
```

### Expected Impact
- Near-zero CPU usage when idle
- Better response times under load
- Improved system resource sharing

## 6. CUDA Stream Optimization

### Current Issue
Sequential execution of preprocessing, inference, and postprocessing.

### Proposed Solution

```cpp
class PipelinedDetector {
private:
    // Triple buffering for pipeline
    struct PipelineStage {
        cv::cuda::GpuMat preprocessed;
        void* inference_buffer;
        Detection* detections;
        cudaEvent_t ready_event;
    };
    
    PipelineStage stages[3];
    int current_stage = 0;
    
    cudaStream_t preprocess_stream;
    cudaStream_t inference_stream;
    cudaStream_t postprocess_stream;
    
public:
    void processPipelined(cv::cuda::GpuMat& input) {
        auto& stage = stages[current_stage];
        
        // Start preprocessing on current frame
        preProcess(input, stage.preprocessed, preprocess_stream);
        cudaEventRecord(stage.ready_event, preprocess_stream);
        
        // If previous frame ready, start inference
        int prev_stage = (current_stage + 2) % 3;
        if (cudaEventQuery(stages[prev_stage].ready_event) == cudaSuccess) {
            cudaStreamWaitEvent(inference_stream, stages[prev_stage].ready_event);
            runInference(stages[prev_stage], inference_stream);
        }
        
        // If inference done on earlier frame, start postprocess
        int post_stage = (current_stage + 1) % 3;
        if (inferenceReady(post_stage)) {
            postProcess(stages[post_stage], postprocess_stream);
        }
        
        current_stage = (current_stage + 1) % 3;
    }
};
```

### Expected Impact
- 20-30% throughput improvement
- Better GPU utilization
- Reduced end-to-end latency

## 7. Optimized Accumulator Pattern

### Current Issue
Float accumulation causing precision loss and irregular movement.

### Proposed Solution

```cpp
class PreciseAccumulator {
private:
    // Use fixed-point arithmetic for sub-pixel precision
    static constexpr int PRECISION_BITS = 16;
    static constexpr int SCALE = 1 << PRECISION_BITS;
    
    int64_t accumulated_x = 0;
    int64_t accumulated_y = 0;
    
public:
    std::pair<int, int> accumulate(float dx, float dy) {
        // Convert to fixed-point
        accumulated_x += static_cast<int64_t>(dx * SCALE);
        accumulated_y += static_cast<int64_t>(dy * SCALE);
        
        // Extract integer part
        int move_x = accumulated_x >> PRECISION_BITS;
        int move_y = accumulated_y >> PRECISION_BITS;
        
        // Keep fractional part
        accumulated_x &= (SCALE - 1);
        accumulated_y &= (SCALE - 1);
        
        return {move_x, move_y};
    }
};
```

### Expected Impact
- Smoother sub-pixel movement
- No precision loss over time
- Better tracking accuracy

## Implementation Priority

1. **Week 1**: GPU Memory Pool + Spatial NMS
2. **Week 2**: Lock-Free Queue + Config Caching  
3. **Week 3**: Thread Optimization + CUDA Streams
4. **Week 4**: Testing and Benchmarking

## Performance Monitoring

Add these metrics to track improvements:

```cpp
struct PerformanceMetrics {
    std::atomic<float> gpu_memory_allocated;
    std::atomic<float> nms_execution_time;
    std::atomic<float> detection_latency;
    std::atomic<float> cpu_usage_percent;
    std::atomic<int> cache_hits;
    std::atomic<int> cache_misses;
    
    void report() {
        // Log or display metrics
    }
};
```

## Testing Strategy

1. **Unit Tests**: For each optimization component
2. **Integration Tests**: Full pipeline performance
3. **Stress Tests**: High detection count scenarios
4. **Regression Tests**: Ensure accuracy maintained

## Expected Overall Impact

- **Latency Reduction**: 30-40% end-to-end
- **Throughput Increase**: 40-50% detections/second
- **CPU Usage**: 50-70% reduction
- **Memory Usage**: 20-30% reduction
- **Code Maintainability**: Significantly improved