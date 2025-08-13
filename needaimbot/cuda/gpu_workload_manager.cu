#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <atomic>
#include <mutex>

namespace cg = cooperative_groups;

// GPU 워크로드 매니저 - CPU 작업을 GPU로 완전 오프로드
class GpuWorkloadManager {
private:
    // GPU 메모리 풀
    struct MemoryPool {
        void* device_pool;
        size_t pool_size;
        std::atomic<size_t> used;
        
        MemoryPool(size_t size) : pool_size(size), used(0) {
            cudaMalloc(&device_pool, size);
            cudaMemset(device_pool, 0, size);
        }
        
        ~MemoryPool() {
            if (device_pool) cudaFree(device_pool);
        }
        
        void* allocate(size_t size) {
            size_t offset = used.fetch_add(size);
            if (offset + size > pool_size) {
                used.fetch_sub(size);
                return nullptr;
            }
            return static_cast<char*>(device_pool) + offset;
        }
        
        void reset() {
            used.store(0);
        }
    };
    
    // 비동기 작업 큐
    struct WorkQueue {
        enum WorkType {
            CAPTURE_PROCESS = 0,
            DETECTION = 1,
            TRACKING = 2,
            PID_CONTROL = 3,
            MOUSE_CALC = 4
        };
        
        struct WorkItem {
            WorkType type;
            void* data;
            size_t data_size;
            cudaEvent_t completion_event;
        };
        
        WorkItem* d_queue;
        int* d_head;
        int* d_tail;
        int max_items;
        
        WorkQueue(int size = 128) : max_items(size) {
            cudaMalloc(&d_queue, sizeof(WorkItem) * max_items);
            cudaMalloc(&d_head, sizeof(int));
            cudaMalloc(&d_tail, sizeof(int));
            cudaMemset(d_head, 0, sizeof(int));
            cudaMemset(d_tail, 0, sizeof(int));
        }
        
        ~WorkQueue() {
            cudaFree(d_queue);
            cudaFree(d_head);
            cudaFree(d_tail);
        }
    };
    
    // 전용 CUDA 스트림들
    cudaStream_t highPriorityStream;   // 마우스 계산용
    cudaStream_t normalStream;          // 일반 처리용
    cudaStream_t backgroundStream;      // 백그라운드 작업용
    
    // 메모리 풀
    std::unique_ptr<MemoryPool> smallPool;   // 작은 할당용 (< 1MB)
    std::unique_ptr<MemoryPool> largePool;   // 큰 할당용 (이미지 등)
    
    // 작업 큐
    std::unique_ptr<WorkQueue> workQueue;
    
    // 성능 카운터
    std::atomic<uint64_t> gpu_tasks_completed{0};
    std::atomic<uint64_t> cpu_tasks_avoided{0};
    
public:
    GpuWorkloadManager() {
        // 높은 우선순위 스트림 생성
        int leastPriority, greatestPriority;
        cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        
        cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, greatestPriority);
        cudaStreamCreateWithPriority(&normalStream, cudaStreamNonBlocking, 0);
        cudaStreamCreateWithPriority(&backgroundStream, cudaStreamNonBlocking, leastPriority);
        
        // 메모리 풀 초기화
        smallPool = std::make_unique<MemoryPool>(10 * 1024 * 1024);  // 10MB
        largePool = std::make_unique<MemoryPool>(100 * 1024 * 1024); // 100MB
        
        // 작업 큐 초기화
        workQueue = std::make_unique<WorkQueue>();
    }
    
    ~GpuWorkloadManager() {
        cudaStreamDestroy(highPriorityStream);
        cudaStreamDestroy(normalStream);
        cudaStreamDestroy(backgroundStream);
    }
    
    // CPU 작업을 GPU로 오프로드
    template<typename KernelFunc>
    void offloadToGpu(KernelFunc kernel, dim3 grid, dim3 block, 
                      void** args, size_t sharedMem = 0, 
                      bool highPriority = false) {
        cudaStream_t stream = highPriority ? highPriorityStream : normalStream;
        
        // 커널 실행
        cudaLaunchKernel((void*)kernel, grid, block, args, sharedMem, stream);
        
        gpu_tasks_completed.fetch_add(1);
        cpu_tasks_avoided.fetch_add(1);
    }
    
    // 스트림 동기화 최소화
    bool checkCompletion(cudaStream_t stream) {
        cudaError_t result = cudaStreamQuery(stream);
        return (result == cudaSuccess);
    }
    
    // 메모리 할당 최적화
    void* allocateGpuMemory(size_t size) {
        if (size < 1024 * 1024) {
            return smallPool->allocate(size);
        } else {
            return largePool->allocate(size);
        }
    }
    
    // 프레임별 리셋
    void resetFrame() {
        smallPool->reset();
        // largePool은 이미지용이므로 유지
    }
    
    // 성능 통계
    void getStats(uint64_t& tasks_completed, uint64_t& cpu_avoided) {
        tasks_completed = gpu_tasks_completed.load();
        cpu_avoided = cpu_tasks_avoided.load();
    }
};

// 배치 처리 커널 - 여러 작업을 한번에 처리
__global__ void batchProcessingKernel(
    float* input_data,
    float* output_data,
    int* work_types,
    int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    int work_type = work_types[tid];
    
    // 작업 타입별 처리
    switch(work_type) {
        case 0: // 이미지 전처리
            // GPU에서 직접 처리
            output_data[tid] = input_data[tid] * 2.0f;
            break;
            
        case 1: // 필터링
            // GPU에서 직접 처리
            output_data[tid] = fmaxf(0.0f, input_data[tid]);
            break;
            
        case 2: // 정규화
            // GPU에서 직접 처리
            output_data[tid] = input_data[tid] / 255.0f;
            break;
    }
}

// 병렬 리덕션 커널 - CPU 대신 GPU에서 집계
__global__ void parallelReductionKernel(
    float* input,
    float* output,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    
    // 첫 번째 리덕션
    float mySum = (i < n) ? input[i] : 0;
    if (i + blockDim.x < n) mySum += input[i + blockDim.x];
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // 블록 내 리덕션
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 결과 저장
    if (tid == 0) output[blockIdx.x] = mySum;
}

// Persistent 커널 - CPU 폴링 제거
__global__ void persistentWorkerKernel(
    volatile int* work_available,
    float* work_data,
    float* results,
    volatile int* should_exit
) {
    __shared__ bool done;
    
    if (threadIdx.x == 0) {
        done = false;
    }
    __syncthreads();
    
    while (!done) {
        if (threadIdx.x == 0) {
            // 작업 확인
            if (*should_exit) {
                done = true;
            } else if (*work_available) {
                // 작업 플래그 리셋
                *work_available = 0;
            }
        }
        __syncthreads();
        
        if (!done && *work_available == 0) {
            // 실제 작업 수행
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            results[tid] = work_data[tid] * 2.0f;
        }
        
        // CPU 부하 감소를 위한 짧은 대기
        __nanosleep(100);
    }
}

// C++ 인터페이스
extern "C" {
    GpuWorkloadManager* createGpuWorkloadManager() {
        return new GpuWorkloadManager();
    }
    
    void destroyGpuWorkloadManager(GpuWorkloadManager* manager) {
        delete manager;
    }
    
    void offloadBatchWork(
        GpuWorkloadManager* manager,
        float* d_input,
        float* d_output,
        int* d_work_types,
        int batch_size
    ) {
        dim3 block(256);
        dim3 grid((batch_size + block.x - 1) / block.x);
        
        void* args[] = { &d_input, &d_output, &d_work_types, &batch_size };
        manager->offloadToGpu(batchProcessingKernel, grid, block, args);
    }
    
    void startPersistentKernel(
        volatile int* d_work_available,
        float* d_work_data,
        float* d_results,
        volatile int* d_should_exit
    ) {
        dim3 block(256);
        dim3 grid(1);
        
        persistentWorkerKernel<<<grid, block>>>(
            d_work_available, d_work_data, d_results, d_should_exit
        );
    }
}