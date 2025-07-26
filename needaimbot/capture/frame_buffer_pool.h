#pragma once

#include "../cuda/simple_cuda_mat.h"
#include <vector>
#include <queue>
#include <mutex>
#include <memory>
#include <atomic>

class FrameBufferPool {
public:
    FrameBufferPool(size_t initial_size = 10);
    ~FrameBufferPool();

    // CPU 버퍼 관리
    SimpleMat acquireCpuBuffer(int height, int width, int channels);
    void releaseCpuBuffer(SimpleMat&& buffer);

    // GPU 버퍼 관리
    SimpleCudaMat acquireGpuBuffer(int height, int width, int channels);
    void releaseGpuBuffer(SimpleCudaMat&& buffer);

    // 풀 상태 조회
    size_t getCpuPoolSize() const { return m_cpuPool.size(); }
    size_t getGpuPoolSize() const { return m_gpuPool.size(); }
    
    // 통계
    struct Stats {
        std::atomic<size_t> cpu_acquires{0};
        std::atomic<size_t> cpu_releases{0};
        std::atomic<size_t> cpu_creates{0};
        std::atomic<size_t> gpu_acquires{0};
        std::atomic<size_t> gpu_releases{0};
        std::atomic<size_t> gpu_creates{0};
    };
    
    const Stats& getStats() const { return m_stats; }

private:
    struct BufferInfo {
        int height;
        int width;
        int channels;
        
        bool matches(int h, int w, int c) const {
            return height == h && width == w && channels == c;
        }
    };

    // CPU 버퍼 풀
    std::vector<std::pair<SimpleMat, BufferInfo>> m_cpuPool;
    mutable std::mutex m_cpuMutex;

    // GPU 버퍼 풀
    std::vector<std::pair<SimpleCudaMat, BufferInfo>> m_gpuPool;
    mutable std::mutex m_gpuMutex;

    // 최대 풀 크기
    const size_t m_maxPoolSize;
    
    // 통계
    mutable Stats m_stats;
};

// 전역 프레임 버퍼 풀 인스턴스
extern std::unique_ptr<FrameBufferPool> g_frameBufferPool;