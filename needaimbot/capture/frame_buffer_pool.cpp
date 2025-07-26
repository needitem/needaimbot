#include "frame_buffer_pool.h"
#include <algorithm>
#include <iostream>

std::unique_ptr<FrameBufferPool> g_frameBufferPool;

FrameBufferPool::FrameBufferPool(size_t initial_size) 
    : m_maxPoolSize(initial_size * 2) {
    // 초기 버퍼들을 미리 할당하지 않음 (첫 사용 시 생성)
}

FrameBufferPool::~FrameBufferPool() {
    // 소멸자에서 자동으로 모든 버퍼 해제
}

SimpleMat FrameBufferPool::acquireCpuBuffer(int height, int width, int channels) {
    m_stats.cpu_acquires++;
    
    std::lock_guard<std::mutex> lock(m_cpuMutex);
    
    // 재사용 가능한 버퍼 찾기
    auto it = std::find_if(m_cpuPool.begin(), m_cpuPool.end(),
        [height, width, channels](const auto& pair) {
            return pair.second.matches(height, width, channels);
        });
    
    if (it != m_cpuPool.end()) {
        SimpleMat buffer = std::move(it->first);
        m_cpuPool.erase(it);
        return buffer;
    }
    
    // 재사용 가능한 버퍼가 없으면 새로 생성
    m_stats.cpu_creates++;
    return SimpleMat(height, width, channels);
}

void FrameBufferPool::releaseCpuBuffer(SimpleMat&& buffer) {
    if (buffer.empty()) return;
    
    m_stats.cpu_releases++;
    
    std::lock_guard<std::mutex> lock(m_cpuMutex);
    
    // 풀이 최대 크기에 도달하지 않았으면 버퍼 저장
    if (m_cpuPool.size() < m_maxPoolSize) {
        BufferInfo info{buffer.rows(), buffer.cols(), buffer.channels()};
        m_cpuPool.emplace_back(std::move(buffer), info);
    }
    // 최대 크기 초과 시 버퍼는 자동으로 해제됨
}

SimpleCudaMat FrameBufferPool::acquireGpuBuffer(int height, int width, int channels) {
    m_stats.gpu_acquires++;
    
    std::lock_guard<std::mutex> lock(m_gpuMutex);
    
    // 재사용 가능한 버퍼 찾기
    auto it = std::find_if(m_gpuPool.begin(), m_gpuPool.end(),
        [height, width, channels](const auto& pair) {
            return pair.second.matches(height, width, channels);
        });
    
    if (it != m_gpuPool.end()) {
        SimpleCudaMat buffer = std::move(it->first);
        m_gpuPool.erase(it);
        return buffer;
    }
    
    // 재사용 가능한 버퍼가 없으면 새로 생성
    m_stats.gpu_creates++;
    SimpleCudaMat buffer;
    buffer.create(height, width, channels);
    return buffer;
}

void FrameBufferPool::releaseGpuBuffer(SimpleCudaMat&& buffer) {
    if (buffer.empty()) return;
    
    m_stats.gpu_releases++;
    
    std::lock_guard<std::mutex> lock(m_gpuMutex);
    
    // 풀이 최대 크기에 도달하지 않았으면 버퍼 저장
    if (m_gpuPool.size() < m_maxPoolSize) {
        BufferInfo info{buffer.rows(), buffer.cols(), buffer.channels()};
        m_gpuPool.emplace_back(std::move(buffer), info);
    }
    // 최대 크기 초과 시 버퍼는 자동으로 해제됨
}