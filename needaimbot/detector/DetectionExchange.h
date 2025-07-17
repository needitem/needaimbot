#ifndef DETECTION_EXCHANGE_H
#define DETECTION_EXCHANGE_H

#include <atomic>
#include <chrono>
#include "detector.h"
#include "../utils/lock_free_queue.h"

// Lock-free detection data structure
struct DetectionPacket {
    Detection bestTarget;
    bool hasTarget;
    int detectionVersion;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    DetectionPacket() : hasTarget(false), detectionVersion(0) {
        memset(&bestTarget, 0, sizeof(Detection));
    }
};

// Lock-free detection exchange system
class DetectionExchange {
private:
    DoubleBuffer<DetectionPacket> detectionBuffer;
    std::atomic<int> currentVersion{0};
    
public:
    // Called by detector thread to publish new detection
    void publishDetection(const Detection& target, bool hasTarget) {
        auto& writeBuffer = detectionBuffer.getWriteBuffer();
        writeBuffer.bestTarget = target;
        writeBuffer.hasTarget = hasTarget;
        writeBuffer.detectionVersion = currentVersion.fetch_add(1, std::memory_order_acq_rel) + 1;
        writeBuffer.timestamp = std::chrono::high_resolution_clock::now();
        detectionBuffer.swapBuffers();
    }
    
    // Called by mouse thread to get latest detection
    bool getLatestDetection(DetectionPacket& packet) {
        return detectionBuffer.getReadBuffer(packet);
    }
    
    // Check if new data is available without consuming
    bool hasNewDetection() const {
        return detectionBuffer.hasNewData();
    }
    
    int getCurrentVersion() const {
        return currentVersion.load(std::memory_order_acquire);
    }
};

// Alternative: Single producer, single consumer ring buffer
template<size_t Size = 64>
class SPSCDetectionQueue {
private:
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
    struct alignas(64) Entry {
        DetectionPacket packet;
        std::atomic<bool> ready{false};
    };
    
    alignas(64) Entry buffer[Size];
    alignas(64) std::atomic<size_t> writePos{0};
    alignas(64) std::atomic<size_t> readPos{0};
    
    static constexpr size_t indexMask = Size - 1;
    
public:
    bool push(const Detection& target, bool hasTarget) {
        size_t currentWrite = writePos.load(std::memory_order_relaxed);
        size_t nextWrite = (currentWrite + 1) & indexMask;
        
        // Check if full
        if (nextWrite == readPos.load(std::memory_order_acquire)) {
            return false;
        }
        
        auto& entry = buffer[currentWrite];
        entry.packet.bestTarget = target;
        entry.packet.hasTarget = hasTarget;
        entry.packet.timestamp = std::chrono::high_resolution_clock::now();
        entry.ready.store(true, std::memory_order_release);
        
        writePos.store(nextWrite, std::memory_order_release);
        return true;
    }
    
    bool pop(DetectionPacket& packet) {
        size_t currentRead = readPos.load(std::memory_order_relaxed);
        
        // Check if empty
        if (currentRead == writePos.load(std::memory_order_acquire)) {
            return false;
        }
        
        auto& entry = buffer[currentRead];
        
        // Wait for data to be ready
        if (!entry.ready.load(std::memory_order_acquire)) {
            return false;
        }
        
        packet = entry.packet;
        entry.ready.store(false, std::memory_order_release);
        
        readPos.store((currentRead + 1) & indexMask, std::memory_order_release);
        return true;
    }
    
    bool empty() const {
        return readPos.load(std::memory_order_acquire) == 
               writePos.load(std::memory_order_acquire);
    }
};

#endif // DETECTION_EXCHANGE_H