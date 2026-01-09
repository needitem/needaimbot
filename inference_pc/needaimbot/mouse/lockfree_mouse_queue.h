#pragma once

#include <atomic>
#include <array>

// Lock-free SPSC (Single Producer Single Consumer) queue for mouse commands
// GPU writes, CPU reads - no mutex needed

enum class MouseCommandType : uint8_t {
    NONE = 0,
    MOVE = 1,
    PRESS = 2,
    RELEASE = 3
};

struct MouseCommand {
    MouseCommandType type;
    int dx;
    int dy;
};

template<size_t Size = 1024>
class LockFreeMouseQueue {
private:
    std::array<MouseCommand, Size> buffer;
    alignas(64) std::atomic<size_t> writeIndex{0};
    alignas(64) std::atomic<size_t> readIndex{0};

public:
    // Producer (GPU thread) writes to queue
    bool push(MouseCommandType type, int dx = 0, int dy = 0) {
        size_t currentWrite = writeIndex.load(std::memory_order_relaxed);
        size_t nextWrite = (currentWrite + 1) % Size;

        if (nextWrite == readIndex.load(std::memory_order_acquire)) {
            return false; // Queue full
        }

        buffer[currentWrite] = {type, dx, dy};
        writeIndex.store(nextWrite, std::memory_order_release);
        return true;
    }

    // Consumer (CPU thread) reads from queue
    bool pop(MouseCommand& cmd) {
        size_t currentRead = readIndex.load(std::memory_order_relaxed);

        if (currentRead == writeIndex.load(std::memory_order_acquire)) {
            return false; // Queue empty
        }

        cmd = buffer[currentRead];
        readIndex.store((currentRead + 1) % Size, std::memory_order_release);
        return true;
    }

    bool isEmpty() const {
        return readIndex.load(std::memory_order_acquire) ==
               writeIndex.load(std::memory_order_acquire);
    }
};
