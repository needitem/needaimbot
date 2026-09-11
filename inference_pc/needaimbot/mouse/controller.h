#pragma once

// Movement/control layer: turns an aim decision (dx, dy) into physical mouse
// motion. Owns the async move-sender thread, the lock-free move queue, and
// no-recoil ticking - decoupled from the capture/inference orchestration in
// simple_main.cpp, which only ever calls submitAimMovement() (from the GPU
// completion callback) and tickNoRecoil() (from its main loop).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

#include "MakcuConnection.h"

namespace controller {

using Clock = std::chrono::steady_clock;

constexpr uint8_t kMakcuLeftMask = 0x01;
constexpr uint8_t kMakcuRightMask = 0x02;
constexpr uint8_t kMakcuMiddleMask = 0x04;
constexpr uint8_t kMakcuSide1Mask = 0x08;
constexpr uint8_t kMakcuSide2Mask = 0x10;

inline bool maskAiming(uint8_t mask) {
    return (mask & (kMakcuRightMask | kMakcuSide2Mask)) != 0;
}

inline bool maskThumbAiming(uint8_t mask) {
    return (mask & kMakcuSide2Mask) != 0;
}

inline bool maskShooting(uint8_t mask) {
    return (mask & kMakcuLeftMask) != 0;
}

struct MoveCommand {
    int dx = 0;
    int dy = 0;
};

struct MoveQueueSlot {
    std::atomic<uint64_t> sequence{0};
    MoveCommand command{};
};

// Lock-free MPSC ring buffer of pending mouse moves, drained by the sender
// thread.
struct MoveQueue {
    static constexpr uint32_t kCapacity = 4096;  // Must stay power-of-two.
    static_assert((kCapacity & (kCapacity - 1)) == 0, "MoveQueue capacity must be power-of-two");

    std::array<MoveQueueSlot, kCapacity> ring{};
    std::atomic<uint64_t> enqueuePos{0};
    std::atomic<uint64_t> dequeuePos{0};

    MoveQueue() {
        for (uint64_t i = 0; i < kCapacity; ++i) {
            ring[static_cast<size_t>(i)].sequence.store(i, std::memory_order_relaxed);
        }
    }

    bool tryPush(const MoveCommand& cmd) {
        uint64_t pos = enqueuePos.load(std::memory_order_relaxed);
        for (;;) {
            MoveQueueSlot& slot = ring[static_cast<size_t>(pos & (kCapacity - 1))];
            const uint64_t seq = slot.sequence.load(std::memory_order_acquire);
            const int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);

            if (diff == 0) {
                if (enqueuePos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    slot.command = cmd;
                    slot.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
                pos = enqueuePos.load(std::memory_order_relaxed);
            }
        }
    }

    bool tryPop(MoveCommand& out) {
        uint64_t pos = dequeuePos.load(std::memory_order_relaxed);
        for (;;) {
            MoveQueueSlot& slot = ring[static_cast<size_t>(pos & (kCapacity - 1))];
            const uint64_t seq = slot.sequence.load(std::memory_order_acquire);
            const int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);

            if (diff == 0) {
                if (dequeuePos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    out = slot.command;
                    slot.sequence.store(pos + kCapacity, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;  // Queue empty
            } else {
                pos = dequeuePos.load(std::memory_order_relaxed);
            }
        }
    }

    bool hasPending() const {
        return dequeuePos.load(std::memory_order_acquire) !=
               enqueuePos.load(std::memory_order_acquire);
    }
};

// Cached settings the controller needs, decoupled from the app-wide Config.
struct Settings {
    bool forceAimOn = false;
    // Already ANDed with (mouseMinIntervalMs <= 0) by the caller.
    bool directAimMoveInCallback = false;
    int mouseMinIntervalMs = 1;

    bool noRecoilEnabled = true;
    float recoilCompX = 0.0f;
    float recoilCompY = 0.8f;
    int recoilTickMs = 10;
    // NOTE: the static shoot-offset aim-shift no longer lives here. It is a
    // reference-point shift applied inside the GPU controller (AimConfig::
    // shoot_offset_x/y), fed per-frame from Config::shootOffsetX/Y while
    // shooting - see simple_main.cpp.
};

class MouseController {
public:
    void configure(MakcuConnection& makcu, const Settings& settings) {
        makcu_ = &makcu;
        settings_ = settings;
        lastRecoilTime_ = Clock::now();
    }

    // onThreadStart runs once on the sender thread before it starts draining
    // the queue - the caller uses it to apply its own realtime/affinity policy
    // (kept out of this header, which stays platform-agnostic).
    void start(std::function<void()> onThreadStart = nullptr) {
        senderRunning_.store(true, std::memory_order_relaxed);
        senderThread_ = std::thread([this, onThreadStart = std::move(onThreadStart)]() {
            if (onThreadStart) onThreadStart();
            senderLoop();
        });
    }

    void stop() {
        {
            // The flag and the notify must be ordered against the sender's
            // predicate evaluation by the same mutex the sender waits on.
            // Storing it outside the lock let the sender read "still running"
            // and an empty queue, then miss the notify and sleep forever - a
            // join() that never returns.
            std::lock_guard<std::mutex> lock(moveQueueCvMutex_);
            senderRunning_.store(false, std::memory_order_relaxed);
        }
        moveQueueCv_.notify_all();
        if (senderThread_.joinable()) {
            senderThread_.join();
        }
    }

    bool directAimMoveInCallback() const { return settings_.directAimMoveInCallback; }

    // Called from the GPU completion callback once a target is found and
    // aiming is active. The static shoot-offset is NOT applied here: it is a
    // reference-point shift folded into the GPU controller's error term (see
    // AimConfig::shoot_offset_x/y), so the aim converges to the offset and
    // holds. Applying it here as a per-frame additive nudge is what made the
    // aim drift/jerk upward while shooting.
    void submitAimMovement(int dx, int dy) {
        if (settings_.directAimMoveInCallback) {
            if (dx != 0 || dy != 0) {
                makcu_->move(dx, dy);
            }
            return;
        }
        const MoveCommand cmd{dx, dy};
        if (moveQueue_.tryPush(cmd)) {
            notifySender();
        } else {
            moveQueueDropped_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Called every ~1ms from the main loop with the current button mask;
    // fires a no-recoil tick every recoilTickMs while shooting + aiming.
    void tickNoRecoil(uint8_t buttonMask) {
        if (!settings_.noRecoilEnabled) return;

        const bool firing = maskShooting(buttonMask) &&
                             (settings_.forceAimOn || maskAiming(buttonMask));
        if (!firing) {
            // Drop any stale sub-pixel carry whenever a firing burst is not
            // active, so it can't survive into the next burst.
            recoilResidualX_ = 0.0f;
            recoilResidualY_ = 0.0f;
            return;
        }

        const auto now = Clock::now();
        const auto tick = std::chrono::milliseconds(settings_.recoilTickMs);
        if (now - lastRecoilTime_ < tick) return;

        emitRecoilTick();
        lastRecoilTime_ += tick;
        if (lastRecoilTime_ < now - tick) {
            lastRecoilTime_ = now;  // fell far behind -> resync, no burst
        }
    }

    uint64_t takeDroppedCount() {
        return moveQueueDropped_.exchange(0, std::memory_order_relaxed);
    }

private:
    // The sender's wait predicate reads moveQueue_.hasPending(), which is not
    // guarded by moveQueueCvMutex_. Pushing and then notifying without taking
    // that mutex therefore races the sender's own predicate check: it can look,
    // find the queue empty, and go to sleep in the window between the push and
    // the notify - a lost wakeup that strands the move until the next one. Only
    // used on the queued path; the shipped config sends straight from the
    // completion callback and never reaches this.
    void notifySender() {
        {
            std::lock_guard<std::mutex> lock(moveQueueCvMutex_);
        }
        moveQueueCv_.notify_one();
    }

    // Best-effort queued move (falls back to an immediate send if the queue is
    // momentarily full) - used for no-recoil, which must never be silently
    // dropped the way an occasional aim-move sample can be.
    void queueMove(int dx, int dy) {
        if (dx == 0 && dy == 0) return;
        const MoveCommand cmd{dx, dy};
        if (moveQueue_.tryPush(cmd)) {
            notifySender();
        } else {
            makcu_->move(dx, dy);
        }
    }

    // Carries the sub-pixel remainder so a fractional comp value (e.g. 1.3)
    // averages out instead of truncating to int every tick.
    void emitRecoilTick() {
        recoilResidualX_ += settings_.recoilCompX;
        recoilResidualY_ += settings_.recoilCompY;
        const int dx = static_cast<int>(recoilResidualX_);
        const int dy = static_cast<int>(recoilResidualY_);
        recoilResidualX_ -= static_cast<float>(dx);
        recoilResidualY_ -= static_cast<float>(dy);
        queueMove(dx, dy);
    }

    void senderLoop() {
        const int senderMinIntervalMs = std::max(0, settings_.mouseMinIntervalMs);

        MoveCommand cmd;
        // Accumulator may exceed the per-send MAKCU range (+-127); the excess
        // is carried into the next flush instead of being clamped away. Cap
        // the raw accumulator to a few frames' worth so a runaway producer
        // can't pile up unbounded.
        constexpr int kPendingAccumCap = 512;
        int pendingDx = 0;
        int pendingDy = 0;
        auto nextSendTime = Clock::now();

        auto hasPendingMove = [&]() { return pendingDx != 0 || pendingDy != 0; };
        auto flushMove = [&](Clock::time_point now) -> bool {
            if (!hasPendingMove()) return false;
            if (senderMinIntervalMs > 0 && now < nextSendTime) return false;
            const int sendDx = std::clamp(pendingDx, -127, 127);
            const int sendDy = std::clamp(pendingDy, -127, 127);
            makcu_->move(sendDx, sendDy);
            // Carry any overflow into the next flush instead of dropping it.
            pendingDx = std::clamp(pendingDx - sendDx, -kPendingAccumCap, kPendingAccumCap);
            pendingDy = std::clamp(pendingDy - sendDy, -kPendingAccumCap, kPendingAccumCap);
            if (senderMinIntervalMs > 0) {
                nextSendTime = now + std::chrono::milliseconds(senderMinIntervalMs);
            }
            return true;
        };

        while (senderRunning_.load(std::memory_order_relaxed) || moveQueue_.hasPending() || hasPendingMove()) {
            while (moveQueue_.tryPop(cmd)) {
                if (cmd.dx != 0 || cmd.dy != 0) {
                    pendingDx = std::clamp(pendingDx + cmd.dx, -kPendingAccumCap, kPendingAccumCap);
                    pendingDy = std::clamp(pendingDy + cmd.dy, -kPendingAccumCap, kPendingAccumCap);
                }
            }

            const auto now = Clock::now();
            if (flushMove(now)) {
                continue;
            }

            if (senderRunning_.load(std::memory_order_relaxed) || moveQueue_.hasPending() || hasPendingMove()) {
                std::unique_lock<std::mutex> lock(moveQueueCvMutex_);
                if (hasPendingMove() && senderMinIntervalMs > 0 && now < nextSendTime) {
                    moveQueueCv_.wait_until(lock, nextSendTime, [&]() {
                        return !senderRunning_.load(std::memory_order_relaxed) || moveQueue_.hasPending();
                    });
                } else {
                    moveQueueCv_.wait(lock, [&]() {
                        return !senderRunning_.load(std::memory_order_relaxed) || moveQueue_.hasPending();
                    });
                }
            } else {
                break;
            }
        }
    }

    MakcuConnection* makcu_ = nullptr;
    Settings settings_;

    MoveQueue moveQueue_;
    std::condition_variable moveQueueCv_;
    std::mutex moveQueueCvMutex_;
    std::atomic<uint64_t> moveQueueDropped_{0};

    std::atomic<bool> senderRunning_{false};
    std::thread senderThread_;

    Clock::time_point lastRecoilTime_{};
    float recoilResidualX_ = 0.0f;
    float recoilResidualY_ = 0.0f;
};

}  // namespace controller
