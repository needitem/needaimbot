#include "udp_capture.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <new>

#ifdef _WIN32
#define SOCKADDR struct sockaddr
#define WSAETIMEDOUT WSAETIMEDOUT
#else
#define SOCKADDR struct sockaddr
#define WSAETIMEDOUT ETIMEDOUT
#include <pthread.h>
#include <sched.h>
#endif

namespace {
bool isFrameIdNewer(uint32_t candidate, uint32_t baseline) {
    return static_cast<int32_t>(candidate - baseline) > 0;
}

constexpr auto kFrameIdResetIdleTimeout = std::chrono::seconds(2);

// Wall-clock (system_clock) epoch microseconds - must match the game PC's
// nowUnixMicros() so ping/pong timestamps are comparable across machines.
int64_t nowUnixMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
}  // namespace

UDPCapture::UDPCapture() = default;

UDPCapture::~UDPCapture() {
    Shutdown();
}

UDPCapture::FrameFragments* UDPCapture::acquireFragment() {
    if (m_freeFragmentCount == 0) {
        return nullptr;
    }
    const int idx = m_freeFragmentStack[--m_freeFragmentCount];
    FrameFragments& frag = m_fragmentStorage[static_cast<size_t>(idx)];
    frag.active = false;
    frag.frameId = 0;
    frag.activeListIndex = -1;
    frag.nextInBucket = -1;
    frag.slotIndex = idx;
    return &frag;
}

void UDPCapture::releaseFragment(FrameFragments* frag) {
    if (!frag) return;
    const int idx = frag->slotIndex;
    if (idx < 0 || idx >= static_cast<int>(MAX_FRAGMENT_SLOTS)) return;

    frag->active = false;
    frag->frameId = 0;
    frag->slotIndex = idx;
    frag->activeListIndex = -1;
    frag->nextInBucket = -1;
    frag->receivedMask = 0;
    frag->useReceivedMask = false;
    frag->totalPackets = 0;
    frag->receivedCount = 0;
    frag->width = 0;
    frag->height = 0;
    frag->bytesPerPixel = 3;
    frag->pixelFormat = UDP_PIXEL_FORMAT_RGB;
    frag->frameBytes = 0;
    frag->bufferIndex = -1;
    frag->dropped = false;
    frag->lastUpdate = std::chrono::steady_clock::time_point{};
    if (m_freeFragmentCount < MAX_FRAGMENT_SLOTS) {
        m_freeFragmentStack[m_freeFragmentCount++] = idx;
    }
}

UDPCapture::FrameFragments* UDPCapture::findFragment(uint32_t frameId) {
    const size_t bucket = static_cast<size_t>(frameId) & (FRAGMENT_BUCKETS - 1);
    int idx = m_bucketHeads[bucket];
    while (idx >= 0) {
        FrameFragments& frag = m_fragmentStorage[static_cast<size_t>(idx)];
        if (frag.active && frag.frameId == frameId) {
            return &frag;
        }
        idx = frag.nextInBucket;
    }
    return nullptr;
}

void UDPCapture::linkFragment(FrameFragments* frag, uint32_t frameId) {
    if (!frag) return;
    const int idx = frag->slotIndex;
    if (idx < 0 || idx >= static_cast<int>(MAX_FRAGMENT_SLOTS)) return;

    const size_t bucket = static_cast<size_t>(frameId) & (FRAGMENT_BUCKETS - 1);
    frag->frameId = frameId;
    frag->active = true;
    frag->activeListIndex = static_cast<int>(m_activeFragmentCount);
    if (m_activeFragmentCount < MAX_FRAGMENT_SLOTS) {
        m_activeFragmentSlots[m_activeFragmentCount] = idx;
    }
    frag->nextInBucket = m_bucketHeads[bucket];
    m_bucketHeads[bucket] = idx;
    ++m_activeFragmentCount;
}

void UDPCapture::unlinkFragment(FrameFragments* frag) {
    if (!frag || !frag->active) return;
    const int idx = frag->slotIndex;
    if (idx < 0 || idx >= static_cast<int>(MAX_FRAGMENT_SLOTS)) {
        frag->active = false;
        return;
    }

    const size_t bucket = static_cast<size_t>(frag->frameId) & (FRAGMENT_BUCKETS - 1);
    int* current = &m_bucketHeads[bucket];
    while (*current >= 0) {
        if (*current == idx) {
            *current = m_fragmentStorage[static_cast<size_t>(idx)].nextInBucket;
            break;
        }
        current = &m_fragmentStorage[static_cast<size_t>(*current)].nextInBucket;
    }

    frag->active = false;
    const int activePos = frag->activeListIndex;
    if (activePos >= 0 && static_cast<size_t>(activePos) < m_activeFragmentCount) {
        const size_t pos = static_cast<size_t>(activePos);
        const size_t last = m_activeFragmentCount - 1;
        const int movedIdx = m_activeFragmentSlots[last];
        m_activeFragmentSlots[pos] = movedIdx;
        if (movedIdx >= 0 && movedIdx < static_cast<int>(MAX_FRAGMENT_SLOTS)) {
            m_fragmentStorage[static_cast<size_t>(movedIdx)].activeListIndex = static_cast<int>(pos);
        }
        m_activeFragmentSlots[last] = -1;
        --m_activeFragmentCount;
    }
    frag->activeListIndex = -1;
    frag->nextInBucket = -1;
}

bool UDPCapture::allocatePinnedBuffers(size_t size) {
    if (size == 0) return false;

    freePinnedBuffers();
    bool pinnedOk = true;
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        // Allocate MAPPED pinned memory (cudaHostAllocMapped) instead of plain
        // cudaMallocHost. On Tegra/Orin (unified memory) this lets the inference
        // engine obtain a device pointer via cudaHostGetDevicePointer() and have
        // the preprocess kernel read the received frame directly - eliminating the
        // per-frame H2D copy. On discrete GPUs the mapped flag is harmless (the
        // engine still stages via H2D). Freed with cudaFreeHost() as before.
        //
        // WriteCombined: this buffer is only ever WRITTEN by the CPU (the per-chunk
        // memcpy in processPacket) and READ by the GPU (preprocess kernel) or the
        // H2D DMA engine (discrete GPU). No CPU code reads the frame back on the hot
        // path, so WC gives fast non-temporal CPU stores and avoids polluting the RT
        // receive core's cache with ~1MB/frame it never re-reads. The only CPU reader
        // is the 1Hz debug frame dump (slow WC reads there, but off the hot path).
        cudaError_t err = cudaHostAlloc(&m_pinnedFrameBuffer[i], size,
                                        cudaHostAllocMapped | cudaHostAllocWriteCombined);
        if (err != cudaSuccess) {
            std::cerr << "[UDPCapture] Pinned alloc failed at buffer " << i
                      << ": " << cudaGetErrorString(err) << "\n";
            pinnedOk = false;
            break;
        }
    }

    if (!pinnedOk) {
        for (int i = 0; i < NUM_BUFFERS; ++i) {
            if (m_pinnedFrameBuffer[i]) {
                cudaFreeHost(m_pinnedFrameBuffer[i]);
                m_pinnedFrameBuffer[i] = nullptr;
            }
        }
        for (int i = 0; i < NUM_BUFFERS; ++i) {
            m_pinnedFrameBuffer[i] = new (std::nothrow) uint8_t[size];
            if (!m_pinnedFrameBuffer[i]) {
                freePinnedBuffers();
                return false;
            }
        }
    }

    m_pinnedBufferSize = size;
    m_usePinnedMemory = pinnedOk;
    m_latestBufferIndex.store(-1, std::memory_order_relaxed);
    m_publishSeq.store(0, std::memory_order_relaxed);
    m_consumedSeq = 0;
    m_reserveCursor = 0;
    m_hasLatestPublishedFrameId = false;
    m_latestPublishedFrameId = 0;
    m_latestPublishTime = std::chrono::steady_clock::time_point{};
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        m_bufferState[i].store(BUFFER_FREE, std::memory_order_relaxed);
        m_bufferWidth[i].store(0, std::memory_order_relaxed);
        m_bufferHeight[i].store(0, std::memory_order_relaxed);
        m_bufferBytesPerPixel[i].store(3, std::memory_order_relaxed);
        m_bufferPixelFormat[i].store(UDP_PIXEL_FORMAT_RGB, std::memory_order_relaxed);
        m_bufferFrameId[i].store(0, std::memory_order_relaxed);
        m_bufferCaptureUnixMicros[i].store(0, std::memory_order_relaxed);
    }

    std::cout << "[UDPCapture] Allocated " << (size / 1024) << "KB x " << NUM_BUFFERS
              << (m_usePinnedMemory ? " pinned buffers\n" : " heap buffers (fallback)\n");
    return true;
}

bool UDPCapture::ensurePinnedCapacity(size_t size) {
    if (m_pinnedBufferSize >= size && m_pinnedFrameBuffer[0]) return true;

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        if (m_bufferState[i].load(std::memory_order_acquire) != BUFFER_FREE) {
            return false;
        }
    }
    return allocatePinnedBuffers(size);
}

void UDPCapture::freePinnedBuffers() {
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        if (m_pinnedFrameBuffer[i]) {
            if (m_usePinnedMemory) {
                cudaFreeHost(m_pinnedFrameBuffer[i]);
            } else {
                delete[] m_pinnedFrameBuffer[i];
            }
            m_pinnedFrameBuffer[i] = nullptr;
        }
        m_bufferState[i].store(BUFFER_FREE, std::memory_order_relaxed);
        m_bufferWidth[i].store(0, std::memory_order_relaxed);
        m_bufferHeight[i].store(0, std::memory_order_relaxed);
        m_bufferBytesPerPixel[i].store(3, std::memory_order_relaxed);
        m_bufferPixelFormat[i].store(UDP_PIXEL_FORMAT_RGB, std::memory_order_relaxed);
        m_bufferFrameId[i].store(0, std::memory_order_relaxed);
        m_bufferCaptureUnixMicros[i].store(0, std::memory_order_relaxed);
    }
    m_pinnedBufferSize = 0;
    m_usePinnedMemory = false;
    m_latestBufferIndex.store(-1, std::memory_order_relaxed);
    m_publishSeq.store(0, std::memory_order_relaxed);
    m_consumedSeq = 0;
    m_reserveCursor = 0;
    m_hasLatestPublishedFrameId = false;
    m_latestPublishedFrameId = 0;
    m_latestPublishTime = std::chrono::steady_clock::time_point{};
}

int UDPCapture::reserveAssemblingBuffer() {
    const int latest = m_latestBufferIndex.load(std::memory_order_acquire);
    for (int attempt = 0; attempt < NUM_BUFFERS; ++attempt) {
        const int idx = (m_reserveCursor + attempt) % NUM_BUFFERS;
        int expected = BUFFER_FREE;
        if (m_bufferState[idx].compare_exchange_strong(
                expected, BUFFER_ASSEMBLING, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            m_reserveCursor = (idx + 1) % NUM_BUFFERS;
            return idx;
        }
    }

    for (int attempt = 0; attempt < NUM_BUFFERS; ++attempt) {
        const int idx = (m_reserveCursor + attempt) % NUM_BUFFERS;
        if (idx == latest) continue;
        int expected = BUFFER_READY;
        if (m_bufferState[idx].compare_exchange_strong(
                expected, BUFFER_ASSEMBLING, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            m_reserveCursor = (idx + 1) % NUM_BUFFERS;
            return idx;
        }
    }

    return -1;
}

void UDPCapture::releaseAssemblingBuffer(int bufferIndex) {
    if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS) return;
    int expected = BUFFER_ASSEMBLING;
    m_bufferState[bufferIndex].compare_exchange_strong(
        expected, BUFFER_FREE, std::memory_order_acq_rel, std::memory_order_relaxed);
}

bool UDPCapture::publishAssembledBuffer(int bufferIndex, uint16_t width, uint16_t height,
                                        uint32_t frameId, uint8_t bytesPerPixel,
                                        uint8_t pixelFormat, uint64_t captureUnixMicros,
                                        std::chrono::steady_clock::time_point publishTime) {
    if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS) return false;

    const bool sourceLikelyRestarted =
        m_hasLatestPublishedFrameId &&
        publishTime != std::chrono::steady_clock::time_point{} &&
        (publishTime - m_latestPublishTime) > kFrameIdResetIdleTimeout;
    if (m_hasLatestPublishedFrameId && !sourceLikelyRestarted &&
        !isFrameIdNewer(frameId, m_latestPublishedFrameId)) {
        releaseAssemblingBuffer(bufferIndex);
        m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    m_bufferWidth[bufferIndex].store(width, std::memory_order_relaxed);
    m_bufferHeight[bufferIndex].store(height, std::memory_order_relaxed);
    m_bufferBytesPerPixel[bufferIndex].store(bytesPerPixel, std::memory_order_relaxed);
    m_bufferPixelFormat[bufferIndex].store(pixelFormat, std::memory_order_relaxed);
    m_bufferFrameId[bufferIndex].store(frameId, std::memory_order_relaxed);
    m_bufferCaptureUnixMicros[bufferIndex].store(captureUnixMicros, std::memory_order_relaxed);

    const int prevLatest = m_latestBufferIndex.exchange(bufferIndex, std::memory_order_acq_rel);
    m_bufferState[bufferIndex].store(BUFFER_READY, std::memory_order_release);

    if (prevLatest >= 0 && prevLatest != bufferIndex) {
        int expected = BUFFER_READY;
        m_bufferState[prevLatest].compare_exchange_strong(
            expected, BUFFER_FREE, std::memory_order_acq_rel, std::memory_order_relaxed);
    }

    m_receivedFrames.fetch_add(1, std::memory_order_relaxed);
    m_latestPublishedFrameId = frameId;
    m_hasLatestPublishedFrameId = true;
    m_latestPublishTime = publishTime;
    {
        std::lock_guard<std::mutex> lock(m_publishCvMutex);
        m_publishSeq.fetch_add(1, std::memory_order_release);
    }
    m_publishCv.notify_one();
    return true;
}

void UDPCapture::clearFragmentState() {
    m_activeFragmentCount = 0;
    for (size_t i = 0; i < MAX_FRAGMENT_SLOTS; ++i) {
        m_activeFragmentSlots[i] = -1;
    }
    for (size_t i = 0; i < FRAGMENT_BUCKETS; ++i) {
        m_bucketHeads[i] = -1;
    }
    m_freeFragmentCount = 0;
    for (size_t i = 0; i < MAX_FRAGMENT_SLOTS; ++i) {
        FrameFragments& frag = m_fragmentStorage[i];
        if (frag.bufferIndex >= 0) {
            releaseAssemblingBuffer(frag.bufferIndex);
        }
        if (frag.received.capacity() == 0) {
            frag.received.reserve(64);
        }
        frag.active = false;
        frag.frameId = 0;
        frag.slotIndex = static_cast<int>(i);
        frag.activeListIndex = -1;
        frag.nextInBucket = -1;
        frag.receivedMask = 0;
        frag.useReceivedMask = false;
        frag.totalPackets = 0;
        frag.receivedCount = 0;
        frag.width = 0;
        frag.height = 0;
        frag.bytesPerPixel = 3;
        frag.pixelFormat = UDP_PIXEL_FORMAT_RGB;
        frag.frameBytes = 0;
        frag.bufferIndex = -1;
        frag.dropped = false;
        frag.lastUpdate = std::chrono::steady_clock::time_point{};
        m_freeFragmentStack[m_freeFragmentCount++] = static_cast<int>(i);
    }
}

void UDPCapture::rememberCreditTarget(const sockaddr_in& addr) {
    // Hot path: called for every valid chunk. Skip the mutex once the target is
    // learned and its address/port are unchanged (the common case for an entire
    // session). Only lock to learn the target or when the sender's addr/port
    // actually changes, keeping the per-packet cost to two relaxed atomic loads.
    const uint32_t addrId = static_cast<uint32_t>(addr.sin_addr.s_addr);
    const uint16_t portId = static_cast<uint16_t>(addr.sin_port);
    if (m_creditTargetLearned.load(std::memory_order_acquire) &&
        m_creditTargetAddrId.load(std::memory_order_relaxed) == addrId &&
        m_creditTargetPortId.load(std::memory_order_relaxed) == portId) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_creditTargetMutex);
    m_creditTargetAddr = addr;
    m_hasCreditTarget = true;
    m_creditTargetAddrId.store(addrId, std::memory_order_relaxed);
    m_creditTargetPortId.store(portId, std::memory_order_relaxed);
    m_creditTargetLearned.store(true, std::memory_order_release);
}

bool UDPCapture::SendFrameCredit(uint32_t minFrameId, uint32_t credits) {
    if (m_recvSocket == INVALID_SOCKET || credits == 0) return false;

    sockaddr_in target{};
    {
        std::lock_guard<std::mutex> lock(m_creditTargetMutex);
        if (!m_hasCreditTarget) return false;
        target = m_creditTargetAddr;
    }

    // Credits are sent back to the source address learned from frame packets;
    // minFrameId is a barrier so the sender never intentionally repeats an old
    // frame after inference has consumed a newer one.
    UDPCreditPacket packet{};
    packet.magic = UDP_CREDIT_MAGIC;
    packet.size = static_cast<uint16_t>(sizeof(UDPCreditPacket));
    packet.flags = 0;
    packet.minFrameId = minFrameId;
    packet.credits = credits;
    packet.sequence = m_creditSeq.fetch_add(1, std::memory_order_relaxed) + 1;

    const int sent = sendto(m_recvSocket, reinterpret_cast<const char*>(&packet),
                            static_cast<int>(sizeof(packet)), 0,
                            reinterpret_cast<SOCKADDR*>(&target), sizeof(target));
    return sent == static_cast<int>(sizeof(packet));
}

void UDPCapture::SendClockSyncPing() {
    if (m_recvSocket == INVALID_SOCKET) return;

    // Self-throttle to ~10/s so callers can invoke this every main-loop tick.
    const auto now = std::chrono::steady_clock::now();
    if (m_lastPingTime.time_since_epoch().count() != 0 &&
        (now - m_lastPingTime) < std::chrono::milliseconds(100)) {
        return;
    }

    sockaddr_in target{};
    {
        std::lock_guard<std::mutex> lock(m_creditTargetMutex);
        if (!m_hasCreditTarget) return;  // no game PC learned yet
        target = m_creditTargetAddr;
    }
    m_lastPingTime = now;

    UDPSyncPing ping{};
    ping.magic = UDP_SYNC_PING_MAGIC;
    ping.seq = ++m_pingSeq;
    ping.t1 = static_cast<uint64_t>(nowUnixMicros());
    sendto(m_recvSocket, reinterpret_cast<const char*>(&ping),
           static_cast<int>(sizeof(ping)), 0,
           reinterpret_cast<SOCKADDR*>(&target), sizeof(target));
}

void UDPCapture::handleSyncPong(const uint8_t* data, int len) {
    if (len < static_cast<int>(sizeof(UDPSyncPong))) return;
    UDPSyncPong pong{};
    std::memcpy(&pong, data, sizeof(pong));
    if (pong.magic != UDP_SYNC_PONG_MAGIC) return;

    const int64_t t1 = static_cast<int64_t>(pong.t1);  // inference send
    const int64_t t2 = static_cast<int64_t>(pong.t2);  // game recv
    const int64_t t3 = static_cast<int64_t>(pong.t3);  // game send
    const int64_t t4 = nowUnixMicros();                // inference recv

    // NTP: offset = ((t2-t1)+(t3-t4))/2, rtt = (t4-t1)-(t3-t2).
    const int64_t rtt = (t4 - t1) - (t3 - t2);
    if (rtt < 0 || rtt > 1000000) return;  // bogus / >1s round trip: ignore
    const int64_t offset = ((t2 - t1) + (t3 - t4)) / 2;

    // Clock filter: keep the offset from the lowest-RTT sample (least queuing
    // noise), and periodically re-open the window so it tracks slow drift.
    const auto now = std::chrono::steady_clock::now();
    if (m_syncBestResetTime.time_since_epoch().count() == 0 ||
        (now - m_syncBestResetTime) > std::chrono::seconds(4)) {
        m_syncBestRttUs = INT64_MAX;
        m_syncBestResetTime = now;
    }
    if (rtt < m_syncBestRttUs) {
        m_syncBestRttUs = rtt;
        m_clockOffsetUs.store(offset, std::memory_order_relaxed);
        m_clockOffsetValid.store(true, std::memory_order_release);
    }
}

int64_t UDPCapture::GetClockOffsetMicros(bool* valid) const {
    if (valid) *valid = m_clockOffsetValid.load(std::memory_order_acquire);
    return m_clockOffsetUs.load(std::memory_order_relaxed);
}

bool UDPCapture::Initialize(unsigned short listenPort) {
    m_listenPort = listenPort;

#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[UDPCapture] WSAStartup failed\n";
        return false;
    }
#endif

    m_recvSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_recvSocket == INVALID_SOCKET) {
        std::cerr << "[UDPCapture] Failed to create receive socket\n";
        return false;
    }

    // Allow immediate rebind after a crash/restart.
    int reuse = 1;
    setsockopt(m_recvSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));

    int recvBufSize = 32 * 1024 * 1024;
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVBUF, (char*)&recvBufSize, sizeof(recvBufSize));
#ifndef _WIN32
    // The kernel clamps SO_RCVBUF to net.core.rmem_max (often ~208KB on stock
    // Jetson) - a too-small buffer silently drops bursts -> dropped frames.
    // Verify the actual size and warn so the cause is visible.
    int actualBuf = 0;
    socklen_t blen = sizeof(actualBuf);
    if (getsockopt(m_recvSocket, SOL_SOCKET, SO_RCVBUF, &actualBuf, &blen) == 0) {
        // Linux reports double the usable size.
        const int usable = actualBuf / 2;
        if (usable < recvBufSize / 2) {
            std::cerr << "[UDPCapture] WARNING: SO_RCVBUF clamped to " << (usable / 1024)
                      << "KB (requested " << (recvBufSize / 1024)
                      << "KB). Raise it: sudo sysctl -w net.core.rmem_max=33554432\n";
        }
    }
#endif

    sockaddr_in bindAddr{};
    bindAddr.sin_family = AF_INET;
    bindAddr.sin_addr.s_addr = INADDR_ANY;
    bindAddr.sin_port = htons(listenPort);
    if (bind(m_recvSocket, (SOCKADDR*)&bindAddr, sizeof(bindAddr)) == SOCKET_ERROR) {
        std::cerr << "[UDPCapture] Failed to bind to port " << listenPort << "\n";
        closesocket(m_recvSocket);
        m_recvSocket = INVALID_SOCKET;
        return false;
    }

    // Short recv timeout so stale-fragment cleanup runs promptly during traffic
    // gaps (frees half-assembled frame slots sooner).
#ifdef _WIN32
    DWORD timeout = 10;
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
#else
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 10000;
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
#endif

    constexpr size_t kDefaultBufferSize = 640 * 640 * 4;
    if (!allocatePinnedBuffers(kDefaultBufferSize)) {
        std::cerr << "[UDPCapture] Failed to allocate capture buffers\n";
        closesocket(m_recvSocket);
        m_recvSocket = INVALID_SOCKET;
        return false;
    }

    std::cout << "[UDPCapture] Initialized, listening on port " << listenPort << "\n";
    return true;
}

void UDPCapture::Shutdown() {
    StopCapture();

    if (m_recvSocket != INVALID_SOCKET) {
        closesocket(m_recvSocket);
        m_recvSocket = INVALID_SOCKET;
    }

    freePinnedBuffers();
    clearFragmentState();

#ifdef _WIN32
    WSACleanup();
#endif
}

bool UDPCapture::StartCapture() {
    if (m_running.load(std::memory_order_relaxed)) return true;
    if (m_recvSocket == INVALID_SOCKET) return false;

    clearFragmentState();

    m_latestBufferIndex.store(-1, std::memory_order_relaxed);
    m_publishSeq.store(0, std::memory_order_relaxed);
    m_consumedSeq = 0;
    m_reserveCursor = 0;
    m_hasLatestPublishedFrameId = false;
    m_latestPublishedFrameId = 0;
    m_latestPublishTime = std::chrono::steady_clock::time_point{};
    {
        std::lock_guard<std::mutex> lock(m_creditTargetMutex);
        m_hasCreditTarget = false;
        m_creditTargetAddr = sockaddr_in{};
    }
    m_creditSeq.store(0, std::memory_order_relaxed);
    m_receivedFrames.store(0, std::memory_order_relaxed);
    m_droppedFrames.store(0, std::memory_order_relaxed);
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        m_bufferState[i].store(BUFFER_FREE, std::memory_order_relaxed);
    }

    m_running.store(true, std::memory_order_relaxed);
    m_recvThread = std::thread(&UDPCapture::receiveThread, this);

    std::cout << "[UDPCapture] Started capture (pinned: "
              << (m_usePinnedMemory ? "enabled" : "disabled") << ")\n";
    return true;
}

void UDPCapture::StopCapture() {
    if (!m_running.load(std::memory_order_relaxed)) return;

    m_running.store(false, std::memory_order_relaxed);
    m_publishCv.notify_all();
    if (m_recvThread.joinable()) {
        m_recvThread.join();
    }

    clearFragmentState();
    std::cout << "[UDPCapture] Stopped capture\n";
}

void UDPCapture::receiveThread() {
#ifndef _WIN32
    // Best-effort priority/affinity hints for lower receive jitter.
    {
        struct sched_param param;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
            param.sched_priority = sched_get_priority_max(SCHED_RR);
            pthread_setschedparam(pthread_self(), SCHED_RR, &param);
        }
    }
    {
        const long cpuCount = sysconf(_SC_NPROCESSORS_ONLN);
        // Configured override wins; otherwise fall back to the built-in default
        // of pinning to the last core. A configured value < 0 means "unpinned".
        int targetCore = (m_receiveAffinityCore != kAffinityUnset)
                             ? m_receiveAffinityCore
                             : (cpuCount > 1 ? static_cast<int>(cpuCount - 1) : -1);
        if (targetCore >= 0 && targetCore < cpuCount) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(targetCore, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        }
    }
#endif

#ifndef __linux__
    std::vector<uint8_t> recvBuffer(65536);
#endif
    // At 144fps frames arrive ~7ms apart and a frame's fragments burst within
    // ~1ms, so 12ms is ample headroom while freeing doomed (partial) frames
    // ~2x sooner than before, cutting the dead-latency a lost frame holds.
    constexpr auto kFragmentStaleTimeout = std::chrono::milliseconds(12);
    constexpr uint32_t kCleanupPacketInterval = 64;
    static_assert((kCleanupPacketInterval & (kCleanupPacketInterval - 1)) == 0,
                  "kCleanupPacketInterval must be power-of-two");
    uint32_t packetsSinceCleanup = 0;
    FrameFragments* cachedFrag = nullptr;
    uint32_t cachedFrameId = 0;

    auto cleanupStaleFragments =
        [this, kFragmentStaleTimeout, &cachedFrag, &cachedFrameId](
            std::chrono::steady_clock::time_point now) {
        const size_t activeSnapshot = m_activeFragmentCount;
        if (activeSnapshot == 0) return;
        std::array<int, MAX_FRAGMENT_SLOTS> activeSlotsSnapshot{};
        const size_t snapshotCount = std::min(activeSnapshot, MAX_FRAGMENT_SLOTS);
        for (size_t i = 0; i < snapshotCount; ++i) {
            activeSlotsSnapshot[i] = m_activeFragmentSlots[i];
        }

        for (size_t i = 0; i < snapshotCount; ++i) {
            const int slotIdx = activeSlotsSnapshot[i];
            if (slotIdx < 0 || slotIdx >= static_cast<int>(MAX_FRAGMENT_SLOTS)) continue;
            FrameFragments& frag = m_fragmentStorage[static_cast<size_t>(slotIdx)];
            if (!frag.active) continue;
            if ((now - frag.lastUpdate) <= kFragmentStaleTimeout) continue;
            if (frag.bufferIndex >= 0) {
                releaseAssemblingBuffer(frag.bufferIndex);
                frag.bufferIndex = -1;
            }
            if (!frag.dropped) {
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
            }
            unlinkFragment(&frag);
            if (cachedFrag == &frag) {
                cachedFrag = nullptr;
                cachedFrameId = 0;
            }
            releaseFragment(&frag);
        }
    };

    auto processPacket = [this, &cachedFrag, &cachedFrameId](
                             const uint8_t* packetData, int packetBytes,
                             std::chrono::steady_clock::time_point packetNow,
                             const sockaddr_in* fromAddr) {
        if (!packetData || packetBytes < 4) return;
        // Clock-sync pong (small control packet) - handle and return before the
        // frame-header path. One magic compare per packet; frame chunks fall
        // straight through.
        {
            uint32_t magic;
            std::memcpy(&magic, packetData, sizeof(magic));
            if (magic == UDP_SYNC_PONG_MAGIC) {
                handleSyncPong(packetData, packetBytes);
                return;
            }
        }
        if (packetBytes < static_cast<int>(sizeof(UDPPacketHeaderV2))) return;

        const auto* v2 = reinterpret_cast<const UDPPacketHeaderV2*>(packetData);
        if (v2->magic != UDP_PACKET_V3_MAGIC ||
            v2->headerSize < sizeof(UDPPacketHeaderV2) ||
            static_cast<size_t>(v2->headerSize) > static_cast<size_t>(packetBytes)) {
            return;
        }
        if (fromAddr) {
            rememberCreditTarget(*fromAddr);
        }

        const uint32_t frameId = v2->frameId;
        const uint16_t chunkIndex = v2->chunkIndex;
        const uint16_t totalChunks = v2->totalChunks;
        const uint32_t chunkSize = v2->chunkSize;
        const uint16_t frameWidth = v2->frameWidth;
        const uint16_t frameHeight = v2->frameHeight;
        const uint8_t bytesPerPixel = v2->bytesPerPixel;
        const uint8_t pixelFormat = v2->pixelFormat;
        const uint64_t captureUnixMicros = v2->captureUnixMicros;
        const size_t payloadOffset = v2->payloadOffset;
        const size_t frameBytes = v2->frameBytes;
        const uint8_t* payload = packetData + v2->headerSize;
        const size_t expectedPayload = static_cast<size_t>(packetBytes) - v2->headerSize;
        if (chunkSize != expectedPayload) return;

        if (totalChunks == 0 || chunkIndex >= totalChunks ||
            frameWidth == 0 || frameHeight == 0 ||
            bytesPerPixel != 3 ||
            pixelFormat != UDP_PIXEL_FORMAT_RGB) {
            return;
        }
        const size_t expectedFrameBytes =
            static_cast<size_t>(frameWidth) * static_cast<size_t>(frameHeight) *
            static_cast<size_t>(bytesPerPixel);
        if (frameBytes != expectedFrameBytes ||
            payloadOffset > frameBytes ||
            static_cast<size_t>(chunkSize) > frameBytes - payloadOffset) {
            return;
        }
        if (m_hasLatestPublishedFrameId &&
            (packetNow - m_latestPublishTime) <= kFrameIdResetIdleTimeout &&
            !isFrameIdNewer(frameId, m_latestPublishedFrameId)) {
            return;
        }

        FrameFragments* frag = nullptr;
        if (cachedFrag && cachedFrag->active && cachedFrameId == frameId &&
            cachedFrag->frameId == frameId) {
            frag = cachedFrag;
        } else {
            frag = findFragment(frameId);
            if (frag) {
                cachedFrag = frag;
                cachedFrameId = frameId;
            }
        }
        if (!frag) {
            frag = acquireFragment();
            if (!frag) {
                // No free fragment slot: drop this packet/frame.
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            frag->totalPackets = totalChunks;
            frag->receivedCount = 0;
            frag->width = frameWidth;
            frag->height = frameHeight;
            frag->bytesPerPixel = bytesPerPixel;
            frag->pixelFormat = pixelFormat;
            frag->frameBytes = frameBytes;
            frag->captureUnixMicros = captureUnixMicros;
            frag->dropped = false;
            frag->bufferIndex = -1;
            frag->useReceivedMask = (totalChunks <= 64);
            if (frag->useReceivedMask) {
                frag->receivedMask = 0;
            } else {
                if (frag->received.size() < totalChunks) {
                    frag->received.resize(totalChunks);
                }
                std::fill(frag->received.begin(), frag->received.begin() + totalChunks, 0);
            }
            frag->lastUpdate = packetNow;

            const size_t frameSize = frag->frameBytes;
            if (!ensurePinnedCapacity(frameSize)) {
                frag->dropped = true;
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
            } else {
                frag->bufferIndex = reserveAssemblingBuffer();
                if (frag->bufferIndex < 0) {
                    frag->dropped = true;
                    m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
                }
            }

            linkFragment(frag, frameId);
            cachedFrag = frag;
            cachedFrameId = frameId;
        }
        if (!frag) return;

        frag->lastUpdate = packetNow;
        if (frag->totalPackets != totalChunks ||
            frag->width != frameWidth ||
            frag->height != frameHeight ||
            frag->bytesPerPixel != bytesPerPixel ||
            frag->pixelFormat != pixelFormat ||
            frag->frameBytes != frameBytes) {
            if (!frag->dropped) {
                if (frag->bufferIndex >= 0) {
                    releaseAssemblingBuffer(frag->bufferIndex);
                    frag->bufferIndex = -1;
                }
                frag->dropped = true;
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        if (frag->dropped || frag->bufferIndex < 0) return;
        if (frag->useReceivedMask) {
            const uint64_t bit = (1ull << chunkIndex);
            if (frag->receivedMask & bit) return;
            frag->receivedMask |= bit;
        } else {
            if (chunkIndex >= frag->received.size() || frag->received[chunkIndex]) return;
            frag->received[chunkIndex] = 1;
        }

        const size_t frameSize = frag->frameBytes;
        if (frameSize == 0) return;
        const size_t offset = payloadOffset;
        if (offset >= frameSize) return;

        const size_t expectedChunkSize = static_cast<size_t>(chunkSize);
        const size_t requestedSize = static_cast<size_t>(chunkSize);
        if (requestedSize != expectedChunkSize) return;

        uint8_t* dst = m_pinnedFrameBuffer[frag->bufferIndex];
        if (!dst) return;
        std::memcpy(dst + offset, payload, requestedSize);
        frag->receivedCount++;

        if (frag->receivedCount == frag->totalPackets) {
            const int publishIdx = frag->bufferIndex;
            const uint16_t publishW = frag->width;
            const uint16_t publishH = frag->height;
            const uint8_t publishBpp = frag->bytesPerPixel;
            const uint8_t publishFormat = frag->pixelFormat;
            const uint64_t publishCaptureUs = frag->captureUnixMicros;
            const bool published = publishAssembledBuffer(
                publishIdx, publishW, publishH, frameId, publishBpp, publishFormat,
                publishCaptureUs, packetNow);

            frag->bufferIndex = -1;
            unlinkFragment(frag);
            if (cachedFrag == frag) {
                cachedFrag = nullptr;
                cachedFrameId = 0;
            }
            releaseFragment(frag);

            // No locks are held at this point (publishAssembledBuffer releases
            // m_publishCvMutex before returning), so the callback can run inline.
            if (published && m_frameReadyCallback) {
                m_frameReadyCallback();
            }
        }
    };

#ifdef __linux__
    constexpr unsigned int kRecvBatchPackets = 32;
    std::array<std::array<uint8_t, 65536>, kRecvBatchPackets> batchBuffers{};
    std::array<sockaddr_in, kRecvBatchPackets> batchFromAddr{};
    std::array<iovec, kRecvBatchPackets> batchIov{};
    std::array<mmsghdr, kRecvBatchPackets> batchMsgs{};
    for (unsigned int i = 0; i < kRecvBatchPackets; ++i) {
        batchIov[i].iov_base = batchBuffers[i].data();
        batchIov[i].iov_len = batchBuffers[i].size();
        batchMsgs[i].msg_hdr.msg_name = &batchFromAddr[i];
        batchMsgs[i].msg_hdr.msg_iov = &batchIov[i];
        batchMsgs[i].msg_hdr.msg_iovlen = 1;
        batchMsgs[i].msg_hdr.msg_control = nullptr;
        batchMsgs[i].msg_hdr.msg_controllen = 0;
        batchMsgs[i].msg_hdr.msg_flags = 0;
        batchMsgs[i].msg_hdr.msg_namelen = sizeof(sockaddr_in);
        batchMsgs[i].msg_len = 0;
    }
#else
    sockaddr_in fromAddr{};
#ifdef _WIN32
    int fromLen = sizeof(fromAddr);
#else
    socklen_t fromLen = sizeof(fromAddr);
#endif
#endif

    while (m_running.load(std::memory_order_relaxed)) {
#ifdef __linux__
        for (unsigned int i = 0; i < kRecvBatchPackets; ++i) {
            batchMsgs[i].msg_hdr.msg_namelen = sizeof(sockaddr_in);
            batchMsgs[i].msg_len = 0;
        }

        const int batchCount = recvmmsg(
            m_recvSocket, batchMsgs.data(), kRecvBatchPackets, MSG_WAITFORONE, nullptr);
        if (batchCount <= 0) {
            int err = errno;
            if (err == ETIMEDOUT || err == EWOULDBLOCK || err == EAGAIN) {
                cleanupStaleFragments(std::chrono::steady_clock::now());
                continue;
            }
            if (err == EINTR) {
                continue;
            }
            continue;
        }

        const auto batchNow = std::chrono::steady_clock::now();
        for (int i = 0; i < batchCount; ++i) {
            if (((++packetsSinceCleanup) & (kCleanupPacketInterval - 1)) == 0) {
                cleanupStaleFragments(batchNow);
            }
            processPacket(batchBuffers[static_cast<size_t>(i)].data(),
                          static_cast<int>(batchMsgs[static_cast<size_t>(i)].msg_len),
                          batchNow,
                          &batchFromAddr[static_cast<size_t>(i)]);
        }
#else
        fromLen = sizeof(fromAddr);
        int ret = recvfrom(m_recvSocket, (char*)recvBuffer.data(), (int)recvBuffer.size(),
                           0, (SOCKADDR*)&fromAddr, &fromLen);

        if (ret <= 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) {
#else
            int err = errno;
            if (err == ETIMEDOUT || err == EWOULDBLOCK || err == EAGAIN) {
#endif
                cleanupStaleFragments(std::chrono::steady_clock::now());
                continue;
            }
            continue;
        }

        const auto packetNow = std::chrono::steady_clock::now();
        if (((++packetsSinceCleanup) & (kCleanupPacketInterval - 1)) == 0) {
            cleanupStaleFragments(packetNow);
        }
        processPacket(recvBuffer.data(), ret, packetNow, &fromAddr);
#endif
    }
}

bool UDPCapture::AcquireFramePinned(void** pinnedRgbData, unsigned int* width,
                                    unsigned int* height, uint64_t* outFrameId,
                                    int* bufferIndex, uint32_t timeoutMs,
                                    uint8_t* bytesPerPixel, uint8_t* pixelFormat,
                                    uint64_t* outCaptureUnixMicros) {
    if (!m_running.load(std::memory_order_relaxed)) return false;

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeoutMs);
    while (m_running.load(std::memory_order_relaxed)) {
        const uint64_t seq = m_publishSeq.load(std::memory_order_acquire);
        if (seq != m_consumedSeq) {
            const int idx = m_latestBufferIndex.load(std::memory_order_acquire);
            if (idx >= 0 && idx < NUM_BUFFERS && m_pinnedFrameBuffer[idx]) {
                int expected = BUFFER_READY;
                if (m_bufferState[idx].compare_exchange_strong(
                        expected, BUFFER_IN_USE, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    m_consumedSeq = seq;
                    if (pinnedRgbData) *pinnedRgbData = m_pinnedFrameBuffer[idx];
                    if (width) *width = m_bufferWidth[idx].load(std::memory_order_relaxed);
                    if (height) *height = m_bufferHeight[idx].load(std::memory_order_relaxed);
                    if (outFrameId) *outFrameId = m_bufferFrameId[idx].load(std::memory_order_relaxed);
                    if (outCaptureUnixMicros) {
                        *outCaptureUnixMicros =
                            m_bufferCaptureUnixMicros[idx].load(std::memory_order_relaxed);
                    }
                    if (bytesPerPixel) {
                        *bytesPerPixel = static_cast<uint8_t>(
                            m_bufferBytesPerPixel[idx].load(std::memory_order_relaxed));
                    }
                    if (pixelFormat) {
                        *pixelFormat = static_cast<uint8_t>(
                            m_bufferPixelFormat[idx].load(std::memory_order_relaxed));
                    }
                    if (bufferIndex) *bufferIndex = idx;
                    return true;
                }
            }
        }

        if (timeoutMs == 0) return false;
        std::unique_lock<std::mutex> lock(m_publishCvMutex);
        if (!m_running.load(std::memory_order_relaxed)) return false;
        if (m_publishCv.wait_until(lock, deadline, [&]() {
                return !m_running.load(std::memory_order_relaxed) ||
                       (m_publishSeq.load(std::memory_order_acquire) != m_consumedSeq);
            })) {
            continue;
        }
        return false;
    }
    return false;
}

void UDPCapture::ReleaseFrame(int bufferIndex) {
    if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS) return;
    int expected = BUFFER_IN_USE;
    m_bufferState[bufferIndex].compare_exchange_strong(
        expected, BUFFER_FREE, std::memory_order_acq_rel, std::memory_order_relaxed);
}
