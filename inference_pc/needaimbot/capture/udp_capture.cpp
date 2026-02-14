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
        cudaError_t err = cudaMallocHost(&m_pinnedFrameBuffer[i], size);
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
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        m_bufferState[i].store(BUFFER_FREE, std::memory_order_relaxed);
        m_bufferWidth[i].store(0, std::memory_order_relaxed);
        m_bufferHeight[i].store(0, std::memory_order_relaxed);
        m_bufferFrameId[i].store(0, std::memory_order_relaxed);
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
        m_bufferFrameId[i].store(0, std::memory_order_relaxed);
    }
    m_pinnedBufferSize = 0;
    m_usePinnedMemory = false;
    m_latestBufferIndex.store(-1, std::memory_order_relaxed);
    m_publishSeq.store(0, std::memory_order_relaxed);
    m_consumedSeq = 0;
    m_reserveCursor = 0;
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

void UDPCapture::publishAssembledBuffer(int bufferIndex, uint16_t width, uint16_t height, uint32_t frameId) {
    if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS) return;

    m_bufferWidth[bufferIndex].store(width, std::memory_order_relaxed);
    m_bufferHeight[bufferIndex].store(height, std::memory_order_relaxed);
    m_bufferFrameId[bufferIndex].store(frameId, std::memory_order_relaxed);

    const int prevLatest = m_latestBufferIndex.exchange(bufferIndex, std::memory_order_acq_rel);
    m_bufferState[bufferIndex].store(BUFFER_READY, std::memory_order_release);

    if (prevLatest >= 0 && prevLatest != bufferIndex) {
        int expected = BUFFER_READY;
        m_bufferState[prevLatest].compare_exchange_strong(
            expected, BUFFER_FREE, std::memory_order_acq_rel, std::memory_order_relaxed);
    }

    m_receivedFrames.fetch_add(1, std::memory_order_relaxed);
    m_publishSeq.fetch_add(1, std::memory_order_release);
    m_publishCv.notify_one();
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
        frag.frameBytes = 0;
        frag.bufferIndex = -1;
        frag.dropped = false;
        frag.lastUpdate = std::chrono::steady_clock::time_point{};
        m_freeFragmentStack[m_freeFragmentCount++] = static_cast<int>(i);
    }
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

    int recvBufSize = 32 * 1024 * 1024;
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVBUF, (char*)&recvBufSize, sizeof(recvBufSize));

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

#ifdef _WIN32
    DWORD timeout = 100;
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
#else
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
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
    m_receivedFrames.store(0, std::memory_order_relaxed);
    m_droppedFrames.store(0, std::memory_order_relaxed);
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        m_bufferState[i].store(BUFFER_FREE, std::memory_order_relaxed);
    }

    m_running.store(true, std::memory_order_relaxed);
    m_isCapturing.store(true, std::memory_order_relaxed);
    m_recvThread = std::thread(&UDPCapture::receiveThread, this);

    std::cout << "[UDPCapture] Started capture (pinned: "
              << (m_usePinnedMemory ? "enabled" : "disabled") << ")\n";
    return true;
}

void UDPCapture::StopCapture() {
    if (!m_running.load(std::memory_order_relaxed)) return;

    m_running.store(false, std::memory_order_relaxed);
    m_isCapturing.store(false, std::memory_order_relaxed);
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
        if (cpuCount > 1) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(static_cast<int>(cpuCount - 1), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        }
    }
#endif

#ifndef __linux__
    std::vector<uint8_t> recvBuffer(65536);
#endif
    constexpr size_t kChunkPayloadBytes = 60000;
    constexpr auto kFragmentStaleTimeout = std::chrono::milliseconds(100);
    constexpr uint32_t kMinPartialPublishRatioPct = 80;
    constexpr uint32_t kCleanupPacketInterval = 64;
    static_assert((kCleanupPacketInterval & (kCleanupPacketInterval - 1)) == 0,
                  "kCleanupPacketInterval must be power-of-two");
    uint32_t packetsSinceCleanup = 0;
    FrameFragments* cachedFrag = nullptr;
    uint32_t cachedFrameId = 0;

    auto cleanupStaleFragments =
        [this, kFragmentStaleTimeout, kChunkPayloadBytes, &cachedFrag, &cachedFrameId](
            std::chrono::steady_clock::time_point now) {
        const size_t activeSnapshot = m_activeFragmentCount;
        if (activeSnapshot == 0) return;
        std::array<int, MAX_FRAGMENT_SLOTS> activeSlotsSnapshot{};
        const size_t snapshotCount = std::min(activeSnapshot, MAX_FRAGMENT_SLOTS);
        for (size_t i = 0; i < snapshotCount; ++i) {
            activeSlotsSnapshot[i] = m_activeFragmentSlots[i];
        }

        auto clearMissingChunks = [kChunkPayloadBytes](FrameFragments& frag, uint8_t* dst) {
            if (!dst || frag.totalPackets == 0 || frag.width == 0 || frag.height == 0) return;

            const size_t frameSize = frag.frameBytes;
            if (frameSize == 0) return;
            auto clearChunk = [&](uint16_t chunkIdx) {
                const size_t offset = static_cast<size_t>(chunkIdx) * kChunkPayloadBytes;
                if (offset >= frameSize) return;
                const size_t clearSize = std::min(kChunkPayloadBytes, frameSize - offset);
                std::memset(dst + offset, 0, clearSize);
            };

            if (frag.useReceivedMask) {
                uint64_t fullMask = ~0ull;
                if (frag.totalPackets < 64) {
                    fullMask = (1ull << frag.totalPackets) - 1ull;
                }
                uint64_t missingMask = (~frag.receivedMask) & fullMask;
                for (uint16_t chunkIdx = 0; chunkIdx < frag.totalPackets; ++chunkIdx) {
                    if ((missingMask & (1ull << chunkIdx)) != 0) {
                        clearChunk(chunkIdx);
                    }
                }
                return;
            }

            const size_t packetCount = static_cast<size_t>(frag.totalPackets);
            for (size_t i = 0; i < packetCount; ++i) {
                if (i >= frag.received.size() || frag.received[i] == 0) {
                    clearChunk(static_cast<uint16_t>(i));
                }
            }
        };

        for (size_t i = 0; i < snapshotCount; ++i) {
            const int slotIdx = activeSlotsSnapshot[i];
            if (slotIdx < 0 || slotIdx >= static_cast<int>(MAX_FRAGMENT_SLOTS)) continue;
            FrameFragments& frag = m_fragmentStorage[static_cast<size_t>(slotIdx)];
            if (!frag.active) continue;
            if ((now - frag.lastUpdate) <= kFragmentStaleTimeout) continue;
            const bool canPublishPartial =
                frag.bufferIndex >= 0 &&
                frag.totalPackets > 0 &&
                frag.receivedCount > 0 &&
                (static_cast<uint32_t>(frag.receivedCount) * 100u >=
                 static_cast<uint32_t>(frag.totalPackets) * kMinPartialPublishRatioPct);
            if (canPublishPartial) {
                uint8_t* dst = m_pinnedFrameBuffer[frag.bufferIndex];
                clearMissingChunks(frag, dst);
                publishAssembledBuffer(frag.bufferIndex, frag.width, frag.height, frag.frameId);
                frag.bufferIndex = -1;
                unlinkFragment(&frag);
                if (cachedFrag == &frag) {
                    cachedFrag = nullptr;
                    cachedFrameId = 0;
                }
                releaseFragment(&frag);
                continue;
            }
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

    auto processPacket = [this, kChunkPayloadBytes, &cachedFrag, &cachedFrameId](
                             const uint8_t* packetData, int packetBytes,
                             std::chrono::steady_clock::time_point packetNow) {
        if (!packetData || packetBytes < static_cast<int>(sizeof(UDPPacketHeader))) return;

        const UDPPacketHeader* header = reinterpret_cast<const UDPPacketHeader*>(packetData);
        const size_t expectedPayload = static_cast<size_t>(packetBytes) - sizeof(UDPPacketHeader);
        if (header->chunkSize != expectedPayload) return;

        const uint32_t frameId = header->frameId;
        const uint16_t chunkIndex = header->chunkIndex;
        const uint16_t totalChunks = header->totalChunks;
        const uint32_t chunkSize = header->chunkSize;
        if (totalChunks == 0 || chunkIndex >= totalChunks ||
            header->frameWidth == 0 || header->frameHeight == 0) {
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
            frag->width = header->frameWidth;
            frag->height = header->frameHeight;
            frag->frameBytes =
                static_cast<size_t>(frag->width) * static_cast<size_t>(frag->height) * 4;
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
            frag->width != header->frameWidth ||
            frag->height != header->frameHeight) {
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
        const size_t offset = static_cast<size_t>(chunkIndex) * kChunkPayloadBytes;
        if (offset >= frameSize) return;

        const size_t remaining = frameSize - offset;
        const size_t expectedChunkSize = std::min(kChunkPayloadBytes, remaining);
        const size_t requestedSize = static_cast<size_t>(chunkSize);
        if (requestedSize != expectedChunkSize) return;

        uint8_t* dst = m_pinnedFrameBuffer[frag->bufferIndex];
        if (!dst) return;
        const uint8_t* payload = packetData + sizeof(UDPPacketHeader);
        std::memcpy(dst + offset, payload, requestedSize);
        frag->receivedCount++;

        if (frag->receivedCount == frag->totalPackets) {
            const int publishIdx = frag->bufferIndex;
            const uint16_t publishW = frag->width;
            const uint16_t publishH = frag->height;
            publishAssembledBuffer(publishIdx, publishW, publishH, frameId);

            frag->bufferIndex = -1;
            unlinkFragment(frag);
            if (cachedFrag == frag) {
                cachedFrag = nullptr;
                cachedFrameId = 0;
            }
            releaseFragment(frag);
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
                          batchNow);
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
        processPacket(recvBuffer.data(), ret, packetNow);
#endif
    }
}

bool UDPCapture::AcquireFramePinned(void** pinnedRgbData, unsigned int* width,
                                    unsigned int* height, uint64_t* outFrameId,
                                    int* bufferIndex, uint32_t timeoutMs) {
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
