#include "udp_capture.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#define SOCKADDR struct sockaddr
#define WSAETIMEDOUT WSAETIMEDOUT
#else
#define SOCKADDR struct sockaddr
#define WSAETIMEDOUT ETIMEDOUT
#endif

UDPCapture::UDPCapture() = default;

UDPCapture::~UDPCapture() {
    Shutdown();
}

UDPCapture::FrameFragments* UDPCapture::acquireFragment() {
    if (!m_freeFragments.empty()) {
        FrameFragments* frag = m_freeFragments.back();
        m_freeFragments.pop_back();
        return frag;
    }
    m_fragmentStorage.emplace_back(std::make_unique<FrameFragments>());
    return m_fragmentStorage.back().get();
}

void UDPCapture::releaseFragment(FrameFragments* frag) {
    if (!frag) return;
    frag->received.clear();
    frag->totalPackets = 0;
    frag->receivedCount = 0;
    frag->width = 0;
    frag->height = 0;
    frag->bufferIndex = -1;
    frag->dropped = false;
    m_freeFragments.push_back(frag);
}

bool UDPCapture::allocatePinnedBuffers(size_t size) {
    if (size == 0) return false;

    freePinnedBuffers();
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        cudaError_t err = cudaMallocHost(&m_pinnedFrameBuffer[i], size);
        if (err != cudaSuccess) {
            std::cerr << "[UDPCapture] Failed to allocate pinned buffer " << i
                      << ": " << cudaGetErrorString(err) << "\n";
            freePinnedBuffers();
            return false;
        }
    }

    m_pinnedBufferSize = size;
    m_usePinnedMemory = true;
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
              << " pinned buffers\n";
    return true;
}

bool UDPCapture::ensurePinnedCapacity(size_t size) {
    if (m_usePinnedMemory && m_pinnedBufferSize >= size) return true;

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
            cudaFreeHost(m_pinnedFrameBuffer[i]);
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
    if (!m_usePinnedMemory) return -1;

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
}

void UDPCapture::clearFragmentState() {
    for (auto& kv : m_fragmentMap) {
        FrameFragments* frag = kv.second;
        if (!frag) continue;
        if (frag->bufferIndex >= 0) {
            releaseAssemblingBuffer(frag->bufferIndex);
        }
        releaseFragment(frag);
    }
    m_fragmentMap.clear();
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

    int recvBufSize = 8 * 1024 * 1024;
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

    constexpr size_t kDefaultPinnedSize = 640 * 640 * 4;
    if (!allocatePinnedBuffers(kDefaultPinnedSize)) {
        std::cerr << "[UDPCapture] Warning: pinned memory unavailable\n";
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
    if (m_fragmentStorage.empty()) {
        constexpr size_t kPreallocatedFragments = 128;
        m_fragmentStorage.reserve(kPreallocatedFragments);
        m_freeFragments.reserve(kPreallocatedFragments);
        for (size_t i = 0; i < kPreallocatedFragments; ++i) {
            m_fragmentStorage.emplace_back(std::make_unique<FrameFragments>());
            m_freeFragments.push_back(m_fragmentStorage.back().get());
        }
    }
    m_fragmentMap.reserve(128);

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
    if (m_recvThread.joinable()) {
        m_recvThread.join();
    }

    clearFragmentState();
    std::cout << "[UDPCapture] Stopped capture\n";
}

void UDPCapture::receiveThread() {
    std::vector<uint8_t> recvBuffer(65536);
    constexpr size_t kChunkPayloadBytes = 60000;
    constexpr auto kFragmentStaleTimeout = std::chrono::milliseconds(100);
    constexpr uint32_t kCleanupPacketInterval = 64;
    uint32_t packetsSinceCleanup = 0;

    auto cleanupStaleFragments = [this, kFragmentStaleTimeout](std::chrono::steady_clock::time_point now) {
        for (auto it = m_fragmentMap.begin(); it != m_fragmentMap.end();) {
            FrameFragments* frag = it->second;
            if (frag && (now - frag->lastUpdate) > kFragmentStaleTimeout) {
                if (frag->bufferIndex >= 0) {
                    releaseAssemblingBuffer(frag->bufferIndex);
                    frag->bufferIndex = -1;
                }
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
                releaseFragment(frag);
                it = m_fragmentMap.erase(it);
            } else {
                ++it;
            }
        }
    };

    sockaddr_in fromAddr{};
#ifdef _WIN32
    int fromLen = sizeof(fromAddr);
#else
    socklen_t fromLen = sizeof(fromAddr);
#endif

    while (m_running.load(std::memory_order_relaxed)) {
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
        if ((++packetsSinceCleanup % kCleanupPacketInterval) == 0) {
            cleanupStaleFragments(packetNow);
        }

        if (ret < (int)sizeof(UDPPacketHeader)) continue;

        const UDPPacketHeader* header = reinterpret_cast<const UDPPacketHeader*>(recvBuffer.data());
        const size_t expectedPayload = static_cast<size_t>(ret) - sizeof(UDPPacketHeader);
        if (header->chunkSize != expectedPayload) continue;

        const uint32_t frameId = header->frameId;
        const uint16_t chunkIndex = header->chunkIndex;
        const uint16_t totalChunks = header->totalChunks;
        const uint32_t chunkSize = header->chunkSize;
        if (totalChunks == 0 || chunkIndex >= totalChunks ||
            header->frameWidth == 0 || header->frameHeight == 0) {
            continue;
        }

        FrameFragments* frag = nullptr;
        auto it = m_fragmentMap.find(frameId);
        if (it == m_fragmentMap.end()) {
            frag = acquireFragment();
            if (!frag) {
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            frag->totalPackets = totalChunks;
            frag->receivedCount = 0;
            frag->width = header->frameWidth;
            frag->height = header->frameHeight;
            frag->dropped = false;
            frag->bufferIndex = -1;
            frag->received.assign(totalChunks, 0);
            frag->lastUpdate = packetNow;

            const size_t frameSize =
                static_cast<size_t>(frag->width) * static_cast<size_t>(frag->height) * 4;
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

            m_fragmentMap.emplace(frameId, frag);
        } else {
            frag = it->second;
        }
        if (!frag) continue;

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
            continue;
        }
        if (frag->dropped || frag->bufferIndex < 0) continue;
        if (chunkIndex >= frag->received.size() || frag->received[chunkIndex]) continue;

        const size_t frameSize =
            static_cast<size_t>(frag->width) * static_cast<size_t>(frag->height) * 4;
        const size_t offset = static_cast<size_t>(chunkIndex) * kChunkPayloadBytes;
        if (offset >= frameSize) continue;

        const size_t remaining = frameSize - offset;
        const size_t expectedChunkSize = std::min(kChunkPayloadBytes, remaining);
        const size_t requestedSize = static_cast<size_t>(chunkSize);
        if (requestedSize != expectedChunkSize) continue;

        uint8_t* dst = m_pinnedFrameBuffer[frag->bufferIndex];
        if (!dst) continue;
        const uint8_t* payload = recvBuffer.data() + sizeof(UDPPacketHeader);
        std::memcpy(dst + offset, payload, requestedSize);
        frag->received[chunkIndex] = 1;
        frag->receivedCount++;

        if (frag->receivedCount == frag->totalPackets) {
            const int publishIdx = frag->bufferIndex;
            const uint16_t publishW = frag->width;
            const uint16_t publishH = frag->height;
            publishAssembledBuffer(publishIdx, publishW, publishH, frameId);

            frag->bufferIndex = -1;
            releaseFragment(frag);
            m_fragmentMap.erase(frameId);
        }
    }
}

bool UDPCapture::AcquireFramePinned(void** pinnedRgbData, unsigned int* width,
                                    unsigned int* height, uint64_t* outFrameId,
                                    int* bufferIndex, uint32_t timeoutMs) {
    if (!m_running.load(std::memory_order_relaxed)) return false;

    const auto start = std::chrono::steady_clock::now();
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
        const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsedMs >= timeoutMs) return false;
        std::this_thread::yield();
    }
    return false;
}

void UDPCapture::ReleaseFrame(int bufferIndex) {
    if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS) return;
    int expected = BUFFER_IN_USE;
    m_bufferState[bufferIndex].compare_exchange_strong(
        expected, BUFFER_FREE, std::memory_order_acq_rel, std::memory_order_relaxed);
}
