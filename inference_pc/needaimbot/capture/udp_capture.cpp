#include "udp_capture.h"

#include <iostream>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
#define SOCKADDR struct sockaddr
#define WSAETIMEDOUT WSAETIMEDOUT
#else
#include <sys/select.h>
#define SOCKADDR struct sockaddr
#define WSAETIMEDOUT ETIMEDOUT
#endif

UDPCapture::UDPCapture()
    : m_startTime(std::chrono::steady_clock::now()) {
}

UDPCapture::~UDPCapture() {
    Shutdown();
}

UDPCapture::FrameFragments* UDPCapture::acquireFragmentLocked() {
    if (!m_freeFragments.empty()) {
        FrameFragments* frag = m_freeFragments.back();
        m_freeFragments.pop_back();
        return frag;
    }
    m_fragmentStorage.emplace_back(std::make_unique<FrameFragments>());
    return m_fragmentStorage.back().get();
}

void UDPCapture::releaseFragmentLocked(FrameFragments* frag) {
    if (!frag) return;
    frag->totalPackets = 0;
    frag->receivedCount = 0;
    frag->width = 0;
    frag->height = 0;
    m_freeFragments.push_back(frag);
}

bool UDPCapture::allocatePinnedBuffers(size_t size) {
    if (m_pinnedBufferSize >= size && m_usePinnedMemory) {
        return true;  // Already allocated enough
    }

    freePinnedBuffers();

    // Allocate double-buffered pinned memory
    for (int i = 0; i < NUM_BUFFERS; i++) {
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
    std::cout << "[UDPCapture] Allocated " << (size / 1024) << "KB x " << NUM_BUFFERS
              << " pinned buffers for zero-copy\n";
    return true;
}

void UDPCapture::freePinnedBuffers() {
    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (m_pinnedFrameBuffer[i]) {
            cudaFreeHost(m_pinnedFrameBuffer[i]);
            m_pinnedFrameBuffer[i] = nullptr;
        }
    }
    m_pinnedBufferSize = 0;
    m_usePinnedMemory = false;
}

bool UDPCapture::Initialize(unsigned short listenPort) {
    m_listenPort = listenPort;

#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[UDPCapture] WSAStartup failed\n";
        return false;
    }
#endif

    // Create receive socket
    m_recvSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_recvSocket == INVALID_SOCKET) {
        std::cerr << "[UDPCapture] Failed to create receive socket\n";
        return false;
    }

    // Set large receive buffer
    int recvBufSize = 8 * 1024 * 1024;  // 8MB
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVBUF,
               (char*)&recvBufSize, sizeof(recvBufSize));

    // Bind to listen port
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

    // Set receive timeout for non-blocking behavior
#ifdef _WIN32
    DWORD timeout = 100;  // 100ms
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVTIMEO,
               (char*)&timeout, sizeof(timeout));
#else
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;  // 100ms
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVTIMEO,
               (char*)&timeout, sizeof(timeout));
#endif

    // Pre-allocate pinned buffers for typical 320x320 BGRA frames
    size_t defaultSize = 320 * 320 * 4;  // BGRA
    if (!allocatePinnedBuffers(defaultSize)) {
        std::cerr << "[UDPCapture] Warning: Failed to allocate pinned memory, "
                  << "falling back to regular memory\n";
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

#ifdef _WIN32
    WSACleanup();
#endif
}

bool UDPCapture::StartCapture() {
    if (m_running.load(std::memory_order_relaxed)) return true;
    if (m_recvSocket == INVALID_SOCKET) return false;

    m_running.store(true, std::memory_order_relaxed);
    m_isCapturing.store(true, std::memory_order_relaxed);
    m_startTime = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(m_fragmentMutex);
        for (auto& it : m_fragmentMap) {
            releaseFragmentLocked(it.second);
        }
        m_fragmentMap.clear();
        m_fragmentMap.reserve(128);
    }

    // Start receive thread
    m_recvThread = std::thread(&UDPCapture::receiveThread, this);

    std::cout << "[UDPCapture] Started capture (pinned memory: "
              << (m_usePinnedMemory ? "enabled" : "disabled") << ")\n";
    return true;
}

void UDPCapture::StopCapture() {
    if (!m_running.load(std::memory_order_relaxed)) return;

    m_running.store(false, std::memory_order_relaxed);
    m_isCapturing.store(false, std::memory_order_relaxed);

    // Wake up any waiting threads
    {
        std::lock_guard<std::mutex> lock(m_bufferMutex);
        m_newFrameAvailable = true;
    }
    m_frameReady.notify_all();

    if (m_recvThread.joinable()) {
        m_recvThread.join();
    }

    std::cout << "[UDPCapture] Stopped capture\n";
}

void UDPCapture::receiveThread() {
    std::vector<uint8_t> recvBuffer(65536);  // Max UDP packet size (up to 60KB chunks)
    constexpr size_t kChunkPayloadBytes = 60000;
    constexpr auto kFragmentStaleTimeout = std::chrono::milliseconds(100);
    constexpr uint32_t kCleanupPacketInterval = 64;
    uint32_t packetsSinceCleanup = 0;

    auto cleanupStaleFragments = [this, kFragmentStaleTimeout](std::chrono::steady_clock::time_point now) {
        std::lock_guard<std::mutex> lock(m_fragmentMutex);
        for (auto it = m_fragmentMap.begin(); it != m_fragmentMap.end();) {
            FrameFragments* frag = it->second;
            if (frag && (now - frag->lastUpdate) > kFragmentStaleTimeout) {
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
                releaseFragmentLocked(frag);
                it = m_fragmentMap.erase(it);
            } else {
                ++it;
            }
        }
    };

    sockaddr_in fromAddr;
#ifdef _WIN32
    int fromLen = sizeof(fromAddr);
#else
    socklen_t fromLen = sizeof(fromAddr);
#endif

    auto releaseCompletedFragment = [this](FrameFragments*& frag) {
        if (!frag) return;
        std::lock_guard<std::mutex> lock(m_fragmentMutex);
        releaseFragmentLocked(frag);
        frag = nullptr;
    };

    while (m_running.load(std::memory_order_relaxed)) {
        fromLen = sizeof(fromAddr);
        int ret = recvfrom(m_recvSocket, (char*)recvBuffer.data(),
                          (int)recvBuffer.size(), 0,
                          (SOCKADDR*)&fromAddr, &fromLen);

        if (ret <= 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) {
#else
            int err = errno;
            if (err == ETIMEDOUT || err == EWOULDBLOCK || err == EAGAIN) {
#endif
                // Cleanup old incomplete frames (older than stale timeout).
                cleanupStaleFragments(std::chrono::steady_clock::now());
                continue;  // Normal timeout, keep waiting
            }
            continue;
        }

        if ((++packetsSinceCleanup % kCleanupPacketInterval) == 0) {
            // Keep fragment map bounded even under continuous traffic.
            cleanupStaleFragments(std::chrono::steady_clock::now());
        }

        // Parse header (new format: 16 bytes)
        if (ret < (int)sizeof(UDPPacketHeader)) {
            continue;  // Too small
        }

        const UDPPacketHeader* header = (const UDPPacketHeader*)recvBuffer.data();

        // Validate chunk size matches received data
        size_t expectedPayload = ret - sizeof(UDPPacketHeader);
        if (header->chunkSize != expectedPayload) {
            continue;  // Size mismatch
        }

        const uint8_t* payload = recvBuffer.data() + sizeof(UDPPacketHeader);
        uint32_t frameId = header->frameId;
        uint16_t chunkIndex = header->chunkIndex;
        uint16_t totalChunks = header->totalChunks;
        uint32_t chunkSize = header->chunkSize;
        if (totalChunks == 0 || chunkIndex >= totalChunks ||
            header->frameWidth == 0 || header->frameHeight == 0) {
            continue;
        }

        bool frameComplete = false;
        uint16_t completedWidth = 0;
        uint16_t completedHeight = 0;
        FrameFragments* completedFrag = nullptr;

        // Assemble fragments (keep fragment lock scope tight).
        {
            std::lock_guard<std::mutex> lock(m_fragmentMutex);

            // Get or create fragment entry
            FrameFragments* frag = nullptr;
            auto fragIt = m_fragmentMap.find(frameId);
            if (fragIt == m_fragmentMap.end()) {
                frag = acquireFragmentLocked();
                m_fragmentMap.emplace(frameId, frag);
            } else {
                frag = fragIt->second;
            }
            if (!frag) {
                continue;
            }

            // Initialize if first packet of this frame
            // BGRA format: width * height * 4 bytes
            const size_t frameSize = static_cast<size_t>(header->frameWidth) * static_cast<size_t>(header->frameHeight) * 4;
            const bool metadataMismatch =
                (frag->totalPackets != 0) &&
                (frag->totalPackets != totalChunks ||
                 frag->width != header->frameWidth ||
                 frag->height != header->frameHeight);
            if (frag->totalPackets == 0 || metadataMismatch) {
                if (frag->data.size() != frameSize) {
                    frag->data.resize(frameSize);
                }
                if (frag->received.size() != totalChunks) {
                    frag->received.resize(totalChunks);
                }
                std::fill(frag->received.begin(), frag->received.end(), 0);
                frag->totalPackets = totalChunks;
                frag->receivedCount = 0;
                frag->width = header->frameWidth;
                frag->height = header->frameHeight;
            }

            frag->lastUpdate = std::chrono::steady_clock::now();

            // Store chunk data if not already received
            if (chunkIndex < frag->received.size() && !frag->received[chunkIndex]) {
                // Calculate offset: each chunk can be up to 60000 bytes
                size_t offset = static_cast<size_t>(chunkIndex) * kChunkPayloadBytes;
                if (offset < frag->data.size()) {
                    size_t remaining = frag->data.size() - offset;
                    size_t expectedChunkSize = std::min(kChunkPayloadBytes, remaining);
                    size_t requestedSize = static_cast<size_t>(chunkSize);
                    if (requestedSize == expectedChunkSize) {
                        memcpy(frag->data.data() + offset, payload, requestedSize);
                        frag->received[chunkIndex] = 1;
                        frag->receivedCount++;
                    }
                }
            }

            // Check if frame is complete
            if (frag->receivedCount == frag->totalPackets) {
                frameComplete = true;
                completedWidth = frag->width;
                completedHeight = frag->height;
                completedFrag = frag;
                m_fragmentMap.erase(frameId);
            }
        }

        if (!frameComplete || !completedFrag) {
            continue;
        }

        // Frame complete! Copy BGRA directly to pinned buffer (GPU does conversion)
        const size_t bgraSize = static_cast<size_t>(completedWidth) * static_cast<size_t>(completedHeight) * 4;
        if (completedFrag->data.size() != bgraSize) {
            m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
            releaseCompletedFragment(completedFrag);
            continue;
        }

        // Ensure pinned buffers are large enough
        if (!m_usePinnedMemory || m_pinnedBufferSize < bgraSize) {
            if (!allocatePinnedBuffers(bgraSize)) {
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
                releaseCompletedFragment(completedFrag);
                continue;
            }
        }

        bool published = false;
        {
            std::lock_guard<std::mutex> bufLock(m_bufferMutex);

            // Get write buffer index
            int writeIdx = m_writeBuffer.load(std::memory_order_relaxed);

            // Find a free pinned buffer for this frame.
            bool foundFreeBuffer = false;
            for (int attempt = 0; attempt < NUM_BUFFERS; ++attempt) {
                int candidate = (writeIdx + attempt) % NUM_BUFFERS;
                if (!m_bufferInUse[candidate].load(std::memory_order_acquire)) {
                    writeIdx = candidate;
                    foundFreeBuffer = true;
                    break;
                }
            }
            if (foundFreeBuffer) {
                uint8_t* dstBuffer = m_pinnedFrameBuffer[writeIdx];
                if (dstBuffer) {
                    // Direct BGRA memcpy to pinned buffer (GPU handles BGRA->CHW conversion)
                    memcpy(dstBuffer, completedFrag->data.data(), bgraSize);

                    // Swap buffers
                    m_readBuffer.store(writeIdx, std::memory_order_release);
                    m_writeBuffer.store((writeIdx + 1) % NUM_BUFFERS, std::memory_order_relaxed);

                    // Update frame info
                    m_frameWidth.store(completedWidth, std::memory_order_relaxed);
                    m_frameHeight.store(completedHeight, std::memory_order_relaxed);
                    m_lastFrameId.store(frameId, std::memory_order_relaxed);
                    m_frameCounter.fetch_add(1, std::memory_order_relaxed);
                    m_receivedFrames.fetch_add(1, std::memory_order_relaxed);

                    m_newFrameAvailable = true;
                    published = true;
                }
            }
        }

        releaseCompletedFragment(completedFrag);

        if (published) {
            m_frameReady.notify_one();
        } else {
            m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

bool UDPCapture::GetLatestFrame(void** frameData, unsigned int* width,
                                 unsigned int* height, unsigned int* size) {
    std::lock_guard<std::mutex> lock(m_bufferMutex);

    int readIdx = m_readBuffer.load(std::memory_order_acquire);
    if (!m_pinnedFrameBuffer[readIdx]) {
        return false;
    }

    if (frameData) *frameData = m_pinnedFrameBuffer[readIdx];
    if (width) *width = m_frameWidth.load();
    if (height) *height = m_frameHeight.load();
    if (size) *size = m_frameWidth.load() * m_frameHeight.load() * 4;  // BGRA

    return true;
}

bool UDPCapture::AcquireFrameSync(void** rgbData, unsigned int* width,
                                   unsigned int* height, uint64_t* outFrameId,
                                   uint32_t timeoutMs) {
    std::unique_lock<std::mutex> lock(m_bufferMutex);

    // Wait for new frame
    if (!m_newFrameAvailable) {
        if (!m_frameReady.wait_for(lock, std::chrono::milliseconds(timeoutMs),
            [this] { return m_newFrameAvailable || !m_running.load(std::memory_order_relaxed); })) {
            return false;  // Timeout
        }
    }

    if (!m_running.load(std::memory_order_relaxed)) {
        return false;
    }

    int readIdx = m_readBuffer.load(std::memory_order_acquire);
    if (!m_pinnedFrameBuffer[readIdx]) {
        return false;
    }

    m_newFrameAvailable = false;

    if (rgbData) *rgbData = m_pinnedFrameBuffer[readIdx];
    if (width) *width = m_frameWidth.load();
    if (height) *height = m_frameHeight.load();
    if (outFrameId) *outFrameId = m_lastFrameId.load();

    return true;
}

bool UDPCapture::AcquireFramePinned(void** pinnedRgbData, unsigned int* width,
                                     unsigned int* height, uint64_t* outFrameId,
                                     int* bufferIndex, uint32_t timeoutMs) {
    std::unique_lock<std::mutex> lock(m_bufferMutex);

    // Wait for new frame
    if (!m_newFrameAvailable) {
        if (!m_frameReady.wait_for(lock, std::chrono::milliseconds(timeoutMs),
            [this] { return m_newFrameAvailable || !m_running.load(std::memory_order_relaxed); })) {
            return false;  // Timeout
        }
    }

    if (!m_running.load(std::memory_order_relaxed)) {
        return false;
    }

    int readIdx = m_readBuffer.load(std::memory_order_acquire);
    if (!m_pinnedFrameBuffer[readIdx]) {
        return false;
    }

    // Mark buffer as in use
    m_bufferInUse[readIdx].store(true, std::memory_order_release);
    m_newFrameAvailable = false;

    if (pinnedRgbData) *pinnedRgbData = m_pinnedFrameBuffer[readIdx];
    if (width) *width = m_frameWidth.load();
    if (height) *height = m_frameHeight.load();
    if (outFrameId) *outFrameId = m_lastFrameId.load();
    if (bufferIndex) *bufferIndex = readIdx;

    return true;
}

void UDPCapture::ReleaseFrame(int bufferIndex) {
    if (bufferIndex >= 0 && bufferIndex < NUM_BUFFERS) {
        m_bufferInUse[bufferIndex].store(false, std::memory_order_release);
    }
}

bool UDPCapture::AcquireFrameToCuda(void* d_rgbBuffer, size_t bufferSize,
                                     unsigned int* width, unsigned int* height,
                                     cudaStream_t stream, uint32_t timeoutMs) {
    void* pinnedData = nullptr;
    unsigned int w, h;
    int bufIdx;

    if (!AcquireFramePinned(&pinnedData, &w, &h, nullptr, &bufIdx, timeoutMs)) {
        return false;
    }

    size_t requiredSize = w * h * 4;  // BGRA
    if (bufferSize < requiredSize) {
        std::cerr << "[UDPCapture] Buffer too small: " << bufferSize
                  << " < " << requiredSize << "\n";
        ReleaseFrame(bufIdx);
        return false;
    }

    // Direct async copy from pinned memory to GPU (zero intermediate copy)
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(d_rgbBuffer, pinnedData, requiredSize,
                              cudaMemcpyHostToDevice, stream);
    } else {
        err = cudaMemcpy(d_rgbBuffer, pinnedData, requiredSize,
                         cudaMemcpyHostToDevice);
    }

    // Release buffer after copy is queued (async) or done (sync)
    ReleaseFrame(bufIdx);

    if (err != cudaSuccess) {
        std::cerr << "[UDPCapture] CUDA memcpy failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    if (width) *width = w;
    if (height) *height = h;
    return true;
}

double UDPCapture::GetReceiveFps() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - m_startTime).count();
    if (elapsed < 0.001) return 0.0;
    return m_receivedFrames.load(std::memory_order_relaxed) / elapsed;
}
