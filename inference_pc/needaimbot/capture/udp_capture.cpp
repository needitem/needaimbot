#include "udp_capture.h"

#include <iostream>
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

bool UDPCapture::Initialize(unsigned short listenPort,
                            const std::string& gamePcIp,
                            unsigned short mouseStatePort) {
    m_listenPort = listenPort;
    m_mouseStatePort = mouseStatePort;

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

    // Create send socket for mouse state
    m_sendSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_sendSocket == INVALID_SOCKET) {
        std::cerr << "[UDPCapture] Failed to create send socket\n";
        closesocket(m_recvSocket);
        m_recvSocket = INVALID_SOCKET;
        return false;
    }

    // Set game PC address if provided
    if (!gamePcIp.empty()) {
        m_gamePcAddr.sin_family = AF_INET;
        m_gamePcAddr.sin_port = htons(mouseStatePort);
        if (inet_pton(AF_INET, gamePcIp.c_str(), &m_gamePcAddr.sin_addr) == 1) {
            m_hasGamePcAddr = true;
            std::cout << "[UDPCapture] Will send mouse state to "
                      << gamePcIp << ":" << mouseStatePort << "\n";
        }
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
    if (m_sendSocket != INVALID_SOCKET) {
        closesocket(m_sendSocket);
        m_sendSocket = INVALID_SOCKET;
    }

    // Free CUDA pinned memory
    if (m_pinnedBuffer) {
        cudaFreeHost(m_pinnedBuffer);
        m_pinnedBuffer = nullptr;
        m_pinnedBufferSize = 0;
    }

#ifdef _WIN32
    WSACleanup();
#endif
}

bool UDPCapture::StartCapture() {
    if (m_running.load()) return true;
    if (m_recvSocket == INVALID_SOCKET) return false;

    m_running.store(true);
    m_isCapturing.store(true);
    m_startTime = std::chrono::steady_clock::now();

    // Start receive thread
    m_recvThread = std::thread(&UDPCapture::receiveThread, this);

    std::cout << "[UDPCapture] Started capture\n";
    return true;
}

void UDPCapture::StopCapture() {
    if (!m_running.load()) return;

    m_running.store(false);
    m_isCapturing.store(false);

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
    std::vector<uint8_t> recvBuffer(2048);  // Max packet size
    sockaddr_in fromAddr;
#ifdef _WIN32
    int fromLen = sizeof(fromAddr);
#else
    socklen_t fromLen = sizeof(fromAddr);
#endif

    while (m_running.load()) {
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
                // Cleanup old incomplete frames (older than 100ms)
                auto now = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(m_fragmentMutex);
                for (auto it = m_fragmentMap.begin(); it != m_fragmentMap.end();) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - it->second.lastUpdate).count();
                    if (elapsed > 100) {
                        m_droppedFrames.fetch_add(1);
                        it = m_fragmentMap.erase(it);
                    } else {
                        ++it;
                    }
                }
                continue;  // Normal timeout, keep waiting
            }
            continue;
        }

        // Auto-detect game PC address from first received packet
        if (!m_hasGamePcAddr) {
            m_gamePcAddr = fromAddr;
            m_gamePcAddr.sin_port = htons(m_mouseStatePort);
            m_hasGamePcAddr = true;

            char ipStr[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &fromAddr.sin_addr, ipStr, INET_ADDRSTRLEN);
            std::cout << "[UDPCapture] Detected game PC at " << ipStr << "\n";
        }

        // Parse header
        if (ret < (int)sizeof(UDPPacketHeader)) {
            continue;  // Too small
        }

        const UDPPacketHeader* header = (const UDPPacketHeader*)recvBuffer.data();

        // Validate magic
        if (header->magic != 0x46524D45) {  // "FRME"
            continue;  // Invalid packet
        }

        // Check data size
        size_t expectedPayload = ret - sizeof(UDPPacketHeader);
        if (header->dataSize != expectedPayload) {
            continue;  // Size mismatch
        }

        const uint8_t* payload = recvBuffer.data() + sizeof(UDPPacketHeader);
        uint32_t frameId = header->frameId;
        uint16_t packetIndex = header->packetIndex;
        uint16_t totalPackets = header->totalPackets;

        // Assemble fragments
        {
            std::lock_guard<std::mutex> lock(m_fragmentMutex);

            // Get or create fragment entry
            auto& frag = m_fragmentMap[frameId];

            // Initialize if first packet of this frame
            if (frag.totalPackets == 0) {
                size_t frameSize = header->width * header->height * 3;
                frag.data.resize(frameSize);
                frag.received.resize(totalPackets, false);
                frag.totalPackets = totalPackets;
                frag.receivedCount = 0;
                frag.width = header->width;
                frag.height = header->height;
            }

            frag.lastUpdate = std::chrono::steady_clock::now();

            // Store packet data if not already received
            if (packetIndex < frag.received.size() && !frag.received[packetIndex]) {
                size_t offset = packetIndex * 1400;  // Max payload per packet
                size_t copySize = std::min((size_t)header->dataSize, frag.data.size() - offset);
                memcpy(frag.data.data() + offset, payload, copySize);
                frag.received[packetIndex] = true;
                frag.receivedCount++;
            }

            // Check if frame is complete
            if (frag.receivedCount == frag.totalPackets) {
                // Frame complete! Move to output buffer
                {
                    std::lock_guard<std::mutex> bufLock(m_bufferMutex);

                    // Resize buffer if needed
                    if (m_frameBuffer[m_writeBuffer].size() != frag.data.size()) {
                        m_frameBuffer[m_writeBuffer].resize(frag.data.size());
                    }

                    // Copy data
                    memcpy(m_frameBuffer[m_writeBuffer].data(), frag.data.data(), frag.data.size());

                    // Swap buffers
                    std::swap(m_writeBuffer, m_readBuffer);

                    // Update frame info
                    m_frameWidth.store(frag.width);
                    m_frameHeight.store(frag.height);
                    m_lastFrameId.store(frameId);
                    m_frameCounter.fetch_add(1);
                    m_receivedFrames.fetch_add(1);

                    m_newFrameAvailable = true;
                }
                m_frameReady.notify_one();

                // Remove from fragment map
                m_fragmentMap.erase(frameId);
            }
        }
    }
}

bool UDPCapture::GetLatestFrame(void** frameData, unsigned int* width,
                                 unsigned int* height, unsigned int* size) {
    std::lock_guard<std::mutex> lock(m_bufferMutex);

    if (m_frameBuffer[m_readBuffer].empty()) {
        return false;
    }

    if (frameData) *frameData = m_frameBuffer[m_readBuffer].data();
    if (width) *width = m_frameWidth.load();
    if (height) *height = m_frameHeight.load();
    if (size) *size = (unsigned int)m_frameBuffer[m_readBuffer].size();

    return true;
}

bool UDPCapture::AcquireFrameSync(void** rgbData, unsigned int* width,
                                   unsigned int* height, uint64_t* outFrameId,
                                   uint32_t timeoutMs) {
    std::unique_lock<std::mutex> lock(m_bufferMutex);

    // Wait for new frame
    if (!m_newFrameAvailable) {
        if (!m_frameReady.wait_for(lock, std::chrono::milliseconds(timeoutMs),
            [this] { return m_newFrameAvailable || !m_running.load(); })) {
            return false;  // Timeout
        }
    }

    if (!m_running.load() || m_frameBuffer[m_readBuffer].empty()) {
        return false;
    }

    m_newFrameAvailable = false;

    if (rgbData) *rgbData = m_frameBuffer[m_readBuffer].data();
    if (width) *width = m_frameWidth.load();
    if (height) *height = m_frameHeight.load();
    if (outFrameId) *outFrameId = m_lastFrameId.load();

    return true;
}

bool UDPCapture::AcquireFrameToCuda(void* d_rgbBuffer, size_t bufferSize,
                                     unsigned int* width, unsigned int* height,
                                     cudaStream_t stream, uint32_t timeoutMs) {
    void* hostData = nullptr;
    unsigned int w, h;

    if (!AcquireFrameSync(&hostData, &w, &h, nullptr, timeoutMs)) {
        return false;
    }

    size_t requiredSize = w * h * 3;
    if (bufferSize < requiredSize) {
        std::cerr << "[UDPCapture] Buffer too small: " << bufferSize
                  << " < " << requiredSize << "\n";
        return false;
    }

    // Ensure pinned buffer for async copy
    if (m_pinnedBufferSize < requiredSize) {
        if (m_pinnedBuffer) {
            cudaFreeHost(m_pinnedBuffer);
        }
        cudaMallocHost(&m_pinnedBuffer, requiredSize);
        m_pinnedBufferSize = requiredSize;
    }

    // Copy to pinned memory then to GPU
    memcpy(m_pinnedBuffer, hostData, requiredSize);

    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(d_rgbBuffer, m_pinnedBuffer, requiredSize,
                              cudaMemcpyHostToDevice, stream);
    } else {
        err = cudaMemcpy(d_rgbBuffer, m_pinnedBuffer, requiredSize,
                         cudaMemcpyHostToDevice);
    }

    if (err != cudaSuccess) {
        std::cerr << "[UDPCapture] CUDA memcpy failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    if (width) *width = w;
    if (height) *height = h;
    return true;
}

void UDPCapture::SendMouseState(bool aimActive, bool shootActive) {
    if (!m_hasGamePcAddr || m_sendSocket == INVALID_SOCKET) {
        return;
    }

    // Only send on state change
    if (aimActive != m_lastAimState) {
        const char* msg = aimActive ? "AIM:START" : "AIM:STOP";
        sendto(m_sendSocket, msg, (int)strlen(msg), 0,
               (SOCKADDR*)&m_gamePcAddr, sizeof(m_gamePcAddr));
        m_lastAimState = aimActive;
    }

    if (shootActive != m_lastShootState) {
        const char* msg = shootActive ? "SHOOT:START" : "SHOOT:STOP";
        sendto(m_sendSocket, msg, (int)strlen(msg), 0,
               (SOCKADDR*)&m_gamePcAddr, sizeof(m_gamePcAddr));
        m_lastShootState = shootActive;
    }
}

double UDPCapture::GetReceiveFps() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - m_startTime).count();
    if (elapsed < 0.001) return 0.0;
    return m_receivedFrames.load() / elapsed;
}
