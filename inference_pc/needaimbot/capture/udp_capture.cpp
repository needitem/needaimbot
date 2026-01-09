#include "udp_capture.h"
#include "lz4.h"

#include <iostream>
#include <cstring>

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

    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[UDPCapture] WSAStartup failed\n";
        return false;
    }

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
    DWORD timeout = 100;  // 100ms
    setsockopt(m_recvSocket, SOL_SOCKET, SO_RCVTIMEO, 
               (char*)&timeout, sizeof(timeout));

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

    WSACleanup();
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
    std::vector<uint8_t> recvBuffer(256 * 1024);  // 256KB receive buffer
    std::vector<uint8_t> decompressBuffer;
    sockaddr_in fromAddr;
    int fromLen = sizeof(fromAddr);

    while (m_running.load()) {
        int ret = recvfrom(m_recvSocket, (char*)recvBuffer.data(), 
                          (int)recvBuffer.size(), 0,
                          (SOCKADDR*)&fromAddr, &fromLen);
        
        if (ret <= 0) {
            if (WSAGetLastError() == WSAETIMEDOUT) {
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
        if (ret < (int)sizeof(UDPFrameHeader)) {
            continue;  // Too small
        }

        const UDPFrameHeader* header = (const UDPFrameHeader*)recvBuffer.data();
        
        // Validate magic
        if (header->magic != 0x46524D45) {  // "FRME"
            continue;  // Invalid packet
        }

        // Check data size
        size_t expectedPayload = ret - sizeof(UDPFrameHeader);
        if (header->compressedSize != expectedPayload) {
            m_droppedFrames.fetch_add(1);
            continue;  // Size mismatch
        }

        const uint8_t* payload = recvBuffer.data() + sizeof(UDPFrameHeader);
        size_t compressedSize = header->compressedSize;
        size_t originalSize = header->originalSize;

        // Decompress LZ4
        if (!decompressLZ4(payload, compressedSize, decompressBuffer, originalSize)) {
            m_droppedFrames.fetch_add(1);
            continue;  // Decompression failed
        }

        // Store frame in write buffer
        {
            std::lock_guard<std::mutex> lock(m_bufferMutex);
            
            // Resize buffer if needed
            if (m_frameBuffer[m_writeBuffer].size() != originalSize) {
                m_frameBuffer[m_writeBuffer].resize(originalSize);
            }
            
            // Copy data
            memcpy(m_frameBuffer[m_writeBuffer].data(), decompressBuffer.data(), originalSize);
            
            // Swap buffers
            std::swap(m_writeBuffer, m_readBuffer);
            
            // Update frame info
            m_frameWidth.store(header->width);
            m_frameHeight.store(header->height);
            m_lastFrameId.store(header->frameId);
            m_frameCounter.fetch_add(1);
            m_receivedFrames.fetch_add(1);
            
            m_newFrameAvailable = true;
        }
        m_frameReady.notify_one();
    }
}

bool UDPCapture::decompressLZ4(const uint8_t* compressed, size_t compressedSize,
                               std::vector<uint8_t>& output, size_t originalSize) {
    output.resize(originalSize);
    
    int result = LZ4_decompress_safe(
        (const char*)compressed,
        (char*)output.data(),
        (int)compressedSize,
        (int)originalSize
    );
    
    return result == (int)originalSize;
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
