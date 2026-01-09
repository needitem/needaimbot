/**
 * GamePC - Screen Capture UDP Streamer
 * 
 * Captures a specified screen region and sends it to the inference PC via UDP.
 * Receives mouse commands back from the inference PC via UDP broadcast.
 */

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

// Configuration
struct Config {
    std::string inferenceIP = "192.168.1.100";  // Inference PC IP
    unsigned short sendPort = 5007;              // Port to send frames
    unsigned short recvPort = 5006;              // Port to receive mouse events
    int captureX = 0;
    int captureY = 0;
    int captureWidth = 640;
    int captureHeight = 640;
    int targetFPS = 144;
    bool compress = true;  // Use simple RLE compression
};

static std::atomic<bool> g_running{true};
static Config g_config;

// Mouse event state (received from inference PC)
static std::atomic<bool> g_aimActive{false};
static std::atomic<bool> g_shootActive{false};

void signalHandler(int) {
    g_running.store(false);
}

// Simple DDA Capture class (minimal version)
class SimpleCapture {
public:
    bool Initialize() {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL obtained;
        
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            flags, levels, 2, D3D11_SDK_VERSION, &m_device, &obtained, &m_context);
        if (FAILED(hr)) return false;
        
        ComPtr<IDXGIDevice> dxgiDevice;
        hr = m_device.As(&dxgiDevice);
        if (FAILED(hr)) return false;
        
        ComPtr<IDXGIAdapter> adapter;
        hr = dxgiDevice->GetAdapter(&adapter);
        if (FAILED(hr)) return false;
        
        hr = adapter->EnumOutputs(0, &m_output);
        if (FAILED(hr)) return false;
        
        ComPtr<IDXGIOutput1> output1;
        hr = m_output.As(&output1);
        if (FAILED(hr)) return false;
        
        hr = output1->DuplicateOutput(m_device.Get(), &m_duplication);
        if (FAILED(hr)) return false;
        
        DXGI_OUTPUT_DESC desc;
        m_output->GetDesc(&desc);
        m_screenWidth = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
        m_screenHeight = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
        
        return true;
    }
    
    bool CaptureFrame(std::vector<uint8_t>& outData, int x, int y, int w, int h) {
        if (!m_duplication) return false;
        
        ComPtr<IDXGIResource> resource;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = m_duplication->AcquireNextFrame(16, &frameInfo, &resource);
        
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return false;
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            // Recreate duplication
            m_duplication.Reset();
            ComPtr<IDXGIOutput1> output1;
            m_output.As(&output1);
            output1->DuplicateOutput(m_device.Get(), &m_duplication);
            return false;
        }
        if (FAILED(hr)) return false;
        
        ComPtr<ID3D11Texture2D> texture;
        hr = resource.As(&texture);
        if (FAILED(hr)) {
            m_duplication->ReleaseFrame();
            return false;
        }
        
        D3D11_TEXTURE2D_DESC texDesc;
        texture->GetDesc(&texDesc);
        
        // Create staging texture if needed
        if (!m_staging || m_stagingWidth != w || m_stagingHeight != h) {
            D3D11_TEXTURE2D_DESC stagingDesc = {};
            stagingDesc.Width = w;
            stagingDesc.Height = h;
            stagingDesc.MipLevels = 1;
            stagingDesc.ArraySize = 1;
            stagingDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            stagingDesc.SampleDesc.Count = 1;
            stagingDesc.Usage = D3D11_USAGE_STAGING;
            stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            
            m_device->CreateTexture2D(&stagingDesc, nullptr, &m_staging);
            m_stagingWidth = w;
            m_stagingHeight = h;
        }
        
        // Copy region
        D3D11_BOX box = {(UINT)x, (UINT)y, 0, (UINT)(x + w), (UINT)(y + h), 1};
        m_context->CopySubresourceRegion(m_staging.Get(), 0, 0, 0, 0, texture.Get(), 0, &box);
        
        // Map and copy data
        D3D11_MAPPED_SUBRESOURCE mapped;
        hr = m_context->Map(m_staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            m_duplication->ReleaseFrame();
            return false;
        }
        
        // Copy to output (BGRA -> RGB, resize if needed)
        outData.resize(w * h * 3);
        uint8_t* dst = outData.data();
        for (int row = 0; row < h; ++row) {
            const uint8_t* src = (const uint8_t*)mapped.pData + row * mapped.RowPitch;
            for (int col = 0; col < w; ++col) {
                dst[0] = src[2];  // R
                dst[1] = src[1];  // G
                dst[2] = src[0];  // B
                dst += 3;
                src += 4;
            }
        }
        
        m_context->Unmap(m_staging.Get(), 0);
        m_duplication->ReleaseFrame();
        return true;
    }
    
    int GetScreenWidth() const { return m_screenWidth; }
    int GetScreenHeight() const { return m_screenHeight; }
    
private:
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<IDXGIOutput> m_output;
    ComPtr<IDXGIOutputDuplication> m_duplication;
    ComPtr<ID3D11Texture2D> m_staging;
    int m_screenWidth = 0;
    int m_screenHeight = 0;
    int m_stagingWidth = 0;
    int m_stagingHeight = 0;
};

// UDP Frame packet header
#pragma pack(push, 1)
struct FrameHeader {
    uint32_t magic;      // 0x46524D45 = "FRME"
    uint32_t frameId;
    uint16_t width;
    uint16_t height;
    uint32_t dataSize;
    uint8_t compressed;
};
#pragma pack(pop)

// Simple RLE compression for screen data
std::vector<uint8_t> compressRLE(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> result;
    result.reserve(data.size());
    
    size_t i = 0;
    while (i < data.size()) {
        uint8_t val = data[i];
        size_t count = 1;
        while (i + count < data.size() && data[i + count] == val && count < 255) {
            ++count;
        }
        
        if (count >= 3 || val == 0xFF) {
            result.push_back(0xFF);  // Escape byte
            result.push_back((uint8_t)count);
            result.push_back(val);
        } else {
            for (size_t j = 0; j < count; ++j) {
                result.push_back(val);
            }
        }
        i += count;
    }
    
    return result;
}

// Receive thread for mouse events from inference PC
void receiveThread(SOCKET sock) {
    char buffer[256];
    sockaddr_in from;
    int fromLen = sizeof(from);
    
    while (g_running.load()) {
        int ret = recvfrom(sock, buffer, sizeof(buffer) - 1, 0,
                          (SOCKADDR*)&from, &fromLen);
        if (ret <= 0) continue;
        
        buffer[ret] = '\0';
        std::string msg(buffer);
        
        // Parse mouse events from inference PC
        if (msg == "AIM:START") {
            g_aimActive.store(true);
        } else if (msg == "AIM:STOP") {
            g_aimActive.store(false);
        } else if (msg == "SHOOT:START") {
            g_shootActive.store(true);
        } else if (msg == "SHOOT:STOP") {
            g_shootActive.store(false);
        }
    }
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --ip <addr>       Inference PC IP (default: 192.168.1.100)\n"
              << "  --port <port>     Send port (default: 5007)\n"
              << "  --recv-port <port> Receive port (default: 5006)\n"
              << "  --region <x,y,w,h> Capture region (default: center 640x640)\n"
              << "  --fps <num>       Target FPS (default: 144)\n"
              << "  --no-compress     Disable compression\n";
}

bool parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--ip" && i + 1 < argc) {
            g_config.inferenceIP = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            g_config.sendPort = (unsigned short)std::stoi(argv[++i]);
        } else if (arg == "--recv-port" && i + 1 < argc) {
            g_config.recvPort = (unsigned short)std::stoi(argv[++i]);
        } else if (arg == "--region" && i + 1 < argc) {
            // Parse x,y,w,h
            std::string region = argv[++i];
            sscanf(region.c_str(), "%d,%d,%d,%d",
                   &g_config.captureX, &g_config.captureY,
                   &g_config.captureWidth, &g_config.captureHeight);
        } else if (arg == "--fps" && i + 1 < argc) {
            g_config.targetFPS = std::stoi(argv[++i]);
        } else if (arg == "--no-compress") {
            g_config.compress = false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (!parseArgs(argc, argv)) {
        return 0;
    }
    
    std::signal(SIGINT, signalHandler);
    
    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return 1;
    }
    
    // Initialize capture
    SimpleCapture capture;
    if (!capture.Initialize()) {
        std::cerr << "Failed to initialize screen capture\n";
        WSACleanup();
        return 1;
    }
    
    std::cout << "Screen: " << capture.GetScreenWidth() << "x" << capture.GetScreenHeight() << "\n";
    
    // Set default capture region to center of screen
    if (g_config.captureX == 0 && g_config.captureY == 0) {
        g_config.captureX = (capture.GetScreenWidth() - g_config.captureWidth) / 2;
        g_config.captureY = (capture.GetScreenHeight() - g_config.captureHeight) / 2;
    }
    
    std::cout << "Capture region: " << g_config.captureX << "," << g_config.captureY
              << " " << g_config.captureWidth << "x" << g_config.captureHeight << "\n";
    
    // Create send socket
    SOCKET sendSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sendSock == INVALID_SOCKET) {
        std::cerr << "Failed to create send socket\n";
        WSACleanup();
        return 1;
    }
    
    // Set send buffer size
    int sendBufSize = 4 * 1024 * 1024;  // 4MB
    setsockopt(sendSock, SOL_SOCKET, SO_SNDBUF, (char*)&sendBufSize, sizeof(sendBufSize));
    
    sockaddr_in destAddr = {};
    destAddr.sin_family = AF_INET;
    destAddr.sin_port = htons(g_config.sendPort);
    inet_pton(AF_INET, g_config.inferenceIP.c_str(), &destAddr.sin_addr);
    
    // Create receive socket
    SOCKET recvSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (recvSock == INVALID_SOCKET) {
        std::cerr << "Failed to create receive socket\n";
        closesocket(sendSock);
        WSACleanup();
        return 1;
    }
    
    sockaddr_in recvAddr = {};
    recvAddr.sin_family = AF_INET;
    recvAddr.sin_addr.s_addr = INADDR_ANY;
    recvAddr.sin_port = htons(g_config.recvPort);
    
    if (bind(recvSock, (SOCKADDR*)&recvAddr, sizeof(recvAddr)) == SOCKET_ERROR) {
        std::cerr << "Failed to bind receive socket\n";
        closesocket(sendSock);
        closesocket(recvSock);
        WSACleanup();
        return 1;
    }
    
    // Set receive timeout
    int timeout = 10;
    setsockopt(recvSock, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    
    // Start receive thread
    std::thread recvThread(receiveThread, recvSock);
    
    std::cout << "GamePC Streamer started\n";
    std::cout << "Sending to: " << g_config.inferenceIP << ":" << g_config.sendPort << "\n";
    std::cout << "Listening on port: " << g_config.recvPort << "\n";
    std::cout << "Target FPS: " << g_config.targetFPS << "\n";
    std::cout << "Press Ctrl+C to exit\n\n";
    
    // Main capture loop
    std::vector<uint8_t> frameData;
    uint32_t frameId = 0;
    auto frameInterval = std::chrono::microseconds(1000000 / g_config.targetFPS);
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    uint64_t totalFrames = 0;
    uint64_t totalBytes = 0;
    auto statsStart = std::chrono::steady_clock::now();
    
    while (g_running.load()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - lastFrameTime;
        
        if (elapsed < frameInterval) {
            std::this_thread::sleep_for(frameInterval - elapsed);
            continue;
        }
        lastFrameTime = now;
        
        // Capture frame
        if (!capture.CaptureFrame(frameData, g_config.captureX, g_config.captureY,
                                   g_config.captureWidth, g_config.captureHeight)) {
            continue;
        }
        
        // Compress if enabled
        std::vector<uint8_t> sendData;
        bool compressed = false;
        if (g_config.compress) {
            sendData = compressRLE(frameData);
            if (sendData.size() < frameData.size() * 0.9) {
                compressed = true;
            } else {
                sendData = frameData;  // Not worth compressing
            }
        } else {
            sendData = frameData;
        }
        
        // Prepare header
        FrameHeader header;
        header.magic = 0x46524D45;  // "FRME"
        header.frameId = frameId++;
        header.width = (uint16_t)g_config.captureWidth;
        header.height = (uint16_t)g_config.captureHeight;
        header.dataSize = (uint32_t)sendData.size();
        header.compressed = compressed ? 1 : 0;
        
        // Send header + data (may need fragmentation for large frames)
        const size_t maxPacket = 65000;  // UDP max - some overhead
        
        if (sizeof(header) + sendData.size() <= maxPacket) {
            // Single packet
            std::vector<uint8_t> packet(sizeof(header) + sendData.size());
            memcpy(packet.data(), &header, sizeof(header));
            memcpy(packet.data() + sizeof(header), sendData.data(), sendData.size());
            
            sendto(sendSock, (char*)packet.data(), (int)packet.size(), 0,
                   (SOCKADDR*)&destAddr, sizeof(destAddr));
        } else {
            // Fragment into multiple packets (simplified - header in first packet)
            // TODO: Implement proper fragmentation with sequence numbers
            std::cerr << "Frame too large, skipping: " << sendData.size() << " bytes\n";
        }
        
        totalFrames++;
        totalBytes += sendData.size();
        
        // Print stats every second
        auto statsDuration = std::chrono::duration_cast<std::chrono::seconds>(now - statsStart);
        if (statsDuration.count() >= 1) {
            double fps = totalFrames / (double)statsDuration.count();
            double mbps = (totalBytes * 8.0) / (statsDuration.count() * 1000000.0);
            std::cout << "\rFPS: " << std::fixed << std::setprecision(1) << fps
                      << " | " << std::setprecision(2) << mbps << " Mbps"
                      << " | Aim: " << (g_aimActive.load() ? "ON" : "OFF")
                      << " | Shoot: " << (g_shootActive.load() ? "ON" : "OFF")
                      << "     " << std::flush;
            
            totalFrames = 0;
            totalBytes = 0;
            statsStart = now;
        }
    }
    
    std::cout << "\nShutting down...\n";
    
    g_running.store(false);
    recvThread.join();
    
    closesocket(sendSock);
    closesocket(recvSock);
    WSACleanup();
    
    return 0;
}
