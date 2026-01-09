/**
 * GamePC - Screen Capture UDP Streamer with LZ4 compression
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
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "lz4.h"

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

// Configuration
struct Config {
    std::string inferenceIP = "192.168.1.100";
    unsigned short sendPort = 5007;
    int captureX = 0;
    int captureY = 0;
    int captureWidth = 320;
    int captureHeight = 320;
    int targetFPS = 90;
};

static std::atomic<bool> g_running{true};
static Config g_config;

void signalHandler(int) {
    g_running.store(false);
}

// Simple DDA Capture class
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
    
    bool CaptureFrame(std::vector<uint8_t>& outData, int x, int y, int w, int h, UINT timeoutMs = 100) {
        if (!m_duplication) return false;
        
        ComPtr<IDXGIResource> resource;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = m_duplication->AcquireNextFrame(timeoutMs, &frameInfo, &resource);
        
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return false;
        if (hr == DXGI_ERROR_ACCESS_LOST) {
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
        
        // BGRA -> RGB
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
    uint32_t magic;           // 0x46524D45 = "FRME"
    uint32_t frameId;
    uint16_t width;
    uint16_t height;
    uint32_t compressedSize;  // LZ4 compressed size
    uint32_t originalSize;    // Original RGB size
};
#pragma pack(pop)

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --ip <addr>       Inference PC IP (default: 192.168.1.100)\n"
              << "  --port <port>     Send port (default: 5007)\n"
              << "  --region <x,y,w,h> Capture region (default: center 320x320)\n"
              << "  --fps <num>       Target FPS (default: 90)\n";
}

bool parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--ip" && i + 1 < argc) {
            g_config.inferenceIP = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            g_config.sendPort = (unsigned short)std::stoi(argv[++i]);
        } else if (arg == "--region" && i + 1 < argc) {
            std::string region = argv[++i];
            sscanf(region.c_str(), "%d,%d,%d,%d",
                   &g_config.captureX, &g_config.captureY,
                   &g_config.captureWidth, &g_config.captureHeight);
        } else if (arg == "--fps" && i + 1 < argc) {
            g_config.targetFPS = std::stoi(argv[++i]);
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
        std::cerr << "Failed to create socket\n";
        WSACleanup();
        return 1;
    }
    
    // Set send buffer size
    int sendBufSize = 1024 * 1024;
    setsockopt(sendSock, SOL_SOCKET, SO_SNDBUF, (char*)&sendBufSize, sizeof(sendBufSize));
    
    sockaddr_in destAddr = {};
    destAddr.sin_family = AF_INET;
    destAddr.sin_port = htons(g_config.sendPort);
    inet_pton(AF_INET, g_config.inferenceIP.c_str(), &destAddr.sin_addr);
    
    std::cout << "GamePC Streamer (LZ4) started\n";
    std::cout << "Sending to: " << g_config.inferenceIP << ":" << g_config.sendPort << "\n";
    std::cout << "Press Ctrl+C to exit\n\n";
    
    // Buffers
    std::vector<uint8_t> frameData;
    const int maxCompressedSize = LZ4_compressBound(g_config.captureWidth * g_config.captureHeight * 3);
    std::vector<char> compressBuffer(maxCompressedSize);
    std::vector<uint8_t> packetBuffer(sizeof(FrameHeader) + maxCompressedSize);
    
    uint32_t frameId = 0;
    uint64_t totalFrames = 0;
    uint64_t totalBytes = 0;
    auto statsStart = std::chrono::steady_clock::now();
    
    // Timing stats
    double totalCaptureMs = 0, totalCompressMs = 0, totalSendMs = 0;
    
    // Wait up to 2 frame times for next frame
    const int captureTimeoutMs = 2000 / g_config.targetFPS;
    
    while (g_running.load()) {
        // Capture frame (blocks until new frame or timeout)
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!capture.CaptureFrame(frameData, g_config.captureX, g_config.captureY,
                                   g_config.captureWidth, g_config.captureHeight, captureTimeoutMs)) {
            continue;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // LZ4 compress
        int compressedSize = LZ4_compress_fast(
            (const char*)frameData.data(),
            compressBuffer.data(),
            (int)frameData.size(),
            maxCompressedSize,
            1  // acceleration (1 = fastest)
        );
        auto t3 = std::chrono::high_resolution_clock::now();
        
        if (compressedSize <= 0) {
            std::cerr << "LZ4 compression failed\n";
            continue;
        }
        
        // Check if fits in single UDP packet
        const size_t maxPacket = 65000;
        if (sizeof(FrameHeader) + compressedSize > maxPacket) {
            std::cerr << "Frame too large after LZ4: " << compressedSize << " bytes\n";
            continue;
        }
        
        // Prepare packet
        FrameHeader* header = (FrameHeader*)packetBuffer.data();
        header->magic = 0x46524D45;  // "FRME"
        header->frameId = frameId++;
        header->width = (uint16_t)g_config.captureWidth;
        header->height = (uint16_t)g_config.captureHeight;
        header->compressedSize = (uint32_t)compressedSize;
        header->originalSize = (uint32_t)frameData.size();
        
        memcpy(packetBuffer.data() + sizeof(FrameHeader), compressBuffer.data(), compressedSize);
        
        // Send
        int packetSize = (int)(sizeof(FrameHeader) + compressedSize);
        sendto(sendSock, (char*)packetBuffer.data(), packetSize, 0,
               (SOCKADDR*)&destAddr, sizeof(destAddr));
        auto t4 = std::chrono::high_resolution_clock::now();
        
        totalCaptureMs += std::chrono::duration<double, std::milli>(t2 - t1).count();
        totalCompressMs += std::chrono::duration<double, std::milli>(t3 - t2).count();
        totalSendMs += std::chrono::duration<double, std::milli>(t4 - t3).count();
        
        totalFrames++;
        totalBytes += compressedSize;
        
        // Print stats every second
        auto statsNow = std::chrono::steady_clock::now();
        auto statsDuration = std::chrono::duration_cast<std::chrono::seconds>(statsNow - statsStart);
        if (statsDuration.count() >= 1 && totalFrames > 0) {
            double fps = totalFrames / (double)statsDuration.count();
            double mbps = (totalBytes * 8.0) / (statsDuration.count() * 1000000.0);
            double avgCapture = totalCaptureMs / totalFrames;
            double avgCompress = totalCompressMs / totalFrames;
            double avgSend = totalSendMs / totalFrames;
            double ratio = (g_config.captureWidth * g_config.captureHeight * 3.0) / (totalBytes / (double)totalFrames);
            
            std::cout << "\rFPS: " << std::fixed << std::setprecision(1) << fps
                      << " | Cap:" << std::setprecision(2) << avgCapture << "ms"
                      << " LZ4:" << avgCompress << "ms"
                      << " Snd:" << avgSend << "ms"
                      << " | " << ratio << ":1"
                      << " | " << mbps << " Mbps"
                      << "     " << std::flush;
            
            totalFrames = 0;
            totalBytes = 0;
            totalCaptureMs = totalCompressMs = totalSendMs = 0;
            statsStart = statsNow;
        }
    }
    
    std::cout << "\nShutting down...\n";
    closesocket(sendSock);
    WSACleanup();
    
    return 0;
}
