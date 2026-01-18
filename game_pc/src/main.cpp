/**
 * GamePC - Screen Capture UDP Streamer with Packet Fragmentation (No Compression)
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
#include <fstream>
#include <iomanip>
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
    std::string inferenceIP = "192.168.1.100";
    unsigned short sendPort = 5007;
    int captureX = 0;
    int captureY = 0;
    int captureWidth = 256;
    int captureHeight = 256;
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

        // BGRA 그대로 복사 (inference_pc에서 RGB로 변환)
        outData.resize(w * h * 4);
        uint8_t* dst = outData.data();
        for (int row = 0; row < h; ++row) {
            const uint8_t* src = (const uint8_t*)mapped.pData + row * mapped.RowPitch;
            memcpy(dst, src, w * 4);
            dst += w * 4;
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

// UDP Packet header - matches inference_pc UDPPacketHeader (16 bytes)
#pragma pack(push, 1)
struct UDPPacketHeader {
    uint32_t frameId;         // 4 bytes - 프레임 번호
    uint16_t chunkIndex;      // 2 bytes - 청크 인덱스 (0부터)
    uint16_t totalChunks;     // 2 bytes - 전체 청크 수
    uint32_t chunkSize;       // 4 bytes - 이 청크의 데이터 크기
    uint16_t frameWidth;      // 2 bytes - 프레임 너비
    uint16_t frameHeight;     // 2 bytes - 프레임 높이
};  // Total: 16 bytes
#pragma pack(pop)

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --ip <addr>       Inference PC IP (default: from config.ini)\n"
              << "  --port <port>     Send port (default: from config.ini)\n"
              << "  --region <x,y,w,h> Capture region (default: from config.ini)\n"
              << "  --fps <num>       Target FPS (default: from config.ini)\n"
              << "\nConfig file: config.ini\n";
}

// Simple INI parser for config.ini
bool loadConfig(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Config file '" << filename << "' not found, using defaults\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == ';' || line[0] == '#' || line[0] == '[') {
            continue;
        }

        // Parse key=value
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        // Trim key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // Set config values
        if (key == "InferenceIP") {
            g_config.inferenceIP = value;
        } else if (key == "SendPort") {
            g_config.sendPort = (unsigned short)std::stoi(value);
        } else if (key == "CaptureX") {
            g_config.captureX = std::stoi(value);
        } else if (key == "CaptureY") {
            g_config.captureY = std::stoi(value);
        } else if (key == "CaptureWidth") {
            g_config.captureWidth = std::stoi(value);
        } else if (key == "CaptureHeight") {
            g_config.captureHeight = std::stoi(value);
        } else if (key == "TargetFPS") {
            g_config.targetFPS = std::stoi(value);
        }
    }

    file.close();
    std::cout << "Loaded config from '" << filename << "'\n";
    return true;
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

std::string getExeDirectory(const char* argv0) {
    std::string path(argv0);
    size_t pos = path.find_last_of("\\/");
    if (pos != std::string::npos) {
        return path.substr(0, pos + 1);
    }
    return "";
}

int main(int argc, char** argv) {
    // Load config from exe directory
    std::string exeDir = getExeDirectory(argv[0]);
    std::string configPath = exeDir + "config.ini";
    loadConfig(configPath.c_str());

    // Command line args override config file
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
    int sendBufSize = 2 * 1024 * 1024;  // Larger buffer for fragmented packets
    setsockopt(sendSock, SOL_SOCKET, SO_SNDBUF, (char*)&sendBufSize, sizeof(sendBufSize));

    sockaddr_in destAddr = {};
    destAddr.sin_family = AF_INET;
    destAddr.sin_port = htons(g_config.sendPort);
    inet_pton(AF_INET, g_config.inferenceIP.c_str(), &destAddr.sin_addr);

    std::cout << "GamePC Streamer (BGRA, 60KB chunks) started\n";
    std::cout << "Sending to: " << g_config.inferenceIP << ":" << g_config.sendPort << "\n";
    std::cout << "Press Ctrl+C to exit\n\n";

    // Buffers
    std::vector<uint8_t> frameData;
    const size_t maxPayloadPerPacket = 60000;  // 큰 청크 (LAN 환경)
    std::vector<uint8_t> packetBuffer(sizeof(UDPPacketHeader) + maxPayloadPerPacket);

    uint32_t frameId = 0;
    uint64_t totalFrames = 0;
    uint64_t totalBytes = 0;
    auto statsStart = std::chrono::steady_clock::now();

    // Timing stats
    double totalCaptureMs = 0, totalSendMs = 0;

    // FPS limiting
    const double frameIntervalUs = 1000000.0 / g_config.targetFPS;
    auto nextFrameTime = std::chrono::steady_clock::now();

    // Wait up to 2 frame times for next frame
    const int captureTimeoutMs = 2000 / g_config.targetFPS;

    while (g_running.load()) {
        // FPS limiting - wait until next frame time
        auto now = std::chrono::steady_clock::now();
        if (now < nextFrameTime) {
            std::this_thread::sleep_until(nextFrameTime);
        }
        nextFrameTime = std::chrono::steady_clock::now() + std::chrono::microseconds((int64_t)frameIntervalUs);

        // Capture frame (blocks until new frame or timeout)
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!capture.CaptureFrame(frameData, g_config.captureX, g_config.captureY,
                                   g_config.captureWidth, g_config.captureHeight, captureTimeoutMs)) {
            continue;
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        // Calculate number of packets needed
        size_t frameSize = frameData.size();
        uint16_t totalPackets = (uint16_t)((frameSize + maxPayloadPerPacket - 1) / maxPayloadPerPacket);

        // Send fragmented packets
        for (uint16_t i = 0; i < totalPackets; i++) {
            size_t offset = i * maxPayloadPerPacket;
            size_t remaining = frameSize - offset;
            uint32_t payloadSize = (uint32_t)std::min(remaining, maxPayloadPerPacket);

            // Prepare packet header (matches inference_pc UDPPacketHeader)
            UDPPacketHeader* header = (UDPPacketHeader*)packetBuffer.data();
            header->frameId = frameId;
            header->chunkIndex = i;
            header->totalChunks = totalPackets;
            header->chunkSize = payloadSize;
            header->frameWidth = (uint16_t)g_config.captureWidth;
            header->frameHeight = (uint16_t)g_config.captureHeight;

            // Copy payload
            memcpy(packetBuffer.data() + sizeof(UDPPacketHeader),
                   frameData.data() + offset,
                   payloadSize);

            // Send packet
            int packetSize = (int)(sizeof(UDPPacketHeader) + payloadSize);
            sendto(sendSock, (char*)packetBuffer.data(), packetSize, 0,
                   (SOCKADDR*)&destAddr, sizeof(destAddr));
        }

        auto t3 = std::chrono::high_resolution_clock::now();

        frameId++;
        totalCaptureMs += std::chrono::duration<double, std::milli>(t2 - t1).count();
        totalSendMs += std::chrono::duration<double, std::milli>(t3 - t2).count();

        totalFrames++;
        totalBytes += frameSize;

        // Print stats every second
        auto statsNow = std::chrono::steady_clock::now();
        auto statsDuration = std::chrono::duration_cast<std::chrono::seconds>(statsNow - statsStart);
        if (statsDuration.count() >= 1 && totalFrames > 0) {
            double fps = totalFrames / (double)statsDuration.count();
            double mbps = (totalBytes * 8.0) / (statsDuration.count() * 1000000.0);
            double avgCapture = totalCaptureMs / totalFrames;
            double avgSend = totalSendMs / totalFrames;

            std::cout << "\rFPS: " << std::fixed << std::setprecision(1) << fps
                      << " | Cap:" << std::setprecision(2) << avgCapture << "ms"
                      << " Snd:" << avgSend << "ms"
                      << " | " << mbps << " Mbps"
                      << " | " << totalPackets << " pkts/frame"
                      << "     " << std::flush;

            totalFrames = 0;
            totalBytes = 0;
            totalCaptureMs = totalSendMs = 0;
            statsStart = statsNow;
        }
    }

    std::cout << "\nShutting down...\n";
    closesocket(sendSock);
    WSACleanup();

    return 0;
}
