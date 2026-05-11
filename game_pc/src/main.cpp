/**
 * GamePC - Screen Capture UDP Streamer with Packet Fragmentation (No Compression)
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mmsystem.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <immintrin.h>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "winmm.lib")

using Microsoft::WRL::ComPtr;

// Configuration
struct Config {
    std::string inferenceIP = "192.168.1.100";
    unsigned short sendPort = 5007;
    std::string localBindIP;
    int captureX = 0;
    int captureY = 0;
    int captureWidth = 256;
    int captureHeight = 256;
    bool useCaptureOffset = false;
    int captureOffsetX = 0;
    int captureOffsetY = 0;
    int targetFPS = 90;
    int outputIndex = 0;
    bool useGUI = true;
    int packetPayloadBytes = 60000;
    int captureTimeoutMs = 5;
    // Credit mode reduces wasted network traffic and stale frames; it can add
    // a tiny wait versus pure continuous-latest push when the LAN is uncongested.
    // Frequent "Rec" in the status line means credit packets are late/lost and
    // this recovery timeout should be tuned for the real network.
    int creditTimeoutMs = 50;
};

static std::atomic<bool> g_running{true};
static Config g_config;

static constexpr int kMinPacketPayloadBytes = 512;
static constexpr int kMaxPacketPayloadBytes = 60000;
static constexpr uint32_t UDP_PACKET_V2_MAGIC = 0x32415047u;  // "GPA2" little-endian
static constexpr uint8_t UDP_PIXEL_FORMAT_RGB = 2;
static constexpr uint32_t UDP_CREDIT_MAGIC = 0x43504147u;  // "GPAC" little-endian

struct OutputInfo {
    int index = 0;
    int width = 0;
    int height = 0;
    int refreshHz = 0;
    bool primary = false;
    std::string deviceName;
    std::string label;
};

struct LatestFrameSlot {
    std::mutex mutex;
    std::condition_variable cv;
    // Capture keeps only the newest DDA BGRA frame. The sender converts that
    // latest frame to RGB with SIMD, so old frames cannot queue behind inference.
    std::vector<uint8_t> bgra;
    uint64_t sequence = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    bool hasFrame = false;
    bool stopped = false;
};

struct StreamStats {
    std::atomic<uint64_t> captureAttempts{0};
    std::atomic<uint64_t> capturedFrames{0};
    std::atomic<uint64_t> skippedFrames{0};
    std::atomic<uint64_t> captureTimeouts{0};
    std::atomic<uint64_t> captureAccessLost{0};
    std::atomic<uint64_t> captureFailures{0};
    std::atomic<uint64_t> outputFrames{0};
    std::atomic<uint64_t> sentFrames{0};
    std::atomic<uint64_t> droppedFrames{0};
    std::atomic<uint64_t> wouldBlockDrops{0};
    std::atomic<uint64_t> creditPackets{0};
    std::atomic<uint64_t> recoverySends{0};
    std::atomic<uint64_t> totalBytes{0};
    std::atomic<uint64_t> totalPackets{0};
    std::atomic<int64_t> totalCaptureUs{0};
    std::atomic<int64_t> totalConvertUs{0};
    std::atomic<int64_t> totalSendUs{0};
    std::atomic<uint32_t> lastCaptureHr{0};
};

void applyWindowsRealtimeHint(const wchar_t* threadName, int idealProcessorOffset) {
    HANDLE thread = GetCurrentThread();
    if (threadName) {
        using SetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PCWSTR);
        HMODULE kernel32 = GetModuleHandleW(L"Kernel32.dll");
        auto setThreadDescription = kernel32
            ? reinterpret_cast<SetThreadDescriptionFn>(GetProcAddress(kernel32, "SetThreadDescription"))
            : nullptr;
        if (setThreadDescription) {
            setThreadDescription(thread, threadName);
        }
    }
    SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
    const DWORD cpuCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (cpuCount > 1) {
        const DWORD offset = static_cast<DWORD>(std::max(0, idealProcessorOffset));
        const DWORD processor = (cpuCount - 1 - std::min(offset, cpuCount - 1));
        SetThreadIdealProcessor(thread, processor);
    }
}

void signalHandler(int) {
    g_running.store(false);
}

void printStatusLine(const std::string& text) {
    HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    if (console == INVALID_HANDLE_VALUE || console == nullptr) {
        std::cout << '\r' << text << std::flush;
        return;
    }

    CONSOLE_SCREEN_BUFFER_INFO csbi{};
    if (!GetConsoleScreenBufferInfo(console, &csbi)) {
        std::cout << '\r' << text << std::flush;
        return;
    }

    std::string clipped = text;
    if (csbi.dwSize.X > 1 && clipped.size() >= static_cast<size_t>(csbi.dwSize.X)) {
        clipped.resize(static_cast<size_t>(csbi.dwSize.X - 1));
    }

    COORD lineStart{};
    lineStart.X = 0;
    lineStart.Y = csbi.dwCursorPosition.Y;

    DWORD written = 0;
    FillConsoleOutputCharacterA(console, ' ', csbi.dwSize.X, lineStart, &written);
    FillConsoleOutputAttribute(console, csbi.wAttributes, csbi.dwSize.X, lineStart, &written);
    SetConsoleCursorPosition(console, lineStart);

    std::cout << clipped << std::flush;
}

std::string toLowerCopy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

int clampPacketPayloadBytes(int bytes) {
    return std::clamp(bytes, kMinPacketPayloadBytes, kMaxPacketPayloadBytes);
}

#if defined(_M_X64) || defined(_M_IX86) || defined(__SSSE3__)
void convertBgraToRgbSsse3(const uint8_t* src, uint8_t* dst, size_t pixelCount) {
    const __m128i shuffleMask = _mm_setr_epi8(
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12,
        static_cast<char>(0x80), static_cast<char>(0x80),
        static_cast<char>(0x80), static_cast<char>(0x80));

    size_t i = 0;
    for (; i + 16 <= pixelCount; i += 16) {
        const __m128i p0 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)), shuffleMask);
        const __m128i p1 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 16)), shuffleMask);
        const __m128i p2 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 32)), shuffleMask);
        const __m128i p3 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 48)), shuffleMask);

        const __m128i out0 = _mm_or_si128(p0, _mm_slli_si128(p1, 12));
        const __m128i out1 = _mm_or_si128(_mm_srli_si128(p1, 4), _mm_slli_si128(p2, 8));
        const __m128i out2 = _mm_or_si128(_mm_srli_si128(p2, 8), _mm_slli_si128(p3, 4));

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), out0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), out1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 32), out2);
        src += 64;
        dst += 48;
    }

    for (; i < pixelCount; ++i) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        src += 4;
        dst += 3;
    }
}
#endif

void convertBgraToRgb(const std::vector<uint8_t>& bgra, std::vector<uint8_t>& rgb,
                      int width, int height) {
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t expectedBgraBytes = pixelCount * 4;
    if (bgra.size() < expectedBgraBytes) {
        rgb.clear();
        return;
    }
    const size_t requiredBytes = pixelCount * 3;
    if (rgb.size() != requiredBytes) {
        rgb.resize(requiredBytes);
    }

    const uint8_t* src = bgra.data();
    uint8_t* dst = rgb.data();
#if defined(_M_X64) || defined(_M_IX86) || defined(__SSSE3__)
    convertBgraToRgbSsse3(src, dst, pixelCount);
#else
    for (size_t i = 0; i < pixelCount; ++i) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        src += 4;
        dst += 3;
    }
#endif
}

std::string wideToUtf8(const wchar_t* wide) {
    if (!wide || wide[0] == L'\0') return "";

    int utf8Size = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (utf8Size <= 1) return "";

    std::string result(static_cast<size_t>(utf8Size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, result.data(), utf8Size, nullptr, nullptr);
    result.pop_back();  // Drop trailing null terminator.
    return result;
}

std::vector<OutputInfo> enumerateCaptureOutputs() {
    std::vector<OutputInfo> outputs;

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL obtained;

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        flags, levels, 2, D3D11_SDK_VERSION, &device, &obtained, &context);
    if (FAILED(hr)) return outputs;

    ComPtr<IDXGIDevice> dxgiDevice;
    hr = device.As(&dxgiDevice);
    if (FAILED(hr)) return outputs;

    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr)) return outputs;

    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(index, &output);
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr)) {
            continue;
        }

        DXGI_OUTPUT_DESC desc{};
        if (FAILED(output->GetDesc(&desc))) {
            continue;
        }

        OutputInfo info{};
        info.index = static_cast<int>(index);
        info.width = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
        info.height = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
        info.deviceName = wideToUtf8(desc.DeviceName);

        MONITORINFO monitorInfo{};
        monitorInfo.cbSize = sizeof(monitorInfo);
        if (GetMonitorInfo(desc.Monitor, &monitorInfo)) {
            info.primary = (monitorInfo.dwFlags & MONITORINFOF_PRIMARY) != 0;
        }

        DEVMODEW dm{};
        dm.dmSize = sizeof(dm);
        if (EnumDisplaySettingsW(desc.DeviceName, ENUM_CURRENT_SETTINGS, &dm) &&
            dm.dmDisplayFrequency > 1) {
            info.refreshHz = static_cast<int>(dm.dmDisplayFrequency);
        }

        const char* primarySuffix = info.primary ? ", Primary" : "";
        if (info.refreshHz > 0) {
            char text[256];
            std::snprintf(text, sizeof(text), "[%d] %s  %dx%d @ %dHz%s",
                          info.index, info.deviceName.c_str(), info.width, info.height,
                          info.refreshHz, primarySuffix);
            info.label = text;
        } else {
            char text[256];
            std::snprintf(text, sizeof(text), "[%d] %s  %dx%d%s",
                          info.index, info.deviceName.c_str(), info.width, info.height,
                          primarySuffix);
            info.label = text;
        }

        outputs.push_back(std::move(info));
    }

    return outputs;
}

enum class CaptureStatus {
    Captured,
    Timeout,
    AccessLost,
    Failed,
};

struct CaptureResult {
    CaptureStatus status = CaptureStatus::Failed;
    HRESULT hr = S_OK;
};

// Simple DDA Capture class
class SimpleCapture {
public:
    bool Initialize(int outputIndex) {
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

        hr = adapter->EnumOutputs(static_cast<UINT>(outputIndex), &m_output);
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

    bool CaptureFrame(std::vector<uint8_t>& outData, int x, int y, int w, int h,
                      UINT timeoutMs = 100, CaptureResult* result = nullptr) {
        auto setResult = [&](CaptureStatus status, HRESULT hr) {
            if (result) {
                result->status = status;
                result->hr = hr;
            }
        };

        if (!m_duplication) {
            setResult(CaptureStatus::Failed, E_POINTER);
            return false;
        }

        ComPtr<IDXGIResource> resource;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = m_duplication->AcquireNextFrame(timeoutMs, &frameInfo, &resource);

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            setResult(CaptureStatus::Timeout, hr);
            return false;
        }
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            m_duplication.Reset();
            ComPtr<IDXGIOutput1> output1;
            m_output.As(&output1);
            output1->DuplicateOutput(m_device.Get(), &m_duplication);
            setResult(CaptureStatus::AccessLost, hr);
            return false;
        }
        if (FAILED(hr)) {
            setResult(CaptureStatus::Failed, hr);
            return false;
        }

        ComPtr<ID3D11Texture2D> texture;
        hr = resource.As(&texture);
        if (FAILED(hr)) {
            m_duplication->ReleaseFrame();
            setResult(CaptureStatus::Failed, hr);
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
            setResult(CaptureStatus::Failed, hr);
            return false;
        }

        // Copy BGRA readback; the sender converts it to RGB before UDP transmission.
        const size_t frameBytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 4;
        if (outData.size() != frameBytes) {
            outData.resize(frameBytes);
        }
        if (mapped.RowPitch == (UINT)(w * 4)) {
            // Contiguous: single memcpy (no row padding)
            memcpy(outData.data(), mapped.pData, frameBytes);
        } else {
            // Padded rows: copy row by row
            uint8_t* dst = outData.data();
            for (int row = 0; row < h; ++row) {
                memcpy(dst, (const uint8_t*)mapped.pData + row * mapped.RowPitch, w * 4);
                dst += w * 4;
            }
        }

        m_context->Unmap(m_staging.Get(), 0);
        m_duplication->ReleaseFrame();
        setResult(CaptureStatus::Captured, S_OK);
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

#pragma pack(push, 1)
struct UDPPacketHeaderV2 {
    uint32_t magic;
    uint16_t headerSize;
    uint16_t flags;
    uint32_t frameId;
    uint32_t payloadOffset;
    uint32_t frameBytes;
    uint16_t chunkIndex;
    uint16_t totalChunks;
    uint32_t chunkSize;
    uint16_t frameWidth;
    uint16_t frameHeight;
    uint8_t pixelFormat;
    uint8_t bytesPerPixel;
    uint16_t reserved;
};
static_assert(sizeof(UDPPacketHeaderV2) == 36, "UDPPacketHeaderV2 must stay wire-compatible");

struct UDPCreditPacket {
    uint32_t magic;
    uint16_t size;
    uint16_t flags;
    uint32_t minFrameId;
    uint32_t credits;
    uint64_t sequence;
};
static_assert(sizeof(UDPCreditPacket) == 24, "UDPCreditPacket must stay wire-compatible");
#pragma pack(pop)

struct FrameSendResult {
    bool sent = false;
    bool wouldBlock = false;
    uint16_t totalPackets = 0;
};

FrameSendResult sendFrameUdp(SOCKET sendSock, const sockaddr_in& destAddr,
                             const uint8_t* sendData, size_t frameSize,
                             uint32_t frameId, uint16_t frameWidth,
                             uint16_t frameHeight, uint8_t pixelFormat,
                             uint8_t bytesPerPixel, size_t maxPayloadPerPacket) {
    FrameSendResult result{};
    if (!sendData || frameSize == 0 || frameSize > UINT32_MAX || maxPayloadPerPacket == 0) {
        return result;
    }

    const size_t totalPacketsSize = (frameSize + maxPayloadPerPacket - 1) / maxPayloadPerPacket;
    if (totalPacketsSize == 0 || totalPacketsSize > UINT16_MAX) {
        return result;
    }
    result.totalPackets = static_cast<uint16_t>(totalPacketsSize);

    for (uint16_t i = 0; i < result.totalPackets; ++i) {
        const size_t offset = static_cast<size_t>(i) * maxPayloadPerPacket;
        const size_t remaining = frameSize - offset;
        const uint32_t payloadSize = static_cast<uint32_t>(std::min(remaining, maxPayloadPerPacket));

        UDPPacketHeaderV2 header{};
        header.magic = UDP_PACKET_V2_MAGIC;
        header.headerSize = static_cast<uint16_t>(sizeof(UDPPacketHeaderV2));
        header.flags = 0;
        header.frameId = frameId;
        header.payloadOffset = static_cast<uint32_t>(offset);
        header.frameBytes = static_cast<uint32_t>(frameSize);
        header.chunkIndex = i;
        header.totalChunks = result.totalPackets;
        header.chunkSize = payloadSize;
        header.frameWidth = frameWidth;
        header.frameHeight = frameHeight;
        header.pixelFormat = pixelFormat;
        header.bytesPerPixel = bytesPerPixel;
        header.reserved = 0;

        WSABUF bufs[2];
        bufs[0].buf = reinterpret_cast<CHAR*>(&header);
        bufs[0].len = sizeof(UDPPacketHeaderV2);
        bufs[1].buf = reinterpret_cast<CHAR*>(const_cast<uint8_t*>(sendData) + offset);
        bufs[1].len = payloadSize;

        DWORD bytesSent = 0;
        const int sendResult = WSASendTo(
            sendSock,
            bufs,
            2,
            &bytesSent,
            0,
            (SOCKADDR*)&destAddr,
            sizeof(destAddr),
            nullptr,
            nullptr
        );
        if (sendResult == SOCKET_ERROR) {
            const int err = WSAGetLastError();
            result.wouldBlock = (err == WSAEWOULDBLOCK || err == WSAENOBUFS);
            return result;
        }

        const DWORD expectedBytes = static_cast<DWORD>(sizeof(UDPPacketHeaderV2) + payloadSize);
        if (bytesSent != expectedBytes) {
            return result;
        }
    }

    result.sent = true;
    return result;
}

bool setSocketNonBlocking(SOCKET sock) {
    u_long nonBlocking = 1;
    return ioctlsocket(sock, FIONBIO, &nonBlocking) == 0;
}

bool readFrameCredit(SOCKET sock, std::atomic<int>& frameCredits,
                     std::atomic<uint32_t>& minCreditFrameId,
                     StreamStats& stats, LatestFrameSlot& latestFrame) {
    UDPCreditPacket packet{};
    sockaddr_in fromAddr{};
    int fromLen = sizeof(fromAddr);
    const int ret = recvfrom(sock, reinterpret_cast<char*>(&packet), sizeof(packet), 0,
                             reinterpret_cast<SOCKADDR*>(&fromAddr), &fromLen);
    if (ret == SOCKET_ERROR) {
        return false;
    }
    if (ret != static_cast<int>(sizeof(packet)) ||
        packet.magic != UDP_CREDIT_MAGIC ||
        packet.size != sizeof(UDPCreditPacket) ||
        packet.credits == 0) {
        return false;
    }

    minCreditFrameId.store(packet.minFrameId, std::memory_order_release);
    frameCredits.store(1, std::memory_order_release);
    stats.creditPackets.fetch_add(1, std::memory_order_relaxed);
    latestFrame.cv.notify_one();
    return true;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --ip <addr>       Inference PC IP (default: from config.ini)\n"
              << "  --port <port>     Send port (default: from config.ini)\n"
              << "  --bind-ip <addr>  Local NIC IP for sender socket bind (optional)\n"
              << "  --region <x,y,w,h> Capture region (default: from config.ini)\n"
              << "  --offset <x,y>    Capture center offset from selected output center\n"
              << "  --output <idx>    Capture monitor index (default: from config.ini)\n"
              << "  --fps <num>       Target FPS (default: from config.ini)\n"
              << "  --payload-bytes <n> UDP payload bytes per packet, 512..60000\n"
              << "  --capture-timeout-ms <n> Desktop Duplication wait, 0..100\n"
              << "  --credit-timeout-ms <n> Recovery send timeout if credit packet is lost\n"
              << "  --gui             Show startup config GUI\n"
              << "  --no-gui          Skip startup config GUI\n"
              << "\nConfig file: config.ini\n";
}

enum : int {
    IDC_MONITOR_COMBO = 1001,
    IDC_FPS_EDIT = 1002
};

struct ConfigDialogState {
    Config workingConfig;
    std::vector<OutputInfo> outputs;
    bool accepted = false;
    HWND monitorCombo = nullptr;
    HWND fpsEdit = nullptr;
};

void setControlFont(HWND control, HFONT font) {
    if (!control || !font) return;
    SendMessage(control, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
}

LRESULT CALLBACK configWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<ConfigDialogState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

    switch (msg) {
    case WM_NCCREATE: {
        auto* cs = reinterpret_cast<CREATESTRUCTA*>(lParam);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        return TRUE;
    }
    case WM_CREATE: {
        if (!state) return -1;

        HFONT font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
        const int margin = 16;
        const int controlWidth = 470;

        HWND monitorLabel = CreateWindowExA(
            0, "STATIC", "Capture Monitor",
            WS_CHILD | WS_VISIBLE,
            margin, 18, 140, 20,
            hwnd, nullptr, nullptr, nullptr
        );
        setControlFont(monitorLabel, font);

        state->monitorCombo = CreateWindowExA(
            0, "COMBOBOX", "",
            WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWNLIST | WS_VSCROLL,
            margin, 40, controlWidth, 260,
            hwnd, reinterpret_cast<HMENU>(IDC_MONITOR_COMBO), nullptr, nullptr
        );
        setControlFont(state->monitorCombo, font);

        for (const auto& output : state->outputs) {
            SendMessageA(state->monitorCombo, CB_ADDSTRING, 0,
                         reinterpret_cast<LPARAM>(output.label.c_str()));
        }

        int selectedIndex = 0;
        for (size_t i = 0; i < state->outputs.size(); ++i) {
            if (state->outputs[i].index == state->workingConfig.outputIndex) {
                selectedIndex = static_cast<int>(i);
                break;
            }
        }
        SendMessageA(state->monitorCombo, CB_SETCURSEL, static_cast<WPARAM>(selectedIndex), 0);

        HWND fpsLabel = CreateWindowExA(
            0, "STATIC", "Target FPS",
            WS_CHILD | WS_VISIBLE,
            margin, 84, 120, 20,
            hwnd, nullptr, nullptr, nullptr
        );
        setControlFont(fpsLabel, font);

        state->fpsEdit = CreateWindowExA(
            WS_EX_CLIENTEDGE, "EDIT", "",
            WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL,
            margin, 106, 120, 24,
            hwnd, reinterpret_cast<HMENU>(IDC_FPS_EDIT), nullptr, nullptr
        );
        setControlFont(state->fpsEdit, font);

        char fpsText[32];
        std::snprintf(fpsText, sizeof(fpsText), "%d", state->workingConfig.targetFPS);
        SetWindowTextA(state->fpsEdit, fpsText);

        HWND tip = CreateWindowExA(
            0, "STATIC",
            "Tip: Capture FPS cannot exceed selected monitor refresh rate.",
            WS_CHILD | WS_VISIBLE,
            margin, 138, 430, 20,
            hwnd, nullptr, nullptr, nullptr
        );
        setControlFont(tip, font);

        HWND startButton = CreateWindowExA(
            0, "BUTTON", "Start",
            WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON,
            320, 172, 80, 28,
            hwnd, reinterpret_cast<HMENU>(IDOK), nullptr, nullptr
        );
        setControlFont(startButton, font);

        HWND cancelButton = CreateWindowExA(
            0, "BUTTON", "Cancel",
            WS_CHILD | WS_VISIBLE | WS_TABSTOP,
            406, 172, 80, 28,
            hwnd, reinterpret_cast<HMENU>(IDCANCEL), nullptr, nullptr
        );
        setControlFont(cancelButton, font);

        return 0;
    }
    case WM_COMMAND: {
        if (!state) return 0;
        const int commandId = LOWORD(wParam);
        if (commandId == IDOK) {
            int selected = static_cast<int>(SendMessageA(state->monitorCombo, CB_GETCURSEL, 0, 0));
            if (selected < 0 || selected >= static_cast<int>(state->outputs.size())) {
                MessageBoxA(hwnd, "Select a monitor.", "Invalid selection", MB_OK | MB_ICONWARNING);
                return 0;
            }

            char fpsText[32] = {};
            GetWindowTextA(state->fpsEdit, fpsText, static_cast<int>(sizeof(fpsText)));
            char* endPtr = nullptr;
            long parsedFps = std::strtol(fpsText, &endPtr, 10);
            if (endPtr == fpsText || *endPtr != '\0' || parsedFps < 1 || parsedFps > 1000) {
                MessageBoxA(hwnd, "FPS must be a number between 1 and 1000.", "Invalid FPS",
                            MB_OK | MB_ICONWARNING);
                return 0;
            }

            state->workingConfig.outputIndex = state->outputs[selected].index;
            state->workingConfig.targetFPS = static_cast<int>(parsedFps);
            state->accepted = true;
            DestroyWindow(hwnd);
            return 0;
        }
        if (commandId == IDCANCEL) {
            DestroyWindow(hwnd);
            return 0;
        }
        break;
    }
    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        break;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

bool showStartupConfigDialog(Config& config, const std::vector<OutputInfo>& outputs) {
    if (outputs.empty()) {
        MessageBoxA(nullptr, "No capture outputs were found.", "GamePC Streamer",
                    MB_OK | MB_ICONERROR);
        return false;
    }

    const char* windowClassName = "GamePCStreamerConfigWindow";
    HINSTANCE hInstance = GetModuleHandle(nullptr);

    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = configWindowProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    wc.lpszClassName = windowClassName;

    RegisterClassExA(&wc);

    ConfigDialogState state{};
    state.workingConfig = config;
    state.outputs = outputs;

    HWND hwnd = CreateWindowExA(
        WS_EX_DLGMODALFRAME,
        windowClassName,
        "GamePC Streamer Settings",
        WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 520, 250,
        nullptr, nullptr, hInstance, &state
    );
    if (!hwnd) {
        return false;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        if (!IsDialogMessage(hwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    if (state.accepted) {
        config = state.workingConfig;
        return true;
    }
    return false;
}

// Simple INI parser for config.ini
bool parseBool(const std::string& text) {
    const std::string lower = toLowerCopy(text);
    return (lower == "1" || lower == "true" || lower == "yes" || lower == "on");
}

bool loadConfig(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Config file '" << filename << "' not found, using defaults\n";
        return false;
    }

    std::string line;
    bool captureOffsetKeySeen = false;
    bool useCaptureOffsetKeySeen = false;
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
        } else if (key == "LocalBindIP") {
            g_config.localBindIP = value;
        } else if (key == "CaptureX") {
            g_config.captureX = std::stoi(value);
        } else if (key == "CaptureY") {
            g_config.captureY = std::stoi(value);
        } else if (key == "CaptureWidth") {
            g_config.captureWidth = std::stoi(value);
        } else if (key == "CaptureHeight") {
            g_config.captureHeight = std::stoi(value);
        } else if (key == "UseCaptureOffset") {
            g_config.useCaptureOffset = parseBool(value);
            useCaptureOffsetKeySeen = true;
        } else if (key == "CaptureOffsetX") {
            g_config.captureOffsetX = std::stoi(value);
            captureOffsetKeySeen = true;
        } else if (key == "CaptureOffsetY") {
            g_config.captureOffsetY = std::stoi(value);
            captureOffsetKeySeen = true;
        } else if (key == "OutputIndex") {
            g_config.outputIndex = std::max(0, std::stoi(value));
        } else if (key == "TargetFPS") {
            g_config.targetFPS = std::stoi(value);
        } else if (key == "UseGUI") {
            g_config.useGUI = parseBool(value);
        } else if (key == "PacketPayloadBytes") {
            g_config.packetPayloadBytes = clampPacketPayloadBytes(std::stoi(value));
        } else if (key == "CaptureTimeoutMs") {
            g_config.captureTimeoutMs = std::clamp(std::stoi(value), 0, 100);
        } else if (key == "CreditTimeoutMs") {
            g_config.creditTimeoutMs = std::clamp(std::stoi(value), 5, 1000);
        }
    }

    if (captureOffsetKeySeen && !useCaptureOffsetKeySeen) {
        g_config.useCaptureOffset = true;
    }

    file.close();
    std::cout << "Loaded config from '" << filename << "'\n";
    return true;
}

bool saveConfig(const char* filename) {
    std::ofstream file(filename, std::ios::trunc);
    if (!file.is_open()) {
        return false;
    }

    file << "[Network]\n";
    file << "InferenceIP=" << g_config.inferenceIP << "\n";
    file << "SendPort=" << g_config.sendPort << "\n";
    file << "LocalBindIP=" << g_config.localBindIP << "\n\n";

    file << "[Capture]\n";
    file << "CaptureX=" << g_config.captureX << "\n";
    file << "CaptureY=" << g_config.captureY << "\n";
    file << "CaptureWidth=" << g_config.captureWidth << "\n";
    file << "CaptureHeight=" << g_config.captureHeight << "\n";
    file << "UseCaptureOffset=" << (g_config.useCaptureOffset ? 1 : 0) << "\n";
    file << "CaptureOffsetX=" << g_config.captureOffsetX << "\n";
    file << "CaptureOffsetY=" << g_config.captureOffsetY << "\n";
    file << "OutputIndex=" << g_config.outputIndex << "\n\n";

    file << "[Performance]\n";
    file << "TargetFPS=" << g_config.targetFPS << "\n";
    file << "UseGUI=" << (g_config.useGUI ? 1 : 0) << "\n";
    file << "PacketPayloadBytes=" << g_config.packetPayloadBytes << "\n";
    file << "CaptureTimeoutMs=" << g_config.captureTimeoutMs << "\n";
    file << "CreditTimeoutMs=" << g_config.creditTimeoutMs << "\n";

    return true;
}

bool parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--ip" && i + 1 < argc) {
            g_config.inferenceIP = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            g_config.sendPort = (unsigned short)std::stoi(argv[++i]);
        } else if (arg == "--bind-ip" && i + 1 < argc) {
            g_config.localBindIP = argv[++i];
        } else if (arg == "--region" && i + 1 < argc) {
            std::string region = argv[++i];
            sscanf(region.c_str(), "%d,%d,%d,%d",
                   &g_config.captureX, &g_config.captureY,
                   &g_config.captureWidth, &g_config.captureHeight);
            g_config.useCaptureOffset = false;
        } else if (arg == "--offset" && i + 1 < argc) {
            std::string offset = argv[++i];
            sscanf(offset.c_str(), "%d,%d", &g_config.captureOffsetX, &g_config.captureOffsetY);
            g_config.useCaptureOffset = true;
        } else if (arg == "--output" && i + 1 < argc) {
            g_config.outputIndex = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--fps" && i + 1 < argc) {
            g_config.targetFPS = std::stoi(argv[++i]);
        } else if (arg == "--payload-bytes" && i + 1 < argc) {
            g_config.packetPayloadBytes = clampPacketPayloadBytes(std::stoi(argv[++i]));
        } else if (arg == "--capture-timeout-ms" && i + 1 < argc) {
            g_config.captureTimeoutMs = std::clamp(std::stoi(argv[++i]), 0, 100);
        } else if (arg == "--credit-timeout-ms" && i + 1 < argc) {
            g_config.creditTimeoutMs = std::clamp(std::stoi(argv[++i]), 5, 1000);
        } else if (arg == "--gui") {
            g_config.useGUI = true;
        } else if (arg == "--no-gui") {
            g_config.useGUI = false;
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
    g_config.packetPayloadBytes = clampPacketPayloadBytes(g_config.packetPayloadBytes);

    auto availableOutputs = enumerateCaptureOutputs();
    if (availableOutputs.empty()) {
        std::cerr << "No capture outputs found\n";
        return 1;
    }

    if (g_config.useGUI) {
        if (!showStartupConfigDialog(g_config, availableOutputs)) {
            std::cout << "Startup canceled by user\n";
            return 0;
        }
        if (!saveConfig(configPath.c_str())) {
            std::cerr << "Warning: failed to save config to '" << configPath << "'\n";
        }
    }

    const OutputInfo* selectedOutput = nullptr;
    for (const auto& output : availableOutputs) {
        if (output.index == g_config.outputIndex) {
            selectedOutput = &output;
            break;
        }
    }
    if (!selectedOutput) {
        selectedOutput = &availableOutputs.front();
        g_config.outputIndex = selectedOutput->index;
        std::cout << "Invalid output index. Falling back to output " << g_config.outputIndex << "\n";
    }

    if (selectedOutput->refreshHz > 0 && g_config.targetFPS > selectedOutput->refreshHz) {
        std::cout << "Warning: target FPS (" << g_config.targetFPS << ") exceeds monitor refresh rate ("
                  << selectedOutput->refreshHz << "Hz)\n";
    }
    if (g_config.targetFPS <= 0) {
        std::cerr << "TargetFPS must be positive\n";
        return 1;
    }
    if (g_config.captureWidth <= 0 || g_config.captureHeight <= 0 ||
        g_config.captureWidth > UINT16_MAX || g_config.captureHeight > UINT16_MAX) {
        std::cerr << "CaptureWidth/CaptureHeight must be in range 1.." << UINT16_MAX << "\n";
        return 1;
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
    if (!capture.Initialize(g_config.outputIndex)) {
        std::cerr << "Failed to initialize screen capture\n";
        WSACleanup();
        return 1;
    }

    std::cout << "Output: " << selectedOutput->label << "\n";
    std::cout << "Screen: " << capture.GetScreenWidth() << "x" << capture.GetScreenHeight() << "\n";

    if (g_config.captureWidth > capture.GetScreenWidth() ||
        g_config.captureHeight > capture.GetScreenHeight()) {
        std::cerr << "Capture region is larger than selected output\n";
        WSACleanup();
        return 1;
    }

    const int outputCenterX = capture.GetScreenWidth() / 2;
    const int outputCenterY = capture.GetScreenHeight() / 2;
    if (g_config.useCaptureOffset) {
        g_config.captureX = outputCenterX + g_config.captureOffsetX - g_config.captureWidth / 2;
        g_config.captureY = outputCenterY + g_config.captureOffsetY - g_config.captureHeight / 2;
    } else if (g_config.captureX == 0 && g_config.captureY == 0) {
        // Set default capture region to center of screen.
        g_config.captureX = (capture.GetScreenWidth() - g_config.captureWidth) / 2;
        g_config.captureY = (capture.GetScreenHeight() - g_config.captureHeight) / 2;
    }

    const int maxCaptureX = capture.GetScreenWidth() - g_config.captureWidth;
    const int maxCaptureY = capture.GetScreenHeight() - g_config.captureHeight;
    const int requestedCaptureX = g_config.captureX;
    const int requestedCaptureY = g_config.captureY;
    g_config.captureX = std::clamp(g_config.captureX, 0, maxCaptureX);
    g_config.captureY = std::clamp(g_config.captureY, 0, maxCaptureY);
    if (g_config.captureX != requestedCaptureX || g_config.captureY != requestedCaptureY) {
        std::cout << "Capture region clamped from " << requestedCaptureX << ","
                  << requestedCaptureY << " to " << g_config.captureX << ","
                  << g_config.captureY << "\n";
    }

    std::cout << "Capture region: " << g_config.captureX << "," << g_config.captureY
              << " " << g_config.captureWidth << "x" << g_config.captureHeight << "\n";
    std::cout << "Capture center: "
              << (g_config.captureX + g_config.captureWidth / 2) << ","
              << (g_config.captureY + g_config.captureHeight / 2)
              << " (output center: " << outputCenterX << "," << outputCenterY << ")\n";
    if (g_config.useCaptureOffset) {
        std::cout << "Capture offset: " << g_config.captureOffsetX << ","
                  << g_config.captureOffsetY << " (from selected output center)\n";
    } else {
        std::cout << "Capture mode: absolute top-left (CaptureX/Y)\n";
    }

    constexpr uint8_t wireBytesPerPixelValue = 3;
    constexpr uint8_t wirePixelFormatValue = UDP_PIXEL_FORMAT_RGB;
    const size_t maxPayloadPerPacket = static_cast<size_t>(
        clampPacketPayloadBytes(g_config.packetPayloadBytes));

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
    if (!setSocketNonBlocking(sendSock)) {
        std::cerr << "Warning: failed to set UDP sender socket to non-blocking mode\n";
    }

    if (!g_config.localBindIP.empty()) {
        sockaddr_in localAddr{};
        localAddr.sin_family = AF_INET;
        localAddr.sin_port = htons(0);  // Any ephemeral source port
        if (inet_pton(AF_INET, g_config.localBindIP.c_str(), &localAddr.sin_addr) != 1) {
            std::cerr << "Invalid LocalBindIP: " << g_config.localBindIP << "\n";
            closesocket(sendSock);
            WSACleanup();
            return 1;
        }
        if (bind(sendSock, (SOCKADDR*)&localAddr, sizeof(localAddr)) == SOCKET_ERROR) {
            int err = WSAGetLastError();
            std::cerr << "Failed to bind sender socket to " << g_config.localBindIP
                      << " (WSA error: " << err << ")\n";
            closesocket(sendSock);
            WSACleanup();
            return 1;
        }
    }

    sockaddr_in destAddr = {};
    destAddr.sin_family = AF_INET;
    destAddr.sin_port = htons(g_config.sendPort);
    if (inet_pton(AF_INET, g_config.inferenceIP.c_str(), &destAddr.sin_addr) != 1) {
        std::cerr << "Invalid InferenceIP: " << g_config.inferenceIP << "\n";
        closesocket(sendSock);
        WSACleanup();
        return 1;
    }

    sockaddr_in boundAddr{};
    int boundAddrLen = sizeof(boundAddr);
    std::string boundIpText = "auto";
    if (getsockname(sendSock, (SOCKADDR*)&boundAddr, &boundAddrLen) == 0) {
        char ipBuf[INET_ADDRSTRLEN] = {};
        if (inet_ntop(AF_INET, &boundAddr.sin_addr, ipBuf, sizeof(ipBuf))) {
            boundIpText = ipBuf;
        }
    }

    std::cout << "GamePC Streamer (RGB, " << maxPayloadPerPacket
              << "B payload, UDP V2) started\n";
    std::cout << "Sending to: " << g_config.inferenceIP << ":" << g_config.sendPort << "\n";
    std::cout << "Local bind IP: " << boundIpText << "\n";
    std::cout << "Target FPS: " << g_config.targetFPS << "\n";
    std::cout << "Capture timeout: " << g_config.captureTimeoutMs << "ms\n";
    std::cout << "Press Ctrl+C to exit\n\n";

    bool highResTimerEnabled = (timeBeginPeriod(1) == TIMERR_NOERROR);
    if (!highResTimerEnabled) {
        std::cerr << "Warning: failed to enable 1ms timer resolution\n";
    }
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    LatestFrameSlot latestFrame;
    StreamStats stats;
    std::atomic<int> frameCredits{1};  // Bootstrap one frame so inference can learn our UDP source.
    std::atomic<uint32_t> minCreditFrameId{0};
    // RGB-only, scatter/gather UDP send, and SIMD conversion cut bandwidth/copy
    // cost. Local 256x256 tests favored 60000B payload for average/p95 latency,
    // but real LAN loss/MTU behavior should still be checked with payload sweep.
    const auto creditRecoveryTimeout =
        std::chrono::milliseconds(std::clamp(g_config.creditTimeoutMs, 5, 1000));
    auto statsStart = std::chrono::steady_clock::now();

    std::thread creditThread([&]() {
        applyWindowsRealtimeHint(L"udp-credit", 1);
        while (g_running.load(std::memory_order_relaxed)) {
            bool gotAnyCredit = false;
            while (readFrameCredit(sendSock, frameCredits, minCreditFrameId, stats, latestFrame)) {
                gotAnyCredit = true;
            }
            if (!gotAnyCredit) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });

    std::thread senderThread([&]() {
        applyWindowsRealtimeHint(L"udp-send", 0);
        std::vector<uint8_t> bgraLocal;
        std::vector<uint8_t> wireData;
        uint32_t frameId = 0;
        uint64_t lastSentSequence = 0;
        auto lastSendAttempt = std::chrono::steady_clock::now() - creditRecoveryTimeout;

        while (g_running.load(std::memory_order_relaxed)) {
            uint16_t sendWidth = 0;
            uint16_t sendHeight = 0;
            bool recoverySend = false;

            {
                std::unique_lock<std::mutex> lock(latestFrame.mutex);
                latestFrame.cv.wait_for(lock, std::chrono::milliseconds(1), [&]() {
                    const bool hasNewFrame =
                        latestFrame.hasFrame && latestFrame.sequence != lastSentSequence;
                    if (!hasNewFrame) return latestFrame.stopped;
                    if (frameCredits.load(std::memory_order_acquire) > 0) return true;
                    return (std::chrono::steady_clock::now() - lastSendAttempt) >= creditRecoveryTimeout;
                });

                if (latestFrame.stopped && !latestFrame.hasFrame) {
                    break;
                }
                if (!latestFrame.hasFrame || latestFrame.sequence == lastSentSequence) {
                    continue;
                }

                const bool hasCredit = frameCredits.load(std::memory_order_acquire) > 0;
                // One credit normally permits one latest frame. Recovery sends
                // prevent a single lost credit packet from stalling the stream.
                recoverySend =
                    !hasCredit &&
                    (std::chrono::steady_clock::now() - lastSendAttempt) >= creditRecoveryTimeout;
                if (!hasCredit && !recoverySend) {
                    continue;
                }
                if (hasCredit) {
                    frameCredits.store(0, std::memory_order_release);
                } else {
                    stats.recoverySends.fetch_add(1, std::memory_order_relaxed);
                }

                sendWidth = latestFrame.width;
                sendHeight = latestFrame.height;
                lastSentSequence = latestFrame.sequence;
                bgraLocal.swap(latestFrame.bgra);
                latestFrame.hasFrame = false;
            }

            if (bgraLocal.empty() || sendWidth == 0 || sendHeight == 0) {
                continue;
            }

            const auto convertStart = std::chrono::high_resolution_clock::now();
            convertBgraToRgb(bgraLocal, wireData, sendWidth, sendHeight);
            const auto convertEnd = std::chrono::high_resolution_clock::now();
            stats.totalConvertUs.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(convertEnd - convertStart).count(),
                std::memory_order_relaxed);

            const uint8_t* sendData = wireData.data();
            const size_t frameSize = wireData.size();
            if (!sendData || frameSize == 0 || frameSize > UINT32_MAX) {
                continue;
            }
            stats.outputFrames.fetch_add(1, std::memory_order_relaxed);
            const uint32_t requestedMinFrameId = minCreditFrameId.load(std::memory_order_acquire);
            if (frameId < requestedMinFrameId) {
                frameId = requestedMinFrameId;
            }

            const auto sendStart = std::chrono::high_resolution_clock::now();
            const FrameSendResult sendResult = sendFrameUdp(
                sendSock,
                destAddr,
                sendData,
                frameSize,
                frameId,
                sendWidth,
                sendHeight,
                wirePixelFormatValue,
                wireBytesPerPixelValue,
                maxPayloadPerPacket);
            const auto sendEnd = std::chrono::high_resolution_clock::now();

            stats.totalSendUs.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(sendEnd - sendStart).count(),
                std::memory_order_relaxed);
            stats.totalPackets.fetch_add(sendResult.totalPackets, std::memory_order_relaxed);

            frameId++;
            lastSendAttempt = std::chrono::steady_clock::now();
            if (sendResult.wouldBlock) {
                stats.wouldBlockDrops.fetch_add(1, std::memory_order_relaxed);
            }
            if (!sendResult.sent) {
                stats.droppedFrames.fetch_add(1, std::memory_order_relaxed);
            } else {
                stats.sentFrames.fetch_add(1, std::memory_order_relaxed);
                stats.totalBytes.fetch_add(frameSize, std::memory_order_relaxed);
            }
        }
    });

    std::vector<uint8_t> frameData;

    // FPS limiting
    const auto frameInterval = std::chrono::microseconds((int64_t)(1000000.0 / g_config.targetFPS));
    auto nextFrameTime = std::chrono::steady_clock::now();

    const UINT captureTimeoutMs =
        static_cast<UINT>(std::clamp(g_config.captureTimeoutMs, 0, 100));

    auto printStatsIfDue = [&](std::chrono::steady_clock::time_point statsNow) {
        const double statsSeconds = std::chrono::duration<double>(statsNow - statsStart).count();
        if (statsSeconds < 1.0) {
            return;
        }

        const uint64_t captureAttempts = stats.captureAttempts.exchange(0, std::memory_order_relaxed);
        const uint64_t capturedFrames = stats.capturedFrames.exchange(0, std::memory_order_relaxed);
        const uint64_t skippedFrames = stats.skippedFrames.exchange(0, std::memory_order_relaxed);
        const uint64_t captureTimeouts = stats.captureTimeouts.exchange(0, std::memory_order_relaxed);
        const uint64_t captureAccessLost = stats.captureAccessLost.exchange(0, std::memory_order_relaxed);
        const uint64_t captureFailures = stats.captureFailures.exchange(0, std::memory_order_relaxed);
        const uint64_t outputFrames = stats.outputFrames.exchange(0, std::memory_order_relaxed);
        const uint64_t sentFrames = stats.sentFrames.exchange(0, std::memory_order_relaxed);
        const uint64_t droppedFrames = stats.droppedFrames.exchange(0, std::memory_order_relaxed);
        const uint64_t wouldBlockDrops = stats.wouldBlockDrops.exchange(0, std::memory_order_relaxed);
        const uint64_t creditPackets = stats.creditPackets.exchange(0, std::memory_order_relaxed);
        const uint64_t recoverySends = stats.recoverySends.exchange(0, std::memory_order_relaxed);
        const uint64_t totalBytes = stats.totalBytes.exchange(0, std::memory_order_relaxed);
        const uint64_t totalPackets = stats.totalPackets.exchange(0, std::memory_order_relaxed);
        const int64_t totalCaptureUs = stats.totalCaptureUs.exchange(0, std::memory_order_relaxed);
        const int64_t totalConvertUs = stats.totalConvertUs.exchange(0, std::memory_order_relaxed);
        const int64_t totalSendUs = stats.totalSendUs.exchange(0, std::memory_order_relaxed);

        double capFps = capturedFrames / statsSeconds;
        double pollFps = captureAttempts / statsSeconds;
        double skipFps = skippedFrames / statsSeconds;
        double outFps = outputFrames / statsSeconds;
        double sendFps = sentFrames / statsSeconds;
        double mbps = (totalBytes * 8.0) / (statsSeconds * 1000000.0);
        double avgCapture = (capturedFrames > 0) ? (static_cast<double>(totalCaptureUs) / capturedFrames / 1000.0) : 0.0;
        double avgConvert = (outputFrames > 0) ? (static_cast<double>(totalConvertUs) / outputFrames / 1000.0) : 0.0;
        double avgSend = (outputFrames > 0) ? (static_cast<double>(totalSendUs) / outputFrames / 1000.0) : 0.0;
        double dropPct = (outputFrames > 0) ? (droppedFrames * 100.0) / outputFrames : 0.0;
        double packetsPerFrame = (sentFrames > 0) ? (static_cast<double>(totalPackets) / sentFrames) : 0.0;

        std::ostringstream line;
        line << "PollFPS: " << std::fixed << std::setprecision(1) << pollFps
             << " | NewCapFPS: " << capFps
             << " | SkipFPS: " << skipFps
             << " | CT:" << captureTimeouts
             << " AL:" << captureAccessLost
             << " CF:" << captureFailures
             << " | OutFPS: " << outFps
             << " | SendFPS: " << sendFps
             << " | Cap:" << std::setprecision(2) << avgCapture << "ms"
             << " Cvt:" << avgConvert << "ms"
             << " Snd:" << avgSend << "ms"
             << " | " << mbps << " Mbps"
             << " | RGB/" << maxPayloadPerPacket << "B"
             << " | " << std::setprecision(1) << packetsPerFrame << " pkts/frame"
             << " | Drop:" << std::setprecision(1) << dropPct << "%"
             << " (WB:" << wouldBlockDrops << ")"
             << " | Credit:" << creditPackets
             << " Rec:" << recoverySends;
        const uint32_t lastHr = stats.lastCaptureHr.load(std::memory_order_relaxed);
        if (lastHr != 0) {
            line << " HR:0x" << std::hex << std::uppercase << lastHr << std::dec;
        }
        printStatusLine(line.str());
        statsStart = statsNow;
    };

    applyWindowsRealtimeHint(L"capture-main", 2);
    while (g_running.load()) {
        // FPS limiting - keep cadence based on accumulated frame intervals.
        nextFrameTime += frameInterval;
        auto now = std::chrono::steady_clock::now();
        if (now < nextFrameTime) {
            std::this_thread::sleep_until(nextFrameTime);
        } else if (now - nextFrameTime > frameInterval * 2) {
            nextFrameTime = now;
        }

        // Capture a new desktop frame if available.
        auto t1 = std::chrono::high_resolution_clock::now();
        CaptureResult captureResult{};
        bool gotNewFrame = capture.CaptureFrame(frameData, g_config.captureX, g_config.captureY,
                                                g_config.captureWidth, g_config.captureHeight,
                                                captureTimeoutMs, &captureResult);
        auto t2 = std::chrono::high_resolution_clock::now();
        stats.captureAttempts.fetch_add(1, std::memory_order_relaxed);

        if (gotNewFrame) {
            stats.capturedFrames.fetch_add(1, std::memory_order_relaxed);
            stats.totalCaptureUs.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(),
                std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(latestFrame.mutex);
                latestFrame.bgra.swap(frameData);
                latestFrame.width = static_cast<uint16_t>(g_config.captureWidth);
                latestFrame.height = static_cast<uint16_t>(g_config.captureHeight);
                latestFrame.hasFrame = true;
                ++latestFrame.sequence;
            }
            latestFrame.cv.notify_one();
        } else {
            stats.skippedFrames.fetch_add(1, std::memory_order_relaxed);
            stats.lastCaptureHr.store(static_cast<uint32_t>(captureResult.hr), std::memory_order_relaxed);
            switch (captureResult.status) {
                case CaptureStatus::Timeout:
                    stats.captureTimeouts.fetch_add(1, std::memory_order_relaxed);
                    break;
                case CaptureStatus::AccessLost:
                    stats.captureAccessLost.fetch_add(1, std::memory_order_relaxed);
                    break;
                case CaptureStatus::Failed:
                    stats.captureFailures.fetch_add(1, std::memory_order_relaxed);
                    break;
                case CaptureStatus::Captured:
                    break;
            }
        }

        printStatsIfDue(std::chrono::steady_clock::now());
    }

    std::cout << "\nShutting down...\n";
    {
        std::lock_guard<std::mutex> lock(latestFrame.mutex);
        latestFrame.stopped = true;
    }
    latestFrame.cv.notify_all();
    if (senderThread.joinable()) {
        senderThread.join();
    }
    if (creditThread.joinable()) {
        creditThread.join();
    }
    if (highResTimerEnabled) {
        timeEndPeriod(1);
    }
    closesocket(sendSock);
    WSACleanup();

    return 0;
}
