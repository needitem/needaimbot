#pragma once

#include "capture_interface.h"
#include "udp_capture.h"
#include <cuda_runtime.h>

// Adapter to make UDPCapture compatible with ICaptureProvider interface
class UDPCaptureAdapter : public ICaptureProvider {
public:
    UDPCaptureAdapter(UDPCapture* udpCapture) : m_udpCapture(udpCapture) {}
    ~UDPCaptureAdapter() override = default;

    bool Initialize(HWND targetWindow = nullptr) override {
        // UDP capture is initialized separately with network settings
        // This is called for compatibility but actual init happens via InitializeNetwork
        return m_udpCapture != nullptr;
    }

    // Actual initialization with network settings
    bool InitializeNetwork(unsigned short listenPort = 5007,
                          const std::string& gamePcIp = "",
                          unsigned short mouseStatePort = 5006) {
        if (!m_udpCapture) return false;
        return m_udpCapture->Initialize(listenPort, gamePcIp, mouseStatePort);
    }

    void Shutdown() override {
        if (m_udpCapture) m_udpCapture->Shutdown();
    }

    bool StartCapture() override {
        if (!m_udpCapture) return false;
        return m_udpCapture->StartCapture();
    }

    void StopCapture() override {
        if (m_udpCapture) m_udpCapture->StopCapture();
    }

    bool IsCapturing() const override {
        return m_udpCapture && m_udpCapture->IsCapturing();
    }

    // UDP capture provides RGB data, not cudaArray_t
    // This method uploads RGB to a temporary CUDA array
    bool AcquireFrameSync(cudaArray_t* cudaArray, unsigned int* width, unsigned int* height,
                         uint64_t* outPresentQpc = nullptr, uint32_t timeoutMs = 8) override {
        if (!m_udpCapture || !cudaArray) return false;

        void* rgbData = nullptr;
        uint64_t frameId = 0;
        if (!m_udpCapture->AcquireFrameSync(&rgbData, width, height, &frameId, timeoutMs)) {
            return false;
        }

        if (outPresentQpc) *outPresentQpc = frameId;

        // Note: cudaArray upload needs to be handled by the pipeline
        // For now, store the RGB pointer for GetLatestFrame
        m_lastRgbData = rgbData;
        m_lastWidth = *width;
        m_lastHeight = *height;
        m_lastSize = (*width) * (*height) * 3;

        return true;
    }

    // Get RGB frame data directly
    bool AcquireFrameSyncRGB(void** rgbData, unsigned int* width, unsigned int* height,
                             uint64_t* outFrameId = nullptr, uint32_t timeoutMs = 16) {
        if (!m_udpCapture) return false;
        return m_udpCapture->AcquireFrameSync(rgbData, width, height, outFrameId, timeoutMs);
    }

    // Upload frame directly to CUDA device memory
    bool AcquireFrameToCuda(void* d_rgbBuffer, size_t bufferSize,
                            unsigned int* width, unsigned int* height,
                            cudaStream_t stream = nullptr, uint32_t timeoutMs = 16) {
        if (!m_udpCapture) return false;
        return m_udpCapture->AcquireFrameToCuda(d_rgbBuffer, bufferSize, width, height, stream, timeoutMs);
    }

    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size) override {
        if (!m_udpCapture) return false;
        return m_udpCapture->GetLatestFrame(frameData, width, height, size);
    }

    bool WaitForNewFrameSince(uint64_t minPresentQpc, uint32_t timeoutMs) override {
        // UDP capture doesn't support this directly
        // Just wait for any new frame
        void* data = nullptr;
        unsigned int w, h;
        uint64_t frameId;
        return m_udpCapture && m_udpCapture->AcquireFrameSync(&data, &w, &h, &frameId, timeoutMs);
    }

    uint64_t GetLastPresentQpc() const override {
        return m_udpCapture ? m_udpCapture->GetLastFrameId() : 0;
    }

    int GetScreenWidth() const override {
        return m_udpCapture ? m_udpCapture->GetScreenWidth() : 0;
    }

    int GetScreenHeight() const override {
        return m_udpCapture ? m_udpCapture->GetScreenHeight() : 0;
    }

    bool SetCaptureRegion(int x, int y, int width, int height) override {
        // Capture region is determined by game_pc
        return false;
    }

    void GetCaptureRegion(int* x, int* y, int* width, int* height) const override {
        if (m_udpCapture) {
            m_udpCapture->GetCaptureRegion(x, y, width, height);
        }
    }

    // Send mouse state back to game PC
    void SendMouseState(bool aimActive, bool shootActive) {
        if (m_udpCapture) {
            m_udpCapture->SendMouseState(aimActive, shootActive);
        }
    }

    // Access underlying UDP capture
    UDPCapture* GetUDPCapture() { return m_udpCapture; }

private:
    UDPCapture* m_udpCapture;
    void* m_lastRgbData = nullptr;
    unsigned int m_lastWidth = 0;
    unsigned int m_lastHeight = 0;
    unsigned int m_lastSize = 0;
};
