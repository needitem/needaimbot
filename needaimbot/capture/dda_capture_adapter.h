#pragma once

#include "capture_interface.h"
#include "dda_capture.h"

class DDACaptureAdapter : public ICaptureProvider {
public:
    explicit DDACaptureAdapter(DDACapture* impl) : m_impl(impl) {}

    bool Initialize(HWND targetWindow = nullptr) override { return m_impl ? m_impl->Initialize(targetWindow) : false; }
    void Shutdown() override { if (m_impl) m_impl->Shutdown(); }
    bool StartCapture() override { return m_impl ? m_impl->StartCapture() : false; }
    void StopCapture() override { if (m_impl) m_impl->StopCapture(); }
    bool IsCapturing() const override { return m_impl ? m_impl->IsCapturing() : false; }

    bool GetLatestFrameGPU(cudaArray_t* cudaArray, unsigned int* width, unsigned int* height) override {
        return m_impl ? m_impl->GetLatestFrameGPU(cudaArray, width, height) : false;
    }

    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size) override {
        return m_impl ? m_impl->GetLatestFrame(frameData, width, height, size) : false;
    }

    bool WaitForNewFrameSince(uint64_t minPresentQpc, uint32_t timeoutMs) override {
        return m_impl ? m_impl->WaitForNewFrameSince(minPresentQpc, timeoutMs) : true;
    }

    uint64_t GetLastPresentQpc() const override {
        return m_impl ? m_impl->GetLastPresentQpc() : 0;
    }

    int GetScreenWidth() const override { return m_impl ? m_impl->GetScreenWidth() : 0; }
    int GetScreenHeight() const override { return m_impl ? m_impl->GetScreenHeight() : 0; }
    bool SetCaptureRegion(int x, int y, int width, int height) override {
        return m_impl ? m_impl->SetCaptureRegion(x, y, width, height) : false;
    }
    void GetCaptureRegion(int* x, int* y, int* width, int* height) const override {
        if (m_impl) m_impl->GetCaptureRegion(x, y, width, height);
    }

private:
    DDACapture* m_impl;
};
