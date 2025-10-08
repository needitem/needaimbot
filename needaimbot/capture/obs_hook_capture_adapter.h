#pragma once

#include "capture_interface.h"
#include "game_capture.h"
#include <atomic>

// Adapter that exposes GameCapture via ICaptureProvider
class ObsHookCaptureAdapter : public ICaptureProvider {
public:
    explicit ObsHookCaptureAdapter(GameCapture* impl) : m_impl(impl) {}

    bool Initialize(HWND /*targetWindow*/ = nullptr) override {
        // GameCapture is initialized via its own constructor or explicit initialize()
        return m_impl ? m_impl->initialize() : false;
    }
    void Shutdown() override {
        // GameCapture has cleanup in destructor; nothing to do here
    }
    bool StartCapture() override { m_started = true; return true; }
    void StopCapture() override { m_started = false; }
    bool IsCapturing() const override { return m_started.load(); }

    bool GetLatestFrameGPU(cudaArray_t* cudaArray, unsigned int* width, unsigned int* height) override {
        return m_impl ? m_impl->GetLatestFrameGPU(cudaArray, width, height) : false;
    }

    bool GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size) override {
        if (!m_impl) return false;
        Image img = m_impl->get_frame();
        if (!img.data || img.width <= 0 || img.height <= 0) return false;
        if (frameData) *frameData = static_cast<void*>(img.data);
        if (width) *width = static_cast<unsigned int>(img.width);
        if (height) *height = static_cast<unsigned int>(img.height);
        if (size) *size = static_cast<unsigned int>(img.pitch * img.height);
        return true;
    }

    bool WaitForNewFrameSince(uint64_t /*minPresentQpc*/, uint32_t /*timeoutMs*/) override {
        // No presentation timing exposed; just return true to proceed.
        return true;
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
    GameCapture* m_impl;
    std::atomic<bool> m_started{false};
};

