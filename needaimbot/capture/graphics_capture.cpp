#include "graphics_capture.h"
#include "nvfbc_capture.h"
#include <iostream>

GraphicsCapture::GraphicsCapture() = default;

GraphicsCapture::~GraphicsCapture() {
    Shutdown();
}

bool GraphicsCapture::Initialize(HWND targetWindow) {
    // Only use NVFBC - no fallback modes
    if (!IsNVFBCAvailable()) {
        std::cerr << "[GraphicsCapture] NVFBC is not available" << std::endl;
        return false;
    }

    m_nvfbcCapture = std::make_unique<NVFBCCapture>();
    if (!m_nvfbcCapture->Initialize(targetWindow)) {
        std::cerr << "[GraphicsCapture] Failed to initialize NVFBC" << std::endl;
        m_nvfbcCapture.reset();
        return false;
    }

    m_captureWidth = m_nvfbcCapture->GetWidth();
    m_captureHeight = m_nvfbcCapture->GetHeight();

    // Set up frame callback forwarding
    m_nvfbcCapture->SetFrameCallback([this](void* data, unsigned int width, unsigned int height, unsigned int size) {
        if (m_frameCallback) {
            m_frameCallback(data, width, height, size);
        }
    });

    m_isInitialized = true;
    std::cout << "[GraphicsCapture] Initialized with NVFBC (" << m_captureWidth << "x" << m_captureHeight << ")" << std::endl;
    return true;
}

void GraphicsCapture::Shutdown() {
    StopCapture();

    if (m_nvfbcCapture) {
        m_nvfbcCapture->Shutdown();
        m_nvfbcCapture.reset();
    }

    m_isInitialized = false;
    std::cout << "[GraphicsCapture] Shutdown complete" << std::endl;
}

bool GraphicsCapture::StartCapture() {
    if (!m_isInitialized || !m_nvfbcCapture) {
        std::cerr << "[GraphicsCapture] Not initialized" << std::endl;
        return false;
    }

    if (m_nvfbcCapture->StartCapture()) {
        m_isCapturing = true;
        std::cout << "[GraphicsCapture] NVFBC capture started" << std::endl;
        return true;
    } else {
        std::cerr << "[GraphicsCapture] Failed to start NVFBC capture" << std::endl;
        return false;
    }
}

void GraphicsCapture::StopCapture() {
    if (m_isCapturing && m_nvfbcCapture) {
        m_nvfbcCapture->StopCapture();
        m_isCapturing = false;
        std::cout << "[GraphicsCapture] Capture stopped" << std::endl;
    }
}

bool GraphicsCapture::GetLatestFrame(void** frameData, unsigned int* width, unsigned int* height, unsigned int* size) {
    if (!m_nvfbcCapture) {
        return false;
    }
    return m_nvfbcCapture->GetLatestFrame(frameData, width, height, size);
}

void GraphicsCapture::SetFrameCallback(std::function<void(void*, unsigned int, unsigned int, unsigned int)> callback) {
    m_frameCallback = callback;
}

bool GraphicsCapture::IsNVFBCAvailable() {
    return NVFBCCapture::IsNVFBCAvailable();
}