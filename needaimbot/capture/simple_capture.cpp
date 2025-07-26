#include "simple_capture.h"
#include "AppContext.h"
#include "../cuda/cuda_image_processing.h"
#include "frame_buffer_pool.h"
#include <iostream>
#include <chrono>
#include <algorithm>  // For std::min
#include <ppl.h>  // For parallel_for

SimpleScreenCapture::SimpleScreenCapture(int width, int height) 
    : m_initialized(false), m_width(width), m_height(height),
      m_screenDC(nullptr), m_memoryDC(nullptr), m_bitmap(nullptr), m_bitmapData(nullptr),
      m_retryCount(0), m_maxRetries(3)
{
    
    // Get screen dimensions
    m_screenWidth = GetSystemMetrics(SM_CXSCREEN);
    m_screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    
    // Get screen DC
    m_screenDC = GetDC(NULL);
    if (!m_screenDC) {
        return;
    }
    
    // Create compatible DC
    m_memoryDC = CreateCompatibleDC(m_screenDC);
    if (!m_memoryDC) {
        ReleaseDC(NULL, m_screenDC);
        return;
    }
    
    // Setup bitmap info
    ZeroMemory(&m_bmpInfo, sizeof(BITMAPINFO));
    m_bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    m_bmpInfo.bmiHeader.biWidth = m_width;
    m_bmpInfo.bmiHeader.biHeight = -m_height; // Negative for top-down bitmap
    m_bmpInfo.bmiHeader.biPlanes = 1;
    m_bmpInfo.bmiHeader.biBitCount = 32; // BGRA
    m_bmpInfo.bmiHeader.biCompression = BI_RGB;
    
    // Create DIB section
    m_bitmap = CreateDIBSection(m_memoryDC, &m_bmpInfo, DIB_RGB_COLORS, 
                               (void**)&m_bitmapData, NULL, 0);
    if (!m_bitmap) {
        DeleteDC(m_memoryDC);
        ReleaseDC(NULL, m_screenDC);
        return;
    }
    
    // Select bitmap into memory DC
    SelectObject(m_memoryDC, m_bitmap);
    
    // Don't use external data wrapper for now - it seems to be causing issues
    // m_hostFrame = SimpleMat(m_height, m_width, 4, m_bitmapData, m_width * 4);
    
    // std::cout << "[SimpleCapture] Host frame created: " << m_hostFrame.cols() << "x" << m_hostFrame.rows() 
    //           << " channels=" << m_hostFrame.channels() << " empty=" << m_hostFrame.empty() << std::endl;
    
    // Pre-allocate GPU memory to avoid allocation during stream capture
    m_deviceFrame.create(m_height, m_width, 4); // BGRA first
    m_tempBgrFrame.create(m_height, m_width, 3); // BGR result
    
    m_initialized = true;
    
}

SimpleScreenCapture::~SimpleScreenCapture()
{
    if (m_bitmap) DeleteObject(m_bitmap);
    if (m_memoryDC) DeleteDC(m_memoryDC);
    if (m_screenDC) ReleaseDC(NULL, m_screenDC);
}

SimpleCudaMat SimpleScreenCapture::GetNextFrameGpu()
{
    // Return CPU frame as GPU frame is problematic during stream capture
    // Let the caller handle GPU upload if needed
    SimpleMat cpuFrame = GetNextFrameCpu();
    if (cpuFrame.empty()) {
        return SimpleCudaMat();
    }
    
    // For now, return empty GpuMat to avoid stream capture conflicts
    // The detector should handle CPU frames or upload manually
    return SimpleCudaMat();
}

SimpleMat SimpleScreenCapture::GetNextFrameCpu()
{
    static int captureAttempts = 0;
    captureAttempts++;
    
    // Only log every 300 frames to reduce spam
    if (captureAttempts % 300 == 1) {
    }
    
    if (!m_initialized) {
        return SimpleMat();
    }
    
    // Initialization check passed (no need to log every frame)
    
    // Calculate center region coordinates
    int startX = (m_screenWidth - m_width) / 2;
    int startY = (m_screenHeight - m_height) / 2;
    
    if (captureAttempts % 60 == 1) {
    }
    
    // 프레임 버퍼 풀에서 버퍼 획듍 (먼저 선언)
    if (!g_frameBufferPool) {
        g_frameBufferPool = std::make_unique<FrameBufferPool>(10);
    }
    SimpleMat bgrFrame = g_frameBufferPool->acquireCpuBuffer(m_height, m_width, 3);
    
    // 에러 복구를 위한 재시도 로직
    BOOL result = FALSE;
    int retries = 0;
    
    while (!result && retries < m_maxRetries) {
        result = BitBlt(m_memoryDC, 0, 0, m_width, m_height, 
                        m_screenDC, startX, startY, SRCCOPY | NOMIRRORBITMAP);
        
        if (!result) {
            DWORD error = GetLastError();
            if (retries < m_maxRetries - 1) {
                // 지수 백오프
                int sleepMs = (1 << retries) * 10; // 10ms, 20ms, 40ms
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
                retries++;
            } else {
                // 최종 실패 시 빈 프레임 반환
                g_frameBufferPool->releaseCpuBuffer(std::move(bgrFrame));
                return SimpleMat();
            }
        }
    }
    
    // Verify bitmap data is valid
    if (!m_bitmapData) {
        return SimpleMat();
    }
    
    // Verify bitmap data periodically
    if (captureAttempts % 300 == 1 && m_bitmapData) {
    }
    
    // 버퍼가 이미 위에서 선언되었음
    
    if (captureAttempts % 60 == 1) {
    }
    
    // Use bitmap data directly
    const uint8_t* srcData = static_cast<const uint8_t*>(m_bitmapData);
    if (!srcData) {
        return SimpleMat();
    }
    
    // 개선된 병렬 색상 변환 - 캐시 효율성 향상
    const int blockHeight = 8; // 캐시 라인 최적화
    concurrency::parallel_for(0, (m_height + blockHeight - 1) / blockHeight, [&](int blockY) {
        int startY = blockY * blockHeight;
        int endY = (std::min)(startY + blockHeight, m_height);
        
        for (int y = startY; y < endY; ++y) {
            const uint8_t* srcRow = srcData + y * (m_width * 4);
            uint8_t* dstRow = bgrFrame.data() + y * bgrFrame.step();
            
            // SIMD-친화적인 루프 구조
            int x = 0;
            for (; x < m_width - 3; x += 4) {
                // 4픽셀을 한 번에 처리
                for (int i = 0; i < 4; ++i) {
                    dstRow[(x + i) * 3 + 0] = srcRow[(x + i) * 4 + 0]; // B
                    dstRow[(x + i) * 3 + 1] = srcRow[(x + i) * 4 + 1]; // G
                    dstRow[(x + i) * 3 + 2] = srcRow[(x + i) * 4 + 2]; // R
                }
            }
            // 나머지 픽셀 처리
            for (; x < m_width; ++x) {
                dstRow[x * 3 + 0] = srcRow[x * 4 + 0];
                dstRow[x * 3 + 1] = srcRow[x * 4 + 1];
                dstRow[x * 3 + 2] = srcRow[x * 4 + 2];
            }
        }
    });
    
    if (captureAttempts % 60 == 1) {
    }
    
    return bgrFrame;
}