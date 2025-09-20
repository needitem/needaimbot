#include "needaimbot/capture/graphics_capture.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "=== NVFBC Test ===" << std::endl;

    // Check if NVFBC is available
    if (GraphicsCapture::IsNVFBCAvailable()) {
        std::cout << "✓ NVFBC is available!" << std::endl;
    } else {
        std::cout << "✗ NVFBC is not available" << std::endl;
    }

    // Test GraphicsCapture initialization
    GraphicsCapture capture;
    if (capture.Initialize()) {
        std::cout << "✓ GraphicsCapture initialized" << std::endl;
        std::cout << "  - Using NVFBC: " << (capture.IsUsingNVFBC() ? "Yes" : "No") << std::endl;
        std::cout << "  - Resolution: " << capture.GetWidth() << "x" << capture.GetHeight() << std::endl;

        // Test capture start/stop
        if (capture.StartCapture()) {
            std::cout << "✓ Capture started" << std::endl;

            // Let it run for a few seconds
            std::this_thread::sleep_for(std::chrono::seconds(3));

            capture.StopCapture();
            std::cout << "✓ Capture stopped" << std::endl;
        } else {
            std::cout << "✗ Failed to start capture" << std::endl;
        }

        capture.Shutdown();
        std::cout << "✓ GraphicsCapture shutdown" << std::endl;
    } else {
        std::cout << "✗ Failed to initialize GraphicsCapture" << std::endl;
    }

    std::cout << "=== Test Complete ===" << std::endl;

    // Keep console open
    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}