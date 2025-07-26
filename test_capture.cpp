#include <iostream>
#include <windows.h>
#include "needaimbot/capture/simple_capture.h"
#include "needaimbot/utils/image_io.h"

int main() {
    std::cout << "Testing SimpleScreenCapture..." << std::endl;
    
    // Test with 320x320 capture
    SimpleScreenCapture capture(320, 320);
    
    if (!capture.IsInitialized()) {
        std::cerr << "Failed to initialize capture!" << std::endl;
        return 1;
    }
    
    std::cout << "Capture initialized successfully" << std::endl;
    
    // Try to capture a frame
    for (int i = 0; i < 5; i++) {
        std::cout << "\nAttempting capture " << (i + 1) << "..." << std::endl;
        
        SimpleMat frame = capture.GetNextFrameCpu();
        
        if (frame.empty()) {
            std::cerr << "Captured frame is empty!" << std::endl;
        } else {
            std::cout << "Captured frame: " << frame.cols() << "x" << frame.rows() 
                      << " channels=" << frame.channels() << std::endl;
            
            // Save first frame
            if (i == 0) {
                std::cout << "Saving test frame..." << std::endl;
                ImageIO::saveImage("test_capture.png", frame);
            }
        }
        
        Sleep(100); // Small delay between captures
    }
    
    std::cout << "\nTest complete!" << std::endl;
    return 0;
}