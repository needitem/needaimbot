// NVIDIA Frame Buffer Capture (NVFBC) Implementation
// Ultra-low latency GPU capture using NVIDIA's hardware encoder

#include <NvFBC/nvFBC.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <iostream>

class NVFBCCapture {
private:
    NVFBC_SESSION_HANDLE m_fbcHandle;
    NVFBC_CREATE_HANDLE_PARAMS m_createParams;
    NVFBC_CREATE_CAPTURE_SESSION_PARAMS m_captureParams;
    
    // CUDA resources
    cudaGraphicsResource_t m_cudaResource;
    void* m_devicePtr;
    
    // Capture dimensions
    int m_width, m_height;
    int m_offsetX, m_offsetY;
    
public:
    NVFBCCapture(int width, int height) 
        : m_width(width), m_height(height), 
          m_fbcHandle(nullptr), m_devicePtr(nullptr) {
        
        // Calculate center offset for 320x320 capture
        m_offsetX = (GetSystemMetrics(SM_CXSCREEN) - width) / 2;
        m_offsetY = (GetSystemMetrics(SM_CYSCREEN) - height) / 2;
    }
    
    bool Initialize() {
        // 1. Load NVFBC library
        NVFBCRESULT result = NVFBC_SUCCESS;
        
        // Create NVFBC instance
        NvFBCCreateInstance_ptr NvFBCCreateInstance = nullptr;
        HMODULE nvfbcLib = LoadLibrary("NvFBC64.dll");
        if (!nvfbcLib) {
            std::cerr << "Failed to load NvFBC64.dll - NVIDIA GPU required" << std::endl;
            return false;
        }
        
        NvFBCCreateInstance = (NvFBCCreateInstance_ptr)GetProcAddress(nvfbcLib, "NvFBCCreateInstance");
        if (!NvFBCCreateInstance) {
            return false;
        }
        
        // 2. Create NVFBC session
        NVFBC_API_FUNCTION_LIST functionList;
        result = NvFBCCreateInstance(&functionList);
        if (result != NVFBC_SUCCESS) {
            std::cerr << "Failed to create NVFBC instance" << std::endl;
            return false;
        }
        
        // 3. Setup capture parameters for SPECIFIC REGION
        NVFBC_CREATE_PARAMS createParams = {};
        createParams.dwVersion = NVFBC_CREATE_PARAMS_VER;
        createParams.dwInterfaceType = NVFBC_TO_CUDA;  // Direct to CUDA!
        createParams.dwMaxDisplayWidth = -1;  // Auto-detect
        createParams.dwMaxDisplayHeight = -1;
        
        result = functionList.nvFBCCreateHandle(&m_fbcHandle, &createParams);
        if (result != NVFBC_SUCCESS) {
            std::cerr << "Failed to create NVFBC handle" << std::endl;
            return false;
        }
        
        // 4. Configure capture session for CROPPED REGION
        NVFBC_TOCUDA_SETUP_PARAMS setupParams = {};
        setupParams.dwVersion = NVFBC_TOCUDA_SETUP_PARAMS_VER;
        setupParams.eBufferFormat = NVFBC_BUFFER_FORMAT_BGRA;
        
        // KEY: Set capture region (not full screen!)
        setupParams.dwOutputWidth = m_width;   // 320
        setupParams.dwOutputHeight = m_height; // 320
        
        result = functionList.nvFBCToCudaSetup(m_fbcHandle, &setupParams);
        if (result != NVFBC_SUCCESS) {
            return false;
        }
        
        // 5. Allocate CUDA memory for direct GPU transfer
        cudaMalloc(&m_devicePtr, m_width * m_height * 4);  // BGRA format
        
        std::cout << "[NVFBC] Initialized for " << m_width << "x" << m_height 
                  << " capture at offset (" << m_offsetX << ", " << m_offsetY << ")" << std::endl;
        
        return true;
    }
    
    bool CaptureFrame(cudaStream_t stream) {
        NVFBC_TOCUDA_GRAB_FRAME_PARAMS grabParams = {};
        grabParams.dwVersion = NVFBC_TOCUDA_GRAB_FRAME_PARAMS_VER;
        
        // CRITICAL: Specify exact region to capture
        grabParams.dwStartX = m_offsetX;
        grabParams.dwStartY = m_offsetY;
        grabParams.dwTargetWidth = m_width;
        grabParams.dwTargetHeight = m_height;
        
        // Hardware cropping flags
        grabParams.dwFlags = NVFBC_TOCUDA_GRAB_FLAGS_CROP |           // Enable cropping
                             NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT |         // Don't wait for vsync
                             NVFBC_TOCUDA_GRAB_FLAGS_FORCE_REFRESH;   // Force new frame
        
        // Output directly to CUDA device memory
        grabParams.pCudaDeviceBuffer = m_devicePtr;
        
        // Capture with hardware acceleration
        NVFBCRESULT result = NvFBCToCudaGrabFrame(m_fbcHandle, &grabParams);
        
        if (result == NVFBC_SUCCESS) {
            // Frame is already in GPU memory, ready for CUDA processing!
            // No CPU involvement at all!
            return true;
        }
        
        return false;
    }
    
    void* GetCudaDevicePointer() const {
        return m_devicePtr;
    }
    
    ~NVFBCCapture() {
        if (m_devicePtr) {
            cudaFree(m_devicePtr);
        }
        if (m_fbcHandle) {
            NvFBCDestroyHandle(m_fbcHandle);
        }
    }
};

// Usage in main capture loop
void RunNVFBCCapture() {
    NVFBCCapture capture(320, 320);  // Only capture 320x320!
    
    if (!capture.Initialize()) {
        std::cerr << "NVFBC not available - need NVIDIA GPU" << std::endl;
        return;
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    while (true) {
        // Capture directly to GPU memory
        if (capture.CaptureFrame(stream)) {
            // Frame is already in CUDA memory at capture.GetCudaDevicePointer()
            // Can be used directly with CUDA kernels - no copy needed!
            
            // Process with your CUDA detection pipeline...
        }
    }
}