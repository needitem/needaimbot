#include "sunone_aimbot_cpp.h"
#include "duplication_api_capture.h"
#include "config.h"
#include "other_tools.h"
#include "capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

template <typename T>
inline void SafeRelease(T **ppInterface)
{
    if (*ppInterface)
    {
        (*ppInterface)->Release();
        *ppInterface = nullptr;
    }
}

struct FrameContext
{
    ID3D11Texture2D *texture = nullptr;
    std::vector<DXGI_OUTDUPL_MOVE_RECT> moveRects;
    std::vector<RECT> dirtyRects;
};

class DDAManager
{
public:
    DDAManager()
        : m_device(nullptr), m_context(nullptr), m_duplication(nullptr), m_output1(nullptr), m_sharedTexture(nullptr), m_cudaResource(nullptr), m_cudaStream(nullptr), m_framePool(5), m_captureDoneEvent(nullptr) // Pre-allocate 5 frames in the pool
    {
        ZeroMemory(&m_duplDesc, sizeof(m_duplDesc));
    }

    ~DDAManager()
    {
        Release();
    }

    HRESULT Initialize(
        int monitorIndex,
        int captureWidth,
        int captureHeight,
        int &outScreenWidth,
        int &outScreenHeight,
        ID3D11Device **outDevice = nullptr,
        ID3D11DeviceContext **outContext = nullptr)
    {
        HRESULT hr = S_OK;

        IDXGIFactory1 *factory = nullptr;
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&factory);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] Failed to create DXGIFactory1 (hr = " << std::hex << hr << ")." << std::endl;
            return hr;
        }

        IDXGIAdapter1 *adapter = nullptr;
        hr = factory->EnumAdapters1(monitorIndex, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            std::cerr << "[DDA] Not found adapter with index " << monitorIndex << ". Error code: DXGI_ERROR_NOT_FOUND." << std::endl;
            factory->Release();
            return hr;
        }
        else if (FAILED(hr))
        {
            std::cerr << "[DDA] EnumAdapters1 return error (hr = " << std::hex << hr << ")." << std::endl;
            factory->Release();
            return hr;
        }

        IDXGIOutput *output = nullptr;
        hr = adapter->EnumOutputs(0, &output);
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            std::cerr << "[DDA] The adapter has no outputs (monitors). Error code: DXGI_ERROR_NOT_FOUND." << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }
        else if (FAILED(hr))
        {
            std::cerr << "[DDA] EnumOutputs returned an error (hr = " << std::hex << hr << ")." << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        {
            D3D_FEATURE_LEVEL featureLevels[] = {D3D_FEATURE_LEVEL_11_0};
            UINT createDeviceFlags = 0;

            hr = D3D11CreateDevice(
                adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                createDeviceFlags,
                featureLevels,
                1,
                D3D11_SDK_VERSION,
                &m_device,
                nullptr,
                &m_context);

            if (FAILED(hr))
            {
                std::cerr << "[DDA] Couldn't create D3D11Device (hr = " << std::hex << hr << ")." << std::endl;
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return hr;
            }
        }

        hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void **)&m_output1);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] QueryInterface on IDXGIOutput1 failed (hr = " << std::hex << hr << ")." << std::endl;
            SafeRelease(&m_context);
            SafeRelease(&m_device);
            SafeRelease(&output);
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        hr = m_output1->DuplicateOutput(m_device, &m_duplication);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] DuplicateOutput failed (hr = " << std::hex << hr << ")." << std::endl;
            SafeRelease(&m_output1);
            SafeRelease(&m_context);
            SafeRelease(&m_device);
            SafeRelease(&output);
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        m_duplication->GetDesc(&m_duplDesc);

        DXGI_OUTPUT_DESC outputDesc{};
        output->GetDesc(&outputDesc);
        outScreenWidth = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
        outScreenHeight = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;

        if (config.verbose)
        {
            std::wcout << L"[DDA] Monitor: " << outputDesc.DeviceName
                       << L", Resolution: " << outScreenWidth << L"x" << outScreenHeight << std::endl;
        }

        {
            D3D11_TEXTURE2D_DESC sharedTexDesc = {};
            sharedTexDesc.Width = captureWidth;
            sharedTexDesc.Height = captureHeight;
            sharedTexDesc.MipLevels = 1;
            sharedTexDesc.ArraySize = 1;
            sharedTexDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            sharedTexDesc.SampleDesc.Count = 1;
            sharedTexDesc.Usage = D3D11_USAGE_DEFAULT;
            sharedTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
            sharedTexDesc.CPUAccessFlags = 0;
            sharedTexDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

            hr = m_device->CreateTexture2D(&sharedTexDesc, nullptr, &m_sharedTexture);
            if (FAILED(hr))
            {
                std::cerr << "[DDA] Couldn't create sharedTexture (hr = " << std::hex << hr << ")." << std::endl;
                SafeRelease(&m_duplication);
                SafeRelease(&m_output1);
                SafeRelease(&m_context);
                SafeRelease(&m_device);
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return hr;
            }
        }

        {
            cudaError_t err = cudaGraphicsD3D11RegisterResource(&m_cudaResource, m_sharedTexture, cudaGraphicsRegisterFlagsNone);
            if (err != cudaSuccess)
            {
                std::cerr << "[DDA] Error registering sharedTexture in CUDA: "
                          << cudaGetErrorString(err) << std::endl;
                Release();
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return E_FAIL;
            }

            cudaStreamCreate(&m_cudaStream);

            // Create the capture done event
            cudaError_t eventErr = cudaEventCreateWithFlags(&m_captureDoneEvent, cudaEventDisableTiming);
            if (eventErr != cudaSuccess)
            {
                std::cerr << "[DDA] Failed to create CUDA event: " << cudaGetErrorString(eventErr) << std::endl;
                // Handle error appropriately, maybe throw or return error code from Initialize
            }
        }

        SafeRelease(&output);
        SafeRelease(&adapter);
        SafeRelease(&factory);

        if (outDevice)
            *outDevice = m_device;
        if (outContext)
            *outContext = m_context;

        return hr;
    }

    // Modified AcquireFrame to only acquire data and metadata, not perform copies.
    // Returns DXGI_OUTDUPL_FRAME_INFO for metadata checks in the caller.
    HRESULT AcquireFrame(FrameContext &frameCtx, DXGI_OUTDUPL_FRAME_INFO& frameInfo, UINT timeout = 100)
    {
        if (!m_duplication)
            return E_FAIL;

        // DXGI_OUTDUPL_FRAME_INFO frameInfo{}; // Provided by caller now
        IDXGIResource *resource = nullptr;

        HRESULT hr = m_duplication->AcquireNextFrame(timeout, &frameInfo, &resource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
        {
            // Caller should handle timeout
            return hr;
        }
        else if (FAILED(hr))
        { // Includes ACCESS_LOST etc.
             if (resource) resource->Release();
             m_duplication->ReleaseFrame(); // Important to release frame state on error
            return hr;
        }

        // --- Get Texture --- 
        hr = resource->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&frameCtx.texture);
        resource->Release(); // Always release the resource
        if (FAILED(hr) || !frameCtx.texture)
        {
             m_duplication->ReleaseFrame(); // Release frame state if texture acquisition fails
             if (frameCtx.texture) { // QueryInterface might succeed but return null? Unlikely but safe. 
                frameCtx.texture->Release();
                frameCtx.texture = nullptr;
             }
            return FAILED(hr) ? hr : E_FAIL; // Return original error or generic failure
        }

        // --- Get Metadata --- 
        frameCtx.moveRects.clear(); // Clear previous frame's metadata
        frameCtx.dirtyRects.clear();

        if (frameInfo.TotalMetadataBufferSize > 0)
        {
            UINT metaDataSize = frameInfo.TotalMetadataBufferSize;
            m_metaDataBuffer.resize(metaDataSize); // Resize member buffer

            // --- Get Move Rects ---
            UINT moveRectsSize = 0;
            HRESULT hrMeta = m_duplication->GetFrameMoveRects(metaDataSize, 
                                                              reinterpret_cast<DXGI_OUTDUPL_MOVE_RECT*>(m_metaDataBuffer.data()), // Use member buffer
                                                              &moveRectsSize);
            if (SUCCEEDED(hrMeta) && moveRectsSize > 0) {
                UINT numMoveRects = moveRectsSize / sizeof(DXGI_OUTDUPL_MOVE_RECT);
                 // Check against member buffer size
                 if (numMoveRects * sizeof(DXGI_OUTDUPL_MOVE_RECT) <= m_metaDataBuffer.size()) { 
                    frameCtx.moveRects.assign(reinterpret_cast<DXGI_OUTDUPL_MOVE_RECT*>(m_metaDataBuffer.data()), // Use member buffer
                                              reinterpret_cast<DXGI_OUTDUPL_MOVE_RECT*>(m_metaDataBuffer.data()) + numMoveRects);
                 }
            }

            // --- Get Dirty Rects ---
            // Calculate expected size based on resized member buffer and actual move rects size
            UINT expectedDirtyRectsSize = (m_metaDataBuffer.size() >= moveRectsSize) ? (m_metaDataBuffer.size() - moveRectsSize) : 0;
            UINT actualDirtyRectsSize = 0; // To store the actual size returned by GetFrameDirtyRects
            if (expectedDirtyRectsSize > 0) {
                BYTE* dirtyRectsDataStart = m_metaDataBuffer.data() + moveRectsSize; 
                hrMeta = m_duplication->GetFrameDirtyRects(expectedDirtyRectsSize, // Pass the available buffer size
                                                      reinterpret_cast<RECT*>(dirtyRectsDataStart), 
                                                      &actualDirtyRectsSize); // Get the actual size
                if (SUCCEEDED(hrMeta) && actualDirtyRectsSize > 0) {
                    UINT numDirtyRects = actualDirtyRectsSize / sizeof(RECT);
                    // Ensure the actual size fits within the expected size derived from the buffer
                    if (numDirtyRects * sizeof(RECT) <= expectedDirtyRectsSize) { 
                        frameCtx.dirtyRects.assign(reinterpret_cast<RECT*>(dirtyRectsDataStart), 
                                                   reinterpret_cast<RECT*>(dirtyRectsDataStart) + numDirtyRects);
                    }
                }
            }
        }
        // m_metaDataBuffer is a member and persists
        
        // Caller is responsible for calling m_duplication->ReleaseFrame() after processing
        return S_OK; // Successfully acquired frame and metadata
    }

    void ReleaseFrame()
    {
        if (m_duplication)
            m_duplication->ReleaseFrame();
    }

    cv::cuda::GpuMat CopySharedTextureToCudaMat(int regionWidth, int regionHeight)
    {
        // Map the shared texture to CUDA
        cudaError_t err = cudaGraphicsMapResources(1, &m_cudaResource, m_cudaStream);
        if (err != cudaSuccess)
        {
            std::cerr << "[DDA] cudaGraphicsMapResources error: " << cudaGetErrorString(err) << std::endl;
            return cv::cuda::GpuMat();
        }

        // Get the mapped array
        cudaArray_t cuArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, m_cudaResource, 0, 0);
        if (err != cudaSuccess)
        {
            std::cerr << "[DDA] cudaGraphicsSubResourceGetMappedArray error: " << cudaGetErrorString(err) << std::endl;
            cudaGraphicsUnmapResources(1, &m_cudaResource, m_cudaStream);
            return cv::cuda::GpuMat();
        }

        // Get a frame from the pool or create a new one if needed
        cv::cuda::GpuMat frameGpu;
        if (!m_framePool.empty())
        {
            frameGpu = m_framePool.back();
            m_framePool.pop_back();

            // Ensure the frame has the correct size
            if (frameGpu.rows != regionHeight || frameGpu.cols != regionWidth)
            {
                frameGpu.release();
                frameGpu = cv::cuda::GpuMat(regionHeight, regionWidth, CV_8UC4);
            }
        }
        else
        {
            frameGpu = cv::cuda::GpuMat(regionHeight, regionWidth, CV_8UC4);
        }

        // Copy from the CUDA array to the GpuMat
        cudaMemcpy2DFromArrayAsync(
            frameGpu.data, frameGpu.step,
            cuArray, 0, 0,
            regionWidth * 4, regionHeight,
            cudaMemcpyDeviceToDevice, m_cudaStream);

        // Unmap the resource
        cudaGraphicsUnmapResources(1, &m_cudaResource, m_cudaStream);

        // Ensure the copy is complete
        // cudaStreamSynchronize(m_cudaStream); // Removed for lower latency. Caller MUST synchronize.

        // Record event on the capture stream to signal copy completion
        if (m_captureDoneEvent)
        {
            cudaEventRecord(m_captureDoneEvent, m_cudaStream);
        }

        return frameGpu;
    }

    // Return a frame to the pool for reuse
    void RecycleFrame(cv::cuda::GpuMat &frame)
    {
        if (m_framePool.size() < 10) // Limit pool size
        {
            m_framePool.push_back(frame);
        }
    }

    void Release()
    {
        if (m_duplication)
        {
            m_duplication->ReleaseFrame();
            m_duplication->Release();
            m_duplication = nullptr;
        }
        SafeRelease(&m_output1);
        SafeRelease(&m_context);
        SafeRelease(&m_device);

        if (m_cudaResource)
        {
            cudaGraphicsUnregisterResource(m_cudaResource);
            m_cudaResource = nullptr;
        }
        SafeRelease(&m_sharedTexture);

        if (m_cudaStream)
        {
            cudaStreamDestroy(m_cudaStream);
            m_cudaStream = nullptr;
        }

        // Destroy the event
        if (m_captureDoneEvent)
        {
            cudaEventDestroy(m_captureDoneEvent);
            m_captureDoneEvent = nullptr;
        }
    }

public:
    ID3D11Device *m_device;
    ID3D11DeviceContext *m_context;
    IDXGIOutputDuplication *m_duplication;
    IDXGIOutput1 *m_output1;
    DXGI_OUTDUPL_DESC m_duplDesc;

    ID3D11Texture2D *m_sharedTexture;
    cudaGraphicsResource *m_cudaResource;
    cudaStream_t m_cudaStream;
    cudaEvent_t m_captureDoneEvent;
    std::vector<cv::cuda::GpuMat> m_framePool;
    UINT m_timeout = 16; // Reduced timeout for lower latency (approx. 60Hz)
    std::vector<BYTE> m_metaDataBuffer; // Reuse metadata buffer
};

DuplicationAPIScreenCapture::DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight)
    : d3dDevice(nullptr), d3dContext(nullptr), deskDupl(nullptr), stagingTexture(nullptr), output1(nullptr), sharedTexture(nullptr), cudaResource(nullptr), cudaStream(nullptr), regionWidth(desiredWidth), regionHeight(desiredHeight), screenWidth(0), screenHeight(0)
{
    m_ddaManager = std::make_unique<DDAManager>();

    HRESULT hr = m_ddaManager->Initialize(
        config.monitor_idx,
        regionWidth,
        regionHeight,
        screenWidth,
        screenHeight,
        &d3dDevice,
        &d3dContext);
    if (FAILED(hr))
    {
        std::cerr << "[Capture] Error initializing DuplicationAPIScreenCapture: hr=0x"
                  << std::hex << hr << std::endl;
        return;
    }
}

DuplicationAPIScreenCapture::~DuplicationAPIScreenCapture()
{
    if (m_ddaManager)
    {
        m_ddaManager->Release();
        m_ddaManager.reset();
    }
    d3dDevice = nullptr;
    d3dContext = nullptr;
    deskDupl = nullptr;
    stagingTexture = nullptr;
    output1 = nullptr;
    sharedTexture = nullptr;
    cudaResource = nullptr;
    cudaStream = nullptr;
}

cv::cuda::GpuMat DuplicationAPIScreenCapture::GetNextFrameGpu()
{
    if (!m_ddaManager || !m_ddaManager->m_duplication)
        return cv::cuda::GpuMat();

    HRESULT hr = S_OK;
    FrameContext frameCtx; // Holds texture, move/dirty rects
    DXGI_OUTDUPL_FRAME_INFO frameInfo = {}; // Holds metadata info

    // Define the target capture area on the screen *early*
    RECT captureScreenRect;
    captureScreenRect.left = (screenWidth - regionWidth) / 2;
    captureScreenRect.top = (screenHeight - regionHeight) / 2;
    captureScreenRect.right = captureScreenRect.left + regionWidth;
    captureScreenRect.bottom = captureScreenRect.top + regionHeight;

    // Acquire the next frame - populates frameCtx and frameInfo
    hr = m_ddaManager->AcquireFrame(frameCtx, frameInfo, m_ddaManager->m_timeout);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        // std::cout << "[Capture] Wait timeout" << std::endl; // Optional: log timeout
        // No frame acquired, frameCtx.texture should be null
        // DDAManager::AcquireFrame doesn't call ReleaseFrame on timeout
        return cv::cuda::GpuMat();
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_REMOVED)
    {
        std::cerr << "[Capture] DDA Access lost/Device reset. Reinitializing..." << std::endl;
        capture_method_changed.store(true); // Signal reinitialization
        // DDAManager::AcquireFrame already called ReleaseFrame internally on these errors
        // frameCtx.texture should be null or already released
        return cv::cuda::GpuMat(); 
    }
    else if (FAILED(hr))
    {
        std::cerr << "[Capture] AcquireFrame failed (hr=0x" << std::hex << hr << ")" << std::endl;
        // DDAManager::AcquireFrame already called ReleaseFrame internally on failure
        // frameCtx.texture should be null or already released
        return cv::cuda::GpuMat();
    }

    // If we reach here, AcquireFrame succeeded and frameCtx.texture is valid.
    // DDAManager::AcquireFrame did NOT call ReleaseFrame yet.

    // --- Start Copy Logic (Simplified: Always copy the full region) ---
    if (m_ddaManager->m_context && m_ddaManager->m_sharedTexture && frameCtx.texture)
    {
        // Always copy the entire target region from the source texture
        D3D11_BOX sourceBox;
        sourceBox.left = captureScreenRect.left; 
        sourceBox.top = captureScreenRect.top;
        sourceBox.front = 0;
        sourceBox.right = captureScreenRect.right;
        sourceBox.bottom = captureScreenRect.bottom;
        sourceBox.back = 1;

        m_ddaManager->m_context->CopySubresourceRegion(
            m_ddaManager->m_sharedTexture, // Dest texture
            0,                             // Dest subresource
            0, 0, 0,                       // Dest X, Y, Z (destination is top-left of shared texture)
            frameCtx.texture,              // Src texture (copy from the newly acquired frame)
            0,                             // Src subresource
            &sourceBox);                   // Src box defined by captureScreenRect
    }
    // --- End Copy Logic ---

    // Release the acquired desktop frame back to DXGI *after* copying from it
    m_ddaManager->ReleaseFrame(); 

    cv::cuda::GpuMat frameGpu = m_ddaManager->CopySharedTextureToCudaMat(regionWidth, regionHeight);

    // We MUST release the texture obtained from QueryInterface
    if (frameCtx.texture)
    {
         frameCtx.texture->Release();
         frameCtx.texture = nullptr;
    }

    // Recycle the previous frame if we have one
    if (!m_previousFrame.empty())
    {
        m_ddaManager->RecycleFrame(m_previousFrame);
    }

    // Store the current frame for recycling next time
    m_previousFrame = frameGpu;

    return frameGpu;
}

cv::Mat DuplicationAPIScreenCapture::GetNextFrameCpu()
{
    if (!m_ddaManager || !m_ddaManager->m_duplication)
        return cv::Mat();

    HRESULT hr = S_OK;
    FrameContext frameCtx;
    DXGI_OUTDUPL_FRAME_INFO frameInfo = {};

    RECT captureScreenRect;
    captureScreenRect.left = (screenWidth - regionWidth) / 2;
    captureScreenRect.top = (screenHeight - regionHeight) / 2;
    captureScreenRect.right = captureScreenRect.left + regionWidth;
    captureScreenRect.bottom = captureScreenRect.top + regionHeight;

    hr = m_ddaManager->AcquireFrame(frameCtx, frameInfo, m_ddaManager->m_timeout);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        return cv::Mat();
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_REMOVED)
    {
        std::cerr << "[Capture] DDA Access lost/Device reset (CPU Path). Reinitializing..." << std::endl;
        capture_method_changed.store(true);
        return cv::Mat();
    }
    else if (FAILED(hr))
    {
        std::cerr << "[Capture] AcquireFrame failed (CPU Path) (hr=0x" << std::hex << hr << ")" << std::endl;
        return cv::Mat();
    }

    // Copy to shared texture
    if (m_ddaManager->m_context && m_ddaManager->m_sharedTexture && frameCtx.texture)
    {
        D3D11_BOX sourceBox;
        sourceBox.left = captureScreenRect.left;
        sourceBox.top = captureScreenRect.top;
        sourceBox.front = 0;
        sourceBox.right = captureScreenRect.right;
        sourceBox.bottom = captureScreenRect.bottom;
        sourceBox.back = 1;

        m_ddaManager->m_context->CopySubresourceRegion(
            m_ddaManager->m_sharedTexture, 0, 0, 0, 0,
            frameCtx.texture, 0, &sourceBox);
    }

    // Release acquired frame
    m_ddaManager->ReleaseFrame();

    // Copy shared texture to CPU Mat via CUDA mapping
    cv::Mat frameCpu(regionHeight, regionWidth, CV_8UC4); // Create CPU Mat
    if (!frameCpu.empty() && m_ddaManager->m_cudaResource && m_ddaManager->m_cudaStream)
    {
        cudaError_t err = cudaGraphicsMapResources(1, &m_ddaManager->m_cudaResource, m_ddaManager->m_cudaStream);
        if (err == cudaSuccess)
        {
            cudaArray_t cuArray;
            err = cudaGraphicsSubResourceGetMappedArray(&cuArray, m_ddaManager->m_cudaResource, 0, 0);
            if (err == cudaSuccess)
            {
                // Copy from CUDA array to Host (CPU) memory
                err = cudaMemcpy2DFromArrayAsync(
                    frameCpu.data,        // Destination: CPU Mat data pointer
                    frameCpu.step,        // Destination: CPU Mat step (bytes per row)
                    cuArray,              // Source: Mapped CUDA array
                    0, 0,                 // Source X, Y offset
                    regionWidth * 4,      // Width in bytes (BGRA)
                    regionHeight,         // Height
                    cudaMemcpyDeviceToHost, // Copy direction
                    m_ddaManager->m_cudaStream);

                if (err == cudaSuccess)
                {
                    // Synchronize the stream to ensure the copy is complete before returning
                    cudaStreamSynchronize(m_ddaManager->m_cudaStream);
                }
                else
                {
                    std::cerr << "[DDA] cudaMemcpy2DFromArrayAsync (D->H) error: " << cudaGetErrorString(err) << std::endl;
                    frameCpu.release(); // Release the mat on error
                }
            }
            else
            {
                std::cerr << "[DDA] cudaGraphicsSubResourceGetMappedArray (CPU Path) error: " << cudaGetErrorString(err) << std::endl;
                frameCpu.release();
            }
            cudaGraphicsUnmapResources(1, &m_ddaManager->m_cudaResource, m_ddaManager->m_cudaStream);
        }
        else
        {
            std::cerr << "[DDA] cudaGraphicsMapResources (CPU Path) error: " << cudaGetErrorString(err) << std::endl;
            frameCpu.release();
        }
    } else {
         if (frameCpu.empty()) std::cerr << "[DDA] Failed to create CPU Mat" << std::endl;
         if (!m_ddaManager->m_cudaResource) std::cerr << "[DDA] CUDA resource invalid for CPU path" << std::endl;
         if (!m_ddaManager->m_cudaStream) std::cerr << "[DDA] CUDA stream invalid for CPU path" << std::endl;
        frameCpu.release(); // Release if setup failed
    }

    // Release the texture obtained from QueryInterface
    if (frameCtx.texture)
    {
         frameCtx.texture->Release();
         frameCtx.texture = nullptr;
    }

    // Note: Frame recycling is currently GPU-based. CPU path doesn't recycle.

    return frameCpu;
}

// Implementation for the event getter function
cudaEvent_t DuplicationAPIScreenCapture::GetCaptureDoneEvent() const
{
   if (m_ddaManager)
   {
       return m_ddaManager->m_captureDoneEvent;
   }
   return nullptr; // Or handle error appropriately
}