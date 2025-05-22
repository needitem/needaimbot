// Define before including Windows/D3D headers to avoid macro collisions
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "needaimbot.h"
#include "duplication_api_capture.h"
#include "config.h"
#include "other_tools.h"
#include "capture.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

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
        : m_device(nullptr), m_context(nullptr), m_duplication(nullptr), m_output1(nullptr), m_sharedTexture(nullptr), m_cudaResource(nullptr), m_cudaStream(nullptr), m_framePool(5), m_captureDoneEvent(nullptr)
        , m_pinnedHostBuffer(nullptr), m_hostCopyStream(nullptr) // Initialize pinned buffer and copy stream
    {
        ZeroMemory(&m_duplDesc, sizeof(m_duplDesc));
    }

    ~DDAManager()
    {
        // Free pinned host buffer and destroy copy stream
        if (m_hostCopyStream) cudaStreamDestroy(m_hostCopyStream);
        if (m_pinnedHostBuffer) cudaFreeHost(m_pinnedHostBuffer);
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
        // std::cout << "[DDA] Initializing DDAManager for monitor index: " << monitorIndex << std::endl;

        IDXGIFactory1 *factory = nullptr;
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&factory);
        // std::cout << "[DDA] CreateDXGIFactory1 result: 0x" << std::hex << hr << std::endl;
        if (FAILED(hr))
        {
            // std::cerr << "[DDA] Failed to create DXGIFactory1 (hr = " << std::hex << hr << ")." << std::endl;
            return hr;
        }

        IDXGIAdapter1 *adapter = nullptr;
        hr = factory->EnumAdapters1(monitorIndex, &adapter);
        // std::cout << "[DDA] EnumAdapters1 result: 0x" << std::hex << hr << " for index " << monitorIndex << std::endl;
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            // std::cerr << "[DDA] Not found adapter with index " << monitorIndex << ". Error code: DXGI_ERROR_NOT_FOUND." << std::endl;
            factory->Release();
            return hr;
        }
        else if (FAILED(hr))
        {
            // std::cerr << "[DDA] EnumAdapters1 return error (hr = " << std::hex << hr << ")." << std::endl;
            factory->Release();
            return hr;
        }
        if (adapter && config.verbose) { // Keep verbose logging
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            std::wcout << L"[DDA] Using Adapter: " << desc.Description << std::endl;
        }

        IDXGIOutput *output = nullptr;
        hr = adapter->EnumOutputs(0, &output);
        // std::cout << "[DDA] EnumOutputs result: 0x" << std::hex << hr << " for output index 0" << std::endl;
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            // std::cerr << "[DDA] The adapter has no outputs (monitors). Error code: DXGI_ERROR_NOT_FOUND." << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }
        else if (FAILED(hr))
        {
            // std::cerr << "[DDA] EnumOutputs returned an error (hr = " << std::hex << hr << ")." << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }
        if (output && config.verbose) { // Keep verbose logging
             DXGI_OUTPUT_DESC desc;
             output->GetDesc(&desc);
             std::wcout << L"[DDA] Using Output: " << desc.DeviceName << std::endl;
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
            // std::cout << "[DDA] D3D11CreateDevice result: 0x" << std::hex << hr << std::endl;

            if (FAILED(hr))
            {
                // std::cerr << "[DDA] Couldn't create D3D11Device (hr = " << std::hex << hr << ")." << std::endl;
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return hr;
            }
        }

        hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void **)&m_output1);
        // std::cout << "[DDA] QueryInterface for IDXGIOutput1 result: 0x" << std::hex << hr << std::endl;
        if (FAILED(hr))
        {
            // std::cerr << "[DDA] QueryInterface on IDXGIOutput1 failed (hr = " << std::hex << hr << ")." << std::endl;
            SafeRelease(&m_context);
            SafeRelease(&m_device);
            SafeRelease(&output);
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        // std::cout << "[DDA] Calling DuplicateOutput with m_device: " << m_device << std::endl;
        hr = m_output1->DuplicateOutput(m_device, &m_duplication);
        // std::cout << "[DDA] DuplicateOutput result: 0x" << std::hex << hr << std::endl;
        if (FAILED(hr))
        {
            // std::cerr << "[DDA] DuplicateOutput failed (hr = " << std::hex << hr << ")." << std::endl;
            if (hr == DXGI_ERROR_UNSUPPORTED) {
                 // std::cerr << "[DDA] Error: DXGI_ERROR_UNSUPPORTED. Desktop Duplication API is not supported on this system/configuration." << std::endl;
            } else if (hr == DXGI_ERROR_ACCESS_DENIED) {
                 // std::cerr << "[DDA] Error: DXGI_ERROR_ACCESS_DENIED. Access denied, possibly due to protected content or insufficient privileges." << std::endl;
            } else if (hr == E_INVALIDARG) {
                 // std::cerr << "[DDA] Error: E_INVALIDARG. Check if the device pointer is valid and belongs to the correct adapter." << std::endl;
            }
            
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

            cudaError_t eventErr = cudaEventCreateWithFlags(&m_captureDoneEvent, cudaEventDisableTiming);
            if (eventErr != cudaSuccess)
            {
                std::cerr << "[DDA] Failed to create CUDA event: " << cudaGetErrorString(eventErr) << std::endl;
            }

            // Allocate pinned host memory for CPU copy and create a dedicated copy stream
            size_t hostBufferSize = static_cast<size_t>(captureWidth) * captureHeight * 4;
            cudaError_t allocErr = cudaHostAlloc(&m_pinnedHostBuffer, hostBufferSize, cudaHostAllocDefault);
            if (allocErr != cudaSuccess)
            {
                std::cerr << "[DDA] Failed to allocate pinned host memory: " << cudaGetErrorString(allocErr) << std::endl;
            }
            cudaStreamCreate(&m_hostCopyStream);
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

    HRESULT AcquireFrame(FrameContext &frameCtx, DXGI_OUTDUPL_FRAME_INFO& frameInfo, UINT timeout = 100)
    {
        if (!m_duplication)
            return E_FAIL;

        IDXGIResource *resource = nullptr;

        HRESULT hr = m_duplication->AcquireNextFrame(timeout, &frameInfo, &resource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
        {
            return hr;
        }
        else if (FAILED(hr))
        {
             if (resource) resource->Release();
             m_duplication->ReleaseFrame();
            return hr;
        }

        hr = resource->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&frameCtx.texture);
        resource->Release();
        if (FAILED(hr) || !frameCtx.texture)
        {
             m_duplication->ReleaseFrame();
             if (frameCtx.texture) {
                frameCtx.texture->Release();
                frameCtx.texture = nullptr;
             }
            return FAILED(hr) ? hr : E_FAIL;
        }

        frameCtx.moveRects.clear();
        frameCtx.dirtyRects.clear();

        if (frameInfo.TotalMetadataBufferSize > 0)
        {
            UINT metaDataSize = frameInfo.TotalMetadataBufferSize;
            m_metaDataBuffer.resize(metaDataSize);

            UINT moveRectsSize = 0;
            HRESULT hrMeta = m_duplication->GetFrameMoveRects(metaDataSize, 
                                                              reinterpret_cast<DXGI_OUTDUPL_MOVE_RECT*>(m_metaDataBuffer.data()),
                                                              &moveRectsSize);
            if (SUCCEEDED(hrMeta) && moveRectsSize > 0) {
                UINT numMoveRects = moveRectsSize / sizeof(DXGI_OUTDUPL_MOVE_RECT);
                 if (numMoveRects * sizeof(DXGI_OUTDUPL_MOVE_RECT) <= m_metaDataBuffer.size()) { 
                    frameCtx.moveRects.assign(reinterpret_cast<DXGI_OUTDUPL_MOVE_RECT*>(m_metaDataBuffer.data()),
                                              reinterpret_cast<DXGI_OUTDUPL_MOVE_RECT*>(m_metaDataBuffer.data()) + numMoveRects);
                 }
            }

            UINT expectedDirtyRectsSize = (m_metaDataBuffer.size() >= moveRectsSize) ? (m_metaDataBuffer.size() - moveRectsSize) : 0;
            UINT actualDirtyRectsSize = 0;
            if (expectedDirtyRectsSize > 0) {
                BYTE* dirtyRectsDataStart = m_metaDataBuffer.data() + moveRectsSize; 
                hrMeta = m_duplication->GetFrameDirtyRects(expectedDirtyRectsSize,
                                                      reinterpret_cast<RECT*>(dirtyRectsDataStart), 
                                                      &actualDirtyRectsSize);
                if (SUCCEEDED(hrMeta) && actualDirtyRectsSize > 0) {
                    UINT numDirtyRects = actualDirtyRectsSize / sizeof(RECT);
                    if (numDirtyRects * sizeof(RECT) <= expectedDirtyRectsSize) { 
                        frameCtx.dirtyRects.assign(reinterpret_cast<RECT*>(dirtyRectsDataStart), 
                                                   reinterpret_cast<RECT*>(dirtyRectsDataStart) + numDirtyRects);
                    }
                }
            }
        }
        
        return S_OK;
    }

    void ReleaseFrame()
    {
        if (m_duplication)
            m_duplication->ReleaseFrame();
    }

    cv::cuda::GpuMat CopySharedTextureToCudaMat(int regionWidth, int regionHeight)
    {
        cudaError_t err = cudaGraphicsMapResources(1, &m_cudaResource, m_cudaStream);
        if (err != cudaSuccess)
        {
            std::cerr << "[DDA] cudaGraphicsMapResources error: " << cudaGetErrorString(err) << std::endl;
            return cv::cuda::GpuMat();
        }

        cudaArray_t cuArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, m_cudaResource, 0, 0);
        if (err != cudaSuccess)
        {
            std::cerr << "[DDA] cudaGraphicsSubResourceGetMappedArray error: " << cudaGetErrorString(err) << std::endl;
            cudaGraphicsUnmapResources(1, &m_cudaResource, m_cudaStream);
            return cv::cuda::GpuMat();
        }

        cv::cuda::GpuMat frameGpu;
        if (!m_framePool.empty())
        {
            frameGpu = m_framePool.back();
            m_framePool.pop_back();

            if (frameGpu.rows != regionHeight || frameGpu.cols != regionWidth)
            {
                frameGpu.release();
                frameGpu = cv::cuda::GpuMat(regionHeight, regionWidth, CV_8UC4);
            }
        }
        else
        {
             // Allow allocation if pool is empty
            frameGpu = cv::cuda::GpuMat(regionHeight, regionWidth, CV_8UC4);
        }

        cudaMemcpy2DFromArrayAsync(
            frameGpu.data, frameGpu.step,
            cuArray, 0, 0,
            regionWidth * 4, regionHeight,
            cudaMemcpyDeviceToDevice, m_cudaStream);

        cudaGraphicsUnmapResources(1, &m_cudaResource, m_cudaStream);

        if (m_captureDoneEvent)
        {
            cudaEventRecord(m_captureDoneEvent, m_cudaStream);
        }

        return frameGpu;
    }

    void RecycleFrame(cv::cuda::GpuMat &frame)
    {
        if (m_framePool.size() < 10)
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

        if (m_captureDoneEvent)
        {
            cudaEventDestroy(m_captureDoneEvent);
            m_captureDoneEvent = nullptr;
        }
    }

    void SetAcquireTimeout(UINT timeout) {
        m_timeout = timeout;
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
    UINT m_timeout = 1; // Default timeout for AcquireNextFrame in milliseconds
    std::vector<BYTE> m_metaDataBuffer;

    // Added for asynchronous pinned host copy
    unsigned char* m_pinnedHostBuffer;
    cudaStream_t m_hostCopyStream;
};

DuplicationAPIScreenCapture::DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight)
    : d3dDevice(nullptr), d3dContext(nullptr), deskDupl(nullptr), stagingTexture(nullptr), output1(nullptr), sharedTexture(nullptr), cudaResource(nullptr), cudaStream(nullptr), regionWidth(desiredWidth), regionHeight(desiredHeight), screenWidth(0), screenHeight(0), m_initialized(false)
{
    // std::cout << "[Capture] DuplicationAPIScreenCapture constructor called." << std::endl;
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

    // Set timeout after DDAManager is successfully initialized
    if (m_ddaManager) {
        m_ddaManager->SetAcquireTimeout(config.capture_timeout_ms);
    }

    m_initialized = true;
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
        return cv::cuda::GpuMat();
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_REMOVED)
    {
        std::cerr << "[Capture] DDA Access lost/Device reset. Reinitializing..." << std::endl;
        return cv::cuda::GpuMat(); 
    }
    else if (FAILED(hr))
    {
        // std::cerr << "[Capture] AcquireFrame failed (hr=0x" << std::hex << hr << ")" << std::endl;
        return cv::cuda::GpuMat();
    }

    if (m_ddaManager->m_context && m_ddaManager->m_sharedTexture && frameCtx.texture)
    {
        // Copy only changed regions if available
        if (!frameCtx.dirtyRects.empty())
        {
            for (const RECT& dirty : frameCtx.dirtyRects)
            {
                LONG left = std::max(dirty.left, captureScreenRect.left);
                LONG top = std::max(dirty.top, captureScreenRect.top);
                LONG right = std::min(dirty.right, captureScreenRect.right);
                LONG bottom = std::min(dirty.bottom, captureScreenRect.bottom);
                if (left < right && top < bottom)
                {
                    D3D11_BOX box = { left, top, 0, right, bottom, 1 };
                    UINT destX = static_cast<UINT>(left - captureScreenRect.left);
                    UINT destY = static_cast<UINT>(top - captureScreenRect.top);
                    m_ddaManager->m_context->CopySubresourceRegion(
                        m_ddaManager->m_sharedTexture,
                        0,
                        destX, destY, 0,
                        frameCtx.texture,
                        0,
                        &box);
                }
            }
        }
        else
        {
            // Fallback to full region copy
            D3D11_BOX box = { captureScreenRect.left, captureScreenRect.top, 0, captureScreenRect.right, captureScreenRect.bottom, 1 };
            m_ddaManager->m_context->CopySubresourceRegion(
                m_ddaManager->m_sharedTexture,
                0,
                0, 0, 0,
                frameCtx.texture,
                0,
                &box);
        }
    }

    m_ddaManager->ReleaseFrame(); 

    cv::cuda::GpuMat frameGpu = m_ddaManager->CopySharedTextureToCudaMat(regionWidth, regionHeight);

    if (frameCtx.texture)
    {
         frameCtx.texture->Release();
         frameCtx.texture = nullptr;
    }

    if (!m_previousFrame.empty())
    {
        m_ddaManager->RecycleFrame(m_previousFrame);
    }

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
        return cv::Mat();
    }
    else if (FAILED(hr))
    {
        return cv::Mat();
    }

    if (m_ddaManager->m_context && m_ddaManager->m_sharedTexture && frameCtx.texture)
    {
        // Copy only changed regions if available
        if (!frameCtx.dirtyRects.empty())
        {
            for (const RECT& dirty : frameCtx.dirtyRects)
            {
                LONG left = std::max(dirty.left, captureScreenRect.left);
                LONG top = std::max(dirty.top, captureScreenRect.top);
                LONG right = std::min(dirty.right, captureScreenRect.right);
                LONG bottom = std::min(dirty.bottom, captureScreenRect.bottom);
                if (left < right && top < bottom)
                {
                    D3D11_BOX box = { left, top, 0, right, bottom, 1 };
                    UINT destX = static_cast<UINT>(left - captureScreenRect.left);
                    UINT destY = static_cast<UINT>(top - captureScreenRect.top);
                    m_ddaManager->m_context->CopySubresourceRegion(
                        m_ddaManager->m_sharedTexture, 0,
                        destX, destY, 0,
                        frameCtx.texture, 0,
                        &box);
                }
            }
        }
        else
        {
            // Fallback to full region copy
            D3D11_BOX box = { captureScreenRect.left, captureScreenRect.top, 0, captureScreenRect.right, captureScreenRect.bottom, 1 };
            m_ddaManager->m_context->CopySubresourceRegion(
                m_ddaManager->m_sharedTexture, 0,
                0, 0, 0,
                frameCtx.texture, 0,
                &box);
        }
    }

    m_ddaManager->ReleaseFrame();

    // Asynchronous pinned host copy
    cv::Mat frameCpu;
    if (m_ddaManager->m_cudaResource && m_ddaManager->m_pinnedHostBuffer)
    {
        unsigned char* hostPtr = m_ddaManager->m_pinnedHostBuffer;
        size_t hostPitch = static_cast<size_t>(regionWidth) * 4;
        cudaError_t err = cudaGraphicsMapResources(1, &m_ddaManager->m_cudaResource, m_ddaManager->m_hostCopyStream);
        if (err == cudaSuccess)
        {
            cudaArray_t cuArray;
            err = cudaGraphicsSubResourceGetMappedArray(&cuArray, m_ddaManager->m_cudaResource, 0, 0);
            if (err == cudaSuccess)
            {
                err = cudaMemcpy2DFromArrayAsync(
                    hostPtr,
                    hostPitch,
                    cuArray,
                    0, 0,
                    regionWidth * 4,
                    regionHeight,
                    cudaMemcpyDeviceToHost,
                    m_ddaManager->m_hostCopyStream);
                if (err == cudaSuccess)
                {
                    cudaStreamSynchronize(m_ddaManager->m_hostCopyStream);
                    frameCpu = cv::Mat(regionHeight, regionWidth, CV_8UC4, hostPtr, hostPitch);
                }
            }
            cudaGraphicsUnmapResources(1, &m_ddaManager->m_cudaResource, m_ddaManager->m_hostCopyStream);
        }
    }

    if (frameCtx.texture)
    {
         frameCtx.texture->Release();
         frameCtx.texture = nullptr;
    }

    return frameCpu;
}

void DuplicationAPIScreenCapture::SetAcquireTimeout(UINT timeout) {
    if (m_ddaManager) {
        m_ddaManager->SetAcquireTimeout(timeout);
    }
}

cudaEvent_t DuplicationAPIScreenCapture::GetCaptureDoneEvent() const
{
   if (m_ddaManager)
   {
       return m_ddaManager->m_captureDoneEvent;
   }
   return nullptr;
}