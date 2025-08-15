// Virtual Camera Capture Implementation
// Captures video from OBS Virtual Camera or other virtual camera devices
// Uses DirectShow for maximum compatibility

#include "virtual_camera_capture.h"
#include "../AppContext.h"
#include <iostream>
#include <vector>

VirtualCameraCapture::VirtualCameraCapture(int width, int height)
    : m_width(width), m_height(height),
      m_cudaResource(nullptr), m_captureStream(nullptr),
      m_isCapturing(false), m_frameAvailable(false),
      m_callback(nullptr) {
    
    // Allocate frame buffer
    m_frameBufferSize = width * height * 4; // BGRA format
    m_frameBuffer = std::make_unique<uint8_t[]>(m_frameBufferSize);
}

VirtualCameraCapture::~VirtualCameraCapture() {
    StopCapture();
    
    if (m_callback) {
        m_callback->Release();
        m_callback = nullptr;
    }
    
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    
    if (m_captureStream) {
        cudaStreamDestroy(m_captureStream);
        m_captureStream = nullptr;
    }
    
    m_stagingTexture.Reset();
    m_d3dContext.Reset();
    m_d3dDevice.Reset();
}

bool VirtualCameraCapture::Initialize(const std::string& cameraName) {
    std::cout << "[VirtualCamera] Initializing with camera: " << cameraName << std::endl;
    
    // Initialize COM
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        std::cerr << "[VirtualCamera] Failed to initialize COM" << std::endl;
        return false;
    }
    
    // Initialize D3D11 for GPU operations
    if (!InitializeD3D11()) {
        std::cerr << "[VirtualCamera] Failed to initialize D3D11" << std::endl;
        return false;
    }
    
    // Initialize DirectShow capture
    if (!InitializeDirectShow(cameraName)) {
        std::cerr << "[VirtualCamera] Failed to initialize DirectShow" << std::endl;
        return false;
    }
    
    // Initialize CUDA interop
    if (!InitializeCUDAInterop()) {
        std::cerr << "[VirtualCamera] Failed to initialize CUDA interop" << std::endl;
        return false;
    }
    
    std::cout << "[VirtualCamera] Initialization complete" << std::endl;
    return true;
}

bool VirtualCameraCapture::InitializeD3D11() {
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    
    D3D_FEATURE_LEVEL featureLevel;
    UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    
#ifdef _DEBUG
    createFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        createFlags,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &m_d3dDevice,
        &featureLevel,
        &m_d3dContext
    );
    
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to create D3D11 device: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // Create staging texture for frame data
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = m_width;
    desc.Height = m_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, &m_stagingTexture);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to create staging texture: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}

bool VirtualCameraCapture::InitializeCUDAInterop() {
    // Create CUDA stream
    cudaError_t err = cudaStreamCreate(&m_captureStream);
    if (err != cudaSuccess) {
        std::cerr << "[VirtualCamera] Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Register D3D11 texture with CUDA
    err = cudaGraphicsD3D11RegisterResource(
        &m_cudaResource,
        m_stagingTexture.Get(),
        cudaGraphicsRegisterFlagsNone
    );
    
    if (err != cudaSuccess) {
        std::cerr << "[VirtualCamera] Failed to register D3D11 resource with CUDA: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool VirtualCameraCapture::InitializeDirectShow(const std::string& cameraName) {
    HRESULT hr;
    
    // Create the Filter Graph Manager
    hr = CoCreateInstance(CLSID_FilterGraph, nullptr, CLSCTX_INPROC_SERVER,
                          IID_IGraphBuilder, (void**)&m_graphBuilder);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to create graph builder" << std::endl;
        return false;
    }
    
    // Create the Capture Graph Builder
    hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, nullptr, CLSCTX_INPROC_SERVER,
                          IID_ICaptureGraphBuilder2, (void**)&m_captureGraphBuilder);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to create capture graph builder" << std::endl;
        return false;
    }
    
    // Set the graph builder
    m_captureGraphBuilder->SetFiltergraph(m_graphBuilder.Get());
    
    // Find the virtual camera device
    m_sourceFilter = FindCaptureDevice(cameraName);
    if (!m_sourceFilter) {
        std::cerr << "[VirtualCamera] Failed to find camera: " << cameraName << std::endl;
        std::cerr << "[VirtualCamera] Available cameras:" << std::endl;
        
        // List available cameras
        ComPtr<ICreateDevEnum> devEnum;
        CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                        IID_ICreateDevEnum, (void**)&devEnum);
        
        ComPtr<IEnumMoniker> enumMoniker;
        devEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumMoniker, 0);
        
        if (enumMoniker) {
            IMoniker* moniker = nullptr;
            while (enumMoniker->Next(1, &moniker, nullptr) == S_OK) {
                ComPtr<IPropertyBag> propBag;
                hr = moniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&propBag);
                if (SUCCEEDED(hr)) {
                    VARIANT var;
                    VariantInit(&var);
                    hr = propBag->Read(L"FriendlyName", &var, 0);
                    if (SUCCEEDED(hr)) {
                        std::wcout << L"  - " << var.bstrVal << std::endl;
                        VariantClear(&var);
                    }
                }
                moniker->Release();
            }
        }
        return false;
    }
    
    // Add source filter to graph
    hr = m_graphBuilder->AddFilter(m_sourceFilter.Get(), L"Video Capture");
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to add source filter" << std::endl;
        return false;
    }
    
    // Create and add sample grabber filter
    hr = CoCreateInstance(CLSID_SampleGrabber, nullptr, CLSCTX_INPROC_SERVER,
                          IID_IBaseFilter, (void**)&m_sampleGrabberFilter);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to create sample grabber" << std::endl;
        return false;
    }
    
    hr = m_graphBuilder->AddFilter(m_sampleGrabberFilter.Get(), L"Sample Grabber");
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to add sample grabber" << std::endl;
        return false;
    }
    
    // Get the ISampleGrabber interface
    hr = m_sampleGrabberFilter->QueryInterface(IID_ISampleGrabber, (void**)&m_sampleGrabber);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to get ISampleGrabber interface" << std::endl;
        return false;
    }
    
    // Set media type for sample grabber (RGB32)
    AM_MEDIA_TYPE mt;
    ZeroMemory(&mt, sizeof(mt));
    mt.majortype = MEDIATYPE_Video;
    mt.subtype = MEDIASUBTYPE_RGB32;
    mt.formattype = FORMAT_VideoInfo;
    
    hr = m_sampleGrabber->SetMediaType(&mt);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to set media type" << std::endl;
        return false;
    }
    
    // Set callback
    m_callback = new SampleGrabberCallback(this);
    hr = m_sampleGrabber->SetCallback(m_callback, 1); // Use BufferCB
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to set callback" << std::endl;
        return false;
    }
    
    // Disable sample buffering
    m_sampleGrabber->SetBufferSamples(FALSE);
    
    // Create null renderer (we don't need to display the video)
    ComPtr<IBaseFilter> nullRenderer;
    hr = CoCreateInstance(CLSID_NullRenderer, nullptr, CLSCTX_INPROC_SERVER,
                          IID_IBaseFilter, (void**)&nullRenderer);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to create null renderer" << std::endl;
        return false;
    }
    
    hr = m_graphBuilder->AddFilter(nullRenderer.Get(), L"Null Renderer");
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to add null renderer" << std::endl;
        return false;
    }
    
    // Connect the filters
    hr = m_captureGraphBuilder->RenderStream(
        &PIN_CATEGORY_CAPTURE,
        &MEDIATYPE_Video,
        m_sourceFilter.Get(),
        m_sampleGrabberFilter.Get(),
        nullRenderer.Get()
    );
    
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to connect filters: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // Get media control interface
    hr = m_graphBuilder->QueryInterface(IID_IMediaControl, (void**)&m_mediaControl);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to get media control" << std::endl;
        return false;
    }
    
    // Get media event interface
    hr = m_graphBuilder->QueryInterface(IID_IMediaEvent, (void**)&m_mediaEvent);
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to get media event" << std::endl;
        return false;
    }
    
    std::cout << "[VirtualCamera] DirectShow initialized successfully" << std::endl;
    return true;
}

IBaseFilter* VirtualCameraCapture::FindCaptureDevice(const std::string& deviceName) {
    ComPtr<ICreateDevEnum> devEnum;
    ComPtr<IEnumMoniker> enumMoniker;
    IMoniker* moniker = nullptr;
    IBaseFilter* captureFilter = nullptr;
    
    // Create device enumerator
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_ICreateDevEnum, (void**)&devEnum);
    if (FAILED(hr)) {
        return nullptr;
    }
    
    // Create enumerator for video input devices
    hr = devEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumMoniker, 0);
    if (hr != S_OK) {
        return nullptr;
    }
    
    // Convert device name to wide string
    std::wstring wideDeviceName(deviceName.begin(), deviceName.end());
    
    // Enumerate devices
    while (enumMoniker->Next(1, &moniker, nullptr) == S_OK) {
        ComPtr<IPropertyBag> propBag;
        hr = moniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&propBag);
        
        if (SUCCEEDED(hr)) {
            VARIANT var;
            VariantInit(&var);
            
            // Get device name
            hr = propBag->Read(L"FriendlyName", &var, 0);
            if (SUCCEEDED(hr)) {
                std::wstring currentDevice = var.bstrVal;
                
                // Check if this is the device we're looking for
                if (currentDevice.find(wideDeviceName) != std::wstring::npos) {
                    // Bind to filter
                    hr = moniker->BindToObject(0, 0, IID_IBaseFilter, (void**)&captureFilter);
                    if (SUCCEEDED(hr)) {
                        VariantClear(&var);
                        moniker->Release();
                        std::wcout << L"[VirtualCamera] Found device: " << currentDevice << std::endl;
                        return captureFilter;
                    }
                }
                VariantClear(&var);
            }
        }
        moniker->Release();
    }
    
    return nullptr;
}

bool VirtualCameraCapture::StartCapture() {
    if (!m_mediaControl) {
        std::cerr << "[VirtualCamera] Media control not initialized" << std::endl;
        return false;
    }
    
    std::cout << "[VirtualCamera] Starting capture..." << std::endl;
    
    // Start the graph
    HRESULT hr = m_mediaControl->Run();
    if (FAILED(hr)) {
        std::cerr << "[VirtualCamera] Failed to start capture: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    m_isCapturing = true;
    std::cout << "[VirtualCamera] Capture started successfully" << std::endl;
    
    return true;
}

void VirtualCameraCapture::StopCapture() {
    if (m_mediaControl && m_isCapturing) {
        std::cout << "[VirtualCamera] Stopping capture..." << std::endl;
        m_mediaControl->Stop();
        m_isCapturing = false;
    }
}

bool VirtualCameraCapture::WaitForNextFrame() {
    static int frameCheckCount = 0;
    frameCheckCount++;
    
    if (!m_isCapturing) {
        return false;
    }
    
    // Check if new frame is available
    bool hasFrame = m_frameAvailable.exchange(false);
    
    if (hasFrame) {
        // Copy frame buffer to D3D11 texture
        std::lock_guard<std::mutex> lock(m_frameMutex);
        
        D3D11_MAPPED_SUBRESOURCE mapped;
        HRESULT hr = m_d3dContext->Map(m_stagingTexture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (SUCCEEDED(hr)) {
            // Copy frame data to texture
            uint8_t* dst = (uint8_t*)mapped.pData;
            uint8_t* src = m_frameBuffer.get();
            
            for (int y = 0; y < m_height; y++) {
                memcpy(dst + y * mapped.RowPitch, 
                       src + y * m_width * 4, 
                       m_width * 4);
            }
            
            m_d3dContext->Unmap(m_stagingTexture.Get(), 0);
        }
        
        if (frameCheckCount <= 10 || frameCheckCount % 100 == 0) {
            std::cout << "[VirtualCamera] Frame #" << frameCheckCount << " received and uploaded" << std::endl;
        }
        
        return true;
    }
    
    return false;
}

void VirtualCameraCapture::OnFrameReceived(BYTE* pBuffer, long bufferLen) {
    if (bufferLen != m_frameBufferSize) {
        static bool warnedOnce = false;
        if (!warnedOnce) {
            std::cerr << "[VirtualCamera] Warning: Frame size mismatch. Expected: " 
                      << m_frameBufferSize << ", Got: " << bufferLen << std::endl;
            warnedOnce = true;
        }
        return;
    }
    
    // Copy frame to buffer
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        memcpy(m_frameBuffer.get(), pBuffer, bufferLen);
    }
    
    // Signal frame available
    m_frameAvailable.store(true);
    
    static int frameCount = 0;
    frameCount++;
    if (frameCount <= 10 || frameCount % 100 == 0) {
        std::cout << "[VirtualCamera] Frame callback #" << frameCount << std::endl;
    }
}

// SampleGrabberCallback implementation
STDMETHODIMP VirtualCameraCapture::SampleGrabberCallback::QueryInterface(REFIID riid, void** ppv) {
    if (riid == IID_ISampleGrabberCB || riid == IID_IUnknown) {
        *ppv = (void*)static_cast<ISampleGrabberCB*>(this);
        AddRef();
        return S_OK;
    }
    return E_NOINTERFACE;
}

STDMETHODIMP_(ULONG) VirtualCameraCapture::SampleGrabberCallback::Release() {
    ULONG refCount = InterlockedDecrement(&m_refCount);
    if (refCount == 0) {
        delete this;
    }
    return refCount;
}

STDMETHODIMP VirtualCameraCapture::SampleGrabberCallback::SampleCB(double SampleTime, IMediaSample* pSample) {
    // Not used, we use BufferCB instead
    return S_OK;
}

STDMETHODIMP VirtualCameraCapture::SampleGrabberCallback::BufferCB(double SampleTime, BYTE* pBuffer, long BufferLen) {
    if (m_parent) {
        m_parent->OnFrameReceived(pBuffer, BufferLen);
    }
    return S_OK;
}