#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <atomic>
#include <string>
#include <mutex>
#include <dshow.h>

#pragma comment(lib, "strmiids.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")

using Microsoft::WRL::ComPtr;

// Define missing DirectShow interfaces if not available
#ifndef __ISampleGrabberCB_INTERFACE_DEFINED__
#define __ISampleGrabberCB_INTERFACE_DEFINED__

MIDL_INTERFACE("0579154A-2B53-4994-B0D0-E773148EFF85")
ISampleGrabberCB : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE SampleCB(double SampleTime, IMediaSample *pSample) = 0;
    virtual HRESULT STDMETHODCALLTYPE BufferCB(double SampleTime, BYTE *pBuffer, long BufferLen) = 0;
};

#endif // __ISampleGrabberCB_INTERFACE_DEFINED__

#ifndef __ISampleGrabber_INTERFACE_DEFINED__
#define __ISampleGrabber_INTERFACE_DEFINED__

MIDL_INTERFACE("6B652FFF-11FE-4fce-92AD-0266B5D7C78F")
ISampleGrabber : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE SetOneShot(BOOL OneShot) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetMediaType(const AM_MEDIA_TYPE *pType) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetConnectedMediaType(AM_MEDIA_TYPE *pType) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetBufferSamples(BOOL BufferThem) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentBuffer(long *pBufferSize, long *pBuffer) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentSample(IMediaSample **ppSample) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetCallback(ISampleGrabberCB *pCallback, long WhichMethodToCallback) = 0;
};

#endif // __ISampleGrabber_INTERFACE_DEFINED__

// Define CLSIDs if not defined
#ifndef CLSID_SampleGrabber
const CLSID CLSID_SampleGrabber = {0xC1F400A0, 0x3F08, 0x11d3, {0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37}};
#endif

#ifndef CLSID_NullRenderer
const CLSID CLSID_NullRenderer = {0xC1F400A4, 0x3F08, 0x11d3, {0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37}};
#endif

class VirtualCameraCapture {
private:
    // DirectShow components
    ComPtr<IGraphBuilder> m_graphBuilder;
    ComPtr<ICaptureGraphBuilder2> m_captureGraphBuilder;
    ComPtr<IBaseFilter> m_sourceFilter;
    ComPtr<IBaseFilter> m_sampleGrabberFilter;
    ComPtr<ISampleGrabber> m_sampleGrabber;
    ComPtr<IMediaControl> m_mediaControl;
    ComPtr<IMediaEvent> m_mediaEvent;
    
    // D3D11 components for GPU interop
    ComPtr<ID3D11Device> m_d3dDevice;
    ComPtr<ID3D11DeviceContext> m_d3dContext;
    ComPtr<ID3D11Texture2D> m_stagingTexture;
    
    // CUDA interop
    cudaGraphicsResource_t m_cudaResource;
    cudaStream_t m_captureStream;
    
    // Capture properties
    int m_width;
    int m_height;
    std::atomic<bool> m_isCapturing;
    std::atomic<bool> m_frameAvailable;
    
    // Frame buffer
    std::unique_ptr<uint8_t[]> m_frameBuffer;
    size_t m_frameBufferSize;
    std::mutex m_frameMutex;
    
    // Callback class for sample grabber
    class SampleGrabberCallback : public ISampleGrabberCB {
    public:
        VirtualCameraCapture* m_parent;
        ULONG m_refCount;
        
        SampleGrabberCallback(VirtualCameraCapture* parent) : m_parent(parent), m_refCount(1) {}
        
        // IUnknown methods
        STDMETHODIMP QueryInterface(REFIID riid, void** ppv);
        STDMETHODIMP_(ULONG) AddRef() { return InterlockedIncrement(&m_refCount); }
        STDMETHODIMP_(ULONG) Release();
        
        // ISampleGrabberCB methods
        STDMETHODIMP SampleCB(double SampleTime, IMediaSample* pSample);
        STDMETHODIMP BufferCB(double SampleTime, BYTE* pBuffer, long BufferLen);
    };
    
    SampleGrabberCallback* m_callback;
    
    bool InitializeD3D11();
    bool InitializeCUDAInterop();
    bool InitializeDirectShow(const std::string& cameraName);
    IBaseFilter* FindCaptureDevice(const std::string& deviceName);
    
public:
    VirtualCameraCapture(int width, int height);
    ~VirtualCameraCapture();
    
    bool Initialize(const std::string& cameraName = "OBS Virtual Camera");
    bool StartCapture();
    void StopCapture();
    bool WaitForNextFrame();
    
    cudaGraphicsResource_t GetCudaResource() const { return m_cudaResource; }
    bool IsCapturing() const { return m_isCapturing; }
    
    void OnFrameReceived(BYTE* pBuffer, long bufferLen);
};