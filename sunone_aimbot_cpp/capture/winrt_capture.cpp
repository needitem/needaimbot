#include "sunone_aimbot_cpp.h"
#include "winrt_capture.h"
#include "config.h"
#include "other_tools.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

WinRTScreenCapture::WinRTScreenCapture(int desiredWidth, int desiredHeight)
{
    regionWidth = desiredWidth;
    regionHeight = desiredHeight;

    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    winrt::check_hresult(
        D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            0,
            createDeviceFlags,
            featureLevels,
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION,
            d3dDevice.put(),
            nullptr,
            d3dContext.put()
        )
    );

    winrt::com_ptr<IDXGIDevice> dxgiDevice;
    winrt::check_hresult(d3dDevice->QueryInterface(IID_PPV_ARGS(dxgiDevice.put())));

    device = CreateDirect3DDevice(dxgiDevice.get());
    if (!device)
    {
        throw std::runtime_error("[WinRTCapture] Failed to create IDirect3DDevice");
    }

    HMONITOR hMonitor = GetMonitorHandleByIndex(config.monitor_idx);
    if (!hMonitor)
    {
        throw std::runtime_error("[WinRTCapture] Failed to get HMONITOR for monitor_idx");
    }

    captureItem = CreateCaptureItemForMonitor(hMonitor);
    if (!captureItem)
    {
        throw std::runtime_error("[WinRTCapture] Can't create GraphicsCaptureItem for monitor.");
    }

    screenWidth = captureItem.Size().Width;
    screenHeight = captureItem.Size().Height;

    regionX = (screenWidth - regionWidth) / 2;
    regionY = (screenHeight - regionHeight) / 2;

    framePool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
        device,
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
        3,
        captureItem.Size()
    );

    session = framePool.CreateCaptureSession(captureItem);

    if (!config.capture_borders)
    {
        session.IsBorderRequired(false);
    }
    if (!config.capture_cursor)
    {
        session.IsCursorCaptureEnabled(false);
    }

    D3D11_TEXTURE2D_DESC sharedTexDesc = {};
    sharedTexDesc.Width = regionWidth;
    sharedTexDesc.Height = regionHeight;
    sharedTexDesc.MipLevels = 1;
    sharedTexDesc.ArraySize = 1;
    sharedTexDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sharedTexDesc.SampleDesc.Count = 1;
    sharedTexDesc.Usage = D3D11_USAGE_DEFAULT;
    sharedTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    sharedTexDesc.CPUAccessFlags = 0;
    sharedTexDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    HRESULT hr = d3dDevice->CreateTexture2D(&sharedTexDesc, nullptr, sharedTexture.put());
    if (FAILED(hr))
    {
        throw std::runtime_error("[WinRTCapture] Failed to create sharedTexture for subresource copy.");
    }

    cudaError_t err = cudaGraphicsD3D11RegisterResource(&cudaResource, sharedTexture.get(), cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("[WinRTCapture] cudaGraphicsD3D11RegisterResource failed.");
    }

    cudaStreamCreate(&cudaStream);

    session.StartCapture();
}

WinRTScreenCapture::~WinRTScreenCapture()
{
    if (cudaResource)
    {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }
    if (cudaStream)
    {
        cudaStreamDestroy(cudaStream);
        cudaStream = nullptr;
    }

    if (session)
    {
        session.Close();
        session = nullptr;
    }
    if (framePool)
    {
        framePool.Close();
        framePool = nullptr;
    }

    sharedTexture = nullptr;
    d3dContext = nullptr;
    d3dDevice = nullptr;
}

cv::cuda::GpuMat WinRTScreenCapture::GetNextFrameGpu()
{
    try
    {
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame lastFrame{ nullptr };
        while (auto tempFrame = framePool.TryGetNextFrame())
        {
            lastFrame = tempFrame;
        }

        if (!lastFrame)
        {
            return cv::cuda::GpuMat();
        }

        auto frameSurface = lastFrame.Surface();
        winrt::com_ptr<ID3D11Texture2D> frameTexture =
            GetDXGIInterfaceFromObject<ID3D11Texture2D>(frameSurface);

        if (!frameTexture)
        {
            throw std::runtime_error("[WinRTCapture] Can't query ID3D11Texture2D from frame surface.");
        }

        D3D11_BOX sourceRegion;
        sourceRegion.left = regionX;
        sourceRegion.top = regionY;
        sourceRegion.front = 0;
        sourceRegion.right = regionX + regionWidth;
        sourceRegion.bottom = regionY + regionHeight;
        sourceRegion.back = 1;

        d3dContext->CopySubresourceRegion(
            sharedTexture.get(),  // pDstResource
            0,                    // DstSubresource
            0, 0, 0,             // DstX, DstY, DstZ
            frameTexture.get(),   // pSrcResource
            0,                    // SrcSubresource
            &sourceRegion
        );

        cudaGraphicsMapResources(1, &cudaResource, cudaStream);

        cudaArray_t cuArray = nullptr;
        cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

        cv::cuda::GpuMat frameGpu(regionHeight, regionWidth, CV_8UC4);

        size_t rowBytes = regionWidth * sizeof(uchar4);
        cudaMemcpy2DFromArrayAsync(
            frameGpu.data,     // dst
            frameGpu.step,     // dpitch
            cuArray,           // src
            0, 0,              // wOffset, hOffset
            rowBytes,          // width in bytes
            regionHeight,      // height
            cudaMemcpyDeviceToDevice,
            cudaStream
        );

        cudaGraphicsUnmapResources(1, &cudaResource, cudaStream);
        cudaStreamSynchronize(cudaStream);

        return frameGpu;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[WinRTCapture] Exception in GetNextFrameGpu(): " << e.what() << std::endl;
        return cv::cuda::GpuMat();
    }
}

cv::Mat WinRTScreenCapture::GetNextFrameCpu()
{
    try
    {
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame lastFrame{ nullptr };
        while (auto tempFrame = framePool.TryGetNextFrame())
        {
            lastFrame = tempFrame;
        }

        if (!lastFrame)
        {
            return cv::Mat(); // Return empty CPU Mat on timeout/no frame
        }

        auto frameSurface = lastFrame.Surface();
        winrt::com_ptr<ID3D11Texture2D> frameTexture =
            GetDXGIInterfaceFromObject<ID3D11Texture2D>(frameSurface);

        if (!frameTexture)
        {
            throw std::runtime_error("[WinRTCapture] Can't query ID3D11Texture2D from frame surface (CPU Path).");
        }

        D3D11_BOX sourceRegion;
        sourceRegion.left = regionX;
        sourceRegion.top = regionY;
        sourceRegion.front = 0;
        sourceRegion.right = regionX + regionWidth;
        sourceRegion.bottom = regionY + regionHeight;
        sourceRegion.back = 1;

        d3dContext->CopySubresourceRegion(
            sharedTexture.get(), 0, 0, 0, 0,
            frameTexture.get(), 0, &sourceRegion);

        // --- CUDA Mapping and Copy to CPU --- 
        cv::Mat frameCpu(regionHeight, regionWidth, CV_8UC4); // Create CPU Mat
        cudaError_t err;

        err = cudaGraphicsMapResources(1, &cudaResource, cudaStream);
        if (err != cudaSuccess) {
            std::cerr << "[WinRTCapture] cudaGraphicsMapResources (CPU Path) error: " << cudaGetErrorString(err) << std::endl;
            return cv::Mat(); // Return empty Mat on error
        }

        cudaArray_t cuArray = nullptr;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
        if (err != cudaSuccess) {
            std::cerr << "[WinRTCapture] cudaGraphicsSubResourceGetMappedArray (CPU Path) error: " << cudaGetErrorString(err) << std::endl;
            cudaGraphicsUnmapResources(1, &cudaResource, cudaStream); // Unmap even on error here
            return cv::Mat();
        }

        // Copy from CUDA array to Host (CPU) memory
        err = cudaMemcpy2DFromArrayAsync(
            frameCpu.data,        // Destination: CPU Mat data pointer
            frameCpu.step,        // Destination: CPU Mat step (bytes per row)
            cuArray,              // Source: Mapped CUDA array
            0, 0,                 // Source X, Y offset
            regionWidth * 4,      // Width in bytes (BGRA)
            regionHeight,         // Height
            cudaMemcpyDeviceToHost, // Copy direction
            cudaStream);

        if (err == cudaSuccess)
        {
            // Synchronize the stream to ensure the copy is complete before returning
            err = cudaStreamSynchronize(cudaStream);
            if (err != cudaSuccess) {
                 std::cerr << "[WinRTCapture] cudaStreamSynchronize (CPU Path) error: " << cudaGetErrorString(err) << std::endl;
                 // Mat might be partially filled, release it to be safe
                 frameCpu.release();
            }
        }
        else
        {
            std::cerr << "[WinRTCapture] cudaMemcpy2DFromArrayAsync (D->H) error: " << cudaGetErrorString(err) << std::endl;
            frameCpu.release(); // Release the mat on copy error
        }

        // Unmap the resource regardless of copy success/failure (if mapping succeeded)
        cudaGraphicsUnmapResources(1, &cudaResource, cudaStream);

        return frameCpu;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[WinRTCapture] Exception in GetNextFrameCpu(): " << e.what() << std::endl;
        return cv::Mat();
    }
    catch (const winrt::hresult_error& e)
    {
        std::cerr << "[WinRTCapture] WinRT hresult_error in GetNextFrameCpu(): " << winrt::to_string(e.message()) << " (Code: 0x" << std::hex << e.code() << ")" << std::endl;
        return cv::Mat();
    }
}

winrt::Windows::Graphics::Capture::GraphicsCaptureItem
WinRTScreenCapture::CreateCaptureItemForMonitor(HMONITOR hMonitor)
{
    auto interopFactory = winrt::get_activation_factory<
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem,
        IGraphicsCaptureItemInterop
    >();

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
    HRESULT hr = interopFactory->CreateForMonitor(
        hMonitor,
        winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
        winrt::put_abi(item)
    );
    if (FAILED(hr))
    {
        throw std::runtime_error("[WinRTCapture] CreateForMonitor failed. HR=" + std::to_string(hr));
    }
    return item;
}

winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice
WinRTScreenCapture::CreateDirect3DDevice(IDXGIDevice* dxgiDevice)
{
    winrt::com_ptr<::IInspectable> inspectable;
    winrt::check_hresult(
        CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, inspectable.put())
    );
    return inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
}

template<typename T>
winrt::com_ptr<T>
WinRTScreenCapture::GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object)
{
    auto dxgiInterfaceAccess = object.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<T> result = nullptr;

    winrt::check_hresult(
        dxgiInterfaceAccess->GetInterface(winrt::guid_of<T>(), result.put_void())
    );

    return result;
}