#ifndef DUPLICATION_API_CAPTURE_H
#define DUPLICATION_API_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <memory>
#include <iostream>
#include <iomanip>

#include "capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

struct ID3D11Device;
struct ID3D11DeviceContext;
struct IDXGIOutput1;
struct IDXGIOutputDuplication;
struct ID3D11Texture2D;
struct _DXGI_OUTDUPL_DESC;
struct cudaGraphicsResource;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

class DDAManager;

class DuplicationAPIScreenCapture : public IScreenCapture
{
public:
    DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight);
    ~DuplicationAPIScreenCapture();
    cv::cuda::GpuMat GetNextFrameGpu() override;
    cv::Mat GetNextFrameCpu() override;
    cudaEvent_t GetCaptureDoneEvent() const override;
    bool IsInitialized() const { return m_initialized; }

private:
    std::unique_ptr<DDAManager> m_ddaManager;

    ID3D11Device *d3dDevice = nullptr;
    ID3D11DeviceContext *d3dContext = nullptr;
    IDXGIOutputDuplication *deskDupl = nullptr;
    ID3D11Texture2D *stagingTexture = nullptr;
    IDXGIOutput1 *output1 = nullptr;
    ID3D11Texture2D *sharedTexture = nullptr;
    cudaGraphicsResource *cudaResource = nullptr;
    cudaStream_t cudaStream = nullptr;
    cudaEvent_t m_captureDoneEvent;

    int screenWidth = 0;
    int screenHeight = 0;
    int regionWidth = 0;
    int regionHeight = 0;

    cv::cuda::GpuMat m_previousFrame;
    bool m_initialized = false;
};

#endif