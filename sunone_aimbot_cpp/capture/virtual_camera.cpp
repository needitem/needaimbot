#include "virtual_camera.h"
#include "sunone_aimbot_cpp.h"

VirtualCameraCapture::VirtualCameraCapture(int width, int height)
    : captureWidth(width)
    , captureHeight(height)
    , cap(nullptr)
{
    captureWidth = (width % 2 == 0) ? width : width + 1;
    captureHeight = (height % 2 == 0) ? height : height + 1;

    std::vector<std::string> cameras = GetAvailableVirtualCameras();

    int cameraIndex = -1;
    for (int i = 0; i < cameras.size(); ++i)
    {
        if (cameras[i] == config.virtual_camera_name)
        {
            cameraIndex = i;
            break;
        }
    }

    if (cameraIndex == -1 && !cameras.empty())
    {
        cameraIndex = 0;
        config.virtual_camera_name = cameras[0];
        config.saveConfig();
    }

    if (cameraIndex != -1)
    {
        std::vector<int> backends = {
            cv::CAP_DSHOW,
            cv::CAP_MSMF,
            cv::CAP_ANY
        };

        bool cameraOpened = false;
        for (int backend : backends)
        {
            cap = new cv::VideoCapture(cameraIndex, backend);

            if (cap->isOpened())
            {
                cameraOpened = true;
                break;
            }

            delete cap;
            cap = nullptr;
        }

        if (cameraOpened)
        {
            cap->set(cv::CAP_PROP_FRAME_WIDTH, captureWidth);
            cap->set(cv::CAP_PROP_FRAME_HEIGHT, captureHeight);

            double actualWidth = cap->get(cv::CAP_PROP_FRAME_WIDTH);
            double actualHeight = cap->get(cv::CAP_PROP_FRAME_HEIGHT);

            if (config.verbose)
            {
                std::cout << "[Virtual camera] Requested size: " << captureWidth << "x" << captureHeight << std::endl;
                std::cout << "[Virtual camera] Actual camera size: " << actualWidth << "x" << actualHeight << std::endl;
            }
        }
        else
        {
            std::cerr << "[Virtual camera] Error: Could not open virtual camera with any backend" << std::endl;
        }
    }
}

VirtualCameraCapture::~VirtualCameraCapture()
{
    if (cap)
    {
        cap->release();
        delete cap;
    }
}

cv::Mat VirtualCameraCapture::GetNextFrameCpu()
{
    if (!cap || !cap->isOpened())
    {
        return cv::Mat();
    }

    cv::Mat frame;
    if (!cap->read(frame))
    {
        return cv::Mat();
    }

    if (frame.empty())
    {
        return cv::Mat();
    }

    cv::Mat processedFrame;
    try
    {
        // Ensure 3 channels (BGR) for consistency
        if (frame.channels() == 1)
        {
            cv::cvtColor(frame, processedFrame, cv::COLOR_GRAY2BGR);
        }
        else if (frame.channels() == 4)
        {
            cv::cvtColor(frame, processedFrame, cv::COLOR_BGRA2BGR);
        }
        else if (frame.channels() == 3)
        {
            processedFrame = frame.clone();
        }
        else
        {
            std::cerr << "[Virtual camera] Unexpected number of channels: " << frame.channels() << std::endl;
            return cv::Mat();
        }

        // Resize to the desired capture size
        cv::Mat resizedFrame;
        cv::resize(processedFrame, resizedFrame,
            cv::Size(captureWidth, captureHeight),
            0, 0, cv::INTER_LINEAR);

        // Ensure even dimensions if necessary (though maybe less critical for CPU path)
        if (resizedFrame.cols % 2 != 0 || resizedFrame.rows % 2 != 0)
        {
            cv::Mat evenFrame;
            cv::resize(resizedFrame, evenFrame,
                cv::Size(
                    resizedFrame.cols + (resizedFrame.cols % 2),
                    resizedFrame.rows + (resizedFrame.rows % 2)
                ),
                0, 0, cv::INTER_LINEAR);
            resizedFrame = evenFrame;
        }

        return resizedFrame;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[Virtual camera] OpenCV exception in GetNextFrameCpu: " << e.what() << std::endl;
        return cv::Mat();
    }
}

cv::cuda::GpuMat VirtualCameraCapture::GetNextFrameGpu()
{
    cv::Mat cpuFrame = GetNextFrameCpu();
    if (cpuFrame.empty())
    {
        return cv::cuda::GpuMat();
    }

    cv::cuda::GpuMat gpuFrame;
    try
    {
        gpuFrame.upload(cpuFrame);
        return gpuFrame;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[Virtual camera] OpenCV exception in GetNextFrameGpu (upload): " << e.what() << std::endl;
        return cv::cuda::GpuMat();
    }
}

std::vector<std::string> VirtualCameraCapture::GetAvailableVirtualCameras()
{
    std::vector<std::string> cameras;

    std::vector<int> backends = {
        cv::CAP_DSHOW,
        cv::CAP_MSMF,
        cv::CAP_ANY
    };

    for (int backend : backends)
    {
        for (int i = 0; i < 10; ++i)
        {
            try
            {
                cv::VideoCapture testCap(i, backend);

                if (testCap.isOpened())
                {
                    std::string deviceName = "Camera " + std::to_string(i) + ":" +
                        (backend == cv::CAP_DSHOW ? "DirectShow" :
                            backend == cv::CAP_MSMF ? "MSMF" : "Any");

                    cameras.push_back(deviceName);

                    testCap.release();
                }
            }
            catch (...)
            {
                continue;
            }
        }
    }

    std::cout << "[Virtual camera] Available cameras:" << std::endl;
    for (const auto& camera : cameras)
    {
        std::cout << camera << std::endl;
    }

    return cameras;
}