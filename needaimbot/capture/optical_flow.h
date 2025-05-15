#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>
#include <opencv2/cudaoptflow.hpp>

class OpticalFlow
{
public:
    OpticalFlow();

    void startOpticalFlowThread();
    void stopOpticalFlowThread();

    void enqueueFrame(const cv::cuda::GpuMat& frame);
    void getMotion(int& xShift, int& yShift);
    void manageOpticalFlowThread(); // Not present in sunone_aimbot_cpp, but seems like a good idea from the .cpp
    void getAngularVelocity(double& angularVelocityXOut, double& angularVelocityYOut);
    void getAngularAcceleration(double& angularAccelerationXOut, double& angularAccelerationYOut);

    bool isOpticalFlowValid() const; // Changed from isFlowValid to follow naming convention
    bool isThreadRunning() const;    // Added to check if thread is active

    std::pair<double, double> getAverageGlobalFlow();
    cv::cuda::GpuMat flow; // Public for drawing
    std::atomic<bool> isFlowValidAtomic; // Public for drawing, made atomic for thread safety

private:
    void computeOpticalFlow(const cv::cuda::GpuMat& frame);
    void opticalFlowLoop();
    void preprocessFrame(cv::cuda::GpuMat& frameGray);

    std::thread m_opticalFlowThread; // Prefixed member variables
    std::atomic<bool> m_shouldExit;

    std::queue<cv::cuda::GpuMat> m_frameQueue;
    std::condition_variable m_frameCV;
    std::mutex m_frameMutex;

    cv::cuda::GpuMat m_prevFrameGray;
    std::mutex m_flowMutex; // To protect flow and related calculation variables
    int m_xShift; // Not clear if still needed with angular velocity
    int m_yShift; // Not clear if still needed with angular velocity

    cv::cuda::GpuMat m_hintFlow;
    int m_flowWidth, m_flowHeight;
    int m_hintWidth, m_hintHeight;
    int m_outputGridSizeValue;
    int m_hintGridSizeValue;

    double m_prevTime;
    double m_prevAngularVelocityX;
    double m_prevAngularVelocityY;
    double m_angularAccelerationX; // Not fully implemented in source
    double m_angularAccelerationY; // Not fully implemented in source
    double m_prevPixelFlowX;
    double m_prevPixelFlowY;

    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> m_cvOpticalFlow; // Renamed to avoid conflict
};

#endif // OPTICAL_FLOW_H 