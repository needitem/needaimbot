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
    void manageOpticalFlowThread(); 
    void getAngularVelocity(double& angularVelocityXOut, double& angularVelocityYOut);
    void getAngularAcceleration(double& angularAccelerationXOut, double& angularAccelerationYOut);

    bool isOpticalFlowValid() const; 
    bool isThreadRunning() const;    

    std::pair<double, double> getAverageGlobalFlow();
    cv::cuda::GpuMat flow; 
    std::atomic<bool> isFlowValidAtomic; 

private:
    void computeOpticalFlow(const cv::cuda::GpuMat& frame);
    void opticalFlowLoop();
    void preprocessFrame(cv::cuda::GpuMat& frameGray);

    std::thread m_opticalFlowThread; 
    std::atomic<bool> m_shouldExit;

    std::queue<cv::cuda::GpuMat> m_frameQueue;
    std::condition_variable m_frameCV;
    std::mutex m_frameMutex;

    cv::cuda::GpuMat m_prevFrameGray;
    std::mutex m_flowMutex; 
    int m_xShift; 
    int m_yShift; 

    cv::cuda::GpuMat m_hintFlow;
    int m_flowWidth, m_flowHeight;
    int m_hintWidth, m_hintHeight;
    int m_outputGridSizeValue;
    int m_hintGridSizeValue;

    double m_prevTime;
    double m_prevAngularVelocityX;
    double m_prevAngularVelocityY;
    double m_angularAccelerationX; 
    double m_angularAccelerationY; 
    double m_prevPixelFlowX;
    double m_prevPixelFlowY;

    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> m_cvOpticalFlow; 
};

#endif 
