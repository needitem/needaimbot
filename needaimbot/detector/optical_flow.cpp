#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath> 
#include <iostream> 

#include "optical_flow.h"
#include "needaimbot.h" 
#include "capture.h" 




void OpticalFlow::preprocessFrame(cv::cuda::GpuMat& frameGray)
{
    if (frameGray.empty()) return;
    
    cv::cuda::GpuMat tmp;
    // static Gaussian filter to avoid reallocation
    static cv::Ptr<cv::cuda::Filter> gaussFilter = cv::cuda::createGaussianFilter(frameGray.type(), frameGray.type(), cv::Size(3, 3), 0);
    gaussFilter->apply(frameGray, tmp);
    
    cv::cuda::equalizeHist(tmp, tmp);
    
    cv::cuda::threshold(tmp, frameGray, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

OpticalFlow::OpticalFlow() : 
    m_shouldExit(false),
    isFlowValidAtomic(false), 
    m_xShift(0), m_yShift(0), 
    m_cvOpticalFlow(nullptr),
    m_prevTime(0.0),
    m_prevAngularVelocityX(0.0), m_prevAngularVelocityY(0.0),
    m_angularAccelerationX(0.0), m_angularAccelerationY(0.0),
    m_prevPixelFlowX(0.0), m_prevPixelFlowY(0.0),
    m_flowWidth(0), m_flowHeight(0),
    m_hintWidth(0), m_hintHeight(0),
    m_outputGridSizeValue(0), m_hintGridSizeValue(0)
{
}

void OpticalFlow::computeOpticalFlow(const cv::cuda::GpuMat& frame)
{
    isFlowValidAtomic = false;

    if (frame.empty()) {
        std::cerr << "OpticalFlow Error: Input frame is empty." << std::endl;
        return;
    }
    
    cv::cuda::GpuMat frameGray;
    if (frame.channels() == 4) 
    {
        
        cv::cuda::cvtColor(frame, frameGray, cv::COLOR_BGRA2GRAY); 
    }
    else if (frame.channels() == 3)
    {
        cv::cuda::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
    }
    else if (frame.channels() == 1)
    {
        frameGray = frame.clone(); 
    }
    else
    {
        std::cerr << "OpticalFlow Error: Frame has unsupported number of channels: " << frame.channels() << std::endl;
        return;
    }

    
    preprocessFrame(frameGray);

    static cv::cuda::GpuMat prevStaticCheck;
    
    float staticThreshold = config.staticFrameThreshold; 

    if (!prevStaticCheck.empty() && frameGray.size() == prevStaticCheck.size() && frameGray.type() == prevStaticCheck.type())
    {
        cv::cuda::GpuMat diffFrame;
        cv::cuda::absdiff(frameGray, prevStaticCheck, diffFrame);
        
        cv::Scalar sumDiff = cv::cuda::sum(diffFrame);
        float meanDiff = static_cast<float>(sumDiff[0] / (diffFrame.rows * diffFrame.cols));

        if (meanDiff < staticThreshold)
        {
            
            std::lock_guard<std::mutex> lock(m_flowMutex);
            flow.release(); 
            m_prevFrameGray.release(); 
            isFlowValidAtomic = false;
            
            m_prevAngularVelocityX = 0.0;
            m_prevAngularVelocityY = 0.0;
            m_prevPixelFlowX = 0.0;
            m_prevPixelFlowY = 0.0;
            prevStaticCheck = frameGray; 
            return;
        }
    }
    else if (prevStaticCheck.empty()) {
         
        prevStaticCheck = frameGray;
    }


    prevStaticCheck = frameGray.clone();
    prevStaticCheck = frameGray;

    if (!m_prevFrameGray.empty() && (m_prevFrameGray.size() != frameGray.size() || m_prevFrameGray.type() != frameGray.type()))
    {
        
        std::lock_guard<std::mutex> lock(m_flowMutex);
        m_prevFrameGray.release();
        if (m_cvOpticalFlow) m_cvOpticalFlow.release(); 
        m_cvOpticalFlow = nullptr; 
        flow.release();
        isFlowValidAtomic = false;
        std::cout << "OpticalFlow: Resolution changed. Reinitializing." << std::endl;
    }

    if (!m_prevFrameGray.empty())
    {
        if (!m_cvOpticalFlow)
        {
            cv::Size imageSize(frameGray.cols, frameGray.rows);
            if (imageSize.width == 0 || imageSize.height == 0) {
                std::cerr << "OpticalFlow Error: Invalid image size for initialization." << std::endl;
                return;
            }
            
            
            auto perfPreset = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_MEDIUM; 
            auto outputGridSize = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
            auto hintGridSize = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_4;
            bool enableTemporalHints = true;
            bool enableExternalHints = false; 
            bool enableCostBuffer = false;   
            int gpuId = 0; 

            m_cvOpticalFlow = cv::cuda::NvidiaOpticalFlow_2_0::create(
                imageSize,
                perfPreset,
                outputGridSize,
                hintGridSize,
                enableTemporalHints,
                enableExternalHints,
                enableCostBuffer,
                gpuId
            );

            
            m_outputGridSizeValue = 4; 
            m_hintGridSizeValue = 4;   

            m_flowWidth = (imageSize.width + m_outputGridSizeValue - 1) / m_outputGridSizeValue;
            m_flowHeight = (imageSize.height + m_outputGridSizeValue - 1) / m_outputGridSizeValue;
            
            
            m_hintWidth = (imageSize.width + m_hintGridSizeValue - 1) / m_hintGridSizeValue;
            m_hintHeight = (imageSize.height + m_hintGridSizeValue - 1) / m_hintGridSizeValue;
            m_hintFlow.create(m_hintHeight, m_hintWidth, CV_16SC2); 
            m_hintFlow.setTo(cv::Scalar::all(0)); 

            std::cout << "OpticalFlow: NvidiaOpticalFlow_2_0 initialized. Flow map: " << m_flowWidth << "x" << m_flowHeight << std::endl;

        }

        try
        {
            
            std::lock_guard<std::mutex> lock(m_flowMutex);
            if (m_prevFrameGray.empty() || frameGray.empty() || !m_cvOpticalFlow) {
                 isFlowValidAtomic = false; return;
            }
            
            if (flow.empty() || flow.cols != m_flowWidth || flow.rows != m_flowHeight) {
                flow.create(m_flowHeight, m_flowWidth, CV_32FC2); 
            }

            m_cvOpticalFlow->calc(
                m_prevFrameGray,
                frameGray,
                flow, 
                cv::cuda::Stream::Null() 
                
            );
             isFlowValidAtomic = true; 
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "OpticalFlow Error during calc: " << e.what() << std::endl;
            std::lock_guard<std::mutex> lock(m_flowMutex);
            flow.release();
            isFlowValidAtomic = false;
            
            m_prevFrameGray.release(); 
            if (m_cvOpticalFlow) m_cvOpticalFlow.release();
            m_cvOpticalFlow = nullptr;
            return;
        }

        
        if (isFlowValidAtomic) {
            cv::Mat flowCpu;
            
            
            {
                std::lock_guard<std::mutex> lock(m_flowMutex); 
                if (flow.empty()) {
                    isFlowValidAtomic = false;
                    m_prevFrameGray = frameGray.clone(); 
                    return;
                }
                flow.download(flowCpu);
            }


            if (flowCpu.empty()) {
                isFlowValidAtomic = false;
                m_prevFrameGray = frameGray.clone();
                return;
            }

            int width = flowCpu.cols;   
            int height = flowCpu.rows; 

            
            double magnitudeScale = 1.0; 
            double dynamicThreshold = config.optical_flow_magnitudeThreshold; 

            double sumAngularVelocityX = 0.0;
            double sumAngularVelocityY = 0.0;
            int validPointsAngular = 0;

            double sumFlowX = 0.0;
            double sumFlowY = 0.0;
            int validPointsFlow = 0;

            for (int y_flow = 0; y_flow < height; y_flow++) 
            {
                for (int x_flow = 0; x_flow < width; x_flow++)
                {
                    cv::Point2f flowAtPoint = flowCpu.at<cv::Point2f>(y_flow, x_flow);

                    if (std::isfinite(flowAtPoint.x) && std::isfinite(flowAtPoint.y) &&
                        std::abs(flowAtPoint.x) < frameGray.cols && 
                        std::abs(flowAtPoint.y) < frameGray.rows)
                    {
                        double flowMagnitude = cv::norm(flowAtPoint);

                        if (flowMagnitude > dynamicThreshold)
                        {
                            
                            
                            
                            
                            
                            
                            
                            

                            
                            
                            
                            double angularVelocityX_rad = (flowAtPoint.x / static_cast<double>(frameGray.cols)) * (config.fovX * CV_PI / 180.0);
                            double angularVelocityY_rad = (flowAtPoint.y / static_cast<double>(frameGray.rows)) * (config.fovY * CV_PI / 180.0);
                            
                            
                            
                            


                            sumAngularVelocityX += angularVelocityX_rad; 
                            sumAngularVelocityY += angularVelocityY_rad;
                            validPointsAngular++;

                            sumFlowX += flowAtPoint.x;
                            sumFlowY += flowAtPoint.y;
                            validPointsFlow++;
                        }
                    }
                }
            }
            
            
            std::lock_guard<std::mutex> lock(m_flowMutex);
            double currentTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
            double deltaTime = (m_prevTime == 0.0) ? (1.0/60.0) : (currentTime - m_prevTime); 
            deltaTime = std::max(deltaTime, 1e-6); 


            if (validPointsAngular > 0)
            {
                double newAngularVelocityX = sumAngularVelocityX / validPointsAngular; 
                double newAngularVelocityY = sumAngularVelocityY / validPointsAngular; 
                                                                                       
                                                                                       
                
                
                m_prevAngularVelocityX = 0.7 * m_prevAngularVelocityX + 0.3 * newAngularVelocityX;
                m_prevAngularVelocityY = 0.7 * m_prevAngularVelocityY + 0.3 * newAngularVelocityY;

                
                double maxAngularVelPerFrame = 1.0; 
                m_prevAngularVelocityX = std::clamp(m_prevAngularVelocityX, -maxAngularVelPerFrame, maxAngularVelPerFrame);
                m_prevAngularVelocityY = std::clamp(m_prevAngularVelocityY, -maxAngularVelPerFrame, maxAngularVelPerFrame);
            } else {
                m_prevAngularVelocityX = 0.0;
                m_prevAngularVelocityY = 0.0;
            }

            if (validPointsFlow > 0)
            {
                double newFlowX = sumFlowX / validPointsFlow; 
                double newFlowY = sumFlowY / validPointsFlow;

                m_prevPixelFlowX = 0.7 * m_prevPixelFlowX + 0.3 * newFlowX;
                m_prevPixelFlowY = 0.7 * m_prevPixelFlowY + 0.3 * newFlowY;

                double maxFlow = static_cast<double>(frameGray.cols); 
                m_prevPixelFlowX = std::clamp(m_prevPixelFlowX, -maxFlow, maxFlow);
                m_prevPixelFlowY = std::clamp(m_prevPixelFlowY, -maxFlow, maxFlow);
            } else {
                m_prevPixelFlowX = 0.0;
                m_prevPixelFlowY = 0.0;
            }
            
            
            

            m_prevTime = currentTime;
        } 
    } 

    m_prevFrameGray = frameGray.clone(); 
    m_prevFrameGray = frameGray;
}


void OpticalFlow::getAngularVelocity(double& angularVelocityXOut, double& angularVelocityYOut)
{
    std::lock_guard<std::mutex> lock(m_flowMutex);
    angularVelocityXOut = m_prevAngularVelocityX; 
    angularVelocityYOut = m_prevAngularVelocityY;
}

void OpticalFlow::getAngularAcceleration(double& angularAccelerationXOut, double& angularAccelerationYOut)
{
    std::lock_guard<std::mutex> lock(m_flowMutex);
    
    angularAccelerationXOut = m_angularAccelerationX; 
    angularAccelerationYOut = m_angularAccelerationY;
}

std::pair<double, double> OpticalFlow::getAverageGlobalFlow()
{
    std::lock_guard<std::mutex> lock(m_flowMutex);
    return { m_prevPixelFlowX, m_prevPixelFlowY }; 
}



void OpticalFlow::getMotion(int& xShiftOut, int& yShiftOut)
{
    std::lock_guard<std::mutex> lock(m_flowMutex);
    xShiftOut = static_cast<int>(std::round(m_prevPixelFlowX));
    yShiftOut = static_cast<int>(std::round(m_prevPixelFlowY));
}

bool OpticalFlow::isOpticalFlowValid() const {
    return isFlowValidAtomic.load();
}

bool OpticalFlow::isThreadRunning() const {
    return m_opticalFlowThread.joinable();
}



void OpticalFlow::enqueueFrame(const cv::cuda::GpuMat& frame)
{
    if (!config.enable_optical_flow) return; 

    std::unique_lock<std::mutex> lock(m_frameMutex);
    if (m_frameQueue.size() < 5) 
    {
        // shallow copy to avoid deep GPU memory copy
        m_frameQueue.push(frame);
    }
    lock.unlock();
    m_frameCV.notify_one();
}

void OpticalFlow::opticalFlowLoop()
{
    std::cout << "OpticalFlow thread started." << std::endl;
    while (!m_shouldExit)
    {
        cv::cuda::GpuMat currentFrame;
        {
            std::unique_lock<std::mutex> lock(m_frameMutex);
            m_frameCV.wait(lock, [this] { return m_shouldExit || !m_frameQueue.empty(); });

            if (m_shouldExit) break;

            currentFrame = m_frameQueue.front();
            m_frameQueue.pop();
        }

        if (!currentFrame.empty())
        {
            computeOpticalFlow(currentFrame);
        }
    }
    std::cout << "OpticalFlow thread stopped." << std::endl;
}

void OpticalFlow::startOpticalFlowThread()
{
    if(m_opticalFlowThread.joinable()) {
        stopOpticalFlowThread(); 
    }
    m_shouldExit = false;
    m_opticalFlowThread = std::thread(&OpticalFlow::opticalFlowLoop, this);
}

void OpticalFlow::stopOpticalFlowThread()
{
    m_shouldExit = true;
    m_frameCV.notify_all(); 
    if (m_opticalFlowThread.joinable())
    {
        m_opticalFlowThread.join();
    }
     
    std::lock_guard<std::mutex> lock(m_flowMutex);
    if (m_cvOpticalFlow) m_cvOpticalFlow.release();
    m_cvOpticalFlow = nullptr;
    m_prevFrameGray.release();
    flow.release();
    m_hintFlow.release();
    isFlowValidAtomic = false;

    
    std::lock_guard<std::mutex> queueLock(m_frameMutex);
    std::queue<cv::cuda::GpuMat> empty;
    std::swap(m_frameQueue, empty);
}




void OpticalFlow::manageOpticalFlowThread()
{
    
    
    
    
    
    
    
    
}


