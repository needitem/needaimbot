#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath> // For std::isfinite, std::abs, std::clamp
#include <iostream> // For std::cerr

#include "optical_flow.h"
#include "needaimbot.h" // Changed from sunone_aimbot_cpp.h
#include "capture.h" // For latestFrameGpu potentially, and config access

// Forward declaration if config is not fully included via needaimbot.h
// extern Config config; 

void OpticalFlow::preprocessFrame(cv::cuda::GpuMat& frameGray)
{
    if (frameGray.empty()) return;

    // The original preprocessFrame converts to CPU, blurs, equalizes, thresholds, then uploads.
    // This can be slow. Consider performing more operations on GPU if possible.
    // For now, keeping original logic.

    cv::Mat frameCPU;
    frameGray.download(frameCPU);

    if (frameCPU.type() != CV_8UC1)
    {
        cv::cvtColor(frameCPU, frameCPU, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blurredFrame, equalizedFrame, thresholdFrame;

    // Parameters for these operations could be made configurable.
    cv::GaussianBlur(frameCPU, blurredFrame, cv::Size(3, 3), 0);
    cv::equalizeHist(blurredFrame, equalizedFrame);
    // cv::threshold(equalizedFrame, thresholdFrame, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); // OTSU might not always be best
    cv::adaptiveThreshold(equalizedFrame, thresholdFrame, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);


    frameGray.upload(thresholdFrame);
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
    if (frame.channels() == 4) // Handle 4-channel input (e.g., BGRA or RGBA)
    {
        // Assuming BGRA, common on Windows. If RGBA, use cv::COLOR_RGBA2GRAY
        cv::cuda::cvtColor(frame, frameGray, cv::COLOR_BGRA2GRAY); 
    }
    else if (frame.channels() == 3)
    {
        cv::cuda::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
    }
    else if (frame.channels() == 1)
    {
        frameGray = frame.clone(); // Clone to ensure it's continuous and owned if it's a ROI
    }
    else
    {
        std::cerr << "OpticalFlow Error: Frame has unsupported number of channels: " << frame.channels() << std::endl;
        return;
    }

    // Preprocessing can be intensive, ensure it's beneficial.
    // preprocessFrame(frameGray); // Consider making this optional via config

    static cv::cuda::GpuMat prevStaticCheck;
    // Ensure config.staticFrameThreshold is accessible. If not, use a default or pass config.
    float staticThreshold = config.staticFrameThreshold; 

    if (!prevStaticCheck.empty() && frameGray.size() == prevStaticCheck.size() && frameGray.type() == prevStaticCheck.type())
    {
        cv::cuda::GpuMat diffFrame;
        cv::cuda::absdiff(frameGray, prevStaticCheck, diffFrame);
        
        cv::Scalar sumDiff = cv::cuda::sum(diffFrame);
        float meanDiff = static_cast<float>(sumDiff[0] / (diffFrame.rows * diffFrame.cols));

        if (meanDiff < staticThreshold)
        {
            // Static frame detected
            std::lock_guard<std::mutex> lock(m_flowMutex);
            flow.release(); // Release the GpuMat
            m_prevFrameGray.release(); // Release previous frame to reinit on motion
            isFlowValidAtomic = false;
            // Reset velocities?
            m_prevAngularVelocityX = 0.0;
            m_prevAngularVelocityY = 0.0;
            m_prevPixelFlowX = 0.0;
            m_prevPixelFlowY = 0.0;
            prevStaticCheck = frameGray.clone(); // Update static check frame
            return;
        }
    }
    else if (prevStaticCheck.empty()) {
         // Initialize prevStaticCheck on the first valid frame
        prevStaticCheck = frameGray.clone();
    }


    prevStaticCheck = frameGray.clone();

    if (!m_prevFrameGray.empty() && (m_prevFrameGray.size() != frameGray.size() || m_prevFrameGray.type() != frameGray.type()))
    {
        // Resolution or type changed, reset optical flow
        std::lock_guard<std::mutex> lock(m_flowMutex);
        m_prevFrameGray.release();
        if (m_cvOpticalFlow) m_cvOpticalFlow.release(); // Release and reinitialize
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
            // Ensure config values for fovX, fovY are accessible
            // These parameters might need tuning
            auto perfPreset = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_MEDIUM; // Balanced
            auto outputGridSize = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
            auto hintGridSize = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_4;
            bool enableTemporalHints = true;
            bool enableExternalHints = false; // External hints not used in original
            bool enableCostBuffer = false;   // Cost buffer not used
            int gpuId = 0; // Assuming default GPU

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

            // Store grid sizes used for flow map dimensions
            m_outputGridSizeValue = 4; // Based on NV_OF_OUTPUT_VECTOR_GRID_SIZE_4
            m_hintGridSizeValue = 4;   // Based on NV_OF_HINT_VECTOR_GRID_SIZE_4

            m_flowWidth = (imageSize.width + m_outputGridSizeValue - 1) / m_outputGridSizeValue;
            m_flowHeight = (imageSize.height + m_outputGridSizeValue - 1) / m_outputGridSizeValue;
            
            // Hint flow initialization (though external hints are false, internal might use this structure)
            m_hintWidth = (imageSize.width + m_hintGridSizeValue - 1) / m_hintGridSizeValue;
            m_hintHeight = (imageSize.height + m_hintGridSizeValue - 1) / m_hintGridSizeValue;
            m_hintFlow.create(m_hintHeight, m_hintWidth, CV_16SC2); // Type for hints
            m_hintFlow.setTo(cv::Scalar::all(0)); // Initialize hints to zero

            std::cout << "OpticalFlow: NvidiaOpticalFlow_2_0 initialized. Flow map: " << m_flowWidth << "x" << m_flowHeight << std::endl;

        }

        try
        {
            // Lock before accessing/modifying flow GpuMat
            std::lock_guard<std::mutex> lock(m_flowMutex);
            if (m_prevFrameGray.empty() || frameGray.empty() || !m_cvOpticalFlow) {
                 isFlowValidAtomic = false; return;
            }
            // Ensure flow GpuMat is allocated before calc if not done by create() or if it was released
            if (flow.empty() || flow.cols != m_flowWidth || flow.rows != m_flowHeight) {
                flow.create(m_flowHeight, m_flowWidth, CV_32FC2); // Standard flow vector format
            }

            m_cvOpticalFlow->calc(
                m_prevFrameGray,
                frameGray,
                flow, // Output flow map
                cv::cuda::Stream::Null() // Using default stream
                // m_hintFlow // Pass hintFlow if enableExternalHints or enableTemporalHints is true and it's used as output too
            );
             isFlowValidAtomic = true; // Set true only after successful calculation
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "OpticalFlow Error during calc: " << e.what() << std::endl;
            std::lock_guard<std::mutex> lock(m_flowMutex);
            flow.release();
            isFlowValidAtomic = false;
            // Consider re-initializing m_cvOpticalFlow or m_prevFrameGray on certain errors
            m_prevFrameGray.release(); // Force re-grab of prev frame
            if (m_cvOpticalFlow) m_cvOpticalFlow.release();
            m_cvOpticalFlow = nullptr;
            return;
        }

        // Post-process flow if valid
        if (isFlowValidAtomic) {
            cv::Mat flowCpu;
            // This download can be slow, do we need it every frame in this thread?
            // Original code uses it to calculate angular velocities.
            {
                std::lock_guard<std::mutex> lock(m_flowMutex); // Protect flow GpuMat during download
                if (flow.empty()) {
                    isFlowValidAtomic = false;
                    m_prevFrameGray = frameGray.clone(); // Prepare for next frame
                    return;
                }
                flow.download(flowCpu);
            }


            if (flowCpu.empty()) {
                isFlowValidAtomic = false;
                m_prevFrameGray = frameGray.clone();
                return;
            }

            int width = flowCpu.cols;   // This is flow_width, not image_width
            int height = flowCpu.rows; // This is flow_height, not image_height

            // Ensure config values are accessible (fovX, fovY, optical_flow_magnitudeThreshold)
            double magnitudeScale = 1.0; // Original: width > height ? width / 640.0 : height / 640.0; Re-evaluate this scaling
            double dynamicThreshold = config.optical_flow_magnitudeThreshold; // Removed magnitudeScale for now, threshold should be absolute

            double sumAngularVelocityX = 0.0;
            double sumAngularVelocityY = 0.0;
            int validPointsAngular = 0;

            double sumFlowX = 0.0;
            double sumFlowY = 0.0;
            int validPointsFlow = 0;

            for (int y_flow = 0; y_flow < height; y_flow++) // Iterate through flow map
            {
                for (int x_flow = 0; x_flow < width; x_flow++)
                {
                    cv::Point2f flowAtPoint = flowCpu.at<cv::Point2f>(y_flow, x_flow);

                    if (std::isfinite(flowAtPoint.x) && std::isfinite(flowAtPoint.y) &&
                        std::abs(flowAtPoint.x) < frameGray.cols && // Flow vectors should be within image bounds (approx)
                        std::abs(flowAtPoint.y) < frameGray.rows)
                    {
                        double flowMagnitude = cv::norm(flowAtPoint);

                        if (flowMagnitude > dynamicThreshold)
                        {
                            // Convert flow vector to angular velocity using FOV
                            // This assumes flow vectors are pixel displacements per frame time
                            // And that the flow map grid maps reasonably to image space for FOV calc.
                            // The original normalized by flow map width/height, this might be error-prone.
                            // Let's use image width/height for normalization before FOV scaling.
                            // Effective pixel location for this flow vector (center of grid cell)
                            // float image_x = (x_flow + 0.5f) * m_outputGridSizeValue;
                            // float image_y = (y_flow + 0.5f) * m_outputGridSizeValue;

                            // Angular velocity contribution
                            // This part of original code seems to convert pixel flow to an angle using FoV.
                            // (flow_pixels / image_pixels_total_fov_span) * fov_degrees
                            double angularVelocityX_rad = (flowAtPoint.x / static_cast<double>(frameGray.cols)) * (config.fovX * CV_PI / 180.0);
                            double angularVelocityY_rad = (flowAtPoint.y / static_cast<double>(frameGray.rows)) * (config.fovY * CV_PI / 180.0);
                            
                            // Convert back to degrees for consistency if other parts of system use degrees
                            // double angularVelocityX_deg = angularVelocityX_rad * 180.0 / CV_PI;
                            // double angularVelocityY_deg = angularVelocityY_rad * 180.0 / CV_PI;


                            sumAngularVelocityX += angularVelocityX_rad; // Store in radians for now
                            sumAngularVelocityY += angularVelocityY_rad;
                            validPointsAngular++;

                            sumFlowX += flowAtPoint.x;
                            sumFlowY += flowAtPoint.y;
                            validPointsFlow++;
                        }
                    }
                }
            }
            
            // Lock before updating shared member variables
            std::lock_guard<std::mutex> lock(m_flowMutex);
            double currentTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
            double deltaTime = (m_prevTime == 0.0) ? (1.0/60.0) : (currentTime - m_prevTime); // Avoid div by zero, assume 60fps if first time
            deltaTime = std::max(deltaTime, 1e-6); // Prevent deltaTime from being too small or zero


            if (validPointsAngular > 0)
            {
                double newAngularVelocityX = sumAngularVelocityX / validPointsAngular; // This is average angular velocity per frame time
                double newAngularVelocityY = sumAngularVelocityY / validPointsAngular; // Should be divided by deltaTime to get rad/sec?
                                                                                       // Original code doesn't divide by deltaTime here.
                                                                                       // Let's assume these are "effective angular change per frame"
                
                // Smoothing (EMA - Exponential Moving Average)
                m_prevAngularVelocityX = 0.7 * m_prevAngularVelocityX + 0.3 * newAngularVelocityX;
                m_prevAngularVelocityY = 0.7 * m_prevAngularVelocityY + 0.3 * newAngularVelocityY;

                // Clamping to reasonable values (radians per frame?)
                double maxAngularVelPerFrame = 1.0; // Radians, adjust as needed
                m_prevAngularVelocityX = std::clamp(m_prevAngularVelocityX, -maxAngularVelPerFrame, maxAngularVelPerFrame);
                m_prevAngularVelocityY = std::clamp(m_prevAngularVelocityY, -maxAngularVelPerFrame, maxAngularVelPerFrame);
            } else {
                m_prevAngularVelocityX = 0.0;
                m_prevAngularVelocityY = 0.0;
            }

            if (validPointsFlow > 0)
            {
                double newFlowX = sumFlowX / validPointsFlow; // Average pixel flow
                double newFlowY = sumFlowY / validPointsFlow;

                m_prevPixelFlowX = 0.7 * m_prevPixelFlowX + 0.3 * newFlowX;
                m_prevPixelFlowY = 0.7 * m_prevPixelFlowY + 0.3 * newFlowY;

                double maxFlow = static_cast<double>(frameGray.cols); // Max pixel flow can be image width/height
                m_prevPixelFlowX = std::clamp(m_prevPixelFlowX, -maxFlow, maxFlow);
                m_prevPixelFlowY = std::clamp(m_prevPixelFlowY, -maxFlow, maxFlow);
            } else {
                m_prevPixelFlowX = 0.0;
                m_prevPixelFlowY = 0.0;
            }
            // Angular acceleration was not fully implemented in original and needs careful thought on derivation from velocity.
            // m_angularAccelerationX = (m_prevAngularVelocityX - old_prevAngularVelocityX) / deltaTime;
            // m_angularAccelerationY = (m_prevAngularVelocityY - old_prevAngularVelocityY) / deltaTime;

            m_prevTime = currentTime;
        } // end if isFlowValidAtomic
    } // end if !m_prevFrameGray.empty()

    m_prevFrameGray = frameGray.clone(); // Store current frame as previous for next iteration
}


void OpticalFlow::getAngularVelocity(double& angularVelocityXOut, double& angularVelocityYOut)
{
    std::lock_guard<std::mutex> lock(m_flowMutex);
    angularVelocityXOut = m_prevAngularVelocityX; // These are per-frame effective angular changes
    angularVelocityYOut = m_prevAngularVelocityY;
}

void OpticalFlow::getAngularAcceleration(double& angularAccelerationXOut, double& angularAccelerationYOut)
{
    std::lock_guard<std::mutex> lock(m_flowMutex);
    // This was not fully implemented in original source
    angularAccelerationXOut = m_angularAccelerationX; 
    angularAccelerationYOut = m_angularAccelerationY;
}

std::pair<double, double> OpticalFlow::getAverageGlobalFlow()
{
    std::lock_guard<std::mutex> lock(m_flowMutex);
    return { m_prevPixelFlowX, m_prevPixelFlowY }; // Average pixel displacement
}

// The getMotion function providing raw xShift, yShift was not clearly implemented from flow,
// replaced by getAverageGlobalFlow which seems more robust.
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


// --- Threading related methods ---
void OpticalFlow::enqueueFrame(const cv::cuda::GpuMat& frame)
{
    if (!config.enable_optical_flow) return; // Only enqueue if OF is globally enabled

    std::unique_lock<std::mutex> lock(m_frameMutex);
    if (m_frameQueue.size() < 5) // Limit queue size
    {
        m_frameQueue.push(frame.clone()); // Clone to ensure data validity if original is reused
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
        stopOpticalFlowThread(); // Ensure previous thread is stopped
    }
    m_shouldExit = false;
    m_opticalFlowThread = std::thread(&OpticalFlow::opticalFlowLoop, this);
}

void OpticalFlow::stopOpticalFlowThread()
{
    m_shouldExit = true;
    m_frameCV.notify_all(); // Wake up the thread if it's waiting
    if (m_opticalFlowThread.joinable())
    {
        m_opticalFlowThread.join();
    }
     // Release resources on stop
    std::lock_guard<std::mutex> lock(m_flowMutex);
    if (m_cvOpticalFlow) m_cvOpticalFlow.release();
    m_cvOpticalFlow = nullptr;
    m_prevFrameGray.release();
    flow.release();
    m_hintFlow.release();
    isFlowValidAtomic = false;

    // Clear queue
    std::lock_guard<std::mutex> queueLock(m_frameMutex);
    std::queue<cv::cuda::GpuMat> empty;
    std::swap(m_frameQueue, empty);
}

// manageOpticalFlowThread is not in the original sunone code explicitly by that name.
// It's essentially the start/stop logic based on config.
// Let's assume this will be called from the main loop when config.enable_optical_flow changes.
void OpticalFlow::manageOpticalFlowThread()
{
    // This function would typically be called when config.enable_optical_flow changes.
    // The actual start/stop should be triggered from the main application logic.
    // For example, in main.cpp or where config changes are monitored.
    // if (config.enable_optical_flow && !m_opticalFlowThread.joinable()) { // Or check some internal 'is_running' flag
    //     startOpticalFlowThread();
    // } else if (!config.enable_optical_flow && m_opticalFlowThread.joinable()) {
    //     stopOpticalFlowThread();
    // }
}

