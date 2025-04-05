#ifndef AIMBOTTARGET_H
#define AIMBOTTARGET_H

#include <opencv2/opencv.hpp>
#include <vector>

class AimbotTarget
{
public:
    int x, y, w, h;
    int classId;

    AimbotTarget(int x, int y, int w, int h, int classId);
};

// Function to find the best target based on pre-calculated scores
AimbotTarget* findBestTarget(const std::vector<cv::Rect>& boxes,      // Original boxes
                             const std::vector<int>& classes,       // Original classes
                             const std::vector<float>& scores,      // Pre-calculated scores from processBatchedBoxes
                             bool disableHeadshot);                  // Headshot config

#endif // AIMBOTTARGET_H