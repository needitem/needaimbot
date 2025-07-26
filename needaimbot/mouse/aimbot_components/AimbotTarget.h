#ifndef AIMBOTTARGET_H
#define AIMBOTTARGET_H

// OpenCV removed - using standard C++ types
#include <vector>
#include <chrono>

class AimbotTarget
{
public:
    int x, y, w, h;
    int classId;
    int id;
    std::chrono::high_resolution_clock::time_point detection_timestamp;

    AimbotTarget(int x, int y, int w, int h, int classId);
};

#endif 
