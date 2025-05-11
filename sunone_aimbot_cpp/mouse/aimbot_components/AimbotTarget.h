#ifndef AIMBOTTARGET_H
#define AIMBOTTARGET_H

#include <opencv2/opencv.hpp>
#include <vector>

class AimbotTarget
{
public:
    int x, y, w, h;
    int classId;
    int id;

    AimbotTarget(int x, int y, int w, int h, int classId);
};

#endif // AIMBOTTARGET_H