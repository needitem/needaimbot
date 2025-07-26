#pragma once

#include <cuda_runtime.h>
#include "../detector/detector.h"

// RGB 범위 필터 함수
extern "C" void launchRGBRangeFilter(
    const uint8_t* image, 
    uint8_t* mask,
    int width, 
    int height, 
    int pitch,
    uint8_t minR, uint8_t maxR,
    uint8_t minG, uint8_t maxG,
    uint8_t minB, uint8_t maxB,
    cudaStream_t stream = 0);

// Detection을 색상 마스크로 필터링
extern "C" cudaError_t filterDetectionsByColorMask(
    const Detection* detections,
    const uint8_t* colorMask,
    Detection* filteredDetections,
    int* filteredCount,
    int numDetections,
    int maskWidth,
    int maskHeight,
    int minPixels,
    bool removeOnMatch,
    int maxOutputDetections,
    cudaStream_t stream = 0);