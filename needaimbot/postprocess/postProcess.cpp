#include <algorithm>
#include <numeric>
#include <iostream>

#include "postProcess.h"
#include "../needaimbot.h"
#include "../detector/detector.h"


void NMS(std::vector<Detection>& detections, float nmsThreshold)
{
    if (detections.empty()) return;
    
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b)
        {
            return a.confidence > b.confidence;
        }
    );

    std::vector<bool> suppress(detections.size(), false);
    std::vector<Detection> result;
    result.reserve(detections.size());

    std::vector<float> areas(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        areas[i] = static_cast<float>(detections[i].width * detections[i].height);
    }

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppress[i]) continue;
        
        result.push_back(detections[i]);
        
        const Detection& det_i = detections[i];
        const float area_i = areas[i];
        const float conf_i = det_i.confidence;
        
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppress[j]) continue;
            
             if (conf_i > detections[j].confidence * 2.0f) {
                suppress[j] = true;
                continue;
            }
            
            const Detection& det_j = detections[j];
            
            if (det_i.x > det_j.x + det_j.width || 
                det_j.x > det_i.x + det_i.width ||
                det_i.y > det_j.y + det_j.height || 
                det_j.y > det_i.y + det_i.height) {
                continue; 
            }
            
            // Calculate intersection manually
            int inter_x = (std::max)(det_i.x, det_j.x);
            int inter_y = (std::max)(det_i.y, det_j.y);
            int inter_w = (std::min)(det_i.x + det_i.width, det_j.x + det_j.width) - inter_x;
            int inter_h = (std::min)(det_i.y + det_i.height, det_j.y + det_j.height) - inter_y;
            
            if (inter_w > 0 && inter_h > 0)
            {
                const float intersection_area = static_cast<float>(inter_w * inter_h);
                const float union_area = area_i + areas[j] - intersection_area;
                if (intersection_area / union_area > nmsThreshold)
                {
                    suppress[j] = true;
                }
            }
        }
    }
    
    detections = std::move(result);
}


std::vector<Detection> decodeYolo10(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float img_scale)
{
    std::vector<Detection> detections;

    int64_t numDetections = shape[1];

    for (int i = 0; i < numDetections; ++i)
    {
        const float* det = output + i * shape[2];
        float confidence = det[4];

        if (confidence > confThreshold)
        {
            int classId = static_cast<int>(det[5]);

            float x1 = det[0];
            float y1 = det[1];
            float x2 = det[2];
            float y2 = det[3];

            int x = static_cast<int>(x1 * img_scale);
            int y = static_cast<int>(y1 * img_scale);
            int width = static_cast<int>((x2 - x1) * img_scale);
            int height = static_cast<int>((y2 - y1) * img_scale);

            
            if (width <= 0 || height <= 0) continue;

            Detection detection(x, y, width, height, confidence, classId);

            detections.push_back(detection);
        }
    }
    return detections;
}

std::vector<Detection> decodeYolo11(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float img_scale)
{
    std::vector<Detection> detections;
    if (shape.size() != 3)
    {
        std::cerr << "[decodeYolo11] Unsupported output shape" << std::endl;
        return detections;
    }

    detections.reserve(shape[2]);

    int rows = static_cast<int>(shape[1]);
    int cols = static_cast<int>(shape[2]);
    
    if (rows < 4 + numClasses)
    {
        std::cerr << "[decodeYolo11] Number of classes exceeds available rows in det_output" << std::endl;
        return detections;
    }
    
    // Direct pointer access instead of cv::Mat
    const float* det_output = output;

    for (int i = 0; i < cols; ++i)
    {
        // Find max score and class id manually
        float max_score = 0.0f;
        int class_id = 0;
        for (int j = 0; j < numClasses; ++j) {
            float score = det_output[(4 + j) * cols + i];
            if (score > max_score) {
                max_score = score;
                class_id = j;
            }
        }
        float score = max_score;

        if (score > confThreshold)
        {
            float cx = det_output[0 * cols + i];
            float cy = det_output[1 * cols + i];
            float ow = det_output[2 * cols + i];
            float oh = det_output[3 * cols + i];

            const float half_ow = 0.5f * ow;
            const float half_oh = 0.5f * oh;
            
            
            if (ow <= 0 || oh <= 0) continue;

            int x = static_cast<int>((cx - half_ow) * img_scale);
            int y = static_cast<int>((cy - half_oh) * img_scale);
            int width = static_cast<int>(ow * img_scale);
            int height = static_cast<int>(oh * img_scale);

            Detection detection(x, y, width, height, static_cast<float>(score), class_id);

            detections.push_back(detection);
        }
    }
    return detections;
}