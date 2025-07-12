#include <algorithm>
#include <numeric>

#include "postProcess.h"
#include "needaimbot.h"
#include "detector.h"


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
        
        const cv::Rect box_i = detections[i].box();
        const float area_i = areas[i];
        const float conf_i = detections[i].confidence;
        
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppress[j]) continue;
            
             if (conf_i > detections[j].confidence * 2.0f) {
                suppress[j] = true;
                continue;
            }
            
            const cv::Rect box_j = detections[j].box();
            
            if (box_i.x > box_j.x + box_j.width || 
                box_j.x > box_i.x + box_i.width ||
                box_i.y > box_j.y + box_j.height || 
                box_j.y > box_i.y + box_i.height) {
                continue; 
            }
            
            const cv::Rect intersection = box_i & box_j;
            
            if (intersection.width > 0 && intersection.height > 0)
            {
                const float intersection_area = static_cast<float>(intersection.area());
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

            cv::Rect box(x, y, width, height);

            Detection detection(box, confidence, classId);

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

    int rows = shape[1];
    int cols = shape[2];
    
    if (rows < 4 + numClasses)
    {
        std::cerr << "[decodeYolo11] Number of classes exceeds available rows in det_output" << std::endl;
        return detections;
    }
    
    cv::Mat det_output(rows, cols, CV_32F, (void*)output);

    for (int i = 0; i < cols; ++i)
    {
        cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + numClasses);

        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > confThreshold)
        {
            float cx = det_output.at<float>(0, i);
            float cy = det_output.at<float>(1, i);
            float ow = det_output.at<float>(2, i);
            float oh = det_output.at<float>(3, i);

            const float half_ow = 0.5f * ow;
            const float half_oh = 0.5f * oh;
            
            
            if (ow <= 0 || oh <= 0) continue;

            cv::Rect box;
            box.x = static_cast<int>((cx - half_ow) * img_scale);
            box.y = static_cast<int>((cy - half_oh) * img_scale);
            box.width = static_cast<int>(ow * img_scale);
            box.height = static_cast<int>(oh * img_scale);

            Detection detection(box, static_cast<float>(score), class_id_point.y);

            detections.push_back(detection);
        }
    }
    return detections;
}
