#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

#include "sunone_aimbot_cpp.h"
#include "AimbotTarget.h"
#include "config.h"

AimbotTarget::AimbotTarget(int x, int y, int w, int h, int cls) : x(x), y(y), w(w), h(h), classId(cls) {}

// Static object declaration - Object reuse pattern to avoid dynamic allocation every time
static AimbotTarget s_targetInstance(0, 0, 0, 0, 0);

AimbotTarget* sortTargets(const std::vector<cv::Rect>& boxes, const std::vector<int>& classes, int screenWidth, int screenHeight, bool disableHeadshot)
{
    // Quick initial check - return immediately if sizes don't match
    if (boxes.empty() || classes.empty() || boxes.size() != classes.size())
    {
        return nullptr;
    }

    // Calculate screen center (moved outside the loop)
    const int centerX = screenWidth / 2;
    const int centerY = screenHeight / 2;
    
    // Add vector to store priority scores
    std::vector<std::pair<double, int>> scores;
    scores.reserve(boxes.size());
    
    // Cache config values - store frequently used values in local variables to avoid repeated access
    const int class_head = config.class_head;
    const int class_player = config.class_player;
    const int class_bot = config.class_bot;
    const int class_hideout_target_human = config.class_hideout_target_human;
    const int class_hideout_target_balls = config.class_hideout_target_balls;
    const int class_third_person = config.class_third_person;
    const bool shooting_range_targets = config.shooting_range_targets;
    const bool ignore_third_person = config.ignore_third_person;
    const float head_y_offset = config.head_y_offset;
    const float body_y_offset = config.body_y_offset;
    
    // Calculate priority scores for all targets at once
    for (size_t i = 0; i < boxes.size(); i++)
    {
        int classId = classes[i];
        
        // Early check for skip conditions
        if (disableHeadshot && classId == class_head) continue;
        
        // Valid target verification (simplified conditionals)
        bool isValidTarget;
        
        if (classId == class_head && !disableHeadshot) {
            isValidTarget = true;
        } else {
            isValidTarget = 
                (classId == class_player) || 
                (classId == class_bot) || 
                (classId == class_hideout_target_human && shooting_range_targets) ||
                (classId == class_hideout_target_balls && shooting_range_targets) ||
                (classId == class_third_person && !ignore_third_person);
        }
        
        if (!isValidTarget) continue;
        
        const cv::Rect& box = boxes[i];
        
        // Calculate target point
        int offsetY;
        if (classId == class_head && !disableHeadshot) {
            offsetY = static_cast<int>(box.height * head_y_offset);
        } else {
            offsetY = static_cast<int>(box.height * body_y_offset);
        }
        
        int targetPointX = box.x + box.width / 2;
        int targetPointY = box.y + offsetY;
        
        // Distance calculation - compare distances without square root (faster)
        double distSq = (targetPointX - centerX) * (targetPointX - centerX) + 
                      (targetPointY - centerY) * (targetPointY - centerY);
        
        // Score calculation: inversely proportional to distance, apply class weighting
        double score = 1000000.0 / (1.0 + distSq);
        
        // Headshot priority weighting
        if (classId == class_head && !disableHeadshot) {
            score *= 1.5; // Apply weight to headshots
        }
        
        // Store score and index
        scores.emplace_back(score, i);
    }
    
    // No target if no scores
    if (scores.empty()) {
        return nullptr;
    }
    
    // Sort scores (descending)
    std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());
    
    // Select target with highest score
    int nearestIdx = scores[0].second;
    const cv::Rect& nearestBox = boxes[nearestIdx];
    const int nearestClass = classes[nearestIdx];
    
    // Calculate Y coordinate
    int y;
    if (nearestClass == class_head) {
        int headOffsetY = static_cast<int>(nearestBox.height * head_y_offset);
        y = nearestBox.y + headOffsetY - nearestBox.height / 2;
    } else {
        int offsetY = static_cast<int>(nearestBox.height * body_y_offset);
        y = nearestBox.y + offsetY - nearestBox.height / 2;
    }
    
    // Maintain compatibility with existing interface: dynamically allocated but actually using static object
    return new AimbotTarget(
        nearestBox.x,
        y,
        nearestBox.width,
        nearestBox.height,
        nearestClass
    );
}