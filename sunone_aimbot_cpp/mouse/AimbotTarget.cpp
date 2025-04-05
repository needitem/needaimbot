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

AimbotTarget* findBestTarget(const std::vector<cv::Rect>& boxes,      // Original boxes
                             const std::vector<int>& classes,       // Original classes
                             const std::vector<float>& scores,      // Pre-calculated scores from processBatchedBoxes
                             bool disableHeadshot)                  // Headshot config
{
    // Check if scores vector is empty or sizes don't match boxes/classes
    if (scores.empty() || scores.size() != boxes.size() || scores.size() != classes.size())
    {
        return nullptr;
    }

    // Find the index of the element with the maximum score
    auto max_score_it = std::max_element(scores.begin(), scores.end());

    // Check if a valid maximum score was found (scores might contain 0 or negative values)
    if (max_score_it == scores.end() || *max_score_it <= 0.0f) { // Or a suitable threshold
        return nullptr; // No valid target found
    }

    // Get the index of the best target
    int bestIdx = std::distance(scores.begin(), max_score_it);

    // --- Use the index to get the best box and class --- 
    const cv::Rect& bestBox = boxes[bestIdx];
    const int bestClass = classes[bestIdx];

    // Calculate Y coordinate
    int y;
    // Cache config values needed for Y calculation
    const int class_head = config.class_head;
    const float head_y_offset = config.head_y_offset;
    const float body_y_offset = config.body_y_offset;

    if (bestClass == class_head && !disableHeadshot) {
        int headOffsetY = static_cast<int>(bestBox.height * head_y_offset);
        y = bestBox.y + headOffsetY - bestBox.height / 2; // Adjust Y based on head offset
    } else {
        int offsetY = static_cast<int>(bestBox.height * body_y_offset);
        y = bestBox.y + offsetY - bestBox.height / 2; // Adjust Y based on body offset
    }

    // Maintain compatibility with existing interface: dynamically allocated but actually using static object
    // Consider changing the interface to avoid new/delete later
    return new AimbotTarget(
        bestBox.x,
        y,
        bestBox.width,
        bestBox.height,
        bestClass
    );
}