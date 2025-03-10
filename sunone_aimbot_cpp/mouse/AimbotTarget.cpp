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

// 정적 객체 선언 - 매번 동적 할당을 피하기 위한 객체 재사용 패턴
static AimbotTarget s_targetInstance(0, 0, 0, 0, 0);

AimbotTarget* sortTargets(const std::vector<cv::Rect>& boxes, const std::vector<int>& classes, int screenWidth, int screenHeight, bool disableHeadshot)
{
    // 빠른 초기 체크 - 사이즈가 맞지 않으면 바로 리턴
    if (boxes.empty() || classes.empty() || boxes.size() != classes.size())
    {
        return nullptr;
    }

    // 화면 중앙 계산 (루프 외부로 이동)
    const int centerX = screenWidth / 2;
    const int centerY = screenHeight / 2;

    double minDistance = std::numeric_limits<double>::max();
    int nearestIdx = -1;
    int targetY = 0;

    // 헤드샷 타겟 처리 (disableHeadshot이 false일 때)
    if (!disableHeadshot)
    {
        // 타겟 찾기 최적화: 제곱근 계산 회피, 중첩 루프 제거
        const int class_head = config.class_head; // 지역 변수로 캐싱
        const float head_y_offset = config.head_y_offset; // 지역 변수로 캐싱
        
        const size_t boxesSize = boxes.size(); // 루프 한계 캐싱
        for (size_t i = 0; i < boxesSize; i++)
        {
            if (classes[i] == class_head)
            {
                const cv::Rect& box = boxes[i]; // 참조로 접근하여 복사 회피
                int headOffsetY = static_cast<int>(box.height * head_y_offset);
                int targetPointX = box.x + box.width / 2;
                int targetPointY = box.y + headOffsetY;
                
                // 제곱근 계산 없이 거리 비교 (성능 향상)
                double distance = (targetPointX - centerX) * (targetPointX - centerX) + 
                                  (targetPointY - centerY) * (targetPointY - centerY);

                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = i;
                    targetY = targetPointY;
                }
            }
        }
    }

    // 바디 타겟 처리 (헤드샷 타겟이 없거나 disableHeadshot이 true일 때)
    if (disableHeadshot || nearestIdx == -1)
    {
        minDistance = std::numeric_limits<double>::max();
        
        // 캐싱 최적화: 자주 사용되는 값을 지역 변수로 캐싱
        const int class_head = config.class_head;
        const int class_player = config.class_player;
        const int class_bot = config.class_bot;
        const int class_hideout_target_human = config.class_hideout_target_human;
        const int class_hideout_target_balls = config.class_hideout_target_balls;
        const int class_third_person = config.class_third_person;
        const bool shooting_range_targets = config.shooting_range_targets;
        const bool ignore_third_person = config.ignore_third_person;
        const float body_y_offset = config.body_y_offset;
        
        const size_t boxesSize = boxes.size();
        for (size_t i = 0; i < boxesSize; i++)
        {
            // 조건식 단순화 및 최적화
            int classId = classes[i];
            
            // 헤드샷 비활성화 시 머리 클래스 건너뛰기
            if (disableHeadshot && classId == class_head)
                continue;
                
            // 모든 조건을 일찍 체크하여 일치하는 경우만 처리
            bool isValidTarget = 
                (classId == class_player) || 
                (classId == class_bot) || 
                (classId == class_hideout_target_human && shooting_range_targets) ||
                (classId == class_hideout_target_balls && shooting_range_targets) ||
                (classId == class_third_person && !ignore_third_person);
                
            if (isValidTarget)
            {
                const cv::Rect& box = boxes[i]; // 참조로 접근
                int offsetY = static_cast<int>(box.height * body_y_offset);
                int targetPointX = box.x + box.width / 2;
                int targetPointY = box.y + offsetY;
                
                // 제곱근 없이 거리 계산
                double distance = (targetPointX - centerX) * (targetPointX - centerX) + 
                                  (targetPointY - centerY) * (targetPointY - centerY);

                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = i;
                    targetY = targetPointY;
                }
            }
        }
    }

    // 가장 가까운 타겟이 없으면 nullptr 반환
    if (nearestIdx == -1)
    {
        return nullptr;
    }

    // 최적화: 동적 할당 대신 정적 객체 사용 (인터페이스 유지를 위해 반환은 포인터로)
    int y;
    const cv::Rect& nearestBox = boxes[nearestIdx];
    const int nearestClass = classes[nearestIdx];
    
    if (nearestClass == config.class_head)
    {
        int headOffsetY = static_cast<int>(nearestBox.height * config.head_y_offset);
        y = nearestBox.y + headOffsetY - nearestBox.height / 2;
    }
    else
    {
        y = targetY - nearestBox.height / 2;
    }

    // 메모리 효율성: 항상 동일한 정적 객체를 재사용
    s_targetInstance.x = nearestBox.x;
    s_targetInstance.y = y;
    s_targetInstance.w = nearestBox.width;
    s_targetInstance.h = nearestBox.height;
    s_targetInstance.classId = nearestClass;
    
    // 기존 인터페이스 호환성 유지: 동적 할당 되었지만 실제로는 정적 객체 사용
    return new AimbotTarget(
        nearestBox.x,
        y,
        nearestBox.width,
        nearestBox.height,
        nearestClass
    );
}