#pragma once

#include <Windows.h>
#include <thread>

class CPUAffinityManager {
public:
    // 효율 코어(E-core)에 할당하여 전력 소비 감소
    static void setEfficiencyCoreAffinity(HANDLE thread) {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        
        // 마지막 절반 코어들을 효율 코어로 가정 (Intel 12세대+)
        DWORD_PTR mask = 0;
        int numCores = sysInfo.dwNumberOfProcessors;
        
        if (numCores > 4) {
            // 후반부 코어 사용 (보통 E-core)
            for (int i = numCores/2; i < numCores; i++) {
                mask |= (1ULL << i);
            }
        } else {
            // 코어가 적으면 모든 코어 사용
            mask = (1ULL << numCores) - 1;
        }
        
        SetThreadAffinityMask(thread, mask);
    }
    
    // 성능 코어(P-core)에 할당
    static void setPerformanceCoreAffinity(HANDLE thread) {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        
        // 처음 절반 코어들을 성능 코어로 가정
        DWORD_PTR mask = 0;
        int numCores = sysInfo.dwNumberOfProcessors;
        
        for (int i = 0; i < numCores/2 && i < 4; i++) {
            mask |= (1ULL << i);
        }
        
        SetThreadAffinityMask(thread, mask);
    }
    
    // GPU 작업과 분리
    static void avoidGPUCore(HANDLE thread) {
        // GPU가 주로 사용하는 코어 0을 피함
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        
        DWORD_PTR mask = 0;
        for (int i = 1; i < sysInfo.dwNumberOfProcessors; i++) {
            mask |= (1ULL << i);
        }
        
        SetThreadAffinityMask(thread, mask);
    }
};