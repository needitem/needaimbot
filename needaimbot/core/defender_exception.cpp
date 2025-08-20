#include "defender_exception.h"
#include <shellapi.h>
#include <shlobj.h>
#include <filesystem>

namespace DefenderException {
    
    bool IsRunAsAdmin() {
        BOOL isElevated = FALSE;
        HANDLE hToken = NULL;
        
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
            TOKEN_ELEVATION elevation;
            DWORD size = sizeof(TOKEN_ELEVATION);
            if (GetTokenInformation(hToken, TokenElevation, &elevation, sizeof(elevation), &size)) {
                isElevated = elevation.TokenIsElevated;
            }
            CloseHandle(hToken);
        }
        
        return isElevated;
    }
    
    bool AddWindowsDefenderException() {
        // 현재 실행 파일 경로 가져오기
        char exePath[MAX_PATH];
        GetModuleFileNameA(NULL, exePath, MAX_PATH);
        
        // 실행 파일의 디렉토리 경로 추출
        std::filesystem::path currentPath(exePath);
        std::string folderPath = currentPath.parent_path().string();
        
        // PowerShell 명령어 구성
        std::string psCommand = "Add-MpPreference -ExclusionPath '" + folderPath + "'";
        
        // PowerShell을 통해 Windows Defender 예외 추가
        SHELLEXECUTEINFOA sei = { sizeof(sei) };
        sei.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NO_CONSOLE;
        sei.lpVerb = "runas";
        sei.lpFile = "powershell.exe";
        
        std::string params = "-WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command \"" + psCommand + "\"";
        sei.lpParameters = params.c_str();
        sei.nShow = SW_HIDE;
        
        if (!ShellExecuteExA(&sei)) {
            DWORD error = GetLastError();
            if (error != ERROR_CANCELLED) {  // 사용자가 취소하지 않은 경우
                std::cerr << "[DefenderException] Failed to add exception. Error: " << error << std::endl;
            }
            return false;
        }
        
        // 프로세스 완료 대기
        if (sei.hProcess) {
            WaitForSingleObject(sei.hProcess, 5000);  // 최대 5초 대기
            CloseHandle(sei.hProcess);
        }
        
        std::cout << "[DefenderException] Successfully added Windows Defender exception for: " << folderPath << std::endl;
        return true;
    }
    
    bool CheckAndRequestException() {
        // 관리자 권한 확인
        if (!IsRunAsAdmin()) {
            // 관리자 권한으로 재실행
            char exePath[MAX_PATH];
            GetModuleFileNameA(NULL, exePath, MAX_PATH);
            
            SHELLEXECUTEINFOA sei = { sizeof(sei) };
            sei.lpVerb = "runas";
            sei.lpFile = exePath;
            sei.lpParameters = "--add-defender-exception";
            sei.nShow = SW_NORMAL;
            
            if (ShellExecuteExA(&sei)) {
                // 원래 프로세스는 종료
                return false;
            }
        }
        
        // 관리자 권한이 있으면 예외 추가 시도
        return AddWindowsDefenderException();
    }
}