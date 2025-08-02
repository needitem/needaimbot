# Windows Defender 오탐지 해결 방법

## 1. Windows Defender 예외 추가 (즉시 해결)
1. Windows 보안 → 바이러스 및 위협 방지
2. "바이러스 및 위협 방지 설정" → "설정 관리"
3. "제외 추가 또는 제거"
4. 다음 폴더 추가:
   - `C:\Users\th072\needaimbot`
   - 빌드 출력 폴더

## 2. 코드 패턴 수정이 필요한 부분
프로그램이 탐지되는 주요 이유:
- **마우스 제어**: `SendInput`, `SetCursorPos` 등 API 사용
- **화면 캡처**: `GetDC`, `BitBlt` 등 사용
- **프로세스 간섭**: 다른 프로세스 화면 읽기
- **커널 드라이버**: kmbox, rzctl 등 하드웨어 제어

## 3. 임시 해결책
```powershell
# PowerShell 관리자 권한으로 실행
Add-MpPreference -ExclusionPath "C:\Users\th072\needaimbot"
Add-MpPreference -ExclusionProcess "needaimbot.exe"
```

## 4. 영구적 해결책
1. **코드 서명 인증서** (연 $200-500)
   - DigiCert, Sectigo 등에서 구매
   - EV 인증서가 더 효과적

2. **Microsoft에 제출**
   - Windows Defender Security Intelligence에 오탐지 신고
   - https://www.microsoft.com/wdsi/filesubmission

3. **안티바이러스 벤더에 화이트리스트 요청**
   - 각 벤더별로 오탐지 신고 가능

## 5. 프로그램 동작 수정 제안
다음 기능들이 특히 의심받습니다:
- 화면 읽기 + 마우스 제어 조합
- 게임 프로세스 메모리 접근
- 커널 레벨 드라이버 사용

이를 회피하려면:
- 오버레이만 표시하는 "분석 모드" 추가
- 마우스 제어 없는 "시각화 전용 모드"
- 설정 파일로 기능 토글