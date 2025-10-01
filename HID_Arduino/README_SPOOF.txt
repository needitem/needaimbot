========================================
Arduino USB Descriptor Spoofing Guide
Logitech G502 HERO 위장
========================================

[설치 순서]

1. spoof_logitech.bat (관리자 권한으로 실행)
   - boards.txt 패치 (VID/PID 변경)
   - USBCore.cpp 패치 (Manufacturer 변경)
   - 한 번만 실행하면 됩니다

2. force_clean_compile.bat
   - 모든 빌드 캐시 삭제
   - precompiled core 삭제
   - spoof_logitech.bat 실행 후 반드시 실행

3. Arduino IDE 재시작
   - Tools > Board > Arduino Leonardo 선택
   - HID_Arduino.ino 열기
   - Sketch > Verify/Compile
   - 로그에서 확인:
     -DUSB_VID=0x046D
     -DUSB_PID=0xC07D
     -DUSB_MANUFACTURER="Logitech"

4. Upload
   - Upload 버튼 클릭
   - 업로드 완료 대기

5. Arduino USB 재연결
   - Arduino IDE 완전히 종료
   - Arduino USB 뽑기
   - 10초 대기
   - Arduino USB 다시 꽂기
   - 10초 대기 (애플리케이션 모드 전환)

6. check_mouse.bat
   - VID: 046D, PID: C07D 확인
   - Serial 포트(VID: 2341)는 무시 (정상)

========================================
[중요 사항]

- VID_2341은 Serial 포트 (정상)
- VID_046D는 마우스 장치 (위장 성공)
- 게임은 마우스 장치만 확인하므로 성공!

========================================
[복원 방법]

1. boards.txt 복원:
   copy /Y "boards.txt.backup" "boards.txt"

2. USBCore.cpp 복원:
   copy /Y "USBCore.cpp.backup" "USBCore.cpp"

3. force_clean_compile.bat 실행

4. Arduino IDE 재시작 후 재업로드

========================================
[파일 설명]

spoof_logitech.bat
  - Arduino 파일 자동 패치
  - 관리자 권한 필요
  - 한 번만 실행

force_clean_compile.bat
  - 빌드 캐시 완전 삭제
  - 패치 후 반드시 실행

check_mouse.bat
  - 마우스 VID/PID 확인
  - 위장 성공 여부 검증

========================================
