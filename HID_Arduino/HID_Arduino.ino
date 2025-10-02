#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")

#ifdef dobogusinclude
#include <spi4teensy3.h>
#endif

#include <SPI.h>
#include <usbhub.h>
#include <hiduniversal.h>

#include "hidcustom.h"

// 로지텍 마우스 기본 DPI 설정
const int MOUSE_DPI = 1200;

// 마우스 데이터 비트 모드 설정 (true: 16비트, false: 8비트)
const bool USE_16BIT_MOUSE = true;
                    
volatile signed char deltaX = 0, deltaY = 0, deltaZ = 0;

USB Usb;
USBHub Hub(&Usb);
HIDBoot<USB_HID_PROTOCOL_MOUSE> HidMouse(&Usb, true);
HIDUniversal HidUniversal(&Usb);
MouseRptParser Prs;


void setup()
{
    Serial.begin(2000000);
    Serial.println("Arduino Started");

    if (Usb.Init() == -1) {
        Serial.println("USB Host Shield ERROR!");
        while(1);
    }
    Serial.println("USB Host Shield OK");

    // 마우스 초기화를 좀 더 명시적으로
    HidMouse.SetReportParser(0, &Prs);
    HidUniversal.SetReportParser(0, &Prs);
    Mouse.begin();
    Serial.println("Mouse initialized");

    // USB 디바이스 재스캔 시도
    delay(1000);
    Serial.println("Scanning for USB devices...");
}

void loop()
{
    Usb.Task();

    // 바이너리 명령어 처리
    if (Serial.available() >= 3)
    {
        uint8_t cmd[3];
        Serial.readBytes(cmd, 3);
        ParseBinaryCommand(cmd);
    }

    // 마우스 움직임 처리 (제한 없음)
    if (deltaX || deltaY || deltaZ)
    {
        // PC로 마우스 이동 전달
        Mouse.move(deltaX, deltaY, deltaZ);
        deltaX = deltaY = deltaZ = 0;
    }
}

void MouseRptParser::Parse(USBHID *hid, bool is_rpt_id, uint8_t len, uint8_t *buf)
{
    // 디버깅 로그 제거 (성능 향상)
    // Serial.print("Data len: ");
    // Serial.print(len);
    // Serial.print(" | ");
    // for (int i = 0; i < min(len, 8); i++) {
    //     Serial.print(buf[i], HEX);
    //     Serial.print(" ");
    // }
    // Serial.println();

    if (USE_16BIT_MOUSE) {
        // 로지텍 8바이트 데이터 직접 파싱
        if (len >= 8) {
            uint8_t buttons = buf[0];
            // 바이트 2-3을 16비트 X로 해석
            int16_t mouseX = (int16_t)((buf[3] << 8) | buf[2]);
            // 바이트 4-5를 16비트 Y로 해석
            int16_t mouseY = (int16_t)((buf[5] << 8) | buf[4]);
            int8_t wheel = (int8_t)buf[6];

            // 버튼 상태 변경 처리
            if (prevState.mouseInfo.buttons != buttons)
            {
                Mouse._buttons = buttons;
                Mouse.move(0, 0, 0);
                prevState.mouseInfo.buttons = buttons;
            }

            // 마우스 움직임 처리
            if (mouseX != 0 || mouseY != 0)
            {
                deltaX = mouseX;
                deltaY = mouseY;
            }

            // 휠 처리
            if (wheel != 0)
            {
                deltaZ = wheel;
            }
        }

    } else {
        // 8비트 마우스 처리
        MOUSEINFO_8BIT *pmi8 = reinterpret_cast<MOUSEINFO_8BIT*>(buf);

        // 버튼 상태 변경 처리
        if (prevState.mouseInfo.buttons != pmi8->buttons)
        {
            Mouse._buttons = pmi8->buttons;
            Mouse.move(0, 0, 0);
        }

        // 마우스 움직임 처리 (8비트 원본값 그대로)
        if (pmi8->dX || pmi8->dY)
        {
            deltaX = pmi8->dX;
            deltaY = pmi8->dY;
        }

        // 휠 처리
        if (pmi8->wheel)
        {
            deltaZ = pmi8->wheel;
        }

        prevState.mouseInfo.buttons = pmi8->buttons;
    }
}

inline void ParseBinaryCommand(uint8_t* cmd)
{
    switch(cmd[0])
    {
        case 0x01:
            Mouse.click();
            Serial.print("[CMD] Click\n");
            break;
        case 0x02:
            Mouse.release();
            Serial.print("[CMD] Release\n");
            break;
        case 0x03:
            Mouse.press();
            Serial.print("[CMD] Press\n");
            break;
        case 0x04:
            {
                int8_t dx = (int8_t)cmd[1];
                int8_t dy = (int8_t)cmd[2];
                Mouse.move(dx, dy);
                Serial.print("[CMD] Move: dx=");
                Serial.print(dx);
                Serial.print(" dy=");
                Serial.print(dy);
                Serial.print("\n");
            }
            break;
    }
}
