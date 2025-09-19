#include <hidboot.h>
#include "ImprovedMouse.h"

#define CHECK_BIT(var, pos) ((var)&pos)

// 8비트 마우스용 구조체
struct MOUSEINFO_8BIT
{
  uint8_t buttons;
  int8_t dX;
  int8_t dY;
  int8_t wheel;
};

// 16비트 마우스용 구조체 (로지텍 8바이트 형식)
struct MOUSEINFO_16BIT
{
  uint8_t buttons;     // 바이트 0
  uint8_t reserved1;   // 바이트 1
  int16_t dX;          // 바이트 2-3
  int16_t dY;          // 바이트 4-5
  int8_t wheel;        // 바이트 6
  uint8_t reserved2;   // 바이트 7
};

// 호환성을 위한 기본 구조체 (8비트)
typedef MOUSEINFO_8BIT MYMOUSEINFO;

class MouseRptParser : public MouseReportParser
{
  union
  {
    MYMOUSEINFO mouseInfo;
    uint16_t bInfo[sizeof(MYMOUSEINFO)];
  } prevState;

protected:
  void Parse(USBHID *hid, bool is_rpt_id, uint8_t len, uint8_t *buf);

};