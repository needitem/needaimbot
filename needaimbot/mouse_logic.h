#ifndef MOUSE_LOGIC_H
#define MOUSE_LOGIC_H

#include "mouse/mouse.h"

namespace MouseLogic {

    void handle_recoil(MouseThread& mouseThread);
    void handle_silent_aim(MouseThread& mouseThread);
    void handle_aiming(MouseThread& mouseThread);

}

#endif // MOUSE_LOGIC_H
