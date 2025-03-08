#ifndef SUNONE_AIMBOT_CPP_H
#define SUNONE_AIMBOT_CPP_H

#include "config.h"
#include "detector.h"
#include "mouse.h"
#include "SerialConnection.h"

extern Config config;
extern Detector detector;
extern MouseThread* globalMouseThread;
extern SerialConnection* arduinoSerial;
extern std::atomic<bool> input_method_changed;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;

#endif // SUNONE_AIMBOT_CPP_H