#pragma once
#include <windows.h>
#include <string>
#include <iostream>

namespace DefenderException {
    bool IsRunAsAdmin();
    bool AddWindowsDefenderException();
    bool CheckAndRequestException();
}