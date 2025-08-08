#include "AutoClicker.h"
#include <iostream>
#include <conio.h>

int main() {
    AutoClicker clicker;
    
    clicker.Start();
    
    // 메인 스레드는 대기
    while (true) {
        if (_kbhit()) {
            if (_getch() == 27) { // ESC
                break;
            }
        }
        Sleep(100);
    }
    
    clicker.Stop();
    
    return 0;
}