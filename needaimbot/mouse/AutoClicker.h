#ifndef AUTO_CLICKER_H
#define AUTO_CLICKER_H

#include <windows.h>
#include <thread>
#include <atomic>
#include <chrono>

class AutoClicker {
private:
    std::atomic<bool> isRunning;
    std::atomic<bool> isClicking;
    std::atomic<int> clickDelay; // milliseconds
    std::thread clickThread;
    std::thread inputThread;
    
    void ClickLoop();
    void InputLoop();
    
public:
    AutoClicker();
    ~AutoClicker();
    
    void Start();
    void Stop();
    void SetClickDelay(int delayMs);
    int GetClickDelay() const { return clickDelay.load(); }
};

#endif // AUTO_CLICKER_H