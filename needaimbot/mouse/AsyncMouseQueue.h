#pragma once
#include <atomic>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <chrono>
#include "input_drivers/input_method.h"

struct MouseCommand {
    enum Type { MOVE, PRESS, RELEASE };
    Type type;
    int dx, dy;
};

class AsyncMouseQueue {
private:
    std::queue<MouseCommand> commands;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> should_stop{false};
    std::thread worker_thread;
    InputMethod* input_method;
    
    void worker() {
        while (!should_stop) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait(lock, [this] { return !commands.empty() || should_stop; });
            
            if (should_stop) break;
            
            // Process all pending commands
            while (!commands.empty()) {
                MouseCommand cmd = commands.front();
                commands.pop();
                lock.unlock();
                
                // Execute command without holding the lock
                if (input_method && input_method->isValid()) {
                    switch (cmd.type) {
                        case MouseCommand::MOVE:
                            input_method->move(cmd.dx, cmd.dy);
                            break;
                        case MouseCommand::PRESS:
                            input_method->press();
                            break;
                        case MouseCommand::RELEASE:
                            input_method->release();
                            break;
                    }
                }
                
                lock.lock();
            }
        }
    }
    
public:
    AsyncMouseQueue(InputMethod* method) : input_method(method) {
        worker_thread = std::thread(&AsyncMouseQueue::worker, this);
    }
    
    ~AsyncMouseQueue() {
        should_stop = true;
        cv.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
    
    void enqueueMove(int dx, int dy) {
        if (dx == 0 && dy == 0) return;
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            commands.push({MouseCommand::MOVE, dx, dy});
        }
        cv.notify_one();
    }
    
    void enqueuePress() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            commands.push({MouseCommand::PRESS, 0, 0});
        }
        cv.notify_one();
    }
    
    void enqueueRelease() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            commands.push({MouseCommand::RELEASE, 0, 0});
        }
        cv.notify_one();
    }
    
    void setInputMethod(InputMethod* method) {
        input_method = method;
    }
};