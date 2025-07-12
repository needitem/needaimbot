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
    
    // Batching configuration
    static constexpr int MAX_BATCH_SIZE = 10;
    static constexpr auto BATCH_TIMEOUT = std::chrono::microseconds(100);
    
    void worker() {
        std::vector<MouseCommand> batch;
        batch.reserve(MAX_BATCH_SIZE);
        
        while (!should_stop) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Wait with timeout for batching
            cv.wait_for(lock, BATCH_TIMEOUT, [this] { 
                return !commands.empty() || should_stop; 
            });
            
            if (should_stop) break;
            
            // Collect commands for batch processing
            batch.clear();
            while (!commands.empty() && batch.size() < MAX_BATCH_SIZE) {
                batch.push_back(commands.front());
                commands.pop();
            }
            lock.unlock();
            
            // Process batch without holding lock
            if (!batch.empty() && input_method && input_method->isValid()) {
                // Combine consecutive MOVE commands
                int accumulated_dx = 0, accumulated_dy = 0;
                
                for (const auto& cmd : batch) {
                    switch (cmd.type) {
                        case MouseCommand::MOVE:
                            accumulated_dx += cmd.dx;
                            accumulated_dy += cmd.dy;
                            break;
                        case MouseCommand::PRESS:
                            // Send accumulated movement first
                            if (accumulated_dx != 0 || accumulated_dy != 0) {
                                input_method->move(accumulated_dx, accumulated_dy);
                                accumulated_dx = accumulated_dy = 0;
                            }
                            input_method->press();
                            break;
                        case MouseCommand::RELEASE:
                            // Send accumulated movement first
                            if (accumulated_dx != 0 || accumulated_dy != 0) {
                                input_method->move(accumulated_dx, accumulated_dy);
                                accumulated_dx = accumulated_dy = 0;
                            }
                            input_method->release();
                            break;
                    }
                }
                
                // Send any remaining movement
                if (accumulated_dx != 0 || accumulated_dy != 0) {
                    input_method->move(accumulated_dx, accumulated_dy);
                }
            }
        }
    }
    
public:
    AsyncMouseQueue(InputMethod* method) : input_method(method) {
        worker_thread = std::thread(&AsyncMouseQueue::worker, this);
        // Set high priority for mouse input thread
        SetThreadPriority(worker_thread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
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