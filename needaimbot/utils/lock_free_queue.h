#pragma once

#include <atomic>
#include <memory>

template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<T*> data;
        std::atomic<Node*> next;
        
        Node() : data(nullptr), next(nullptr) {}
    };
    
    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    
public:
    LockFreeQueue() {
        Node* dummy = new Node;
        head.store(dummy);
        tail.store(dummy);
    }
    
    ~LockFreeQueue() {
        // Properly clean up all nodes and their data
        Node* current = head.load();
        while (current != nullptr) {
            Node* next = current->next.load();
            T* data = current->data.load();
            if (data != nullptr) {
                delete data;
            }
            delete current;
            current = next;
        }
    }
    
    void enqueue(T item) {
        Node* newNode = new Node;
        T* data = new T(std::move(item));
        newNode->data.store(data);
        
        Node* prevTail = tail.exchange(newNode);
        prevTail->next.store(newNode);
    }
    
    bool dequeue(T& result) {
        Node* head_node = head.load();
        Node* next = head_node->next.load();
        
        if (next == nullptr) {
            return false;
        }
        
        T* data = next->data.exchange(nullptr);
        if (data != nullptr) {
            result = std::move(*data);
            delete data;
        }
        
        head.store(next);
        delete head_node;
        
        return true;
    }
    
    bool empty() const {
        Node* head_node = head.load();
        return head_node->next.load() == nullptr;
    }
};

// Double buffer for lock-free detection data exchange
template<typename T>
class DoubleBuffer {
private:
    T buffers[2];
    std::atomic<int> writeIndex{0};
    std::atomic<int> readIndex{1};
    std::atomic<bool> newDataAvailable{false};
    
public:
    T& getWriteBuffer() {
        return buffers[writeIndex.load(std::memory_order_acquire)];
    }
    
    void swapBuffers() {
        int currentWrite = writeIndex.load(std::memory_order_acquire);
        int currentRead = readIndex.load(std::memory_order_acquire);
        
        writeIndex.store(currentRead, std::memory_order_release);
        readIndex.store(currentWrite, std::memory_order_release);
        newDataAvailable.store(true, std::memory_order_release);
    }
    
    bool getReadBuffer(T& result) {
        if (!newDataAvailable.load(std::memory_order_acquire)) {
            return false;
        }
        
        result = buffers[readIndex.load(std::memory_order_acquire)];
        newDataAvailable.store(false, std::memory_order_release);
        return true;
    }
    
    bool hasNewData() const {
        return newDataAvailable.load(std::memory_order_acquire);
    }
};