#pragma once

#include <atomic>
#include <memory>
#include <chrono>

// Lock-free single-producer single-consumer (SPSC) queue
// Optimized for low-latency mouse input handling
template<typename T>
class LocklessQueue {
private:
    struct Node {
        std::atomic<T*> data;
        std::atomic<Node*> next;
        
        Node() : data(nullptr), next(nullptr) {}
    };
    
    // Cache line padding to prevent false sharing
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    alignas(CACHE_LINE_SIZE) std::atomic<Node*> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<Node*> tail_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> size_;
    
    // Pre-allocated node pool for zero-allocation operation
    struct NodePool {
        static constexpr size_t POOL_SIZE = 256;
        Node nodes[POOL_SIZE];
        std::atomic<size_t> freeIndex{0};
        
        Node* allocate() {
            size_t idx = freeIndex.fetch_add(1, std::memory_order_relaxed);
            if (idx < POOL_SIZE) {
                return &nodes[idx];
            }
            return new Node(); // Fallback to heap allocation
        }
        
        void deallocate(Node* node) {
            // In production, implement proper node recycling
            if (node < nodes || node >= nodes + POOL_SIZE) {
                delete node;
            }
        }
    };
    
    NodePool nodePool_;
    
public:
    LocklessQueue() {
        Node* dummy = nodePool_.allocate();
        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
        size_.store(0, std::memory_order_relaxed);
    }
    
    ~LocklessQueue() {
        // Clean up remaining nodes
        while (Node* oldHead = head_.load(std::memory_order_relaxed)) {
            head_.store(oldHead->next, std::memory_order_relaxed);
            T* data = oldHead->data.load(std::memory_order_relaxed);
            delete data;
            nodePool_.deallocate(oldHead);
        }
    }
    
    // Producer thread - enqueue operation
    bool enqueue(T item) {
        Node* newNode = nodePool_.allocate();
        if (!newNode) return false;
        
        T* data = new T(std::move(item));
        newNode->data.store(data, std::memory_order_relaxed);
        newNode->next.store(nullptr, std::memory_order_relaxed);
        
        Node* prevTail = tail_.exchange(newNode, std::memory_order_acq_rel);
        prevTail->next.store(newNode, std::memory_order_release);
        
        size_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    
    // Consumer thread - dequeue operation
    bool dequeue(T& item) {
        Node* head = head_.load(std::memory_order_relaxed);
        Node* next = head->next.load(std::memory_order_acquire);
        
        if (next == nullptr) {
            return false; // Queue is empty
        }
        
        T* data = next->data.load(std::memory_order_relaxed);
        if (data == nullptr) {
            return false;
        }
        
        item = std::move(*data);
        delete data;
        
        head_.store(next, std::memory_order_release);
        nodePool_.deallocate(head);
        
        size_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
    
    // Try dequeue with timeout (non-blocking)
    bool tryDequeue(T& item, int timeoutMs = 0) {
        if (timeoutMs == 0) {
            return dequeue(item);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto timeout = std::chrono::milliseconds(timeoutMs);
        
        while (true) {
            if (dequeue(item)) {
                return true;
            }
            
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            if (elapsed >= timeout) {
                return false;
            }
            
            // Yield to prevent spinning
            std::this_thread::yield();
        }
    }
    
    size_t size() const {
        return size_.load(std::memory_order_relaxed);
    }
    
    bool empty() const {
        return size() == 0;
    }
};

// Specialized version for mouse commands with built-in batching
struct MouseCommand {
    enum Type {
        MOVE,
        PRESS,    // Changed from CLICK to match existing code
        RELEASE
    };
    
    Type type;
    int dx;
    int dy;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    MouseCommand() : type(MOVE), dx(0), dy(0) {
        timestamp = std::chrono::high_resolution_clock::now();
    }
    
    MouseCommand(Type t, int x, int y) 
        : type(t), dx(x), dy(y) {
        timestamp = std::chrono::high_resolution_clock::now();
    }
};

using MouseCommandQueue = LocklessQueue<MouseCommand>;