#ifndef CONFIG_CACHE_H
#define CONFIG_CACHE_H

#include <atomic>
#include <chrono>
#include "../config/config.h"

// Thread-safe config cache to reduce mutex contention in hot paths
class ConfigCache {
private:
    // Frequently accessed values cached as atomics
    std::atomic<float> cached_move_scale_x{1.0f};
    std::atomic<float> cached_move_scale_y{1.0f};
    std::atomic<float> cached_fov_x{100.0f};
    std::atomic<float> cached_fov_y{100.0f};
    std::atomic<float> cached_smoothing{0.0f};
    std::atomic<bool> cached_auto_aim{false};
    std::atomic<bool> cached_silent_aim{false};
    std::atomic<int> cached_aim_mode{0};
    std::atomic<float> cached_trigger_delay_ms{0.0f};
    
    // Cache invalidation timestamp
    std::atomic<std::chrono::steady_clock::time_point> last_update;
    static constexpr auto CACHE_DURATION = std::chrono::milliseconds(100);
    
    // Singleton instance
    static ConfigCache& getInstance() {
        static ConfigCache instance;
        return instance;
    }
    
public:
    // Update cache from main config (called periodically or on config change)
    static void updateCache(const Config& config) {
        auto& cache = getInstance();
        cache.cached_move_scale_x.store(config.move_scale_x, std::memory_order_relaxed);
        cache.cached_move_scale_y.store(config.move_scale_y, std::memory_order_relaxed);
        cache.cached_fov_x.store(config.fov_x, std::memory_order_relaxed);
        cache.cached_fov_y.store(config.fov_y, std::memory_order_relaxed);
        cache.cached_smoothing.store(config.smoothing, std::memory_order_relaxed);
        cache.cached_auto_aim.store(config.auto_aim, std::memory_order_relaxed);
        cache.cached_silent_aim.store(config.silent_aim, std::memory_order_relaxed);
        cache.cached_aim_mode.store(config.aim_mode, std::memory_order_relaxed);
        cache.cached_trigger_delay_ms.store(config.trigger_delay_ms, std::memory_order_relaxed);
        cache.last_update.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
    }
    
    // Fast getters (no mutex required)
    static float getMoveScaleX() { 
        return getInstance().cached_move_scale_x.load(std::memory_order_relaxed); 
    }
    
    static float getMoveScaleY() { 
        return getInstance().cached_move_scale_y.load(std::memory_order_relaxed); 
    }
    
    static float getFovX() { 
        return getInstance().cached_fov_x.load(std::memory_order_relaxed); 
    }
    
    static float getFovY() { 
        return getInstance().cached_fov_y.load(std::memory_order_relaxed); 
    }
    
    static float getSmoothing() { 
        return getInstance().cached_smoothing.load(std::memory_order_relaxed); 
    }
    
    static bool getAutoAim() { 
        return getInstance().cached_auto_aim.load(std::memory_order_relaxed); 
    }
    
    static bool getSilentAim() { 
        return getInstance().cached_silent_aim.load(std::memory_order_relaxed); 
    }
    
    static int getAimMode() { 
        return getInstance().cached_aim_mode.load(std::memory_order_relaxed); 
    }
    
    static float getTriggerDelayMs() { 
        return getInstance().cached_trigger_delay_ms.load(std::memory_order_relaxed); 
    }
    
    // Check if cache needs refresh
    static bool needsRefresh() {
        auto& cache = getInstance();
        auto now = std::chrono::steady_clock::now();
        auto last = cache.last_update.load(std::memory_order_relaxed);
        return (now - last) > CACHE_DURATION;
    }
    
    // Structured config subset for batch reads
    struct CachedValues {
        float move_scale_x;
        float move_scale_y;
        float fov_x;
        float fov_y;
        float smoothing;
        bool auto_aim;
        bool silent_aim;
        int aim_mode;
        float trigger_delay_ms;
    };
    
    static CachedValues getAll() {
        auto& cache = getInstance();
        return {
            cache.cached_move_scale_x.load(std::memory_order_relaxed),
            cache.cached_move_scale_y.load(std::memory_order_relaxed),
            cache.cached_fov_x.load(std::memory_order_relaxed),
            cache.cached_fov_y.load(std::memory_order_relaxed),
            cache.cached_smoothing.load(std::memory_order_relaxed),
            cache.cached_auto_aim.load(std::memory_order_relaxed),
            cache.cached_silent_aim.load(std::memory_order_relaxed),
            cache.cached_aim_mode.load(std::memory_order_relaxed),
            cache.cached_trigger_delay_ms.load(std::memory_order_relaxed)
        };
    }
};

#endif // CONFIG_CACHE_H