#ifndef TARGET_H
#define TARGET_H

#include <type_traits>

// Define __host__ __device__ for CUDA compatibility
#ifdef __CUDACC__
    #define CUDA_HOSTDEV __host__ __device__
#else
    #define CUDA_HOSTDEV
#endif

/**
 * @brief Simple POD structure for detection and tracking
 * 
 * This is a Plain Old Data structure that can be safely copied
 * between GPU and CPU memory using cudaMemcpy.
 * Contains only essential data for detection display and tracking.
 */
struct Target {
    // === Core bounding box data ===
    int classId;         // Object class ID
    int x;               // Bounding box top-left X
    int y;               // Bounding box top-left Y  
    int width;           // Bounding box width
    int height;          // Bounding box height
    
    // === Tracking data ===
    int id;              // Unique tracking ID (-1 if not tracked)
    
    // === Computed fields for compatibility ===
    float confidence;    // Keep for compatibility but not essential
    float center_x;      // Center point X
    float center_y;      // Center point Y
    float velocity_x;    // Estimated velocity in X 
    float velocity_y;    // Estimated velocity in Y
    
    // === Helper methods (inline for POD compatibility) ===
    
    // Default constructor
    CUDA_HOSTDEV Target() : 
        classId(-1), x(-1), y(-1), width(-1), height(-1),
        id(-1), confidence(0.0f),
        center_x(0), center_y(0),
        velocity_x(0), velocity_y(0)
    {
        updateCenter();
    }
    
    // Constructor for detection
    CUDA_HOSTDEV Target(int x_, int y_, int width_, int height_, float conf, int cls) :
        classId(cls), x(x_), y(y_), width(width_), height(height_),
        id(-1), confidence(conf),
        velocity_x(0), velocity_y(0)
    {
        updateCenter();
    }
    
    // Update center point based on bounding box
    CUDA_HOSTDEV void updateCenter() {
        center_x = x + width / 2.0f;
        center_y = y + height / 2.0f;
    }
    
    // Check if this target has valid detection data
    CUDA_HOSTDEV bool hasValidDetection() const {
        return x >= 0 && y >= 0 && width > 0 && height > 0;
    }
};

// Static assert to ensure Target is POD (Plain Old Data)
// This ensures it can be safely copied with cudaMemcpy
static_assert(std::is_trivially_copyable<Target>::value, 
              "Target must be trivially copyable for CUDA memcpy");
static_assert(std::is_standard_layout<Target>::value,
              "Target must have standard layout for CUDA compatibility");

#endif // TARGET_H