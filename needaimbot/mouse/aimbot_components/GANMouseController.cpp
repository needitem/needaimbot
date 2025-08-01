#include "GANMouseController.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

GANMouseController::GANMouseController()
    : noise_scale_(0.3f)
    , path_complexity_(0.7f)
    , human_variability_(0.5f)
    , reaction_time_(0.15f)
    , acceleration_profile_(2) // human-like by default
    , rng_(std::random_device{}())
    , gaussian_(0.0f, 1.0f)
    , uniform_(0.0f, 1.0f) {
}

std::vector<std::pair<int, int>> GANMouseController::calculatePath(
    float start_x, float start_y, float end_x, float end_y) {
    
    std::vector<std::pair<int, int>> path;
    
    // Generate latent vector (simulating GAN input)
    LatentVector latent = generateLatentVector(start_x, start_y, end_x, end_y);
    
    // Generate path from latent vector (simulating GAN generator)
    auto raw_path = generatePathFromLatent(latent, start_x, start_y, end_x, end_y);
    
    // Add human-like imperfections
    addMicroCorrections(raw_path);
    addHumanImperfections(raw_path);
    
    // Apply fatigue model for longer movements
    float distance = std::sqrt(std::pow(end_x - start_x, 2) + std::pow(end_y - start_y, 2));
    float duration = distance / 500.0f; // Approximate duration based on distance
    applyFatigueModel(raw_path, duration);
    
    // Convert to integer coordinates
    for (const auto& point : raw_path) {
        int x = static_cast<int>(std::round(point.x()));
        int y = static_cast<int>(std::round(point.y()));
        
        // Only add point if it's different from the last one
        if (path.empty() || path.back().first != x || path.back().second != y) {
            path.push_back({x, y});
        }
    }
    
    return path;
}

GANMouseController::LatentVector GANMouseController::generateLatentVector(
    float start_x, float start_y, float end_x, float end_y) {
    
    LatentVector latent;
    
    // Distance and direction features
    float dx = end_x - start_x;
    float dy = end_y - start_y;
    float distance = std::sqrt(dx * dx + dy * dy);
    float angle = std::atan2(dy, dx);
    
    // Fill latent vector with meaningful features
    latent.features[0] = distance / 1000.0f; // Normalized distance
    latent.features[1] = angle / M_PI; // Normalized angle
    latent.features[2] = path_complexity_; // Path complexity
    latent.features[3] = human_variability_; // Human variability
    
    // Add noise to simulate GAN's random input
    for (int i = 4; i < 8; ++i) {
        latent.features[i] = gaussian_(rng_) * noise_scale_;
    }
    
    return latent;
}

std::vector<Eigen::Vector2f> GANMouseController::generatePathFromLatent(
    const LatentVector& latent, float start_x, float start_y, float end_x, float end_y) {
    
    std::vector<Eigen::Vector2f> path;
    
    // Simulate GAN generator layers
    auto layer1_output = generatorLayer1(latent.features);
    auto layer2_output = generatorLayer2(layer1_output);
    
    // Calculate number of points based on distance and complexity
    float distance = std::sqrt(std::pow(end_x - start_x, 2) + std::pow(end_y - start_y, 2));
    int num_points = static_cast<int>(distance * path_complexity_ * 0.1f) + 10;
    num_points = std::max(5, std::min(num_points, 100));
    
    // Generate path points
    auto generated_points = generatorOutput(layer2_output, num_points);
    
    // Transform generated points to actual path
    for (int i = 0; i < num_points; ++i) {
        float t = static_cast<float>(i) / (num_points - 1);
        
        // Apply human-like acceleration profile
        float adjusted_t = humanLikeAcceleration(t);
        
        // Linear interpolation with generated offset
        float x = start_x + adjusted_t * (end_x - start_x);
        float y = start_y + adjusted_t * (end_y - start_y);
        
        // Add generated offset
        if (i < generated_points.size()) {
            x += generated_points[i].x() * distance * 0.1f;
            y += generated_points[i].y() * distance * 0.1f;
        }
        
        path.push_back(Eigen::Vector2f(x, y));
    }
    
    return path;
}

std::vector<float> GANMouseController::generatorLayer1(const std::vector<float>& input) {
    // Simulate first generator layer (8 -> 16 features)
    std::vector<float> output(16);
    
    for (int i = 0; i < 16; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input.size(); ++j) {
            // Random weights simulation
            float weight = std::sin(i * 7.0f + j * 13.0f) * 0.5f;
            sum += input[j] * weight;
        }
        // ReLU activation
        output[i] = std::max(0.0f, sum + gaussian_(rng_) * 0.1f);
    }
    
    return output;
}

std::vector<float> GANMouseController::generatorLayer2(const std::vector<float>& input) {
    // Simulate second generator layer (16 -> 32 features)
    std::vector<float> output(32);
    
    for (int i = 0; i < 32; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input.size(); ++j) {
            // Random weights simulation
            float weight = std::cos(i * 5.0f + j * 11.0f) * 0.3f;
            sum += input[j] * weight;
        }
        // Tanh activation for final layer
        output[i] = std::tanh(sum);
    }
    
    return output;
}

std::vector<Eigen::Vector2f> GANMouseController::generatorOutput(
    const std::vector<float>& features, int num_points) {
    
    std::vector<Eigen::Vector2f> points;
    
    for (int i = 0; i < num_points; ++i) {
        float t = static_cast<float>(i) / num_points;
        
        // Use features to generate offsets
        float x_offset = 0.0f;
        float y_offset = 0.0f;
        
        // Combine features with different frequencies
        for (int j = 0; j < features.size(); ++j) {
            float freq = 1.0f + j * 0.5f;
            x_offset += features[j] * std::sin(t * freq * 2 * M_PI) * 0.1f;
            y_offset += features[j] * std::cos(t * freq * 2 * M_PI) * 0.1f;
        }
        
        points.push_back(Eigen::Vector2f(x_offset, y_offset));
    }
    
    return points;
}

float GANMouseController::humanLikeAcceleration(float t) {
    switch (acceleration_profile_) {
        case 0: // Linear
            return t;
        case 1: // Ease-in-out
            return t * t * (3.0f - 2.0f * t);
        case 2: // Human-like (slow start, fast middle, slow end)
        default:
            // Reaction time simulation
            if (t < reaction_time_) {
                return 0.0f;
            }
            
            // Adjust t to account for reaction time
            t = (t - reaction_time_) / (1.0f - reaction_time_);
            
            // S-curve with human characteristics
            float ease = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
            
            // Add slight overshoot and correction
            if (t > 0.8f && t < 0.95f) {
                ease += std::sin((t - 0.8f) * 20.0f) * 0.02f;
            }
            
            return ease;
    }
}

void GANMouseController::addMicroCorrections(std::vector<Eigen::Vector2f>& path) {
    // Simulate small corrections humans make during movement
    for (size_t i = 1; i < path.size() - 1; ++i) {
        if (uniform_(rng_) < 0.3f * human_variability_) {
            // Add small correction
            float correction_x = gaussian_(rng_) * 0.5f;
            float correction_y = gaussian_(rng_) * 0.5f;
            
            path[i].x() += correction_x;
            path[i].y() += correction_y;
        }
    }
}

void GANMouseController::addHumanImperfections(std::vector<Eigen::Vector2f>& path) {
    // Add various human imperfections
    for (size_t i = 0; i < path.size(); ++i) {
        // Subpixel jitter
        path[i].x() = addSubpixelJitter(path[i].x());
        path[i].y() = addSubpixelJitter(path[i].y());
        
        // Occasional slight tremor
        if (uniform_(rng_) < 0.1f * human_variability_) {
            float tremor = gaussian_(rng_) * 0.3f;
            path[i].x() += tremor;
            path[i].y() += tremor * 0.7f; // Slightly less tremor in Y
        }
    }
}

float GANMouseController::addSubpixelJitter(float value) {
    // Add very small jitter to simulate sensor noise
    return value + (uniform_(rng_) - 0.5f) * 0.2f;
}

void GANMouseController::applyFatigueModel(std::vector<Eigen::Vector2f>& path, float duration) {
    // Simulate fatigue effects on longer movements
    if (duration > 0.5f) {
        float fatigue_factor = std::min(1.0f, (duration - 0.5f) * 0.5f);
        
        for (size_t i = path.size() / 2; i < path.size(); ++i) {
            float t = static_cast<float>(i - path.size() / 2) / (path.size() / 2);
            float fatigue = fatigue_factor * t * human_variability_;
            
            // Add slight drift due to fatigue
            path[i].x() += gaussian_(rng_) * fatigue * 2.0f;
            path[i].y() += gaussian_(rng_) * fatigue * 2.0f;
        }
    }
}

float GANMouseController::calculateVelocityProfile(float t, float distance) {
    // Human-like velocity profile
    float base_velocity = 500.0f; // pixels per second
    
    // Adjust for distance (Fitts's law approximation)
    float distance_factor = std::log(1.0f + distance / 100.0f);
    
    // Apply acceleration profile
    float accel_factor = humanLikeAcceleration(t);
    
    return base_velocity * distance_factor * accel_factor;
}