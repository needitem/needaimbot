#ifndef GAN_MOUSE_CONTROLLER_H
#define GAN_MOUSE_CONTROLLER_H

#include <vector>
#include <array>
#include <random>
#include <memory>
#include "../modules/eigen/include/Eigen/Core"
#include "../modules/eigen/include/Eigen/Dense"

class GANMouseController {
public:
    GANMouseController();
    ~GANMouseController() = default;

    // Main function to calculate mouse movement
    std::vector<std::pair<int, int>> calculatePath(float start_x, float start_y, 
                                                   float end_x, float end_y);

    // Configuration parameters
    void setNoiseScale(float scale) { noise_scale_ = scale; }
    void setPathComplexity(float complexity) { path_complexity_ = complexity; }
    void setHumanVariability(float variability) { human_variability_ = variability; }
    void setReactionTime(float time) { reaction_time_ = time; }
    void setAccelerationProfile(int profile) { acceleration_profile_ = profile; }
    
    // Get configuration
    float getNoiseScale() const { return noise_scale_; }
    float getPathComplexity() const { return path_complexity_; }
    float getHumanVariability() const { return human_variability_; }
    float getReactionTime() const { return reaction_time_; }
    int getAccelerationProfile() const { return acceleration_profile_; }

private:
    // GAN-inspired parameters
    float noise_scale_;
    float path_complexity_;
    float human_variability_;
    float reaction_time_;
    int acceleration_profile_; // 0: linear, 1: ease-in-out, 2: human-like
    
    // Random generators
    std::mt19937 rng_;
    std::normal_distribution<float> gaussian_;
    std::uniform_real_distribution<float> uniform_;
    
    // Neural network-inspired components
    struct LatentVector {
        std::vector<float> features;
        LatentVector() : features(8, 0.0f) {}
    };
    
    // Helper functions
    LatentVector generateLatentVector(float start_x, float start_y, float end_x, float end_y);
    std::vector<Eigen::Vector2f> generatePathFromLatent(const LatentVector& latent, 
                                                        float start_x, float start_y,
                                                        float end_x, float end_y);
    float humanLikeAcceleration(float t);
    void addMicroCorrections(std::vector<Eigen::Vector2f>& path);
    void addHumanImperfections(std::vector<Eigen::Vector2f>& path);
    
    // Simulated GAN generator components
    std::vector<float> generatorLayer1(const std::vector<float>& input);
    std::vector<float> generatorLayer2(const std::vector<float>& input);
    std::vector<Eigen::Vector2f> generatorOutput(const std::vector<float>& features,
                                                 int num_points);
    
    // Human behavior modeling
    float calculateVelocityProfile(float t, float distance);
    float addSubpixelJitter(float value);
    void applyFatigueModel(std::vector<Eigen::Vector2f>& path, float duration);
};

#endif // GAN_MOUSE_CONTROLLER_H