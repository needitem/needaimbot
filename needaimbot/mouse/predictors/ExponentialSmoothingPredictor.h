#ifndef EXPONENTIAL_SMOOTHING_PREDICTOR_H
#define EXPONENTIAL_SMOOTHING_PREDICTOR_H

#include "IPredictor.h" // Include the interface definition
#include <chrono>

// Point2D is defined in IPredictor.h

class ExponentialSmoothingPredictor : public IPredictor { // Inherit from IPredictor
public:
    ExponentialSmoothingPredictor();
    ~ExponentialSmoothingPredictor() override = default; // Override virtual destructor

    /**
     * @brief Configures the predictor.
     * Note: Specific to ExponentialSmoothingPredictor, called after creation.
     * @param alpha The smoothing factor (0.01 to 1.0). Higher values favor recent data.
     * @param prediction_ms Time ahead to predict in milliseconds.
     */
    void configure(float alpha, float prediction_ms);

    // Override interface methods
    void update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) override;
    Point2D predict() const override;
    void reset() override;

private:
    float alpha_; // Smoothing factor for level
    // float beta_; // Optional: Separate smoothing factor for trend (velocity)
    float prediction_time_seconds_;
    
    Point2D smoothed_position_;
    Point2D smoothed_velocity_; // Trend component
    std::chrono::steady_clock::time_point last_timestamp_;
    bool is_initialized_; // To handle the first update
};

#endif // EXPONENTIAL_SMOOTHING_PREDICTOR_H 