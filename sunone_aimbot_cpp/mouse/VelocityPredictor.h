#ifndef VELOCITY_PREDICTOR_H
#define VELOCITY_PREDICTOR_H

#include "IPredictor.h" // Include the interface definition
#include <chrono> // For time points

// Point2D struct is now defined in IPredictor.h

class VelocityPredictor : public IPredictor { // Inherit from IPredictor
public:
    VelocityPredictor();
    ~VelocityPredictor() override = default; // Override virtual destructor

    /**
     * @brief Configures the prediction time.
     * Note: This is specific to VelocityPredictor, called after creation.
     * @param prediction_ms Time ahead to predict in milliseconds.
     */
    void configure(float prediction_ms);

    // Override interface methods
    void update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) override;
    Point2D predict() const override;
    void reset() override;

private:
    float prediction_time_seconds_;
    Point2D last_position_;
    Point2D current_velocity_;
    std::chrono::steady_clock::time_point last_timestamp_;
    bool has_previous_update_; // To handle the first update where velocity cannot be calculated
};

#endif // VELOCITY_PREDICTOR_H 