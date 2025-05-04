#ifndef IPREDICTOR_H
#define IPREDICTOR_H

#include <chrono>
#include <string> // Needed for configure potentially, or elsewhere
#include <vector> // Might be needed

// Common struct for 2D points/vectors, moved here from VelocityPredictor.h
struct Point2D {
    float x;
    float y;
};

/**
 * @brief Interface for target position predictors.
 */
class IPredictor {
public:
    // Virtual destructor is essential for base classes with virtual functions
    virtual ~IPredictor() = default;

    /**
     * @brief Configures the predictor with specific parameters.
     * Note: Parameter handling might need refinement (e.g., using a struct or variant).
     * This interface might change depending on how configuration is managed.
     */
    // virtual void configure(...) = 0; // Configuration might be handled at creation time

    /**
     * @brief Updates the predictor's internal state with the latest observation.
     * @param position The latest observed target position.
     * @param timestamp The timestamp of the observation.
     */
    virtual void update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) = 0;

    /**
     * @brief Predicts the future position based on the internal state.
     * @return The predicted Point2D.
     */
    virtual Point2D predict() const = 0;

    /**
     * @brief Resets the internal state of the predictor.
     * Useful when the target changes or the filter needs re-initialization.
     */
    virtual void reset() = 0;
};

#endif // IPREDICTOR_H 