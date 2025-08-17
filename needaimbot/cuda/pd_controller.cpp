// PD Controller CPU implementation
#include "pd_controller.h"
#include "pd_controller_shared.h"
#include <unordered_map>
#include <cmath>

namespace needaimbot {
namespace cuda {

// Host-side state management for per-target tracking
class PDControllerHost {
private:
    std::unordered_map<int, PDState> target_states;
    PDState global_state;
    static constexpr int MAX_TRACKED_TARGETS = 10;
    
public:
    PDControllerHost() : global_state{0.0f, 0.0f, 0.0f, 0.0f, false} {}
    
    // Get or create state for a target ID
    PDState* getTargetState(int target_id) {
        // Cleanup old targets if too many
        if (target_states.size() > MAX_TRACKED_TARGETS) {
            // Simple cleanup: remove half of the entries
            auto it = target_states.begin();
            int to_remove = static_cast<int>(target_states.size() / 2);
            while (to_remove > 0 && it != target_states.end()) {
                it = target_states.erase(it);
                to_remove--;
            }
        }
        
        return &target_states[target_id];
    }
    
    // Get global state for no-ID case
    PDState* getGlobalState() {
        return &global_state;
    }
    
    // Clear all states
    void reset() {
        target_states.clear();
        global_state = {0.0f, 0.0f, 0.0f, 0.0f, false};
    }
    
    // Remove specific target
    void removeTarget(int target_id) {
        target_states.erase(target_id);
    }
};

// Singleton instance
static PDControllerHost& getPDControllerHost() {
    static PDControllerHost instance;
    return instance;
}

// Reset a specific target's state
void resetTargetPDState(int target_id) {
    if (target_id >= 0) {
        getPDControllerHost().removeTarget(target_id);
    } else {
        PDState* global = getPDControllerHost().getGlobalState();
        global->prev_error_x = 0.0f;
        global->prev_error_y = 0.0f;
        global->filtered_derivative_x = 0.0f;
        global->filtered_derivative_y = 0.0f;
        global->initialized = false;
    }
}

// Reset all PD states
void resetAllPDStates() {
    getPDControllerHost().reset();
}

// PD calculation function (CPU version)
static void calculatePDControl(
    float error_x,
    float error_y,
    float& output_x,
    float& output_y,
    PDState& state,
    const PDConfig& config,
    float dt)
{
    // Apply deadzone
    if (fabsf(error_x) < config.deadzone_x) error_x = 0.0f;
    if (fabsf(error_y) < config.deadzone_y) error_y = 0.0f;
    
    // Calculate proportional terms
    float p_term_x = config.kp_x * error_x;
    float p_term_y = config.kp_y * error_y;
    
    // Calculate derivative terms
    float d_term_x = 0.0f;
    float d_term_y = 0.0f;
    
    if (state.initialized && dt > 0.0f) {
        // Raw derivative
        float raw_derivative_x = (error_x - state.prev_error_x) / dt;
        float raw_derivative_y = (error_y - state.prev_error_y) / dt;
        
        // Apply exponential moving average filter to derivative
        if (config.derivative_filter_alpha > 0.0f) {
            // Filter the derivative using exponential moving average
            state.filtered_derivative_x = config.derivative_filter_alpha * state.filtered_derivative_x + 
                                         (1.0f - config.derivative_filter_alpha) * raw_derivative_x;
            state.filtered_derivative_y = config.derivative_filter_alpha * state.filtered_derivative_y + 
                                         (1.0f - config.derivative_filter_alpha) * raw_derivative_y;
            
            d_term_x = config.kd_x * state.filtered_derivative_x;
            d_term_y = config.kd_y * state.filtered_derivative_y;
        } else {
            // No filtering
            d_term_x = config.kd_x * raw_derivative_x;
            d_term_y = config.kd_y * raw_derivative_y;
            
            // Store unfiltered derivative for consistency
            state.filtered_derivative_x = raw_derivative_x;
            state.filtered_derivative_y = raw_derivative_y;
        }
    }
    
    // Calculate total output
    output_x = p_term_x + d_term_x;
    output_y = p_term_y + d_term_y;
    
    // Apply output limits
    if (output_x > config.max_output_x) output_x = config.max_output_x;
    if (output_x < -config.max_output_x) output_x = -config.max_output_x;
    if (output_y > config.max_output_y) output_y = config.max_output_y;
    if (output_y < -config.max_output_y) output_y = -config.max_output_y;
    
    // Update state for next iteration
    state.prev_error_x = error_x;
    state.prev_error_y = error_y;
    state.initialized = true;
}

// Host function to calculate PD with target ID
void calculatePDControlWithID(
    int target_id,
    float error_x,
    float error_y,
    float& output_x,
    float& output_y,
    const PDConfig& config,
    float dt)
{
    auto& controller = getPDControllerHost();
    PDState* state = nullptr;
    
    if (target_id >= 0) {
        // Use per-target state
        state = controller.getTargetState(target_id);
    } else {
        // Use global state
        state = controller.getGlobalState();
    }
    
    if (state) {
        calculatePDControl(error_x, error_y, output_x, output_y, 
                          *state, config, dt);
    }
}

} // namespace cuda
} // namespace needaimbot