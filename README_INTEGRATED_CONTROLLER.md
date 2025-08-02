# Integrated Mouse Controller System

A sophisticated mouse control system that combines PID-based target tracking with recoil compensation, featuring intelligent conflict resolution and smooth movement interpolation.

## Features

### Core Capabilities
- **PID Target Tracking**: Precise 2D PID controller for smooth target following
- **Recoil Compensation**: Advanced weapon recoil patterns with customizable profiles
- **Conflict Resolution**: Intelligent handling of opposing movements
- **Movement Smoothing**: Velocity-based interpolation for natural mouse movement
- **Performance Monitoring**: Real-time metrics and optimization
- **Thread Safety**: Safe for multi-threaded environments

### Key Advantages
- **Independent Operation**: Recoil control works even without targets
- **Smart Combining**: Automatically coordinates PID and recoil movements
- **Adaptive Behavior**: Adjusts to different conflict scenarios
- **High Performance**: Sub-millisecond processing times
- **Configurable**: Extensive customization options

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 IntegratedMouseController                   │
├─────────────────┬─────────────────┬─────────────────────────┤
│   PIDController │ RecoilController│    ConflictResolver     │
│   • Target track│ • Weapon patterns│ • Movement analysis    │
│   • Error calc  │ • Fire rate sync │ • Priority weighting   │
│   • Smooth resp │ • Pattern index  │ • Strategy selection   │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
                    SmoothingEngine
                    • Velocity interp
                    • Acceleration limit
                    • Natural movement
```

## Installation and Setup

### Prerequisites
- Python 3.7+
- NumPy (for vector math)
- Optional: matplotlib (for visualizations)

### Quick Start
```python
from integrated_controller import IntegratedMouseController

# Create controller with default settings
controller = IntegratedMouseController()

# Basic usage
controller.update_target(target_x, target_y, screen_center_x, screen_center_y)
controller.start_firing()

# Get movement
movement, debug = controller.calculate_movement()
dx, dy = int(movement.x), int(movement.y)
```

## Configuration

### JSON Configuration File
The system uses `controller_config.json` for all settings:

```json
{
    "pid_kp_x": 0.8,        // PID proportional gain X
    "pid_ki_x": 0.02,       // PID integral gain X  
    "pid_kd_x": 0.15,       // PID derivative gain X
    "recoil_strength": 1.2, // Recoil compensation strength
    "smoothing_factor": 0.25,// Movement smoothing amount
    "fire_rate": 12.0       // Weapon fire rate (shots/sec)
}
```

### PID Tuning Guidelines
- **Kp (Proportional)**: Controls responsiveness (0.5-1.5)
  - Higher = faster response, may overshoot
  - Lower = slower, more stable
- **Ki (Integral)**: Eliminates steady-state error (0.01-0.05)
  - Higher = faster error correction, may oscillate
  - Lower = slower correction, more stable
- **Kd (Derivative)**: Reduces overshoot (0.05-0.2)
  - Higher = more damping, may be sluggish
  - Lower = less damping, may overshoot

## Usage Examples

### Scenario 1: Recoil Control Only
```python
controller = IntegratedMouseController(recoil_strength=1.5)

# Start firing without target
controller.start_firing()

while firing:
    movement, _ = controller.calculate_movement(fire_rate=15.0)
    move_mouse(int(movement.x), int(movement.y))
    time.sleep(1/120)  # 120 FPS
```

### Scenario 2: Target Tracking Only
```python
controller = IntegratedMouseController()

# Set target position
controller.update_target(target_x, target_y, screen_center_x, screen_center_y)

while tracking:
    movement, debug = controller.calculate_movement()
    if debug['resolution_strategy'] == 'pid_only':
        move_mouse(int(movement.x), int(movement.y))
    time.sleep(1/120)
```

### Scenario 3: Combined Control
```python
controller = IntegratedMouseController()

# Set target and start firing
controller.update_target(target_x, target_y, screen_center_x, screen_center_y)
controller.start_firing(weapon_pattern=custom_recoil_pattern)

while engaging:
    movement, debug = controller.calculate_movement(fire_rate=weapon_fire_rate)
    
    # Movement automatically combines PID tracking and recoil compensation
    move_mouse(int(movement.x), int(movement.y))
    
    print(f"Strategy: {debug['resolution_strategy']}")
    time.sleep(1/120)
```

## Conflict Resolution Strategies

The system automatically detects and resolves conflicts between PID and recoil movements:

### 1. Aligned Movements (Dot Product > 0.7)
- **Strategy**: `aligned_combine`
- **Action**: Add movements together
- **Use Case**: Target and recoil move in same direction

### 2. Opposing Movements (Dot Product < -0.7)
- **Strategy**: `conflict_weighted`
- **Action**: Weighted average favoring recoil on Y-axis
- **Use Case**: Target up but recoil pulls down

### 3. Perpendicular Movements
- **Strategy**: `perpendicular_weighted`
- **Action**: Combine with slight bias toward PID on X-axis
- **Use Case**: Target left/right while recoil goes up

### 4. Single Movement
- **Strategy**: `pid_only` or `recoil_only`
- **Action**: Use the only active movement
- **Use Case**: No target or not firing

## Performance Optimization

### Target Performance
- **Processing Time**: < 1ms per frame
- **Frame Rate**: 120+ FPS capability
- **Memory Usage**: < 10MB
- **CPU Usage**: < 5% single core

### Optimization Features
- Efficient vector mathematics
- Minimal memory allocations
- Thread-safe operations
- Configurable update rates
- Smart conflict detection

## Integration with C++ System

### Bridge Interface
```python
# Create bridge for C++ integration
bridge = IntegratedControllerBridge("controller_config.json")
bridge.start()

# Update from C++ detection
target = AimbotTarget(x=100, y=50, w=30, h=40)
bridge.update_target(target)

# Update firing state
weapon_profile = WeaponRecoilProfile(base_strength=1.5, fire_rate=12.0)
bridge.set_firing_state(True, weapon_profile)

# Get movement for C++ mouse driver
dx, dy, debug = bridge.get_mouse_movement()
```

### Shared Memory Interface
For high-performance integration, use shared memory:

```cpp
// C++ side
struct SharedControllerData {
    float target_x, target_y;
    bool has_target;
    bool is_firing;
    float recoil_strength;
    int movement_dx, movement_dy;
};
```

## Testing and Validation

### Unit Tests
```bash
python test_integrated_controller.py
```

### Performance Benchmark
```bash
python -c "from test_integrated_controller import run_performance_benchmark; run_performance_benchmark()"
```

### Visualization
```bash
python -c "from test_integrated_controller import visualize_movement_patterns; visualize_movement_patterns()"
```

## Troubleshooting

### Common Issues

1. **Jittery Movement**
   - Increase `smoothing_factor` (0.3-0.5)
   - Reduce PID gains
   - Check update rate consistency

2. **Slow Response**
   - Increase Kp gain
   - Reduce `smoothing_factor`
   - Check for performance bottlenecks

3. **Overshooting Target**
   - Increase Kd gain
   - Reduce Kp gain
   - Add more smoothing

4. **Recoil Not Working**
   - Verify `start_firing()` called
   - Check recoil pattern validity
   - Ensure proper fire rate setting

### Debug Information
Enable debug output to analyze behavior:

```python
movement, debug = controller.calculate_movement()
print(f"PID: {debug['pid_movement']}")
print(f"Recoil: {debug['recoil_movement']}")
print(f"Strategy: {debug['resolution_strategy']}")
print(f"Processing: {debug['processing_time']*1000:.2f}ms")
```

## Advanced Configuration

### Custom Recoil Patterns
```python
# Define weapon-specific pattern
ak47_pattern = [
    Vector2D(0.0, 2.5),   # Initial kick
    Vector2D(0.8, 2.2),   # Right drift
    Vector2D(-0.6, 2.0),  # Left correction
    # ... continue pattern
]

controller.start_firing(weapon_pattern=ak47_pattern)
```

### Dynamic Parameter Adjustment
```python
# Adjust PID gains based on target distance
def adjust_for_distance(distance):
    if distance > 100:
        return {'kp_x': 0.6, 'kp_y': 0.6}  # Reduce for far targets
    else:
        return {'kp_x': 1.0, 'kp_y': 1.0}  # Normal for close targets

controller.update_pid_parameters(**adjust_for_distance(target_distance))
```

### Performance Monitoring
```python
metrics = controller.get_performance_metrics()
print(f"Avg processing time: {metrics.avg_processing_time*1000:.2f}ms")
print(f"Conflicts resolved: {metrics.conflicts_resolved}")
print(f"Total movements: {metrics.total_movements}")
```

## License and Credits

This integrated controller system is designed to work with the existing C++ aimbot framework while providing enhanced functionality and better performance through intelligent movement combination and conflict resolution.

For questions or issues, refer to the test suite and examples provided.