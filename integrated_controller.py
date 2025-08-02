"""
Integrated Mouse Controller System
Combines PID aiming and recoil control with intelligent conflict resolution.

This system provides:
- Vector-based movement combining
- Conflict resolution for opposing movements  
- Smooth movement interpolation
- Performance monitoring
- Independent recoil control
- Coordinated target tracking
"""

import time
import threading
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from collections import deque
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovementType(Enum):
    """Types of mouse movements"""
    PID_AIMING = "pid_aiming"
    RECOIL_CONTROL = "recoil_control"
    COMBINED = "combined"
    IDLE = "idle"


@dataclass
class Vector2D:
    """2D Vector for movement calculations"""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self) -> 'Vector2D':
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)
    
    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y


@dataclass
class MovementCommand:
    """Represents a movement command with metadata"""
    vector: Vector2D
    movement_type: MovementType
    priority: int
    timestamp: float
    confidence: float = 1.0


@dataclass
class PerformanceMetrics:
    """Real-time performance monitoring"""
    total_movements: int = 0
    pid_movements: int = 0
    recoil_movements: int = 0
    combined_movements: int = 0
    conflicts_resolved: int = 0
    avg_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    
    def update_timing(self, processing_time: float):
        self.total_movements += 1
        self.avg_processing_time = (
            (self.avg_processing_time * (self.total_movements - 1) + processing_time) 
            / self.total_movements
        )
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.min_processing_time = min(self.min_processing_time, processing_time)


class PIDController2D:
    """2D PID Controller for target tracking"""
    
    def __init__(self, kp_x: float, ki_x: float, kd_x: float,
                 kp_y: float, ki_y: float, kd_y: float):
        self.kp_x, self.ki_x, self.kd_x = kp_x, ki_x, kd_x
        self.kp_y, self.ki_y, self.kd_y = kp_y, ki_y, kd_y
        
        self.prev_error = Vector2D(0, 0)
        self.integral = Vector2D(0, 0)
        self.last_time = time.time()
        
        # Performance constants
        self.max_integral = 100.0
        self.integral_decay = 0.95
    
    def calculate(self, error: Vector2D) -> Vector2D:
        """Calculate PID output for given error"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.001  # Minimum delta time
        
        # Integral with decay and clamping
        self.integral = self.integral * self.integral_decay
        self.integral.x += error.x * dt
        self.integral.y += error.y * dt
        
        # Clamp integral to prevent windup
        self.integral.x = max(-self.max_integral, min(self.max_integral, self.integral.x))
        self.integral.y = max(-self.max_integral, min(self.max_integral, self.integral.y))
        
        # Derivative
        derivative = Vector2D(
            (error.x - self.prev_error.x) / dt,
            (error.y - self.prev_error.y) / dt
        )
        
        # PID output
        output = Vector2D(
            self.kp_x * error.x + self.ki_x * self.integral.x + self.kd_x * derivative.x,
            self.kp_y * error.y + self.ki_y * self.integral.y + self.kd_y * derivative.y
        )
        
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID state"""
        self.prev_error = Vector2D(0, 0)
        self.integral = Vector2D(0, 0)
        self.last_time = time.time()


class RecoilController:
    """Advanced recoil compensation system"""
    
    def __init__(self, base_strength: float = 1.0):
        self.base_strength = base_strength
        self.recoil_pattern = []
        self.pattern_index = 0
        self.fire_start_time = 0
        self.is_firing = False
        self.accumulated_recoil = Vector2D(0, 0)
        
        # Weapon-specific recoil patterns (can be extended)
        self.default_pattern = [
            Vector2D(0, 2.0),    # Initial kick
            Vector2D(0.5, 1.8),  # Slight right drift
            Vector2D(-0.3, 1.6), # Left correction
            Vector2D(0.2, 1.4),  # Stabilizing
            Vector2D(0, 1.2),    # Steady climb
        ]
        
        self.current_pattern = self.default_pattern
    
    def start_firing(self):
        """Signal start of firing sequence"""
        self.is_firing = True
        self.fire_start_time = time.time()
        self.pattern_index = 0
        self.accumulated_recoil = Vector2D(0, 0)
    
    def stop_firing(self):
        """Signal end of firing sequence"""
        self.is_firing = False
        self.pattern_index = 0
        self.accumulated_recoil = Vector2D(0, 0)
    
    def calculate_compensation(self, fire_rate: float = 10.0) -> Vector2D:
        """Calculate recoil compensation based on current firing state"""
        if not self.is_firing:
            return Vector2D(0, 0)
        
        current_time = time.time()
        fire_duration = current_time - self.fire_start_time
        
        # Calculate which shot we're on based on fire rate
        shot_number = int(fire_duration * fire_rate)
        
        if shot_number < len(self.current_pattern):
            base_recoil = self.current_pattern[shot_number]
        else:
            # Use last pattern entry for sustained fire
            base_recoil = self.current_pattern[-1]
        
        # Apply strength multiplier and invert for compensation
        compensation = Vector2D(
            -base_recoil.x * self.base_strength,
            -base_recoil.y * self.base_strength
        )
        
        return compensation
    
    def set_weapon_pattern(self, pattern: list):
        """Set weapon-specific recoil pattern"""
        if pattern:
            self.current_pattern = pattern
        else:
            self.current_pattern = self.default_pattern


class ConflictResolver:
    """Intelligent conflict resolution for opposing movements"""
    
    def __init__(self):
        self.movement_history = deque(maxlen=10)
        self.conflict_threshold = 0.7  # Threshold for detecting conflicts
        
    def resolve_movements(self, pid_movement: Vector2D, recoil_movement: Vector2D) -> Tuple[Vector2D, str]:
        """
        Resolve conflicts between PID and recoil movements
        Returns: (final_movement, resolution_strategy)
        """
        start_time = time.time()
        
        # Calculate movement magnitudes
        pid_mag = pid_movement.magnitude()
        recoil_mag = recoil_movement.magnitude()
        
        # If either movement is negligible, use the other
        if pid_mag < 0.1:
            return recoil_movement, "recoil_only"
        if recoil_mag < 0.1:
            return pid_movement, "pid_only"
        
        # Check for directional conflict using dot product
        pid_norm = pid_movement.normalize()
        recoil_norm = recoil_movement.normalize()
        dot_product = pid_norm.dot(recoil_norm)
        
        # If movements are aligned (dot product > threshold), combine them
        if dot_product > self.conflict_threshold:
            combined = pid_movement + recoil_movement
            return combined, "aligned_combine"
        
        # If movements are opposed (dot product < -threshold), use weighted approach
        elif dot_product < -self.conflict_threshold:
            # Prioritize recoil control for Y-axis, PID for X-axis
            if abs(recoil_movement.y) > abs(pid_movement.y):
                # Strong vertical recoil - prioritize recoil control
                weight_recoil = 0.8
                weight_pid = 0.2
            else:
                # Moderate recoil - balance both
                weight_recoil = 0.6
                weight_pid = 0.4
            
            resolved = Vector2D(
                pid_movement.x * weight_pid + recoil_movement.x * weight_recoil,
                pid_movement.y * weight_pid + recoil_movement.y * weight_recoil
            )
            return resolved, "conflict_weighted"
        
        # Perpendicular movements - combine with slight PID preference for X
        else:
            combined = Vector2D(
                pid_movement.x * 0.7 + recoil_movement.x * 0.3,
                pid_movement.y * 0.3 + recoil_movement.y * 0.7
            )
            return combined, "perpendicular_weighted"


class SmoothingEngine:
    """Advanced movement smoothing and interpolation"""
    
    def __init__(self, smoothing_factor: float = 0.3):
        self.smoothing_factor = smoothing_factor
        self.last_movement = Vector2D(0, 0)
        self.velocity = Vector2D(0, 0)
        self.acceleration_limit = 50.0  # pixels per frame squared
        
    def smooth_movement(self, target_movement: Vector2D, dt: float) -> Vector2D:
        """Apply smoothing to movement using velocity-based interpolation"""
        
        # Calculate desired velocity change
        desired_velocity = target_movement * (1.0 / max(dt, 0.001))
        velocity_change = desired_velocity - self.velocity
        
        # Limit acceleration to prevent jittery movement
        acceleration = velocity_change * (1.0 / max(dt, 0.001))
        if acceleration.magnitude() > self.acceleration_limit:
            acceleration = acceleration.normalize() * self.acceleration_limit
            velocity_change = acceleration * dt
        
        # Update velocity
        self.velocity = self.velocity + velocity_change
        
        # Apply smoothing factor
        smoothed_velocity = self.velocity * self.smoothing_factor + self.last_movement * (1 - self.smoothing_factor) * (1.0 / max(dt, 0.001))
        
        # Calculate final movement
        smoothed_movement = smoothed_velocity * dt
        
        # Update last movement for next frame
        self.last_movement = smoothed_movement
        
        return smoothed_movement
    
    def reset(self):
        """Reset smoothing state"""
        self.last_movement = Vector2D(0, 0)
        self.velocity = Vector2D(0, 0)


class IntegratedMouseController:
    """Main integrated controller combining PID aiming and recoil control"""
    
    def __init__(self, 
                 kp_x: float = 0.5, ki_x: float = 0.01, kd_x: float = 0.1,
                 kp_y: float = 0.5, ki_y: float = 0.01, kd_y: float = 0.1,
                 recoil_strength: float = 1.0,
                 smoothing_factor: float = 0.3):
        
        # Core components
        self.pid_controller = PIDController2D(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y)
        self.recoil_controller = RecoilController(recoil_strength)
        self.conflict_resolver = ConflictResolver()
        self.smoothing_engine = SmoothingEngine(smoothing_factor)
        
        # State management
        self.has_target = False
        self.is_firing = False
        self.last_update_time = time.time()
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Movement queue for async processing
        self.movement_queue = deque(maxlen=5)
        
        logger.info("IntegratedMouseController initialized successfully")
    
    def update_target(self, target_x: float, target_y: float, 
                     screen_center_x: float, screen_center_y: float):
        """Update target position for PID tracking"""
        with self.lock:
            self.has_target = True
            self.target_error = Vector2D(
                target_x - screen_center_x,
                target_y - screen_center_y
            )
    
    def clear_target(self):
        """Clear current target"""
        with self.lock:
            self.has_target = False
            self.pid_controller.reset()
    
    def start_firing(self, weapon_pattern: Optional[list] = None):
        """Start firing sequence with optional weapon pattern"""
        with self.lock:
            self.is_firing = True
            self.recoil_controller.start_firing()
            if weapon_pattern:
                self.recoil_controller.set_weapon_pattern(weapon_pattern)
    
    def stop_firing(self):
        """Stop firing sequence"""
        with self.lock:
            self.is_firing = False
            self.recoil_controller.stop_firing()
    
    def calculate_movement(self, fire_rate: float = 10.0) -> Tuple[Vector2D, Dict[str, Any]]:
        """
        Calculate optimal mouse movement combining PID and recoil control
        Returns: (movement_vector, debug_info)
        """
        start_time = time.time()
        
        with self.lock:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Calculate PID movement (only if target exists)
            pid_movement = Vector2D(0, 0)
            if self.has_target and hasattr(self, 'target_error'):
                pid_movement = self.pid_controller.calculate(self.target_error)
            
            # Calculate recoil compensation (independent of target)
            recoil_movement = self.recoil_controller.calculate_compensation(fire_rate)
            
            # Resolve conflicts between movements
            if pid_movement.magnitude() > 0.1 or recoil_movement.magnitude() > 0.1:
                final_movement, resolution_strategy = self.conflict_resolver.resolve_movements(
                    pid_movement, recoil_movement
                )
            else:
                final_movement = Vector2D(0, 0)
                resolution_strategy = "no_movement"
            
            # Apply smoothing
            smoothed_movement = self.smoothing_engine.smooth_movement(final_movement, dt)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.update_timing(processing_time)
            
            if pid_movement.magnitude() > 0.1:
                self.metrics.pid_movements += 1
            if recoil_movement.magnitude() > 0.1:
                self.metrics.recoil_movements += 1
            if resolution_strategy.startswith("conflict"):
                self.metrics.conflicts_resolved += 1
            
            # Debug information
            debug_info = {
                'pid_movement': pid_movement,
                'recoil_movement': recoil_movement,
                'final_movement': final_movement,
                'smoothed_movement': smoothed_movement,
                'resolution_strategy': resolution_strategy,
                'has_target': self.has_target,
                'is_firing': self.is_firing,
                'processing_time': processing_time,
                'dt': dt
            }
            
            return smoothed_movement, debug_info
    
    def update_pid_parameters(self, kp_x: float, ki_x: float, kd_x: float,
                             kp_y: float, ki_y: float, kd_y: float):
        """Update PID controller parameters"""
        with self.lock:
            self.pid_controller = PIDController2D(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y)
            logger.info(f"PID parameters updated: KP=({kp_x},{kp_y}), KI=({ki_x},{ki_y}), KD=({kd_x},{kd_y})")
    
    def update_recoil_strength(self, strength: float):
        """Update recoil compensation strength"""
        with self.lock:
            self.recoil_controller.base_strength = strength
            logger.info(f"Recoil strength updated: {strength}")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self.lock:
            return self.metrics
    
    def reset_all(self):
        """Reset all controller states"""
        with self.lock:
            self.pid_controller.reset()
            self.recoil_controller.stop_firing()
            self.smoothing_engine.reset()
            self.has_target = False
            self.is_firing = False
            logger.info("All controller states reset")


# Example usage and testing
if __name__ == "__main__":
    # Create integrated controller
    controller = IntegratedMouseController(
        kp_x=0.8, ki_x=0.02, kd_x=0.15,
        kp_y=0.8, ki_y=0.02, kd_y=0.15,
        recoil_strength=1.2,
        smoothing_factor=0.25
    )
    
    print("=== Integrated Mouse Controller Demo ===")
    
    # Simulate different scenarios
    scenarios = [
        {
            'name': 'Recoil Only (No Target)',
            'has_target': False,
            'is_firing': True,
            'duration': 2.0
        },
        {
            'name': 'PID Only (Target, No Firing)',
            'has_target': True,
            'is_firing': False,
            'duration': 2.0
        },
        {
            'name': 'Combined (Target + Firing)',
            'has_target': True,
            'is_firing': True,
            'duration': 3.0
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Setup scenario
        if scenario['has_target']:
            controller.update_target(100, 50, 960, 540)  # Target offset from center
        else:
            controller.clear_target()
        
        if scenario['is_firing']:
            controller.start_firing()
        else:
            controller.stop_firing()
        
        # Run scenario
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < scenario['duration']:
            movement, debug = controller.calculate_movement(fire_rate=12.0)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"  Movement: ({movement.x:.2f}, {movement.y:.2f}) | "
                      f"Strategy: {debug['resolution_strategy']} | "
                      f"Processing: {debug['processing_time']*1000:.2f}ms")
            
            time.sleep(1/60)  # 60 FPS simulation
        
        # Stop firing for next scenario
        controller.stop_firing()
    
    # Print final metrics
    metrics = controller.get_performance_metrics()
    print(f"\n=== Performance Metrics ===")
    print(f"Total movements: {metrics.total_movements}")
    print(f"PID movements: {metrics.pid_movements}")
    print(f"Recoil movements: {metrics.recoil_movements}")
    print(f"Conflicts resolved: {metrics.conflicts_resolved}")
    print(f"Avg processing time: {metrics.avg_processing_time*1000:.2f}ms")
    print(f"Max processing time: {metrics.max_processing_time*1000:.2f}ms")
    print(f"Min processing time: {metrics.min_processing_time*1000:.2f}ms")