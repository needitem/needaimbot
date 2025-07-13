import numpy as np
from typing import Tuple, Optional
import math
from bezier_trajectory import BezierTrajectoryPlanner, MotionProfileGenerator

class HybridAimController:
    """
    Hybrid controller that combines Bezier trajectory planning with PID control
    to eliminate overshooting while maintaining responsiveness.
    """
    
    def __init__(self, pid_controller):
        self.pid = pid_controller
        self.bezier_planner = BezierTrajectoryPlanner()
        self.motion_profiler = MotionProfileGenerator()
        
        # Mode switching thresholds
        self.bezier_distance_threshold = 100  # Use Bezier for medium-long range
        self.profile_distance_threshold = 30   # Use motion profiles for close range
        self.sticky_zone_radius = 5           # "Magnetic" zone around target
        
        # State tracking
        self.last_position = None
        self.velocity_estimate = (0.0, 0.0)
        self.in_sticky_zone = False
        self.overshoot_detector = OvershootDetector()
        
    def calculate_movement(self, 
                          current_pos: Tuple[float, float],
                          target_pos: Tuple[float, float],
                          target_size: float = 10.0,
                          dt: float = 0.016) -> Tuple[float, float]:
        """
        Calculate optimal movement using hybrid approach to prevent overshooting.
        """
        # Update velocity estimate
        if self.last_position:
            self.velocity_estimate = (
                (current_pos[0] - self.last_position[0]) / dt,
                (current_pos[1] - self.last_position[1]) / dt
            )
        self.last_position = current_pos
        
        # Calculate error and distance
        error_x = target_pos[0] - current_pos[0]
        error_y = target_pos[1] - current_pos[1]
        distance = math.sqrt(error_x**2 + error_y**2)
        
        # Check if we're in sticky zone
        if distance < self.sticky_zone_radius:
            return self._sticky_zone_movement(error_x, error_y, distance)
        
        # Detect potential overshoot
        if self.overshoot_detector.check_overshoot_risk(
            current_pos, target_pos, self.velocity_estimate):
            return self._emergency_brake(error_x, error_y, distance)
        
        # Choose control method based on distance
        if distance > self.bezier_distance_threshold:
            # Long range: Use Bezier trajectory
            return self.bezier_planner.get_next_movement(
                current_pos, target_pos, self.velocity_estimate, dt
            )
        elif distance > self.profile_distance_threshold:
            # Medium range: Blend Bezier with PID
            bezier_move = self.bezier_planner.get_next_movement(
                current_pos, target_pos, self.velocity_estimate, dt
            )
            pid_move = self.pid.update(error_x, error_y, dt)
            
            # Adaptive blending based on distance
            blend_factor = (distance - self.profile_distance_threshold) / (
                self.bezier_distance_threshold - self.profile_distance_threshold
            )
            
            return (
                bezier_move[0] * blend_factor + pid_move[0] * (1 - blend_factor),
                bezier_move[1] * blend_factor + pid_move[1] * (1 - blend_factor)
            )
        else:
            # Close range: Use motion profile with PID
            return self._profiled_movement(error_x, error_y, distance, dt)
    
    def _sticky_zone_movement(self, error_x: float, error_y: float, distance: float):
        """
        Special movement in sticky zone - very slow, precise movements.
        """
        # Exponential decay as we get closer to center
        decay_factor = math.exp(-distance / 2.0)
        strength = 0.1 * (1.0 - decay_factor)
        
        return (error_x * strength, error_y * strength)
    
    def _emergency_brake(self, error_x: float, error_y: float, distance: float):
        """
        Emergency braking when overshoot is detected.
        """
        # Move in opposite direction of velocity to brake
        brake_x = -self.velocity_estimate[0] * 0.5
        brake_y = -self.velocity_estimate[1] * 0.5
        
        # Add small correction towards target
        correction_strength = 0.1
        brake_x += error_x * correction_strength
        brake_y += error_y * correction_strength
        
        return (brake_x, brake_y)
    
    def _profiled_movement(self, error_x: float, error_y: float, 
                          distance: float, dt: float):
        """
        Use S-curve motion profile for close-range precision.
        """
        # Generate motion profile
        max_velocity = self.pid.kp_x * distance  # Scale with PID gain
        profile = self.motion_profiler.generate_profile(distance, max_velocity)
        
        # Find current position in profile based on distance
        profile_position = 1.0 - (distance / self.profile_distance_threshold)
        
        # Get velocity from profile
        velocity_magnitude = self._interpolate_profile(profile, profile_position)
        
        # Apply velocity in error direction
        if distance > 0:
            move_x = (error_x / distance) * velocity_magnitude * dt
            move_y = (error_y / distance) * velocity_magnitude * dt
        else:
            move_x = move_y = 0
            
        return (move_x, move_y)
    
    def _interpolate_profile(self, profile: dict, position: float) -> float:
        """Interpolate velocity from motion profile."""
        phases = profile['phases']
        
        # Find surrounding points
        for i in range(len(phases) - 1):
            if phases[i]['position'] <= position <= phases[i+1]['position']:
                # Linear interpolation
                t = (position - phases[i]['position']) / (
                    phases[i+1]['position'] - phases[i]['position']
                )
                return (phases[i]['velocity'] * (1 - t) + 
                       phases[i+1]['velocity'] * t)
        
        return 0.0  # Default to stopped


class OvershootDetector:
    """
    Detects when the cursor is likely to overshoot the target.
    """
    
    def __init__(self):
        self.history_size = 5
        self.position_history = []
        
    def check_overshoot_risk(self, 
                           current_pos: Tuple[float, float],
                           target_pos: Tuple[float, float],
                           velocity: Tuple[float, float]) -> bool:
        """
        Predict if current trajectory will overshoot target.
        """
        # Add to history
        self.position_history.append(current_pos)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        # Need at least 2 positions to detect
        if len(self.position_history) < 2:
            return False
        
        # Calculate if we're approaching target
        curr_dist = math.sqrt(
            (current_pos[0] - target_pos[0])**2 + 
            (current_pos[1] - target_pos[1])**2
        )
        prev_dist = math.sqrt(
            (self.position_history[-2][0] - target_pos[0])**2 + 
            (self.position_history[-2][1] - target_pos[1])**2
        )
        
        # Not approaching = no overshoot risk
        if curr_dist >= prev_dist:
            return False
        
        # Calculate stopping distance based on current velocity
        velocity_magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # Estimate stopping distance (simplified physics)
        max_deceleration = 5000.0  # pixels/s²
        stopping_distance = (velocity_magnitude**2) / (2 * max_deceleration)
        
        # Add safety margin
        safety_factor = 1.5
        
        return stopping_distance * safety_factor > curr_dist


class VirtualInertiaSystem:
    """
    Simulates physical inertia to create more natural movements.
    This prevents abrupt changes that lead to overshooting.
    """
    
    def __init__(self, mass: float = 1.0, damping: float = 0.8):
        self.mass = mass
        self.damping = damping
        self.velocity = np.array([0.0, 0.0])
        
    def apply_force(self, force: Tuple[float, float], dt: float) -> Tuple[float, float]:
        """
        Apply force to virtual mass and return resulting movement.
        """
        # F = ma, so a = F/m
        acceleration = np.array(force) / self.mass
        
        # Update velocity with damping
        self.velocity = self.velocity * self.damping + acceleration * dt
        
        # Calculate displacement
        displacement = self.velocity * dt
        
        return (displacement[0], displacement[1])
    
    def reset(self):
        """Reset the system when target changes significantly."""
        self.velocity = np.array([0.0, 0.0])