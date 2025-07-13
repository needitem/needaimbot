import numpy as np
from typing import Tuple, List
import math

class BezierTrajectoryPlanner:
    """
    Advanced trajectory planning using cubic Bezier curves to prevent overshooting.
    This creates smooth, predictable paths that naturally decelerate near targets.
    """
    
    def __init__(self):
        self.control_point_factor = 0.3  # How aggressive the curve is
        self.brake_distance = 50  # Distance to start braking
        self.min_speed = 0.1  # Minimum movement speed
        self.trajectory_cache = []
        self.current_t = 0.0
        
    def calculate_bezier_trajectory(self, 
                                  current_pos: Tuple[float, float], 
                                  target_pos: Tuple[float, float],
                                  current_velocity: Tuple[float, float],
                                  target_size: float = 10.0) -> List[Tuple[float, float]]:
        """
        Generate a cubic Bezier curve trajectory that naturally prevents overshooting.
        
        The key insight: Bezier curves provide guaranteed bounds and smooth deceleration.
        """
        # Calculate distance and direction
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.5:  # Already at target
            return [current_pos]
        
        # P0: Start point
        p0 = current_pos
        
        # P3: End point (slightly before target to prevent overshoot)
        overshoot_prevention = 1.0 - (target_size / distance) * 0.5
        p3 = (current_pos[0] + dx * overshoot_prevention,
              current_pos[1] + dy * overshoot_prevention)
        
        # P1: First control point (based on current velocity)
        # This ensures smooth transition from current movement
        velocity_influence = min(distance * 0.3, 20.0)
        p1 = (current_pos[0] + current_velocity[0] * velocity_influence,
              current_pos[1] + current_velocity[1] * velocity_influence)
        
        # P2: Second control point (creates deceleration curve)
        # Positioned to create natural braking
        brake_factor = 1.0 - (self.brake_distance / max(distance, self.brake_distance))
        p2 = (p3[0] - dx * self.control_point_factor * brake_factor,
              p3[1] - dy * self.control_point_factor * brake_factor)
        
        # Generate trajectory points
        trajectory = []
        steps = max(int(distance / 2), 10)  # Adaptive step count
        
        for i in range(steps + 1):
            t = i / steps
            # Cubic Bezier formula
            point = self._bezier_point(p0, p1, p2, p3, t)
            trajectory.append(point)
            
        return trajectory
    
    def _bezier_point(self, p0, p1, p2, p3, t):
        """Calculate point on cubic Bezier curve at parameter t."""
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        
        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        
        return (x, y)
    
    def get_next_movement(self, 
                         current_pos: Tuple[float, float],
                         target_pos: Tuple[float, float],
                         current_velocity: Tuple[float, float],
                         dt: float = 0.016) -> Tuple[float, float]:
        """
        Get the next movement vector following the Bezier trajectory.
        Includes anticipatory braking based on approach velocity.
        """
        # Generate or update trajectory
        if not self.trajectory_cache or self.current_t >= 1.0:
            self.trajectory_cache = self.calculate_bezier_trajectory(
                current_pos, target_pos, current_velocity
            )
            self.current_t = 0.0
        
        # Find closest point on trajectory
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(self.trajectory_cache):
            dist = math.sqrt((point[0] - current_pos[0])**2 + 
                           (point[1] - current_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Look ahead on trajectory for next target
        lookahead = min(closest_idx + 3, len(self.trajectory_cache) - 1)
        next_point = self.trajectory_cache[lookahead]
        
        # Calculate movement vector
        move_x = next_point[0] - current_pos[0]
        move_y = next_point[1] - current_pos[1]
        
        # Anticipatory braking based on approach velocity
        distance_to_target = math.sqrt(
            (target_pos[0] - current_pos[0])**2 + 
            (target_pos[1] - current_pos[1])**2
        )
        
        if distance_to_target < self.brake_distance:
            # Calculate braking factor using smooth S-curve
            brake_progress = 1.0 - (distance_to_target / self.brake_distance)
            brake_factor = self._smooth_step(brake_progress)
            
            # Apply stronger braking as we approach
            move_x *= (1.0 - brake_factor * 0.9)
            move_y *= (1.0 - brake_factor * 0.9)
        
        return (move_x, move_y)
    
    def _smooth_step(self, x):
        """Smooth S-curve for natural acceleration/deceleration."""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)


class MotionProfileGenerator:
    """
    Generates S-curve motion profiles for smooth acceleration and deceleration.
    This prevents abrupt movements that lead to overshooting.
    """
    
    def __init__(self):
        self.max_acceleration = 5000.0  # pixels/s²
        self.max_jerk = 10000.0  # pixels/s³
        self.profile_cache = {}
        
    def generate_profile(self, distance: float, max_velocity: float) -> dict:
        """
        Generate an S-curve motion profile that guarantees no overshoot.
        Returns time-parameterized velocity and position profiles.
        """
        # Check cache
        cache_key = (round(distance, 2), round(max_velocity, 2))
        if cache_key in self.profile_cache:
            return self.profile_cache[cache_key]
        
        # Calculate profile phases
        # Phase 1: Increasing acceleration (jerk limited)
        # Phase 2: Constant acceleration
        # Phase 3: Decreasing acceleration
        # Phase 4: Constant velocity (if reached)
        # Phase 5-7: Mirror of 1-3 for deceleration
        
        # Simplified S-curve for smooth movement
        t_accel = min(max_velocity / self.max_acceleration, 
                     math.sqrt(distance / self.max_acceleration))
        
        profile = {
            'total_time': t_accel * 2,
            'accel_time': t_accel,
            'phases': []
        }
        
        # Generate time samples
        samples = 50
        for i in range(samples + 1):
            t = i / samples * profile['total_time']
            
            if t <= t_accel:
                # Acceleration phase with S-curve
                phase_progress = t / t_accel
                s_factor = self._s_curve(phase_progress)
                velocity = max_velocity * s_factor
                position = distance * s_factor * phase_progress / 2
            else:
                # Deceleration phase
                decel_progress = (t - t_accel) / t_accel
                s_factor = self._s_curve(1.0 - decel_progress)
                velocity = max_velocity * s_factor
                position = distance * (1.0 - (1.0 - s_factor) * (1.0 - decel_progress) / 2)
            
            profile['phases'].append({
                'time': t,
                'position': position,
                'velocity': velocity
            })
        
        self.profile_cache[cache_key] = profile
        return profile
    
    def _s_curve(self, x):
        """Generate S-curve for smooth acceleration."""
        return 0.5 * (1.0 + math.tanh(6.0 * (x - 0.5)))