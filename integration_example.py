"""
Integration Example: Connecting Python Controller to C++ Aimbot
Demonstrates how to use the IntegratedMouseController with the existing C++ system.
"""

import ctypes
import time
import threading
import json
from ctypes import Structure, c_float, c_int, c_bool, POINTER
from integrated_controller import IntegratedMouseController, Vector2D


# Define C structures that match the C++ code
class Point2D(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


class AimbotTarget(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float), 
        ("w", c_float),
        ("h", c_float),
        ("confidence", c_float),
        ("class_id", c_int)
    ]


class WeaponRecoilProfile(Structure):
    _fields_ = [
        ("base_strength", c_float),
        ("fire_rate_multiplier", c_float),
        ("scope_mult_1x", c_float),
        ("scope_mult_2x", c_float),
        ("scope_mult_3x", c_float),
        ("scope_mult_4x", c_float),
        ("recoil_ms", c_float)
    ]


class IntegratedControllerBridge:
    """
    Bridge class that connects the Python IntegratedMouseController 
    to the C++ aimbot system via shared memory or IPC.
    """
    
    def __init__(self, config_file: str = "controller_config.json"):
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize integrated controller
        self.controller = IntegratedMouseController(
            kp_x=self.config.get('pid_kp_x', 0.8),
            ki_x=self.config.get('pid_ki_x', 0.02),
            kd_x=self.config.get('pid_kd_x', 0.15),
            kp_y=self.config.get('pid_kp_y', 0.8),
            ki_y=self.config.get('pid_ki_y', 0.02),
            kd_y=self.config.get('pid_kd_y', 0.15),
            recoil_strength=self.config.get('recoil_strength', 1.2),
            smoothing_factor=self.config.get('smoothing_factor', 0.25)
        )
        
        # Screen configuration
        self.screen_center_x = self.config.get('screen_center_x', 960)
        self.screen_center_y = self.config.get('screen_center_y', 540)
        
        # State tracking
        self.current_target = None
        self.is_firing = False
        self.current_weapon_profile = None
        
        # Performance monitoring
        self.last_update_time = time.time()
        self.update_count = 0
        
        # Thread for continuous updates
        self.running = False
        self.update_thread = None
        
        print("IntegratedControllerBridge initialized")
    
    def load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default config
            default_config = {
                "pid_kp_x": 0.8,
                "pid_ki_x": 0.02,
                "pid_kd_x": 0.15,
                "pid_kp_y": 0.8,
                "pid_ki_y": 0.02,
                "pid_kd_y": 0.15,
                "recoil_strength": 1.2,
                "smoothing_factor": 0.25,
                "screen_center_x": 960,
                "screen_center_y": 540,
                "fire_rate": 12.0,
                "update_rate_hz": 120
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            print(f"Created default config: {config_file}")
            return default_config
    
    def start(self):
        """Start the controller bridge"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        print("Controller bridge started")
    
    def stop(self):
        """Stop the controller bridge"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        print("Controller bridge stopped")
    
    def update_target(self, target: AimbotTarget):
        """Update current target from C++ detection"""
        if target:
            self.current_target = target
            self.controller.update_target(
                target.x + target.w / 2,  # Target center X
                target.y + target.h / 2,  # Target center Y
                self.screen_center_x,
                self.screen_center_y
            )
        else:
            self.current_target = None
            self.controller.clear_target()
    
    def set_firing_state(self, is_firing: bool, weapon_profile: WeaponRecoilProfile = None):
        """Update firing state and weapon profile"""
        self.is_firing = is_firing
        self.current_weapon_profile = weapon_profile
        
        if is_firing:
            # Convert C++ weapon profile to Python recoil pattern
            weapon_pattern = None
            if weapon_profile:
                weapon_pattern = self._convert_weapon_profile(weapon_profile)
            
            self.controller.start_firing(weapon_pattern)
        else:
            self.controller.stop_firing()
    
    def get_mouse_movement(self) -> tuple:
        """
        Get calculated mouse movement for current frame
        Returns: (dx, dy, debug_info)
        """
        fire_rate = self.config.get('fire_rate', 12.0)
        movement, debug = self.controller.calculate_movement(fire_rate)
        
        # Convert to integer pixel movement
        dx = int(round(movement.x))
        dy = int(round(movement.y))
        
        return dx, dy, debug
    
    def update_config(self, new_config: dict):
        """Update controller configuration dynamically"""
        self.config.update(new_config)
        
        # Update PID parameters if provided
        if any(key.startswith('pid_') for key in new_config.keys()):
            self.controller.update_pid_parameters(
                self.config.get('pid_kp_x', 0.8),
                self.config.get('pid_ki_x', 0.02),
                self.config.get('pid_kd_x', 0.15),
                self.config.get('pid_kp_y', 0.8),
                self.config.get('pid_ki_y', 0.02),
                self.config.get('pid_kd_y', 0.15)
            )
        
        # Update recoil strength if provided
        if 'recoil_strength' in new_config:
            self.controller.update_recoil_strength(new_config['recoil_strength'])
        
        print(f"Configuration updated: {new_config}")
    
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics as dictionary"""
        metrics = self.controller.get_performance_metrics()
        
        return {
            'total_movements': metrics.total_movements,
            'pid_movements': metrics.pid_movements,
            'recoil_movements': metrics.recoil_movements,
            'combined_movements': metrics.combined_movements,
            'conflicts_resolved': metrics.conflicts_resolved,
            'avg_processing_time_ms': metrics.avg_processing_time * 1000,
            'max_processing_time_ms': metrics.max_processing_time * 1000,
            'min_processing_time_ms': metrics.min_processing_time * 1000,
            'update_rate_hz': self.update_count / max(time.time() - self.last_update_time, 0.001)
        }
    
    def _convert_weapon_profile(self, profile: WeaponRecoilProfile) -> list:
        """Convert C++ WeaponRecoilProfile to Python recoil pattern"""
        # This is a simplified conversion - in practice, you'd have
        # weapon-specific patterns stored or calculated
        base_strength = profile.base_strength
        
        # Generate a basic recoil pattern based on weapon profile
        pattern = []
        for i in range(10):  # First 10 shots
            # Simulate increasing vertical recoil with slight horizontal drift
            vertical = base_strength * (1.0 + i * 0.1)
            horizontal = base_strength * 0.2 * ((-1) ** i) * (i * 0.1)
            pattern.append(Vector2D(horizontal, vertical))
        
        return pattern
    
    def _update_loop(self):
        """Continuous update loop for real-time processing"""
        update_rate = self.config.get('update_rate_hz', 120)
        frame_time = 1.0 / update_rate
        
        while self.running:
            start_time = time.time()
            
            # Get movement for current frame
            dx, dy, debug = self.get_mouse_movement()
            
            # In a real implementation, you would send this movement
            # to the C++ system via shared memory, pipes, or other IPC
            
            # Update performance counters
            self.update_count += 1
            
            # Sleep to maintain update rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)


def simulate_c_integration():
    """
    Simulate integration with C++ aimbot system
    This demonstrates how the bridge would be used in practice
    """
    print("=== C++ Integration Simulation ===")
    
    # Create bridge
    bridge = IntegratedControllerBridge()
    bridge.start()
    
    # Simulate detection and targeting loop
    try:
        simulation_time = 10.0
        start_time = time.time()
        
        while time.time() - start_time < simulation_time:
            current_time = time.time() - start_time
            
            # Simulate target detection (like from YOLO)
            if current_time > 1.0 and current_time < 8.0:
                # Simulate moving target
                target_x = 100 + 50 * math.sin(current_time)
                target_y = 50 + 25 * math.cos(current_time * 0.8)
                
                target = AimbotTarget()
                target.x = target_x - 25  # Top-left corner
                target.y = target_y - 25
                target.w = 50
                target.h = 50
                target.confidence = 0.9
                target.class_id = 0
                
                bridge.update_target(target)
            else:
                bridge.update_target(None)
            
            # Simulate firing state
            is_firing = current_time > 2.0 and current_time < 7.0
            
            if is_firing != bridge.is_firing:
                # Create weapon profile
                weapon_profile = WeaponRecoilProfile()
                weapon_profile.base_strength = 1.5
                weapon_profile.fire_rate_multiplier = 1.0
                weapon_profile.scope_mult_1x = 1.0
                weapon_profile.recoil_ms = 50.0
                
                bridge.set_firing_state(is_firing, weapon_profile if is_firing else None)
            
            # Get movement (this would be sent to mouse driver)
            dx, dy, debug = bridge.get_mouse_movement()
            
            # Print status every second
            if int(current_time) != int(current_time - 0.016):  # Approximately every second
                has_target = "Yes" if bridge.current_target else "No"
                firing = "Yes" if bridge.is_firing else "No"
                print(f"Time: {current_time:4.1f}s | Target: {has_target:3} | "
                      f"Firing: {firing:3} | Movement: ({dx:3}, {dy:3}) | "
                      f"Strategy: {debug['resolution_strategy']}")
            
            time.sleep(1/60)  # 60 FPS simulation
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted")
    
    finally:
        bridge.stop()
    
    # Print final metrics
    metrics = bridge.get_performance_metrics()
    print(f"\n=== Final Metrics ===")
    for key, value in metrics.items():
        if 'time' in key:
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


def create_dll_interface():
    """
    Example of how to create a DLL interface for C++ integration
    This would be compiled as a separate DLL that the C++ code can call
    """
    
    # Global controller instance
    global_bridge = None
    
    # C-style function exports for DLL
    @ctypes.WINFUNCTYPE(ctypes.c_bool)
    def initialize_controller():
        """Initialize the integrated controller"""
        global global_bridge
        try:
            global_bridge = IntegratedControllerBridge()
            global_bridge.start()
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    @ctypes.WINFUNCTYPE(None)
    def shutdown_controller():
        """Shutdown the integrated controller"""
        global global_bridge
        if global_bridge:
            global_bridge.stop()
            global_bridge = None
    
    @ctypes.WINFUNCTYPE(None, POINTER(AimbotTarget))
    def update_target_from_cpp(target_ptr):
        """Update target from C++ detection"""
        global global_bridge
        if global_bridge and target_ptr:
            target = target_ptr.contents
            global_bridge.update_target(target)
    
    @ctypes.WINFUNCTYPE(None, c_bool, POINTER(WeaponRecoilProfile))
    def set_firing_state_from_cpp(is_firing, weapon_profile_ptr):
        """Set firing state from C++"""
        global global_bridge
        if global_bridge:
            weapon_profile = weapon_profile_ptr.contents if weapon_profile_ptr else None
            global_bridge.set_firing_state(is_firing, weapon_profile)
    
    @ctypes.WINFUNCTYPE(None, POINTER(c_int), POINTER(c_int))
    def get_mouse_movement_for_cpp(dx_ptr, dy_ptr):
        """Get mouse movement for C++"""
        global global_bridge
        if global_bridge and dx_ptr and dy_ptr:
            dx, dy, _ = global_bridge.get_mouse_movement()
            dx_ptr.contents = c_int(dx)
            dy_ptr.contents = c_int(dy)
    
    print("DLL interface functions defined (compile with ctypes.windll for actual DLL)")
    return {
        'initialize_controller': initialize_controller,
        'shutdown_controller': shutdown_controller,
        'update_target_from_cpp': update_target_from_cpp,
        'set_firing_state_from_cpp': set_firing_state_from_cpp,
        'get_mouse_movement_for_cpp': get_mouse_movement_for_cpp
    }


if __name__ == "__main__":
    import math
    
    print("=== Integrated Controller Bridge Example ===")
    
    # Run simulation
    simulate_c_integration()
    
    # Show DLL interface example
    print("\n=== DLL Interface Example ===")
    dll_functions = create_dll_interface()
    print(f"Created {len(dll_functions)} DLL interface functions")
    
    print("\n=== Integration Complete ===")