"""
Test Suite for Integrated Mouse Controller
Comprehensive testing of PID aiming and recoil control integration.
"""

import time
import unittest
import matplotlib.pyplot as plt
import numpy as np
from integrated_controller import (
    IntegratedMouseController, Vector2D, MovementType, 
    PIDController2D, RecoilController, ConflictResolver
)


class TestIntegratedController(unittest.TestCase):
    """Unit tests for integrated controller components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.controller = IntegratedMouseController(
            kp_x=0.5, ki_x=0.01, kd_x=0.1,
            kp_y=0.5, ki_y=0.01, kd_y=0.1,
            recoil_strength=1.0,
            smoothing_factor=0.3
        )
    
    def test_vector2d_operations(self):
        """Test Vector2D mathematical operations"""
        v1 = Vector2D(3, 4)
        v2 = Vector2D(1, 2)
        
        # Test magnitude
        self.assertAlmostEqual(v1.magnitude(), 5.0, places=5)
        
        # Test addition
        v3 = v1 + v2
        self.assertEqual(v3.x, 4)
        self.assertEqual(v3.y, 6)
        
        # Test normalization
        v_norm = v1.normalize()
        self.assertAlmostEqual(v_norm.magnitude(), 1.0, places=5)
        
        # Test dot product
        dot = v1.dot(v2)
        self.assertEqual(dot, 11)  # 3*1 + 4*2
    
    def test_pid_controller(self):
        """Test PID controller functionality"""
        pid = PIDController2D(1.0, 0.1, 0.05, 1.0, 0.1, 0.05)
        
        # Test with consistent error
        error = Vector2D(10, 5)
        output1 = pid.calculate(error)
        
        # Should have proportional response
        self.assertGreater(abs(output1.x), 0)
        self.assertGreater(abs(output1.y), 0)
        
        # Test reset
        pid.reset()
        output2 = pid.calculate(error)
        
        # After reset, output should be similar to first calculation
        self.assertAlmostEqual(output1.x, output2.x, delta=0.1)
    
    def test_recoil_controller(self):
        """Test recoil controller functionality"""
        recoil = RecoilController(base_strength=1.0)
        
        # Test without firing
        compensation = recoil.calculate_compensation()
        self.assertEqual(compensation.magnitude(), 0)
        
        # Test with firing
        recoil.start_firing()
        compensation = recoil.calculate_compensation(fire_rate=10.0)
        self.assertGreater(compensation.magnitude(), 0)
        
        # Test stop firing
        recoil.stop_firing()
        compensation = recoil.calculate_compensation()
        self.assertEqual(compensation.magnitude(), 0)
    
    def test_conflict_resolver(self):
        """Test conflict resolution logic"""
        resolver = ConflictResolver()
        
        # Test aligned movements
        pid_mov = Vector2D(5, 5)
        recoil_mov = Vector2D(2, 2)
        result, strategy = resolver.resolve_movements(pid_mov, recoil_mov)
        self.assertEqual(strategy, "aligned_combine")
        self.assertGreater(result.magnitude(), pid_mov.magnitude())
        
        # Test opposing movements
        pid_mov = Vector2D(10, 0)
        recoil_mov = Vector2D(-8, 0)
        result, strategy = resolver.resolve_movements(pid_mov, recoil_mov)
        self.assertEqual(strategy, "conflict_weighted")
        self.assertLess(result.magnitude(), pid_mov.magnitude())
    
    def test_integrated_scenarios(self):
        """Test integrated controller scenarios"""
        # Test recoil only (no target)
        self.controller.clear_target()
        self.controller.start_firing()
        
        movement, debug = self.controller.calculate_movement()
        self.assertEqual(debug['resolution_strategy'], "recoil_only")
        self.assertGreater(debug['recoil_movement'].magnitude(), 0)
        
        # Test PID only (target, no firing)
        self.controller.stop_firing()
        self.controller.update_target(100, 50, 0, 0)
        
        movement, debug = self.controller.calculate_movement()
        self.assertEqual(debug['resolution_strategy'], "pid_only")
        self.assertGreater(debug['pid_movement'].magnitude(), 0)
        
        # Test combined scenario
        self.controller.start_firing()
        movement, debug = self.controller.calculate_movement()
        self.assertIn(debug['resolution_strategy'], 
                     ["aligned_combine", "conflict_weighted", "perpendicular_weighted"])


def run_performance_benchmark():
    """Run performance benchmark tests"""
    print("\n=== Performance Benchmark ===")
    
    controller = IntegratedMouseController()
    iterations = 1000
    
    # Benchmark different scenarios
    scenarios = [
        ("Idle", False, False),
        ("PID Only", True, False), 
        ("Recoil Only", False, True),
        ("Combined", True, True)
    ]
    
    for scenario_name, has_target, is_firing in scenarios:
        if has_target:
            controller.update_target(50, 25, 0, 0)
        else:
            controller.clear_target()
            
        if is_firing:
            controller.start_firing()
        else:
            controller.stop_firing()
        
        # Run benchmark
        start_time = time.time()
        for _ in range(iterations):
            controller.calculate_movement()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        print(f"{scenario_name:12}: {avg_time:.3f}ms avg, {1000/avg_time:.1f} FPS capability")
        
        controller.stop_firing()
    
    # Print final metrics
    metrics = controller.get_performance_metrics()
    print(f"\nTotal operations: {metrics.total_movements}")
    print(f"Conflicts resolved: {metrics.conflicts_resolved}")


def visualize_movement_patterns():
    """Create visualizations of movement patterns"""
    print("\n=== Creating Movement Visualizations ===")
    
    controller = IntegratedMouseController(
        kp_x=0.8, ki_x=0.02, kd_x=0.15,
        kp_y=0.8, ki_y=0.02, kd_y=0.15,
        recoil_strength=1.5
    )
    
    # Test scenarios with different patterns
    scenarios = [
        {
            'name': 'Recoil Pattern Only',
            'target': None,
            'firing': True,
            'color': 'red',
            'duration': 3.0
        },
        {
            'name': 'PID Tracking Only', 
            'target': (100, 50),
            'firing': False,
            'color': 'blue',
            'duration': 3.0
        },
        {
            'name': 'Combined Control',
            'target': (80, 40),
            'firing': True,
            'color': 'green',
            'duration': 4.0
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Integrated Mouse Controller Movement Patterns', fontsize=16)
    
    for i, scenario in enumerate(scenarios):
        # Setup scenario
        if scenario['target']:
            controller.update_target(scenario['target'][0], scenario['target'][1], 0, 0)
        else:
            controller.clear_target()
            
        if scenario['firing']:
            controller.start_firing()
        else:
            controller.stop_firing()
        
        # Collect movement data
        movements = []
        timestamps = []
        debug_data = []
        
        start_time = time.time()
        while time.time() - start_time < scenario['duration']:
            movement, debug = controller.calculate_movement(fire_rate=12.0)
            movements.append((movement.x, movement.y))
            timestamps.append(time.time() - start_time)
            debug_data.append(debug)
            time.sleep(1/120)  # 120 FPS simulation
        
        controller.stop_firing()
        controller.reset_all()
        
        # Plot movement trajectory
        ax = axes[i//2, i%2] if i < 3 else axes[1, 1]
        x_coords = [m[0] for m in movements]
        y_coords = [m[1] for m in movements]
        
        ax.plot(x_coords, y_coords, color=scenario['color'], alpha=0.7, linewidth=1)
        ax.scatter(x_coords[0], y_coords[0], color='green', s=50, label='Start', zorder=5)
        ax.scatter(x_coords[-1], y_coords[-1], color='red', s=50, label='End', zorder=5)
        ax.set_title(scenario['name'])
        ax.set_xlabel('X Movement (pixels)')
        ax.set_ylabel('Y Movement (pixels)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
    
    # Fourth subplot: Combined comparison
    if len(scenarios) == 3:
        ax = axes[1, 1]
        ax.clear()
        
        # Re-run all scenarios for comparison
        for scenario in scenarios:
            if scenario['target']:
                controller.update_target(scenario['target'][0], scenario['target'][1], 0, 0)
            else:
                controller.clear_target()
                
            if scenario['firing']:
                controller.start_firing()
            else:
                controller.stop_firing()
            
            movements = []
            start_time = time.time()
            while time.time() - start_time < min(2.0, scenario['duration']):
                movement, _ = controller.calculate_movement(fire_rate=12.0)
                movements.append((movement.x, movement.y))
                time.sleep(1/120)
            
            x_coords = [m[0] for m in movements[:100]]  # Limit points for clarity
            y_coords = [m[1] for m in movements[:100]]
            ax.plot(x_coords, y_coords, color=scenario['color'], 
                   label=scenario['name'], alpha=0.8, linewidth=2)
            
            controller.stop_firing()
            controller.reset_all()
        
        ax.set_title('Movement Pattern Comparison')
        ax.set_xlabel('X Movement (pixels)')
        ax.set_ylabel('Y Movement (pixels)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('movement_patterns.png', dpi=300, bbox_inches='tight')
    print("Movement patterns saved as 'movement_patterns.png'")
    
    # Create performance metrics visualization
    plt.figure(figsize=(12, 8))
    
    # Collect timing data for different scenarios
    timing_data = {}
    
    test_scenarios = [
        ("Idle", lambda: (controller.clear_target(), controller.stop_firing())),
        ("PID Only", lambda: (controller.update_target(50, 25, 0, 0), controller.stop_firing())),
        ("Recoil Only", lambda: (controller.clear_target(), controller.start_firing())),
        ("Combined", lambda: (controller.update_target(50, 25, 0, 0), controller.start_firing()))
    ]
    
    for name, setup_func in test_scenarios:
        setup_func()
        
        times = []
        for _ in range(200):
            start = time.time()
            controller.calculate_movement()
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        timing_data[name] = times
        controller.stop_firing()
        controller.reset_all()
    
    # Create box plot
    plt.boxplot([timing_data[name] for name in timing_data.keys()], 
                labels=timing_data.keys())
    plt.title('Processing Time Distribution by Scenario')
    plt.ylabel('Processing Time (ms)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("Performance metrics saved as 'performance_metrics.png'")


def stress_test():
    """Run stress test with rapid scenario changes"""
    print("\n=== Stress Test ===")
    
    controller = IntegratedMouseController()
    test_duration = 10.0
    scenario_changes = 0
    
    start_time = time.time()
    last_change = start_time
    
    while time.time() - start_time < test_duration:
        current_time = time.time()
        
        # Change scenario every 0.5 seconds
        if current_time - last_change > 0.5:
            scenario_changes += 1
            
            # Randomly switch between scenarios
            import random
            scenario = random.choice([
                lambda: (controller.clear_target(), controller.stop_firing()),
                lambda: (controller.update_target(
                    random.uniform(-100, 100), random.uniform(-50, 50), 0, 0
                ), controller.stop_firing()),
                lambda: (controller.clear_target(), controller.start_firing()),
                lambda: (controller.update_target(
                    random.uniform(-100, 100), random.uniform(-50, 50), 0, 0
                ), controller.start_firing())
            ])
            
            scenario()
            last_change = current_time
        
        # Calculate movement
        movement, debug = controller.calculate_movement()
        time.sleep(1/240)  # 240 FPS simulation
    
    metrics = controller.get_performance_metrics()
    print(f"Stress test completed:")
    print(f"  Duration: {test_duration:.1f}s")
    print(f"  Scenario changes: {scenario_changes}")
    print(f"  Total movements: {metrics.total_movements}")
    print(f"  Avg processing time: {metrics.avg_processing_time*1000:.3f}ms")
    print(f"  Max processing time: {metrics.max_processing_time*1000:.3f}ms")
    print(f"  Conflicts resolved: {metrics.conflicts_resolved}")


if __name__ == "__main__":
    print("=== Integrated Mouse Controller Test Suite ===")
    
    # Run unit tests
    print("\n--- Running Unit Tests ---")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    # Create visualizations (optional - requires matplotlib)
    try:
        visualize_movement_patterns()
    except ImportError:
        print("\nSkipping visualizations (matplotlib not available)")
    except Exception as e:
        print(f"\nVisualization error: {e}")
    
    # Run stress test
    stress_test()
    
    print("\n=== Test Suite Complete ===")