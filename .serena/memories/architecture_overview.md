# Architecture Overview

## Current Architecture Issues
The codebase suffers from several architectural problems that need refactoring:

### God Object Problem
- **AppContext.h** contains 80+ member variables
- Single class manages capture, detection, mouse, performance metrics
- Violates Single Responsibility Principle
- Makes testing and maintenance difficult

### Singleton Pattern Overuse
- Multiple singleton instances throughout codebase
- Creates hidden dependencies
- Makes unit testing difficult
- Complicates lifecycle management

### Tight Coupling
- Direct dependencies between modules
- Hard to replace or test individual components
- Circular dependencies in some areas

## Target Architecture (from IMPLEMENTATION_PLAN.md)

### Week 1: State Separation
1. **CaptureState** - Manages capture buffers and synchronization
2. **DetectionState** - Handles target information and detection state
3. **PerformanceMetrics** - Centralized metrics collection with history
4. **MouseState** - Mouse movement queue and state management
5. **ConfigManager** - Configuration management with change notifications

### Week 2: Dependency Injection
1. **ServiceLocator** - Type-safe service registration and retrieval
2. **IModule Interface** - Standardized module lifecycle
3. **Application Class** - Main application orchestration
4. **EventBus** - Decoupled communication between modules

### Week 3: Pipeline Pattern
1. **Pipeline Framework** - Stage-based processing
2. **DetectionPipeline** - Preprocess → Inference → PostProcess
3. **LockFreeRingBuffer** - High-performance inter-thread communication

### Week 4: Testing & Migration
1. **Google Test Integration**
2. **Feature Flags** - Gradual migration support
3. **Unit Tests** - Component-level testing
4. **Integration Tests** - End-to-end validation

## Key Directories

### `/needaimbot/`
- `AppContext.h` - Main God Object (to be refactored)
- `needaimbot.cpp` - Main entry point

### `/needaimbot/core/`
- Core utilities and common types
- Target.h - Detection result structure
- Performance monitoring utilities

### `/needaimbot/detector/`
- TensorRT inference engine
- Model loading and optimization

### `/needaimbot/capture/`
- Screen capture implementations
- GPU buffer management

### `/needaimbot/mouse/`
- Input device drivers
- Movement queue management

### `/needaimbot/cuda/`
- CUDA kernels for GPU acceleration
- Memory management
- Image processing pipelines

### `/needaimbot/overlay/`
- ImGui-based user interface
- Real-time configuration