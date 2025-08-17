# Code Style and Conventions

## General Guidelines
- Follow existing C++17 conventions
- Use RAII and smart pointers where appropriate
- Maintain consistent naming conventions
- Add comments for complex algorithms

## Naming Conventions
- **Functions**: camelCase (e.g., `getWriteBuffer()`, `notifyFrameReady()`)
- **Classes**: PascalCase (e.g., `CaptureState`, `DetectionState`)
- **Variables**: camelCase for local, g_ prefix for globals (e.g., `g_current_inference_time_ms`)
- **Constants**: UPPER_SNAKE_CASE
- **Private members**: trailing underscore (e.g., `gpuBuffers_`, `frameReady_`)

## Code Structure
- Use namespace scoping (e.g., `Core::`, `Events::`)
- Prefer composition over inheritance
- Use atomic types for thread-safe variables
- Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) for memory management

## Performance Considerations
- Profile before optimizing
- Prefer CUDA kernels for parallel operations
- Use pinned memory for CPU-GPU transfers
- Minimize memory allocations in hot paths
- Use lock-free structures where possible

## Thread Safety
- Use `std::atomic` for simple shared state
- Use `std::mutex` with RAII locks
- Use `std::condition_variable` for thread coordination
- Avoid global mutable state

## Documentation
- Use header comments for complex classes
- Document public API functions
- Explain non-obvious algorithms
- Include performance characteristics for critical paths