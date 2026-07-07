#pragma once

// GPU-side motor-synergy trajectory generator: host-callable boundary for the
// device kernel in motor_synergy.cuh. Owns persistent device buffers (sized
// once, reused per call) so generating a trajectory during an active aim
// move never pays a cudaMalloc/cudaFree round trip - mirrors how
// SimpleInference (simple_inference.h) pre-allocates its device buffers
// instead of allocating per call. Config/RNG assembly lives host-side in
// needaimbot/mouse/motor_synergy.hpp; this file is only the shared data
// layout plus the launcher.

#include <cstdint>
#include <cuda_runtime.h>

namespace motor_synergy {
namespace gpu {

constexpr int kMaxCorrections = 2;

struct Correction {
    float D = 0.0f;
    float t0 = 0.0f;
    float mu = 0.0f;
    float sigma = 0.0f;
    float dir_x = 0.0f;
    float dir_y = 0.0f;
};

// One trajectory's sampled shape - assembled host-side once per movement from
// the CPU RNG draws (Fitts timing, primary submovement, corrections,
// curvature, tremor), then read-only on the device by every sample thread.
struct TrajectoryParams {
    float x0 = 0.0f, y0 = 0.0f;
    float tx = 0.0f, ty = 0.0f;  // unit vector along the straight path
    float nx = 0.0f, ny = 0.0f;  // unit normal (curvature bow direction)

    float primary_D = 0.0f;
    float primary_mu = 0.0f;
    float primary_sigma = 0.0f;

    int num_corrections = 0;
    Correction corrections[kMaxCorrections];

    float curv_amp = 0.0f;

    float tremor_freq = 10.0f;
    float tremor_amp = 0.0f;
    float tremor_phase_x = 0.0f;
    float tremor_phase_y = 0.0f;

    float sdn_k = 0.04f;

    uint64_t seed = 0;
};

struct TrajectoryPoint {
    float x = 0.0f, y = 0.0f, t = 0.0f;
};

// Owns fixed-capacity device buffers for one trajectory's worth of samples.
// Not thread-safe; each calling thread should keep its own instance.
class TrajectoryGenerator {
public:
    explicit TrajectoryGenerator(int capacity = 512);
    ~TrajectoryGenerator();

    TrajectoryGenerator(const TrajectoryGenerator&) = delete;
    TrajectoryGenerator& operator=(const TrajectoryGenerator&) = delete;

    int capacity() const { return m_capacity; }

    // times/ouX/ouY are host arrays of length numSamples (<= capacity()).
    // ouX/ouY carry the (sequential, host-integrated) Ornstein-Uhlenbeck path
    // noise; everything else the kernel needs is in `params`. Blocks until
    // `out` (host array, length numSamples) is filled.
    cudaError_t compute(
        const TrajectoryParams& params,
        const float* times, const float* ouX, const float* ouY,
        int numSamples, TrajectoryPoint* out,
        cudaStream_t stream = 0);

private:
    int m_capacity;
    float* m_d_times = nullptr;
    float* m_d_ouX = nullptr;
    float* m_d_ouY = nullptr;
    TrajectoryPoint* m_d_out = nullptr;
    // First cudaMalloc failure hit in the constructor, if any - constructors
    // can't return an error code, so compute() checks and returns this
    // instead of launching against a possibly-null buffer.
    cudaError_t m_initError = cudaSuccess;
};

}  // namespace gpu
}  // namespace motor_synergy
