#pragma once

// Device-side motor-synergy math: one thread per trajectory sample turns the
// host-sampled shape (motor_synergy.h) into a position. This is the
// embarrassingly-parallel remainder of what used to be a single CPU loop in
// needaimbot/mouse/motor_synergy.hpp - only the OU noise term has a genuine
// cross-sample dependency (each step needs the last), so that stays
// host-integrated and arrives here as a precomputed per-sample array.

#include <curand_kernel.h>

#include "motor_synergy.h"

namespace motor_synergy {
namespace gpu {

__device__ __forceinline__ float normalCdf(float x) {
    return 0.5f * (1.0f + erff(x * 0.70710678f));  // 1/sqrt(2)
}

__device__ __forceinline__ float lognormalCdf(float t, float t0, float mu, float sigma) {
    if (t <= t0) return 0.0f;
    return normalCdf((logf(t - t0) - mu) / sigma);
}

__device__ __forceinline__ float lognormalPdf(float t, float t0, float mu, float sigma) {
    if (t <= t0) return 0.0f;
    const float dt = t - t0;
    const float z = (logf(dt) - mu) / sigma;
    return 1.0f / (sigma * 2.50662827f * dt) * expf(-0.5f * z * z);  // sqrt(2*pi)
}

// s^2*(1-s)^3 normalized to peak=1.0 at s=0.4 - maximal curvature during the
// acceleration phase.
__device__ __forceinline__ float curvatureProfile(float s) {
    if (s <= 0.0f || s >= 1.0f) return 0.0f;
    const float v = s * s * (1.0f - s) * (1.0f - s) * (1.0f - s);
    constexpr float norm = 0.4f * 0.4f * 0.6f * 0.6f * 0.6f;
    return v / norm;
}

__global__ void computeTrajectoryKernel(
    TrajectoryParams params,
    const float* __restrict__ times,
    const float* __restrict__ ouX,
    const float* __restrict__ ouY,
    int numSamples,
    TrajectoryPoint* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSamples) return;

    const float t = times[i];
    const float s = lognormalCdf(t, 0.0f, params.primary_mu, params.primary_sigma);

    float bx = params.x0 + params.tx * params.primary_D * s;
    float by = params.y0 + params.ty * params.primary_D * s;
    bx += params.nx * params.curv_amp * curvatureProfile(s);
    by += params.ny * params.curv_amp * curvatureProfile(s);

    float speed = params.primary_D * lognormalPdf(t, 0.0f, params.primary_mu, params.primary_sigma);

    #pragma unroll
    for (int c = 0; c < kMaxCorrections; ++c) {
        if (c >= params.num_corrections) break;
        const Correction corr = params.corrections[c];
        const float cs = lognormalCdf(t, corr.t0, corr.mu, corr.sigma);
        bx += corr.dir_x * corr.D * cs;
        by += corr.dir_y * corr.D * cs;
        speed += corr.D * lognormalPdf(t, corr.t0, corr.mu, corr.sigma);
    }

    const float t_s = t / 1000.0f;
    const float tremMod = 1.0f / (1.0f + speed * 0.3f);
    const float trX = params.tremor_amp * tremMod
        * sinf(6.28318531f * params.tremor_freq * t_s + params.tremor_phase_x);
    const float trY = params.tremor_amp * tremMod
        * sinf(6.28318531f * params.tremor_freq * t_s + params.tremor_phase_y);

    // Independent per-sample noise (signal-dependent noise): own curand
    // substream per sample index, so this stays embarrassingly parallel with
    // no cross-thread state.
    curandStatePhilox4_32_10_t rng;
    curand_init(params.seed, static_cast<uint64_t>(i), 0, &rng);
    const float sdnX = params.sdn_k * speed * curand_normal(&rng);
    const float sdnY = params.sdn_k * speed * curand_normal(&rng);

    out[i].x = bx + ouX[i] + trX + sdnX;
    out[i].y = by + ouY[i] + trY + sdnY;
    out[i].t = t;
}

}  // namespace gpu
}  // namespace motor_synergy
