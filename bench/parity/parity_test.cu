// CUDA <-> Python parity test.
//
// Every tuning decision this session was made in bench/aim_opt.py, then hand-
// ported into needaimbot/cuda/pd_controller.cuh. If the two ever disagree the
// tuning is meaningless, and until now that was only checked by eye. This runs
// the REAL device function over a deterministic input sequence and dumps the
// emitted deltas so the Python model can be compared frame by frame.
//
// Build:  nvcc -O2 -I../../inference_pc/needaimbot/cuda parity_test.cu -o parity_test
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "pd_controller.cuh"

using namespace gpa;

// One thread runs the whole sequence so the state evolution is exactly the
// per-frame recurrence the real kernel performs.
__global__ void runSequence(const float* rx, const float* ry, const int* cls,
                            int n, AimConfig cfg, float sc, float scale,
                            int* out_dx, int* out_dy) {
    AimState st;                       // default-initialised, as the real state is
    for (int i = 0; i < n; ++i) {
        int dx = 0, dy = 0;
        computeAimMovement(rx[i], ry[i], sc, sc, scale, scale, cfg, &st, cls[i], dx, dy);
        out_dx[i] = dx;
        out_dy[i] = dy;
    }
}

int main(int argc, char** argv) {
    // Config = the shipped values (must mirror the Python side exactly).
    AimConfig cfg;
    cfg.kp_x = 0.75f;  cfg.kp_y = 0.82f;
    cfg.p_softness_x = 9.0f; cfg.p_softness_y = 8.0f;
    cfg.kd_x = 0.05f;  cfg.kd_y = 0.06f;
    cfg.max_step = 25.0f;
    cfg.inflight_comp = 1.0f;
    cfg.deadtime_frames = 1.0f;
    cfg.ff_gain = 1.4f;
    cfg.ff_ego_lag = 2.25f;
    cfg.ff_v_ema = 0.2f;
    cfg.lead_vgate = 9.0f;
    cfg.lead_err_gate = 18.0f;
    cfg.predict_frames = 2.6f;
    cfg.oneeuro_enabled = 1.0f;
    cfg.oneeuro_min_cutoff = 0.3f;
    cfg.oneeuro_beta = 0.02f;
    cfg.shoot_offset_x = 0.0f; cfg.shoot_offset_y = 0.0f;
    cfg.class_switch_reject = (argc > 1) ? (float)atoi(argv[1]) : 1.0f;

    const float SC = 160.0f, SCALE = 1.0f;

    // Deterministic pseudo-random input sequence (a tiny LCG so Python can
    // reproduce it bit-for-bit without sharing RNG state).
    const int N = 400;
    std::vector<float> hrx(N), hry(N);
    std::vector<int>   hcls(N);
    unsigned s = 12345u;
    auto nextf = [&]() { s = 1103515245u*s + 12345u; return (float)((s >> 16) & 0x7fff) / 32767.0f; };
    float tx = SC + 40.0f, ty = SC + 15.0f;
    for (int i = 0; i < N; ++i) {
        tx += 1.3f;  ty += 0.4f;                       // moving target
        if (tx > SC + 90.0f) tx = SC - 90.0f;
        hrx[i] = tx + (nextf() - 0.5f) * 16.0f;        // + noise
        hry[i] = ty + (nextf() - 0.5f) * 12.0f;
        hcls[i] = (nextf() < 0.08f) ? 1 : 0;           // occasional head/body flip
    }

    float *drx, *dry; int *dcls, *ddx, *ddy;
    cudaMalloc(&drx, N*sizeof(float)); cudaMalloc(&dry, N*sizeof(float));
    cudaMalloc(&dcls, N*sizeof(int));
    cudaMalloc(&ddx, N*sizeof(int));   cudaMalloc(&ddy, N*sizeof(int));
    cudaMemcpy(drx, hrx.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dry, hry.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dcls, hcls.data(), N*sizeof(int), cudaMemcpyHostToDevice);

    runSequence<<<1,1>>>(drx, dry, dcls, N, cfg, SC, SCALE, ddx, ddy);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<int> odx(N), ody(N);
    cudaMemcpy(odx.data(), ddx, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(ody.data(), ddy, N*sizeof(int), cudaMemcpyDeviceToHost);

    // stdout: the exact inputs and the emitted deltas, for the Python comparison
    for (int i = 0; i < N; ++i)
        printf("%.7f %.7f %d %d %d\n", hrx[i], hry[i], hcls[i], odx[i], ody[i]);
    return 0;
}
