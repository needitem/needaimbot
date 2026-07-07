#include "motor_synergy.cuh"

namespace motor_synergy {
namespace gpu {

TrajectoryGenerator::TrajectoryGenerator(int capacity) : m_capacity(capacity) {
    if (m_initError == cudaSuccess)
        m_initError = cudaMalloc(&m_d_times, static_cast<size_t>(m_capacity) * sizeof(float));
    if (m_initError == cudaSuccess)
        m_initError = cudaMalloc(&m_d_ouX, static_cast<size_t>(m_capacity) * sizeof(float));
    if (m_initError == cudaSuccess)
        m_initError = cudaMalloc(&m_d_ouY, static_cast<size_t>(m_capacity) * sizeof(float));
    if (m_initError == cudaSuccess)
        m_initError = cudaMalloc(&m_d_out, static_cast<size_t>(m_capacity) * sizeof(TrajectoryPoint));
}

TrajectoryGenerator::~TrajectoryGenerator() {
    // cudaFree(nullptr) is a documented no-op, so this is safe even if
    // construction bailed out partway through.
    cudaFree(m_d_times);
    cudaFree(m_d_ouX);
    cudaFree(m_d_ouY);
    cudaFree(m_d_out);
}

cudaError_t TrajectoryGenerator::compute(
    const TrajectoryParams& params,
    const float* times, const float* ouX, const float* ouY,
    int numSamples, TrajectoryPoint* out,
    cudaStream_t stream) {
    if (numSamples <= 0) return cudaSuccess;
    if (numSamples > m_capacity) return cudaErrorInvalidValue;
    if (m_initError != cudaSuccess) return m_initError;

    const size_t bytesF = static_cast<size_t>(numSamples) * sizeof(float);
    const size_t bytesPt = static_cast<size_t>(numSamples) * sizeof(TrajectoryPoint);

    cudaError_t err = cudaSuccess;
    if ((err = cudaMemcpyAsync(m_d_times, times, bytesF, cudaMemcpyHostToDevice, stream)) != cudaSuccess)
        return err;
    if ((err = cudaMemcpyAsync(m_d_ouX, ouX, bytesF, cudaMemcpyHostToDevice, stream)) != cudaSuccess)
        return err;
    if ((err = cudaMemcpyAsync(m_d_ouY, ouY, bytesF, cudaMemcpyHostToDevice, stream)) != cudaSuccess)
        return err;

    constexpr int kBlockSize = 128;
    const int blocks = (numSamples + kBlockSize - 1) / kBlockSize;
    computeTrajectoryKernel<<<blocks, kBlockSize, 0, stream>>>(
        params, m_d_times, m_d_ouX, m_d_ouY, numSamples, m_d_out);
    // Catches launch-configuration errors (bad grid/block size, etc.); actual
    // kernel execution errors only surface at the synchronize below.
    if ((err = cudaGetLastError()) != cudaSuccess) return err;

    if ((err = cudaMemcpyAsync(out, m_d_out, bytesPt, cudaMemcpyDeviceToHost, stream)) != cudaSuccess)
        return err;

    // Authoritative check: surfaces any async error from the calls above,
    // including in-kernel faults (e.g. an illegal memory access).
    return cudaStreamSynchronize(stream);
}

}  // namespace gpu
}  // namespace motor_synergy
