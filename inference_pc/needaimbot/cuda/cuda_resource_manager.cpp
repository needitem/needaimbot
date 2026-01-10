#include "cuda_resource_manager.h"

// Static member definitions for CUDA compilation compatibility
// CUDA 11.x doesn't fully support inline variables
CudaResourceManager* CudaResourceManager::instance_ = nullptr;
std::mutex CudaResourceManager::instance_mutex_;
std::atomic<bool> CudaResourceManager::shutdown_initiated_{false};
