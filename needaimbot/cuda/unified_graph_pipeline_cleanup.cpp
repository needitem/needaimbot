#include "unified_graph_pipeline.h"
#include "cuda_resource_manager.h"
#include <iostream>
#include <mutex>

namespace needaimbot {

UnifiedGraphPipeline::~UnifiedGraphPipeline() {
    shutdown();
    
    // Clean up any remaining events that aren't handled in shutdown
    if (m_state.startEvent) {
        cudaEventDestroy(m_state.startEvent);
        m_state.startEvent = nullptr;
    }
    if (m_state.endEvent) {
        cudaEventDestroy(m_state.endEvent);
        m_state.endEvent = nullptr;
    }
    if (m_previewReadyEvent) {
        cudaEventDestroy(m_previewReadyEvent);
        m_previewReadyEvent = nullptr;
    }
    if (m_lastFrameEnd) {
        cudaEventDestroy(m_lastFrameEnd);
        m_lastFrameEnd = nullptr;
    }
    if (m_copyEvent) {
        cudaEventDestroy(m_copyEvent);
        m_copyEvent = nullptr;
    }
}

void UnifiedGraphPipeline::shutdown() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    std::cout << "[UnifiedGraphPipeline] Shutting down..." << std::endl;
    
    // Stop any ongoing processing
    m_state.graphReady = false;
    
    // Free TensorRT bindings memory
    for (auto& [name, ptr] : m_inputBindings) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    m_inputBindings.clear();
    
    for (auto& [name, ptr] : m_outputBindings) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    m_outputBindings.clear();
    
    // Destroy CUDA graph if exists
    if (m_graph) {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
    }
    
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }
    
    // Destroy events are handled in destructor
    
    // Also destroy primary stream if it exists
    if (m_primaryStream) {
        cudaStreamSynchronize(m_primaryStream);
        cudaStreamDestroy(m_primaryStream);
        m_primaryStream = nullptr;
    }
    
    // Clear TensorRT resources (unique_ptr will handle deletion)
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
    
    // Clear all GPU buffers (SimpleCudaMat destructor will handle memory)
    m_captureBuffer.release();
    m_preprocessBuffer.release();
    
    // Clear pipeline components
    m_tripleBuffer.reset();
    
    // Clean up graph and buffers
    cleanupGraph();
    deallocateBuffers();
    
    // Clear D3D11 resources
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    
    // Clear state
    m_state.graphReady = false;
    
    std::cout << "[UnifiedGraphPipeline] Shutdown complete." << std::endl;
}

} // namespace needaimbot