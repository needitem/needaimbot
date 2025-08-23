#include "unified_graph_pipeline.h"
#include "cuda_resource_manager.h"
#include <iostream>
#include <mutex>

namespace needaimbot {

UnifiedGraphPipeline::~UnifiedGraphPipeline() {
    shutdown();
    // RAII wrappers will automatically clean up events and streams
}

void UnifiedGraphPipeline::shutdown() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    std::cout << "[UnifiedGraphPipeline] Shutting down..." << std::endl;
    
    // Stop any ongoing processing
    m_state.graphReady = false;
    
    // Clear TensorRT bindings - RAII handles memory deallocation
    m_inputBindings.clear();
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
    
    // RAII wrappers automatically handle events and streams cleanup
    m_primaryStream.reset();  // This will synchronize and destroy the stream
    
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