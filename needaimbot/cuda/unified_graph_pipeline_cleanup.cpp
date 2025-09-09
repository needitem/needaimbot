#include "unified_graph_pipeline.h"
#include "cuda_resource_manager.h"
#include <iostream>
#include <mutex>
#include <cuda_runtime.h>
#include <d3d11.h>

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
    
    // Synchronize all GPU operations first
    if (m_pipelineStream) {
        cudaStreamSynchronize(m_pipelineStream->get());
    }
    // cudaDeviceSynchronize() removed for performance - stream sync is sufficient
    
    // Clear TensorRT L2 persistent cache explicitly
    if (m_context) {
        // TensorRT context doesn't directly expose cache control,
        // but we can ensure all async operations are complete
        // and the context is properly synchronized before destruction
        cudaStreamSynchronize(m_pipelineStream ? m_pipelineStream->get() : nullptr);
        
        // The context destructor will handle internal cleanup
        // We rely on proper destruction order below
    }
    
    // Clear TensorRT bindings - RAII handles memory deallocation
    m_inputBindings.clear();
    m_outputBindings.clear();
    
    // Destroy CUDA graph and clear graph cache
    if (m_graphExec) {
        // Synchronize graph execution before destruction
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }
    
    if (m_graph) {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        
        // Clear CUDA graph memory pool to prevent fragmentation
        cudaDeviceGraphMemTrim(0);
    }
    
    // Clear D3D11 resources explicitly before CUDA cleanup
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    
    // Release D3D11 texture references
    if (m_captureTextureD3D) {
        ID3D11Texture2D* texture = static_cast<ID3D11Texture2D*>(m_captureTextureD3D);
        if (texture) {
            texture->Release();
            m_captureTextureD3D = nullptr;
        }
    }
    
    // Release D3D11 context and device references
    if (m_d3dContext) {
        ID3D11DeviceContext* context = static_cast<ID3D11DeviceContext*>(m_d3dContext);
        if (context) {
            context->ClearState();  // Clear all bindings
            context->Flush();       // Flush pending commands
            context->Release();
            m_d3dContext = nullptr;
        }
    }
    
    if (m_d3dDevice) {
        ID3D11Device* device = static_cast<ID3D11Device*>(m_d3dDevice);
        if (device) {
            device->Release();
            m_d3dDevice = nullptr;
        }
    }
    
    // RAII wrappers automatically handle stream cleanup  
    m_pipelineStream.reset();  // Synchronize and destroy pipeline stream
    
    // Clear TensorRT resources with proper cleanup order
    m_context.reset();  // Destroy context first
    m_engine.reset();   // Then engine
    m_runtime.reset();  // Finally runtime
    
    // m_unifiedCaptureBuffer removed - using m_captureBuffer
    
    // Clear preview buffers if allocated
    if (m_preview.enabled) {
        m_preview.previewBuffer.release();
        m_preview.finalTargets.clear();
    }
    
    // Clear capture buffer
    m_captureBuffer.release();
    
    // Clean up graph and buffers
    cleanupGraph();
    deallocateBuffers();
    
    // Clear GPU memory pools but don't reset device
    cudaError_t err = cudaMemPoolTrimTo(nullptr, 0);
    if (err == cudaSuccess) {
        std::cout << "[UnifiedGraphPipeline] GPU memory pools trimmed." << std::endl;
    }
    
    // Clear state
    m_state.graphReady = false;
    
    std::cout << "[UnifiedGraphPipeline] Shutdown complete." << std::endl;
}

} // namespace needaimbot