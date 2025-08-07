#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <memory>
#include <unordered_map>

// Dynamic CUDA Graph Manager with parameter update support
class DynamicCudaGraph {
private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    cudaStream_t stream_ = nullptr;
    
    // Node handles for dynamic updates
    std::unordered_map<std::string, cudaGraphNode_t> kernelNodes_;
    std::unordered_map<std::string, void*> kernelParams_;
    
    // Graph state
    bool isCapturing_ = false;
    bool isInstantiated_ = false;
    
public:
    DynamicCudaGraph(cudaStream_t stream) : stream_(stream) {}
    
    ~DynamicCudaGraph() {
        if (graphExec_) cudaGraphExecDestroy(graphExec_);
        if (graph_) cudaGraphDestroy(graph_);
    }
    
    // Begin graph capture
    cudaError_t beginCapture() {
        if (isCapturing_) return cudaErrorInvalidValue;
        
        cudaError_t err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        if (err == cudaSuccess) {
            isCapturing_ = true;
        }
        return err;
    }
    
    // End graph capture and instantiate
    cudaError_t endCapture() {
        if (!isCapturing_) return cudaErrorInvalidValue;
        
        cudaError_t err = cudaStreamEndCapture(stream_, &graph_);
        if (err != cudaSuccess) {
            isCapturing_ = false;
            return err;
        }
        
        // Get all nodes for later updates
        size_t numNodes;
        cudaGraphGetNodes(graph_, nullptr, &numNodes);
        std::vector<cudaGraphNode_t> nodes(numNodes);
        cudaGraphGetNodes(graph_, nodes.data(), &numNodes);
        
        // Store kernel nodes for dynamic updates
        for (auto node : nodes) {
            cudaGraphNodeType nodeType;
            cudaGraphNodeGetType(node, &nodeType);
            
            if (nodeType == cudaGraphNodeTypeKernel) {
                // Store node for later parameter updates
                // In production, you'd identify nodes by kernel name or ID
                static int nodeId = 0;
                kernelNodes_["kernel_" + std::to_string(nodeId++)] = node;
            }
        }
        
        // Instantiate the graph
        err = cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0);
        if (err == cudaSuccess) {
            isInstantiated_ = true;
        }
        
        isCapturing_ = false;
        return err;
    }
    
    // Update kernel parameters without recreating graph
    cudaError_t updateKernelParams(const std::string& nodeName, 
                                   const cudaKernelNodeParams& params) {
        if (!isInstantiated_) return cudaErrorInvalidValue;
        
        auto it = kernelNodes_.find(nodeName);
        if (it == kernelNodes_.end()) return cudaErrorInvalidValue;
        
        // Update parameters in the executable graph
        return cudaGraphExecKernelNodeSetParams(graphExec_, it->second, &params);
    }
    
    // Launch the graph
    cudaError_t launch() {
        if (!isInstantiated_) return cudaErrorInvalidValue;
        return cudaGraphLaunch(graphExec_, stream_);
    }
    
    // Check if update is needed and update if possible
    bool tryUpdate(cudaGraph_t newGraph) {
        if (!graphExec_ || !newGraph) return false;
        
        cudaGraphExecUpdateResult updateResult;
        cudaGraphNode_t errorNode;
        
        cudaError_t err = cudaGraphExecUpdate(graphExec_, newGraph, 
                                              &errorNode, &updateResult);
        
        if (err == cudaSuccess && updateResult == cudaGraphExecUpdateSuccess) {
            return true;
        }
        
        // If update failed, need to recreate
        if (updateResult == cudaGraphExecUpdateErrorTopologyChanged ||
            updateResult == cudaGraphExecUpdateErrorNodeTypeChanged) {
            cudaGraphExecDestroy(graphExec_);
            cudaGraphInstantiate(&graphExec_, newGraph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph_);
            graph_ = newGraph;
            return true;
        }
        
        return false;
    }
};

// Optimized pipeline with dynamic graph updates
class OptimizedPipelineGraph {
private:
    DynamicCudaGraph preprocessGraph_;
    DynamicCudaGraph inferenceGraph_;
    DynamicCudaGraph postprocessGraph_;
    
    // Kernel parameters that can be updated
    struct PreprocessParams {
        float scaleX, scaleY;
        int srcWidth, srcHeight;
        int dstWidth, dstHeight;
    } preprocessParams_;
    
    struct PostprocessParams {
        float confThreshold;
        float nmsThreshold;
        int maxDetections;
    } postprocessParams_;
    
public:
    OptimizedPipelineGraph(cudaStream_t preprocessStream,
                           cudaStream_t inferenceStream,
                           cudaStream_t postprocessStream)
        : preprocessGraph_(preprocessStream),
          inferenceGraph_(inferenceStream),
          postprocessGraph_(postprocessStream) {}
    
    // Update preprocessing parameters without graph recreation
    void updatePreprocessParams(int srcW, int srcH, int dstW, int dstH) {
        preprocessParams_.srcWidth = srcW;
        preprocessParams_.srcHeight = srcH;
        preprocessParams_.dstWidth = dstW;
        preprocessParams_.dstHeight = dstH;
        preprocessParams_.scaleX = (float)srcW / dstW;
        preprocessParams_.scaleY = (float)srcH / dstH;
        
        // Update kernel parameters in the graph
        cudaKernelNodeParams kernelParams = {0};
        void* kernelArgs[] = {
            &preprocessParams_.scaleX,
            &preprocessParams_.scaleY,
            &preprocessParams_.srcWidth,
            &preprocessParams_.srcHeight,
            &preprocessParams_.dstWidth,
            &preprocessParams_.dstHeight
        };
        kernelParams.func = nullptr; // Set to actual kernel function
        kernelParams.kernelParams = kernelArgs;
        kernelParams.extra = nullptr;
        
        preprocessGraph_.updateKernelParams("preprocess_kernel", kernelParams);
    }
    
    // Update postprocess parameters dynamically
    void updatePostprocessParams(float confThresh, float nmsThresh, int maxDet) {
        postprocessParams_.confThreshold = confThresh;
        postprocessParams_.nmsThreshold = nmsThresh;
        postprocessParams_.maxDetections = maxDet;
        
        cudaKernelNodeParams kernelParams = {0};
        void* kernelArgs[] = {
            &postprocessParams_.confThreshold,
            &postprocessParams_.nmsThreshold,
            &postprocessParams_.maxDetections
        };
        kernelParams.kernelParams = kernelArgs;
        
        postprocessGraph_.updateKernelParams("postprocess_kernel", kernelParams);
    }
    
    // Execute all graphs in pipeline
    void execute() {
        preprocessGraph_.launch();
        inferenceGraph_.launch();
        postprocessGraph_.launch();
    }
};

// Helper function to create optimized graph with dynamic updates
extern "C" void* createOptimizedPipelineGraph(
    cudaStream_t preprocessStream,
    cudaStream_t inferenceStream,
    cudaStream_t postprocessStream)
{
    return new OptimizedPipelineGraph(preprocessStream, inferenceStream, postprocessStream);
}

// Update graph parameters without recreation
extern "C" void updateGraphParameters(
    void* graphHandle,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    float confThreshold,
    float nmsThreshold,
    int maxDetections)
{
    auto* pipeline = static_cast<OptimizedPipelineGraph*>(graphHandle);
    if (pipeline) {
        pipeline->updatePreprocessParams(srcWidth, srcHeight, dstWidth, dstHeight);
        pipeline->updatePostprocessParams(confThreshold, nmsThreshold, maxDetections);
    }
}

// Execute the optimized pipeline
extern "C" void executePipelineGraph(void* graphHandle)
{
    auto* pipeline = static_cast<OptimizedPipelineGraph*>(graphHandle);
    if (pipeline) {
        pipeline->execute();
    }
}

// Destroy the pipeline graph
extern "C" void destroyPipelineGraph(void* graphHandle)
{
    auto* pipeline = static_cast<OptimizedPipelineGraph*>(graphHandle);
    delete pipeline;
}