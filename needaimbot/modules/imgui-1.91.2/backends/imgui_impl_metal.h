














#include "imgui.h"      
#ifndef IMGUI_DISABLE





#ifdef __OBJC__

@class MTLRenderPassDescriptor;
@protocol MTLDevice, MTLCommandBuffer, MTLRenderCommandEncoder;


IMGUI_IMPL_API bool ImGui_ImplMetal_Init(id<MTLDevice> device);
IMGUI_IMPL_API void ImGui_ImplMetal_Shutdown();
IMGUI_IMPL_API void ImGui_ImplMetal_NewFrame(MTLRenderPassDescriptor* renderPassDescriptor);
IMGUI_IMPL_API void ImGui_ImplMetal_RenderDrawData(ImDrawData* drawData,
                                                   id<MTLCommandBuffer> commandBuffer,
                                                   id<MTLRenderCommandEncoder> commandEncoder);


IMGUI_IMPL_API bool ImGui_ImplMetal_CreateFontsTexture(id<MTLDevice> device);
IMGUI_IMPL_API void ImGui_ImplMetal_DestroyFontsTexture();
IMGUI_IMPL_API bool ImGui_ImplMetal_CreateDeviceObjects(id<MTLDevice> device);
IMGUI_IMPL_API void ImGui_ImplMetal_DestroyDeviceObjects();

#endif








#ifdef IMGUI_IMPL_METAL_CPP
#include <Metal/Metal.hpp>
#ifndef __OBJC__


IMGUI_IMPL_API bool ImGui_ImplMetal_Init(MTL::Device* device);
IMGUI_IMPL_API void ImGui_ImplMetal_Shutdown();
IMGUI_IMPL_API void ImGui_ImplMetal_NewFrame(MTL::RenderPassDescriptor* renderPassDescriptor);
IMGUI_IMPL_API void ImGui_ImplMetal_RenderDrawData(ImDrawData* draw_data,
                                                   MTL::CommandBuffer* commandBuffer,
                                                   MTL::RenderCommandEncoder* commandEncoder);


IMGUI_IMPL_API bool ImGui_ImplMetal_CreateFontsTexture(MTL::Device* device);
IMGUI_IMPL_API void ImGui_ImplMetal_DestroyFontsTexture();
IMGUI_IMPL_API bool ImGui_ImplMetal_CreateDeviceObjects(MTL::Device* device);
IMGUI_IMPL_API void ImGui_ImplMetal_DestroyDeviceObjects();

#endif
#endif



#endif 

